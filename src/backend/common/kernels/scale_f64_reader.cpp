// SPDX-License-Identifier: Apache-2.0
//
// Scale computation: data-movement kernel that finds the maximum absolute value of each row / column on a
// RISC-V core, working directly on the FP64 bit patterns
// ---------------------------------------------------------------------------
// Role: reads the FP64 operand from DRAM and, for each output tile (32 rows for A, 32 columns for B),
// goes through the Kt tiles along K to compute
//     sA[m] = max_k |A[m,k]|   (MODE 0, per row)
//     sB[n] = max_k |B[k,n]|   (MODE 1, per column)
// and writes their BF16 representation to DRAM as a tile. expdelta.cpp turns it into the delta that is
// added to the exponent.
// The same kernel is placed on both data-movement RISC-Vs of one core (RISCV_0 and RISCV_1), and each
// takes half of the output tiles assigned to the core (scale_build on the host; the work-area CBs are
// separate).
//
// The maximum is taken in FP64 and only the 32 outputs are converted to BF16; neither a compute kernel nor
// a writer is used.
//
// Comparison and conversion
//   The maximum of |x| is taken in FP64, and the BF16 conversion f (round down, or round up with CEIL
//   below) is applied only once, to the maximum.
//   |x| values are compared as unsigned integers on the 63 bits without the sign bit, high word first and
//   then low word (for finite non-negative FP64 values the order of the bit patterns equals the order of
//   the values; NaN sorts above inf, but f maps both to the same marker 0xFFC0). The low word is compared
//   as well because the CEIL decision looks at the dropped significand bits, which include the low word.
//   f maps a nonzero maximum below 2^-126 to the out-of-range marker 0xFF80 ("Accepted magnitude range"
//   below). 0xFF80 preserves the order neither as a BF16 value nor as a uint16, so f is not monotonic and
//   cannot be applied before the comparison; because the maximum is taken in FP64 and f is applied once
//   afterwards, 0xFF80 never enters a comparison with other elements.
//
// Rounding up (CEIL)
//   With the compile-time argument CEIL set, the BF16 value is moved up by one when any dropped bit is
//   nonzero.
//   In Ozaki scheme II, a normalized value above 1 violates |A'| <= 2^b and leaves the CRT window, so
//   the maximum must never be below the true value. In Ozaki scheme I, rounding down is enough for the
//   formats whose expansion bound R = d/(beta-1) exceeds the target by at least 1/128; only INT8
//   (R = 127/253) and BFP2 (R = 1), which have little margin, round up (bf_ceil in src/ozaki1.cpp).
//   Rounding up acts only on maxima that fall in the normalized range of BF16.
//   CEIL=2 does what CEIL=1 does and also moves the value up by one when the maximum is exactly a power
//   of two (all significand bits 0). The BF16 value itself is at least the maximum but not necessarily
//   strictly larger (1.5 stays 1.5). What becomes strictly larger than the maximum is the scale 2^e that
//   expdelta.cpp derives from it as e = ceil(log2(.)): 2^e is the smallest power of two strictly larger
//   than the maximum, 2^(floor(log2 max)+1) (the scale differs from CEIL=1 only in the rows / columns
//   whose maximum is exactly a power of two, where it is one step higher).
//   It is used by BFP2 (R = target = 1), which cannot represent the target value 2^nshift itself with
//   a single digit: when the maximum element is exactly the target, its expansion never terminates, with
//   digit 1 and residual 1 at every step (tau_above in src/ozaki1.cpp). A row / column whose maximum is
//   just below a power of two (1-delta after normalization) also starts with a run of about log2(1/delta)
//   digits 1, but the run is finite, so it is handled by the slice count p and not here in the scale (the
//   default p = 59 of BFP2 is chosen accordingly).
//
// Accepted magnitude range (the limits imposed by this path)
//   The host (check_values in src/context.cpp) rejects NaN and infinity and every nonzero row / column
//   maximum outside [2^-126, largest finite BF16] before the device runs, so the markers and flags below
//   are defensive and are not reached through the public API.
//   Upper end: an element of 2^128 or more makes the maximum inf (with CEIL set, rounding up produces
//   inf as soon as the maximum exceeds the largest finite BF16 value 2^128 - 2^120).
//   expdelta.cpp sees the exponent 255 and raises a flag, and expdelta_check stops after C has been
//   read back, so no wrong answer is returned (the caller is expected to divide the whole operand by a
//   power of two before passing it). The upper limit of FP64 is 2^1024, so this range is narrower than
//   the "finite and normalized FP64" stated in f64_tf.h.
//   Inf / NaN (FP64 exponent 2047) sort above every finite value in the comparison of the high word
//   without its sign, so they remain as the maximum of the row / column. Dividing by a power of two does
//   not cure this, so a marker different from the upper-end inf is written, 0xFFC0 (sign 1, exponent 255,
//   significand 0x40; distinguished from the lower-end marker 0xFF80 by the significand), and
//   expdelta.cpp raises a separate flag (bit 2).
//   Lower end: a nonzero maximum below 2^-126 cannot be represented as a normalized BF16 number. If it
//   were turned into 0, a row (column) whose elements are all that small would get the maximum 0;
//   expdelta.cpp could not tell it from an all-zero row and would use the scale 2^0, all slices would
//   be 0, and the row would silently come out as zero (with A = 2^-127, B = 2^127, K = 64 every element
//   of the true result is 64, yet the output would be 0; a small input does not imply a small
//   contribution to the product).
//   Therefore, as at the upper end, the case is flagged as out of range and stopped: when the maximum is
//   nonzero (the high word or the low word is nonzero; FP64 subnormals included) and below 2^-126, the
//   marker 0xFF80 (sign 1, exponent 255), which never appears as the BF16 of an |x|, is written;
//   expdelta.cpp distinguishes it from the upper-end inf (0x7F80) by the sign bit and raises a separate
//   flag. After C has been read back, expdelta_check (common/scale.hpp) stops and reports the reason (the
//   caller is expected to multiply the whole operand by a power of two before passing it). Elements that
//   are small in only part of a row do not determine the maximum, so they do not hit this limit.
//   An all-zero row (column) stays 0 and raises no flag.
//
// Reading
//   Two work areas of one FP64 tile each (OZ_F64_TILE_BYTES) are kept; tile kt+1 is read while tile kt is
//   being examined.
//
// Shape of the output tile: expdelta.cpp reads only column 0 of each row, nf(k,0), for A, and each
// column of row 0, nf(0,k), for B (the same positions where the reduction of the matrix engine places
// its result). The other elements are set to 0.
//
// Compile-time arguments (layout: ReaderCT in scale_args.h): the TensorAccessor arguments of the input and
//   the output, CEIL (0 round down, 1 round up, 2 round up and move a power of two one step up), MODE
//   (0 = rows of A, 1 = columns of B), the two work-area CBs (two FP64 tiles, one BF16 result tile)
// Runtime arguments (ReaderRT in the same file): in_addr, out_addr, nout, Kt, stride, start_o (Kt >= 1;
//   scale_build on the host rejects empty matrices. With Kt=0 the kernel would end with tile 0
//   prefetched and without waiting for the read to complete)
//   The input tile index is og*Kt+kt in MODE 0 and kt*stride+og in MODE 1 (og = start_o + o).
//   The output tile index is og.

#include "api/dataflow/dataflow_api.h"
#include "l1_order.h"
#include "tile_nf.h"   // (row, column) -> running index nf within a tile
#include "f64_tf.h"    // OZ_F64_TILE_BYTES
#include "scale_args.h"
#include "scale_bits.h"   // oz_absf64_to_bf16 (shared with the host model)

// Updates (mh, ml) with the maximum |x| of 16 elements (st words apart). The maximum hm of the high words
// of |x| (sign bit cleared) is taken first (one maxu instruction each; there is no branch, so the loads
// can run ahead), and only when hm reaches the current maximum are the low words of the elements whose
// high word equals hm examined (most rows end at the first comparison).
// The result equals the maximum of the elements under lexicographic comparison of (high word, low word).
inline void upd16(const uint32_t* e, uint32_t st, uint32_t& mh, uint32_t& ml) {
    uint32_t hm = 0;
#pragma GCC unroll 16
    for (uint32_t k = 0; k < 16; ++k) {                // little endian: low word first, high word second
        const uint32_t h = e[st * k + 1] & 0x7FFFFFFFu;
        hm = (h > hm) ? h : hm;
    }
    if (hm < mh) return;
    uint32_t lm = (hm == mh) ? ml : 0u;
    for (uint32_t k = 0; k < 16; ++k)
        if ((e[st * k + 1] & 0x7FFFFFFFu) == hm) { const uint32_t l = e[st * k]; lm = (l > lm) ? l : lm; }
    mh = hm; ml = lm;
}

// The conversion of an FP64 |x| (high word h, low word l) to BF16 is oz_absf64_to_bf16 in scale_bits.h
// (the host model that selects the slice count from the input, ozaki1/auto_precision.hpp, uses the same
// function).

void kernel_main() {
    using namespace scale_args;
    const uint32_t in_addr  = get_arg_val<uint32_t>(R_IN_ADDR);
    const uint32_t out_addr = get_arg_val<uint32_t>(R_OUT_ADDR);
    const uint32_t nout     = get_arg_val<uint32_t>(R_NOUT);
    const uint32_t Kt       = get_arg_val<uint32_t>(R_KT);
    const uint32_t stride   = get_arg_val<uint32_t>(R_STRIDE);
    const uint32_t start_o  = get_arg_val<uint32_t>(R_START_O);

    constexpr auto i_args = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(i_args, in_addr);
    constexpr auto o_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(o_args, out_addr);
    constexpr uint32_t CT0  = o_args.next_compile_time_args_offset();
    constexpr uint32_t CEIL = get_compile_time_arg_val(CT0 + CT_CEIL);
    constexpr uint32_t MODE = get_compile_time_arg_val(CT0 + CT_MODE);
    // The same kernel is placed on two RISC-Vs, so each one receives its own work-area CBs:
    // cb_f holds two FP64 tiles (OZ_F64_TILE_BYTES each), cb_o one BF16 result tile
    constexpr uint32_t cb_f = get_compile_time_arg_val(CT0 + CT_CB_F64);
    constexpr uint32_t cb_o = get_compile_time_arg_val(CT0 + CT_CB_OUT);
    const uint32_t op = get_write_ptr(cb_o);

    // Two work areas (reused without push/pop)
    const uint32_t fb0 = get_write_ptr(cb_f);
    const uint32_t fbuf[2] = {fb0, fb0 + OZ_F64_TILE_BYTES};

    uint32_t mh[32], ml[32];
    for (uint32_t o = 0; o < nout; o++) {
        const uint32_t og = start_o + o;
        for (uint32_t k = 0; k < 32; ++k) { mh[k] = 0u; ml[k] = 0u; }
        auto tile_id = [&](uint32_t kt) { return (MODE == 0) ? (og * Kt + kt) : (kt * stride + og); };
        noc_async_read_tile(tile_id(0), in, fbuf[0]);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            noc_async_read_barrier();                        // tile kt has arrived
            if (kt + 1 < Kt) noc_async_read_tile(tile_id(kt + 1), in, fbuf[(kt + 1) & 1u]);
            // The compiler cannot see the completion of the read, so it must not move L1 loads before here
            asm volatile("" ::: "memory");
            const uint32_t* src = reinterpret_cast<const uint32_t*>(fbuf[kt & 1u]);
            for (uint32_t f = 0; f < 4; ++f) {               // for each 16x16 face
                const uint32_t* fp = src + 512u * f;         // one face is 256 elements = 512 words
                if constexpr (MODE == 0) {                   // the 16 elements of row (f/2)*16 + rr
                    const uint32_t rb = (f >> 1) * 16u;
                    for (uint32_t rr = 0; rr < 16; ++rr) {
                        // one row of the face (16 consecutive elements)
                        upd16(fp + 32u * rr, 2u, mh[rb + rr], ml[rb + rr]);
                    }
                } else {                                     // the 16 elements of column (f%2)*16 + cc
                    const uint32_t cb = (f & 1u) * 16u;
                    for (uint32_t cc = 0; cc < 16; ++cc) {
                        // one column of the face (elements 16 apart)
                        upd16(fp + 2u * cc, 32u, mh[cb + cc], ml[cb + cc]);
                    }
                }
            }
            // Finish the L1 loads above before the next iteration starts a read into this work area
            asm volatile("" ::: "memory");
        }
        volatile tt_l1_ptr uint32_t* z = (volatile tt_l1_ptr uint32_t*)op;
        for (uint32_t i = 0; i < 512; ++i) z[i] = 0u;        // zero the 1024 BF16 elements
        volatile tt_l1_ptr uint16_t* dst = (volatile tt_l1_ptr uint16_t*)op;
        for (uint32_t k = 0; k < 32; ++k)
            dst[(MODE == 0) ? nf(k, 0) : nf(0, k)] = oz_absf64_to_bf16<CEIL>(mh[k], ml[k]);
        // Let the NoC read only after the written values have landed in L1 (l1_order.h; reads back the last
        // word written)
        l1_writes_landed((op + 2u * ((MODE == 0) ? nf(31, 0) : nf(0, 31))) & ~3u);
        noc_async_write_tile(og, out, op);
        // Wait before the work tile is rewritten for the next output
        noc_async_write_barrier();
    }
}
