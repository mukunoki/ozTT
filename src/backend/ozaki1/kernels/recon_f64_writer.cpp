// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: data-movement kernel that turns the result of the reconstruction into FP64 and writes it
// ---------------------------------------------------------------------------
// Assembles the three limbs of the triple-float produced by the compute kernel (c_16 high / c_18 middle /
// c_17 low) into FP64 and writes the result. Only integer arithmetic is used: the RISC-V core of the device
// has no FP64 arithmetic unit (it has only part of the single-precision F extension and no double-precision
// D extension), so code written with double would become soft-float (about 76 ns/op).
//
// The final value is
//     C = (hi + mid + lo) * radix^-(L+2) * s_A * s_B
// where s_A = 2^-dA and s_B = 2^-dB are the divisors actually used in the normalization (dA and dB are the
// values shift-e of the exponent-delta tiles of the rows and of the columns. e is the exponent of 2^e, the
// BF16 representation of the maximum of the row or column (the rounding direction depends on the path; CEIL
// in scale_f64_reader.cpp) raised to a power of two, and s is 2^-shift times that, i.e. 2^(e+1) for the
// default shift=-1). Since they are powers of two, adding -(dA+dB) to the exponent is enough. radix^-(L+2)
// is in general not a power of two (radix 14 of BFP4, 254 of INT8), so for all formats alike it is received
// from the host split into a 64-bit significand W and an exponent qw (W is exactly 2^63 when the radix is a
// power of two). The step that multiplies by W and rounds after the limbs have been gathered is shared with
// the writer of Ozaki scheme II and is pack_f64 in common/kernels/f64_tf.h.
//
// The last four compile-time arguments change the behavior:
//   EMIT_LIMBS  Writes out the three limbs. With K blocking, the reconstruction of the next block reads
//               them and adds them in, so it is set for every block but the last.
//   EMIT_F64    Assembles FP64. This takes several tens of cycles per element, so it is done only for the
//               last block.
//   SF          Measurement mode that assembles with C double expressions (soft-float). Diagnostic selector
//               (not used by the library; always 0). It serves the comparison that measures the grounds for
//               doing the conversion to and from FP64 with bit manipulation in integer instructions.
//   PROBE       Diagnostic selector (the library always passes 0). A nonzero value removes work and
//               breaks the answer: 1 the assembly (the amount written does not change), 2 also the reads
//               of the exponent deltas, 3 also the write.
//
// Runtime arguments (order: WriterRT in args.h; the limbs are in the order hi, lo, mid): hi_addr, num_tiles,
//             start_id, lo_addr, mid_addr, out_addr, dltA_addr, dltB_addr, Nt, Whi, Wlo, qw,
//             Wdlo, Wdhi (wscale of the soft-float version), out_off (output offset of a row band)
// Compile-time arguments: the TensorAccessorArgs in the order of WriterTA in args.h (hi, lo, mid, out, dltA,
//             dltB), followed by the four above (WriterCT)

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/l1_order.h"
#include "../../common/kernels/f64_tf.h"
#include "args.h"
using namespace oz1_args;

constexpr uint32_t cb_hi  = CB_LIMB_HI;
constexpr uint32_t cb_lo  = CB_LIMB_LO;
constexpr uint32_t cb_mid = CB_LIMB_MID;
constexpr uint32_t cb_out = CB_F64;      // work area: FP64 tile (OZ_F64_TILE_BYTES)
constexpr uint32_t cb_da  = CB_DLT_A;    // work area: exponent deltas of the rows
constexpr uint32_t cb_db  = CB_DLT_B;    // work area: exponent deltas of the columns

// Soft-float version for comparison: written plainly with C double expressions. The three widenings, two
// additions and two multiplications amount to 7 soft-float calls. The constant wscale is a double, so this
// path is inferior to the integer version not only in speed but also in accuracy (a relative error of 2^-53
// is added uniformly).
inline uint64_t assemble_softfloat(uint32_t hb, uint32_t mb, uint32_t lb, int32_t g,
                                   uint32_t Wdlo, uint32_t Wdhi) {
    union { float f; uint32_t u; } a, b, c;
    a.u = hb; b.u = mb; c.u = lb;
    union { double d; uint64_t u; } w, s, r;
    w.u = ((uint64_t)Wdhi << 32) | Wdlo;                 // wscale (double)
    const int32_t eb = g + 1023;
    s.u = (eb > 0 && eb < 2047) ? ((uint64_t)(uint32_t)eb << 52) : 0ull;   // 2^g
    r.d = (((double)a.f + (double)b.f) + (double)c.f) * w.d * s.d;
    return r.u;
}

void kernel_main() {
    const uint32_t hi_addr   = get_arg_val<uint32_t>(RW_HI_ADDR);
    const uint32_t num_tiles = get_arg_val<uint32_t>(RW_NUM);
    const uint32_t start_id  = get_arg_val<uint32_t>(RW_START);
    const uint32_t lo_addr   = get_arg_val<uint32_t>(RW_LO_ADDR);
    const uint32_t mid_addr  = get_arg_val<uint32_t>(RW_MID_ADDR);
    const uint32_t out_addr  = get_arg_val<uint32_t>(RW_OUT_ADDR);
    const uint32_t dA_addr   = get_arg_val<uint32_t>(RW_DLT_A_ADDR);
    const uint32_t dB_addr   = get_arg_val<uint32_t>(RW_DLT_B_ADDR);
    const uint32_t Nt        = get_arg_val<uint32_t>(RW_NT);
    const uint32_t Whi       = get_arg_val<uint32_t>(RW_WHI);
    const uint32_t Wlo       = get_arg_val<uint32_t>(RW_WLO);
    const int32_t  qw        = (int32_t)get_arg_val<uint32_t>(RW_QW);
    const uint32_t Wdlo      = get_arg_val<uint32_t>(RW_WDLO);   // used by the soft-float version:
    const uint32_t Wdhi      = get_arg_val<uint32_t>(RW_WDHI);   // bit pattern of wscale as a double
    // RW_OUT_OFF = tile offset on the output side. When the whole output is cut into row bands for the
    // reconstruction (a configuration that avoids holding the accumulation planes of the diagonals for the
    // whole output), the limb buffers are indexed with the local index within the band, whereas the final
    // output and the exponent deltas of the rows and columns must be indexed with the global index. The
    // difference between the two is passed here. In the default configuration, which is not cut into bands,
    // it is 0 and the local and global indices coincide.
    const uint32_t out_off   = get_arg_val<uint32_t>(RW_OUT_OFF);

    // The TensorAccessorArgs are in the order of WriterTA (TA_HI, TA_LO, TA_MID, TA_OUT, TA_DLT_A, TA_DLT_B)
    constexpr auto h_args = TensorAccessorArgs<0>();
    const auto ah = TensorAccessor(h_args, hi_addr);
    constexpr auto l_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    const auto al = TensorAccessor(l_args, lo_addr);
    constexpr auto m_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    const auto am = TensorAccessor(m_args, mid_addr);
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto ao = TensorAccessor(o_args, out_addr);
    constexpr auto da_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto ada = TensorAccessor(da_args, dA_addr);
    constexpr auto db_args = TensorAccessorArgs<da_args.next_compile_time_args_offset()>();
    const auto adb = TensorAccessor(db_args, dB_addr);
    static_assert(TA_COUNT == 6, "the count of WriterTA does not match the TensorAccessorArgs above");
    // The order above (hi -> lo -> mid -> out -> dltA -> dltB) is written by hand, so it is also checked
    // against the order of the indices in the table (recon_build_prog on the host arranges them by indexing
    // with the names of the table. Swapping two entries of the table keeps the count at 6 and silently
    // writes the limbs to the wrong planes)
    static_assert(TA_HI == 0 && TA_LO == 1 && TA_MID == 2 && TA_OUT == 3 && TA_DLT_A == 4 && TA_DLT_B == 5,
                  "the order of WriterTA does not match the order of the TensorAccessorArgs above");
    // The last four compile-time arguments (WriterCT): whether to write the limbs (for the accumulation
    // over K blocks), whether to assemble FP64 (last block only), whether to use the soft-float version
    // (for measurement), and PROBE (diagnostic)
    constexpr uint32_t CT0 = db_args.next_compile_time_args_offset();
    constexpr uint32_t EMIT_LIMBS = get_compile_time_arg_val(CT0 + RW_CT_EMIT_LIMBS);
    constexpr uint32_t EMIT_F64   = get_compile_time_arg_val(CT0 + RW_CT_EMIT_F64);
    constexpr uint32_t SF         = get_compile_time_arg_val(CT0 + RW_CT_SF);
    constexpr uint32_t PROBE      = get_compile_time_arg_val(CT0 + RW_CT_PROBE);

    // Work areas (not pushed or popped; the write position is fixed and reused)
    const uint32_t op = get_write_ptr(cb_out);
    const uint32_t ap = get_write_ptr(cb_da);
    const uint32_t bp = get_write_ptr(cb_db);

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_hi, 1);
        cb_wait_front(cb_mid, 1);
        cb_wait_front(cb_lo, 1);
        // The three limbs are kept only when the next K block reads them
        if constexpr (EMIT_LIMBS) {
            noc_async_write_tile(i, ah, get_read_ptr(cb_hi));
            noc_async_write_tile(i, am, get_read_ptr(cb_mid));
            noc_async_write_tile(i, al, get_read_ptr(cb_lo));
        }
        if constexpr (!EMIT_F64) {
            noc_async_write_barrier();
            cb_pop_front(cb_hi, 1); cb_pop_front(cb_mid, 1); cb_pop_front(cb_lo, 1);
            continue;
        }
        // Fetch the exponent deltas (shift-e) of the row and of the column (indexed with the global index)
        const uint32_t gi = i + out_off;
        if constexpr (PROBE < 2) {   // PROBE>=2 removes these reads
            noc_async_read_tile(gi / Nt, ada, ap);
            noc_async_read_tile(gi % Nt, adb, bp);
            noc_async_read_barrier();
        }
        // This RISC-V core reads the limbs (cb_hi, cb_mid, cb_lo) written by the compute kernel (the packer),
        // so the stale contents of L0 are discarded. cb_wait_front does not discard L0. The
        // noc_async_read_barrier above also discards it internally, but relying on that would silently lose
        // the precondition under a change that omits the reads of the exponent deltas (PROBE>=2, or an
        // optimization that reuses the read for a whole row), so it is done explicitly here.
        invalidate_l1_cache();

        volatile tt_l1_ptr uint32_t* ph = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_hi);
        volatile tt_l1_ptr uint32_t* pm = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_mid);
        volatile tt_l1_ptr uint32_t* pl = (volatile tt_l1_ptr uint32_t*)get_read_ptr(cb_lo);
        volatile tt_l1_ptr int32_t*  da = (volatile tt_l1_ptr int32_t*)ap;
        volatile tt_l1_ptr int32_t*  db = (volatile tt_l1_ptr int32_t*)bp;
        volatile tt_l1_ptr uint32_t* po = (volatile tt_l1_ptr uint32_t*)op;
        if constexpr (PROBE == 0) {
            for (uint32_t j = 0; j < 1024; ++j) {
                // The delta tiles hold shift-e, so the exponent to multiply back is -(dA+dB)
                const int32_t g = -(da[j] + db[j]);
                uint64_t u;
                if constexpr (SF) u = assemble_softfloat(ph[j], pm[j], pl[j], g, Wdlo, Wdhi);
                else              u = assemble_tf_f64(ph[j], pm[j], pl[j], g, Whi, Wlo, qw);
                po[2 * j]     = (uint32_t)u;        // little endian: the low word comes first
                po[2 * j + 1] = (uint32_t)(u >> 32);
            }
        } else {
            // The assembly is removed. The limb CBs are only waited for and popped (to keep the dependency
            // on the compute kernel) and their contents are not read, so the reads of the limbs and of the
            // exponent deltas from L1 are removed together with the assembly. The output is filled with a
            // constant. The amount written is one tile (OZ_F64_TILE_BYTES), the same as with PROBE==0.
            (void)ph; (void)pm; (void)pl; (void)da; (void)db;
            for (uint32_t j = 0; j < 1024; ++j) { po[2 * j] = 0u; po[2 * j + 1] = 0u; }
        }
        if constexpr (PROBE < 3) {   // PROBE==3 also removes the write
            // Let the NoC read the assembled FP64 only after it has landed in L1 (l1_order.h)
            l1_writes_landed(op + OZ_F64_TILE_BYTES - 4u);
            noc_async_write_tile(i + out_off, ao, op);
            noc_async_write_barrier();
        }
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_mid, 1);
        cb_pop_front(cb_lo, 1);
    }
}
