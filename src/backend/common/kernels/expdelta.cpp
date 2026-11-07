// SPDX-License-Identifier: Apache-2.0
//
// Finishing step of the scaling: builds the "delta added to the exponent" tiles from the tiles of maxima
// ---------------------------------------------------------------------------
// Role: from the maximum mx (BF16) produced by the per-row / per-column reduction, computes
//         e = ceil(log2(mx))   with e = 0 when mx <= 0
//         d = shift - e
//       and writes an Int32 tile in which d is broadcast over the whole tile.
//       The slicer adds this d, as an integer, to the exponent field of the FP64 values.
//       mx is the FP64 maximum converted to BF16 by scale_f64_reader.cpp, and the rounding direction
//       (round down / round up / move a power of two one step up; CEIL in that file) is decided there.
//       This kernel only looks at the BF16 of mx.
//
// By design, this implementation does not communicate with the host between the transfer of A and B
// and the return of C, so this finishing step (taking ceil(log2(.)) of the maxima and expanding it into
// tiles) is also done on the device, without reading anything back to the host.
//
// No log2 is needed: ceil(log2(mx)) is obtained with integer operations from the exponent and the
// significand of the BF16 (the derivation and the treatment of 0 and subnormals are in oz_bf16_scale_exp
// in scale_bits.h; the host model that selects the slice count from the input uses the same function).
// E=255 is an out-of-range marker, and a flag is raised so that the later stages do not silently return
// a wrong answer. Three cases are distinguished by the sign bit and the significand:
// 0x7F80 (inf): the maximum exceeds the BF16 range (about 2^128) (flag bit 0); 0xFF80: the nonzero
// maximum is below 2^-126 (flag bit 1); 0xFFC0: the row / column has an Inf / NaN element (flag bit 2)
// (oz_absf64_to_bf16 in scale_bits.h; the BF16 of an |x| has sign 0, so the latter two never coincide
// with an ordinary value).
// The flag word is the bitwise OR, and expdelta_check on the host reports the reasons separately.
//
// There are only M+N values to process, so the SFPU is not used and the tiles are built directly on the
// RISC-V.
//
// The layout of the compile-time arguments, runtime arguments and CB indices is expdelta_args in
// scale_args.h (row direction or not; the buffers of the maxima, the output and the flags; the number of
// tiles handled, the first tile, shift, and the flag slot (core index))
#include "api/dataflow/dataflow_api.h"
#include "l1_order.h"
#include "tile_nf.h"   // (row, column) -> running index nf within a tile
#include "scale_args.h"
#include "scale_bits.h"   // oz_bf16_scale_exp (shared with the host model)
using namespace expdelta_args;

constexpr uint32_t ROWWISE = get_compile_time_arg_val(CT_ROWWISE);

void kernel_main() {
    const uint32_t max_addr = get_arg_val<uint32_t>(R_MAX_ADDR);
    const uint32_t out_addr = get_arg_val<uint32_t>(R_OUT_ADDR);
    const uint32_t flg_addr = get_arg_val<uint32_t>(R_FLAG_ADDR);
    const uint32_t num      = get_arg_val<uint32_t>(R_NUM);
    const uint32_t start    = get_arg_val<uint32_t>(R_START);
    const int32_t  shift    = (int32_t)get_arg_val<uint32_t>(R_SHIFT);
    const uint32_t slot     = get_arg_val<uint32_t>(R_SLOT);   // flag slot dedicated to this core

    constexpr uint32_t cb_in = CB_MAX;    // tile of maxima (BF16, 2048 B)
    constexpr uint32_t cb_ot = CB_OUT;    // output tile (Int32, 4096 B)
    constexpr auto m_args = TensorAccessorArgs<CT_COUNT>();
    const auto am = TensorAccessor(m_args, max_addr);
    constexpr auto o_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    const auto ao = TensorAccessor(o_args, out_addr);
    constexpr auto f_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    const auto af = TensorAccessor(f_args, flg_addr);

    const uint32_t ip = get_write_ptr(cb_in), op = get_write_ptr(cb_ot);
    volatile tt_l1_ptr uint16_t* mx = (volatile tt_l1_ptr uint16_t*)ip;
    volatile tt_l1_ptr int32_t*  dt = (volatile tt_l1_ptr int32_t*)op;
    uint32_t bad = 0;

    for (uint32_t t = start; t < start + num; ++t) {
        noc_async_read_tile(t, am, ip);
        noc_async_read_barrier();
        for (uint32_t k = 0; k < 32; ++k) {
            // When out of range, a flag (upper end 1, lower end 2, Inf/NaN 4) is ORed into bad and e = 0
            const int32_t e = oz_bf16_scale_exp(mx[ROWWISE ? nf(k, 0) : nf(0, k)], bad);
            const int32_t d = shift - e;
            for (uint32_t j = 0; j < 32; ++j)
                dt[ROWWISE ? nf(k, j) : nf(j, k)] = d;
        }
        // The last element written. The NoC reads only after the writes have landed in L1 (l1_order.h)
        l1_writes_landed(op + 4u * nf(31, 31));
        noc_async_write_tile(t, ao, op);
        noc_async_write_barrier();
    }
    // Each core raises its flag in a tile of its own. The host looks at all slots after it has read C
    // (outside the window without host communication). No tile is written by more than one core, so
    // nothing depends on the behavior of overlapping NoC writes.
    if (bad) {
        dt[0] = (int32_t)bad;
        l1_writes_landed(op);
        noc_async_write_tile(slot, af, op);
        noc_async_write_barrier();
    }
}
