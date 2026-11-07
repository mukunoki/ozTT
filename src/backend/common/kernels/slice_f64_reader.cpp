// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernel that reads the input of the slicer (FP64 input version; shared by Ozaki schemes I
// and II)
// ---------------------------------------------------------------------------
// The host sends the FP64 operands as they are (8-byte elements) in tile order.
// This kernel performs the FP64 -> triple-float conversion and hands the result to the compute kernel.
// The conversion uses integer operations only: the RISC-V of the device has no FP64 arithmetic unit (even
// the single-precision F extension is only partly implemented, and there is no double-precision D
// extension; Blackhole ISA documentation, BabyRISCV/InstructionSet.md), so code written with double would
// become soft-float (about 76 ns per operation).
//
// How FP64 is split into a triple-float, and why the normalization can be folded into the exponent, are
// explained in the shared header common/kernels/f64_tf.h (the reader of Ozaki scheme II uses the same
// header).
//
// The normalization (per-row / per-column power-of-two scale) is fused here, in the form of adding the
// integer delta d to the exponent field of the FP64 value. Therefore no tile-wise multiplication by a
// scale tile (2^-e in FP32) is needed, and there is no constraint that the scale must fit in the
// exponent range of FP32 (the normalized value satisfies |x/tau| <= 1 in scheme I and, including the
// digit alignment, is at most 2^20 in scheme II (the entry condition of sfpu_tf_proc.h), so the limbs
// fit in FP32 in both).
//
// With the compile-time argument SPLIT set, the second half of each tile (512 elements) is split by the
// writer of the same core (slice_writer.cpp) (slice_f64_split.h). When a single RISC-V does all the
// splitting, that work determines the time of the slicing stage of scheme I, so sharing it between two
// RISC-Vs shortens the stage. In scheme II the time is determined by the SFPU operations of the slicer
// and the writer is a different kernel (modslice_writer.cpp), so SPLIT is not set there.
//
// A soft-float version is kept for comparison (compile-time argument SF). It is written plainly with C
// double expressions and makes 8 libgcc soft-float calls per element.
// It exists only to show how much slower the conversion is without integer bit operations; SF is a
// diagnostic selector (not used by the library; always 0).
//
// Runtime arguments: f64_addr, dlt_addr, num_tiles, start_id, dlt_stride, mode (the layout and the CB
// indices are in slice_f64_args.h; the host fills them with f64_reader_ct / f64_reader_rt in
// common/slice_f64_host.hpp)

#include "api/dataflow/dataflow_api.h"
#include "f64_tf.h"
#include "slice_f64_split.h"
#include "l1_order.h"
#include "slice_f64_args.h"
using namespace slice_f64_args;

constexpr uint32_t cb_h  = CB_HI;      // hi
constexpr uint32_t cb_l  = CB_CRAW;    // craw
constexpr uint32_t cb_s  = CB_SCALE;   // s (scale of the low limb)
constexpr uint32_t cb_m  = CB_BRAW;    // braw
constexpr uint32_t cb_f  = CB_F64;     // work area: FP64 tile (OZ_F64_TILE_BYTES)
constexpr uint32_t cb_d  = CB_DLT;     // work area: exponent-delta tile (Int32)

// Soft-float version: the plain procedure that computes x*2^d in double and splits it into three terms by
// rounding to float and taking differences, written with C double expressions.
// The single normalization x*2^d and the split into the three terms take 8 soft-float operations in total.
inline void split_softfloat(uint32_t w1, uint32_t w0, int32_t d,
                            uint32_t& hb, uint32_t& mb, uint32_t& lb) {
    union { double d; uint64_t u; } xu;  xu.u = ((uint64_t)w1 << 32) | w0;
    union { double d; uint64_t u; } su;                 // 2^d (normalization factor)
    const int32_t eb = d + 1023;
    su.u = (eb > 0 && eb < 2047) ? ((uint64_t)(uint32_t)eb << 52) : 0ull;
    const double x   = xu.d * su.d;                     // 1: muldf3
    const float  hi  = (float)x;                        // 2: truncdfsf2
    const double rem = x - (double)hi;                  // 3,4: extendsfdf2, subdf3
    const float  mid = (float)rem;                      // 5: truncdfsf2
    const float  lo  = (float)(rem - (double)mid);      // 6,7,8: extend, sub, trunc
    union { float f; uint32_t u; } c;
    c.f = hi;  hb = c.u;
    c.f = mid; mb = c.u;
    c.f = lo;  lb = c.u;
}

void kernel_main() {
    const uint32_t f64_addr   = get_arg_val<uint32_t>(R_F64_ADDR);
    const uint32_t dlt_addr   = get_arg_val<uint32_t>(R_DLT_ADDR);
    const uint32_t num_tiles  = get_arg_val<uint32_t>(R_NUM);
    const uint32_t start_id   = get_arg_val<uint32_t>(R_START);
    const uint32_t dlt_stride = get_arg_val<uint32_t>(R_DLT_STRIDE);
    const uint32_t mode       = get_arg_val<uint32_t>(R_MODE);

    constexpr auto f_args = TensorAccessorArgs<0>();
    const auto fa = TensorAccessor(f_args, f64_addr);
    constexpr auto d_args = TensorAccessorArgs<f_args.next_compile_time_args_offset()>();
    const auto da = TensorAccessor(d_args, dlt_addr);
    // 1 selects the soft-float version (diagnostic selector; not used by the library, always 0).
    // The default 0 is the integer bit-operation version.
    constexpr uint32_t SF = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + CT_SF);
    // 1 lets the writer split the second half (explained above; not used with the soft-float version)
    constexpr uint32_t SPLIT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + CT_SPLIT) && !SF;

    // c_5 / c_6 are work areas that are not pushed/popped; the write position is fixed and reused (the CBs
    // are created by the host). With SPLIT=1 the writer also reads the second half from the
    // addresses it received through the mailbox, so they may be overwritten only after the writer's pop
    // (the reserve below; slice_f64_split.h).
    const uint32_t fp = get_write_ptr(cb_f);
    const uint32_t dp = get_write_ptr(cb_d);

    // For A (row direction) the exponent-delta tile is common to the Kt tiles of the same tile row, so it
    // is read only when it changes (this reduces the amount read; the stage is not DRAM-bound).
    uint32_t last_vi = 0xFFFFFFFFu;
    const uint32_t end_id = start_id + num_tiles;
    if constexpr (SPLIT && MBOX_PREFILL)   // probe 2 only (slice_f64_split.h)
        mbox_write_job((volatile tt_l1_ptr SliceJob*)get_write_ptr(SLICE_MBOX_CB), fp, dp, cb_h, cb_m, cb_l, cb_s);
    for (uint32_t i = start_id; i < end_id; ++i) {
        noc_async_read_tile(i, fa, fp);
        // There is one exponent-delta tile per row (A) or per column (B), so it is looked up from the tile
        // index
        const uint32_t vi = (mode == 0) ? (i / dlt_stride) : (i % dlt_stride);
        if (vi != last_vi) { noc_async_read_tile(vi, da, dp); last_vi = vi; }
        noc_async_read_barrier();

        cb_reserve_back(cb_h, 1); cb_reserve_back(cb_m, 1);
        cb_reserve_back(cb_l, 1); cb_reserve_back(cb_s, 1);
        if constexpr (SPLIT) {
            // Put the job into the mailbox (the writer has already popped the previous job: the reserve
            // below waited for it), then split the first half
            cb_reserve_back(SLICE_MBOX_CB, 1);
            volatile tt_l1_ptr SliceJob* job = (volatile tt_l1_ptr SliceJob*)get_write_ptr(SLICE_MBOX_CB);
            mbox_reader_before_write();
            // Notify only after the job contents have landed in L1 (a fence alone does not mean they have
            // landed; l1_order.h)
            mbox_write_job(job, fp, dp, cb_h, cb_m, cb_l, cb_s);
            mbox_reader_notify();
            slice_split_range(fp, dp, get_write_ptr(cb_h), get_write_ptr(cb_m),
                              get_write_ptr(cb_l), get_write_ptr(cb_s), 0, SLICE_SPLIT_J);
            // Wait until the writer has finished splitting the second half and returns the mailbox. The
            // writer pops only after the second half has landed in L1 (slice_writer.cpp), so once the
            // mailbox is back the second half is in L1 as well.
            // The returned page is used as is for the job of the next tile.
            cb_reserve_back(SLICE_MBOX_CB, 1);
            // Hand over to the slicer only after this core's own first half has landed in L1 too (the last
            // write is element 511 of s)
            l1_writes_landed(get_write_ptr(cb_s) + 4u * (SLICE_SPLIT_J - 1));
            cb_push_back(cb_h, 1); cb_push_back(cb_m, 1);
            cb_push_back(cb_l, 1); cb_push_back(cb_s, 1);
            continue;
        }
        const uint32_t ohp = get_write_ptr(cb_h), omp = get_write_ptr(cb_m), olp = get_write_ptr(cb_l),
                       osp = get_write_ptr(cb_s);
        if constexpr (SF) {                     // diagnostic: C double expressions (soft-float)
            volatile tt_l1_ptr uint32_t* src = (volatile tt_l1_ptr uint32_t*)fp;
            volatile tt_l1_ptr int32_t*  dl  = (volatile tt_l1_ptr int32_t*)dp;
            volatile tt_l1_ptr uint32_t* oh  = (volatile tt_l1_ptr uint32_t*)ohp;
            volatile tt_l1_ptr uint32_t* om  = (volatile tt_l1_ptr uint32_t*)omp;
            volatile tt_l1_ptr uint32_t* ol  = (volatile tt_l1_ptr uint32_t*)olp;
            volatile tt_l1_ptr uint32_t* os  = (volatile tt_l1_ptr uint32_t*)osp;
            for (uint32_t j = 0; j < 1024; ++j) {
                const uint32_t w0 = src[2 * j];     // little endian: low word first
                const uint32_t w1 = src[2 * j + 1];
                uint32_t hi, braw, craw;
                split_softfloat(w1, w0, dl[j], hi, braw, craw);
                // s is not used, but it is written so that the output has the same shape
                oh[j] = hi; om[j] = braw; ol[j] = craw; os[j] = 0x3F800000u;
            }
        } else {
            // Split the whole tile with integer bit operations (the same procedure as the first half and
            // the writer's second half of the shared version; slice_f64_split.h)
            slice_split_range(fp, dp, ohp, omp, olp, osp, 0, 1024);
        }
        // Hand over to the slicer only after the four tiles have landed in L1 (the last write is element
        // 1023 of s; l1_order.h)
        l1_writes_landed(osp + 4u * 1023);
        cb_push_back(cb_h, 1); cb_push_back(cb_m, 1);
        cb_push_back(cb_l, 1); cb_push_back(cb_s, 1);
    }
}
