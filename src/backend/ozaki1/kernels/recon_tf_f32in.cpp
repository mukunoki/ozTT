// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: reconstruction on the device (non-deferred path that reads FP32 partial products,
// triple-float)
// ---------------------------------------------------------------------------
// The FP formats normally accumulate the partial products in Int32 inside the GEMM (pair loop or fused
// writer) and reconstruct once at the end with recon_tf_i32in.cpp (deferred reconstruction). This kernel
// serves the configurations in which that is not possible: several K blocks with an inner dimension so long
// that the Int32 accumulation could exceed 2^31 (oz_defer_fits in ozaki1/format.hpp). The Horner evaluation
// then runs for every K block, and the result is carried from one K block to the next with ACC.
// Role: adds up the partial sums P_d of the diagonals with their weights and recovers the original
//       high-precision product.
//
//   S = sum_d radix^(L-d) P_d = (((P_0)*radix + P_1)*radix + ...) + P_L
//
//   This Horner accumulation is carried out with a triple of FP32 values (triple-float, working precision
//   of about 72 bits). A single FP32 saturates at about 24 bits and two at about 48 bits, whereas three
//   reach the about 53 bits equivalent to FP64. The output is S as three FP32 tiles (high, middle, low),
//   and the following stage recon_f64_writer.cpp assembles R = (hi+mid+lo) * radix^-(L+2) * s_A * s_B into
//   FP64 (on the device; the host only reorders. s_A and s_B are the divisors 2^-dA and 2^-dB actually used
//   in the normalization of the rows and columns; they are defined at the top of recon_f64_writer.cpp).
//
// Difference from the Int32-input version (recon_tf_i32in.cpp; deferred reconstruction and INT8)
//   P_d of the 7 formats other than INT8 is an exact integer in FP32 (at most 2^24), so no decoding into
//   a pair is needed and it can be added to the accumulator as a scalar as is.
//
// Compile-time arguments (order: ReconCT in args.h): ndiag (L+1), format value of the partial products,
//                   format value of Float32, bit pattern of radix as a float, ACC (0 = initialize,
//                   1 = accumulate)
//                   (the same layout as the Int32-input version. The Int32-input version does not read ACC,
//                   and this one does not read SIGNMAG and NSINGLE_CAST)
// Runtime arguments: ntiles (ReconRT in args.h)
// DST (8 tiles with fp32_dest_acc. recon_build_prog on the host sets dst_full_sync_en, so the 8 tiles can be
//   used as one section; with half sync, tiles 4..7 would overlap the next section). The names are in
//   namespace dst of sfpu_tf_proc.h: accumulator triple = acc_hi, acc_mid, acc_lo; the P_d to add is term
//   (the rest are work registers of tfMulRadix and tfAddS)

#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "recon_pack.h"
#include "args.h"
using namespace oz1_args;

constexpr uint32_t C_RDX = get_compile_time_arg_val(RC_RADIX);   // bit pattern of radix as a float
// ACC=1: with K blocking, the accumulation up to the previous block (hi, mid, lo) is read from CB_ACC and
// added. The Horner evaluation is linear in P_d, so each block may be evaluated separately and the results
// added together.
constexpr uint32_t ACC = get_compile_time_arg_val(RC_ACC);

constexpr auto cb_in  = tt::CBIndex(CB_RECON_IN);   // input: partial sums P_d of the diagonals (FP32)
constexpr auto cb_acc = tt::CBIndex(CB_ACC);        // input: accumulation up to the previous block (3 tiles)
constexpr auto cb_hi  = tt::CBIndex(CB_LIMB_HI);

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(RCR_NTILES);
    constexpr uint32_t ndiag = get_compile_time_arg_val(RC_NDIAG);  // L+1

    for (uint32_t t = 0; t < ntiles; t++) {
        cb_wait_front(cb_in, ndiag);
        init_sfpu(cb_in, cb_hi);
        tile_regs_acquire();

        using namespace dst;
        copy_tile_to_dst_init_short(cb_in);
        copy_tile(cb_in, 0, acc_hi);       // P_0 (first term of the Horner evaluation)
        Sub(acc_hi, acc_hi, acc_mid); Sub(acc_hi, acc_hi, acc_lo);   // accumulator = (P_0, 0, 0)

        for (uint32_t d = 1; d < ndiag; d++) {
            // One Horner step: acc *= radix (tfMulP2 when the radix is a power of two; sfpu_tf.h)
            tfMulRadix<C_RDX>();
            copy_tile_to_dst_init_short(cb_in);
            copy_tile(cb_in, d, term);      // P_d
            tfAddS<3>();                       // acc += P_d (P_d is an exact integer of at most 2^24)
        }
        if constexpr (ACC) {
            // Add in the triple up to the previous block, starting from the low limb
            cb_wait_front(cb_acc, 3);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 2, term); tfAddS<3>();   // lo
            copy_tile(cb_acc, 1, term); tfAddS<3>();   // mid
            copy_tile(cb_acc, 0, term); tfAddS<3>();   // hi
            cb_pop_front(cb_acc, 3);
        }

        recon_pack_triple();   // accumulator triple to the writer (recon_pack.h)

        cb_pop_front(cb_in, ndiag);
    }
}
