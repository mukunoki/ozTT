// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme I: reconstruction on the device (Int32 input, triple-float)
// ---------------------------------------------------------------------------
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
// Decoding of the input
//   The Int32 input can exceed 2^24, the limit up to which FP32 represents integers exactly. It is
//   therefore split into two exact FP32 values (Vh, Vl), which are added to the accumulator one at a time
//   (each of them alone is an exact FP32).
//
//   On Blackhole, the representation of an Int32 in DST depends on where it came from. The compile-time
//   argument SIGNMAG selects one of the two:
//     SIGNMAG=1  sign-magnitude: an Int32 packed by the matrix engine and loaded with copy_tile
//                (the partial sums P_d of the INT8 path).
//                  magnitude = (raw & 0x7FFFFF00) + (raw & 0xFF), sign = sign(float(raw))
//                The sign is extracted separately and multiplied back after the split.
//     SIGNMAG=0  two's complement: produced by typecast<Float32,Int32>
//                (S_d accumulated in Int32 by the deferred reconstruction, fused in the GEMM).
//                  v = (v & 0xFFFFFF00) + (v & 0xFF) holds as is. The upper part is a multiple of 256
//                  (256*floor(v/256)) with at most 24 significant bits over the whole range of Int32, and
//                  the lower part is at most 255; FP32 represents both exactly. The sign need not be
//                  extracted and multiplied back (the typecast that moves the upper and lower parts to
//                  FP32 is performed regardless of SIGNMAG).
//                  The number of operations of a steady-state step, not counting the loading of tiles, is
//                  radix multiplication + decoding 4 + tfAddS 15 + Cpy 1 + tfAddS 15, where the radix
//                  multiplication is the 19 of tfMulP2 when the radix is a power of two (BF16, FP16, TF32,
//                  BFP8, FP8, BFP2; 54 in total) and the 44 of tfMulR otherwise (BFP4, INT8; 79 in
//                  total) (tfMulRadix in sfpu_tf.h. In the value range of actual runs the result is the
//                  same either way; see the condition for agreement at tfMulP2 in sfpu_tf_proc.h).
//                  With SIGNMAG=1, the typecast, the sign test and the multiplication back add 4.
//   Mixing them up corrupts negative values (the result fails with a relative error on the order of 1e4).
//   The host must pass the same SIGNMAG also to recon_reader_i32in.cpp, which pushes the masks.
//
// Shortcut for small diagonals (NSINGLE_CAST; two's-complement input only)
//   S_d of the deferred reconstruction is bounded by |S_d| <= (d+1) K d_max^2 (d+1 pairs x K terms x
//   product of digits). The host evaluates this bound for each diagonal and passes NSINGLE_CAST, the number
//   of leading diagonals that stay within 2^24 (recon_nsmall in plan.hpp). For those, S_d itself is an exact
//   integer in FP32, so it is moved to FP32 with a single typecast, without the split into two, and added
//   only once. The loading of the two mask tiles, two bitwise ANDs, one typecast, one Cpy and the second
//   tfAddS (15 operations) disappear. The value is read exactly, as in the split case, but the number of
//   additions goes from two to one, so the number of roundings of the lowest limb of the triple-float
//   (Add(acc_lo, tmp2, acc_lo) of tfAddS) changes, and the lowest bits of the result do not necessarily
//   agree with the split case (differences of 1 ulp of FP64 in a tiny fraction of the elements).
//   NSMALL_FRAG of recon_crt.cpp in Ozaki scheme II is "the number of trailing fragments added with
//   tfAddSmall" and is a different quantity.
//
// The order of the compile-time and runtime arguments is ReconCT and ReconRT in args.h (this kernel does not
// read RC_ACC. For RC_NSINGLE_CAST the host passes 0 when SIGNMAG=1)
// DST (8 tiles with fp32_dest_acc. recon_build_prog on the host sets dst_full_sync_en, so the 8 tiles can be
//   used as one section; with half sync, tiles 4..7 would overlap the next section). The names are in
//   namespace dst of sfpu_tf_proc.h: accumulator triple = acc_hi, acc_mid, acc_lo. Of the two decoded
//   values, Vh is placed in term (the value that tfAddS adds) and Vl in keep (which tfAddS does not
//   clobber); the two masks and the sign are in tmp0, tmp1, tmp2

#include <cstdint>
#include "../../common/kernels/sfpu_tf.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/binary_bitwise_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "recon_pack.h"
#include "args.h"
using namespace oz1_args;

constexpr uint32_t I32   = get_compile_time_arg_val(RC_IN_FMT);   // format value of Int32
constexpr uint32_t F32   = get_compile_time_arg_val(RC_F32);      // format value of Float32
constexpr uint32_t C_RDX = get_compile_time_arg_val(RC_RADIX);    // bit pattern of radix as a float
constexpr uint32_t SIGNMAG = get_compile_time_arg_val(RC_SIGNMAG); // 1: sign-magnitude
// Number of leading diagonals that are read with a single typecast
constexpr uint32_t NSINGLE_CAST = get_compile_time_arg_val(RC_NSINGLE_CAST);
// The shortcut is for two's-complement input only (a typecast does not turn sign-magnitude into the value)
static_assert(!SIGNMAG || NSINGLE_CAST == 0, "the shortcut is for two's-complement input only");

constexpr auto cb_in   = tt::CBIndex(CB_RECON_IN);   // input: partial sums P_d of the diagonals (Int32)
constexpr auto cb_mask = tt::CBIndex(CB_MASK);       // input: masks used for the split into a pair
constexpr auto cb_hi   = tt::CBIndex(CB_LIMB_HI);

// Decodes P of cb_in[idx] into two signed, exact FP32 values (rh, rl).
// The masks mh (0x7FFFFF00 if SIGNMAG=1, 0xFFFFFF00 if 0) and ml (0xFF) must already be loaded. sgn is a
// work register.
ALWI void decode(uint32_t idx, uint32_t rh, uint32_t rl,
                 uint32_t mh, uint32_t ml, uint32_t sgn) {
    if constexpr (SIGNMAG) {   // keep the sign aside (it is multiplied back onto the magnitude below)
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, sgn);
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(sgn);
    sign_tile_init(); sign_tile(sgn);                       // sgn = +1/-1/0
    }
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, rh);
    binary_bitwise_tile_init();
    bitwise_and_binary_tile<DataFormat::Int32>(rh, mh, rh); // raw & mh (see the note above)
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(rh);
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, idx, rl);
    binary_bitwise_tile_init();
    bitwise_and_binary_tile<DataFormat::Int32>(rl, ml, rl); // raw&0xFF
    typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(rl);
    if constexpr (SIGNMAG) {
        mul_binary_tile_init();
        mul_binary_tile(rh, sgn, rh);
        mul_binary_tile(rl, sgn, rl);
    }
}

void kernel_main() {
    uint32_t ntiles = get_arg_val<uint32_t>(RCR_NTILES);
    // The reader builds the two mask tiles once in L1 and pushes them (they are constants, so they are not
    // popped)
    cb_wait_front(cb_mask, 2);
    constexpr uint32_t ndiag = get_compile_time_arg_val(RC_NDIAG);  // L+1

    for (uint32_t t = 0; t < ntiles; t++) {
        cb_wait_front(cb_in, ndiag);
        init_sfpu(cb_in, cb_hi);   // reconfigure the int32 unpack-to-dest for every tile
        tile_regs_acquire();

        using namespace dst;
        for (uint32_t d = 0; d < ndiag; d++) {
            // One Horner step: acc *= radix (tfMulP2 when the radix is a power of two)
            if (d > 0) tfMulRadix<C_RDX>();
            if (d < NSINGLE_CAST) {
                // |S_d| <= 2^24 (the bound shown by the host): exact conversion to FP32 with a single typecast
                copy_tile_to_dst_init_short(cb_in);
                // The first one is placed directly in the high limb of the accumulator
                const uint32_t r = (d == 0) ? acc_hi : term;
                copy_tile(cb_in, d, r);
                typecast_tile_init<I32, F32>(); typecast_tile<I32, F32>(r);
                // d == 0: set the accumulator to (S_0, 0, 0)
                if (d == 0) { Sub(acc_hi, acc_hi, acc_mid); Sub(acc_hi, acc_hi, acc_lo); }
                else        tfAddS<3>();                      // acc += S_d
                continue;
            }
            copy_tile_to_dst_init_short(cb_mask);
            copy_tile(cb_mask, 0, tmp0); copy_tile(cb_mask, 1, tmp1);
            decode(d, term, keep, tmp0, tmp1, tmp2);      // decode P_d into Vh (term) and Vl (keep)
            // The first time, clear the accumulator to 0
            if (d == 0) { Sub(term, term, acc_hi); Sub(term, term, acc_mid); Sub(term, term, acc_lo); }
            tfAddS<3>();                                  // acc += Vh; Vl in keep is preserved
            Cpy(keep, term);                              // move Vl into term
            tfAddS<3>();                                  // acc += Vl
        }

        recon_pack_triple();   // accumulator triple to the writer (recon_pack.h)

        cb_pop_front(cb_in, ndiag);
    }
}
