// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Common part of the triple-float slicers (used by the three slicers of Ozaki schemes I and II)
// ---------------------------------------------------------------------------
// Every slicer repeats "receive the four tiles that the reader split from FP64 and assemble the
// triple-float, multiply by the radix, round, and take the residual". They differ only in how the radix
// is applied and in the encoding and destination of the extracted values, so the input hand-off and the
// carrying of the residual are collected here.
//   scheme I  the 7 formats other than INT8 : ozaki1/kernels/slice_tf_num.cpp (body in slice_tf_body.h)
//   scheme I  INT8                          : ozaki1/kernels/slice_tf_sm8.cpp (same)
//   scheme II                               : ozaki2/kernels/digits.h
//
// The CB assignment mirrors the destinations written by slice_f64_reader.cpp (both sides and the host
// take the indices from the single table in slice_f64_args.h).
//   c_0 hi   c_4 braw   c_1 craw   c_2 s (scale; see tf_load_f64 below)
//   c_24 residual limbs 0,1   c_25 residual limb 2
//
// Use of DST (the names are those of namespace dst in sfpu_tf_proc.h): the assembled triple-float and the
// residual are the three terms acc_hi, acc_mid, acc_lo; the scale s of the low limb is placed temporarily
// in term.
//
// Why the residual is split over two CBs: when tiles make a round trip through a CB inside a kernel,
// passing 3 tiles at once (FP32 with dst_full_sync) deadlocks nondeterministically on this hardware.
// Splitting them into 2 + 1 avoids it. All three slicers share this constraint, so the round trip is
// done only by the two functions here.

#include "sfpu_tf.h"
#include "f64_tf.h"
#include "slice_f64_args.h"
#include "api/compute/pack.h"

constexpr auto cb_tf_hi    = tt::CBIndex(slice_f64_args::CB_HI);
constexpr auto cb_tf_lo    = tt::CBIndex(slice_f64_args::CB_CRAW);
// s = +-2^(elsb+7) (elsb is the exponent of the least significant bit of the significand in split_f64 of
// f64_tf.h)
constexpr auto cb_tf_scale = tt::CBIndex(slice_f64_args::CB_SCALE);
constexpr auto cb_tf_mid   = tt::CBIndex(slice_f64_args::CB_BRAW);
constexpr auto cb_tf_res   = tt::CBIndex(slice_f64_args::CB_RES01);
constexpr auto cb_tf_res2  = tt::CBIndex(slice_f64_args::CB_RES2);

// Wait until the four tiles of one input tile (hi, braw, craw and s) are available / discard them after
// use
ALWI void tf_wait_input() {
    cb_wait_front(cb_tf_hi, 1);
    cb_wait_front(cb_tf_lo, 1);
    cb_wait_front(cb_tf_mid, 1);
    cb_wait_front(cb_tf_scale, 1);
}
ALWI void tf_pop_input() {
    cb_pop_front(cb_tf_hi, 1);
    cb_pop_front(cb_tf_lo, 1);
    cb_pop_front(cb_tf_mid, 1);
    cb_pop_front(cb_tf_scale, 1);
}

// The reader (slice_f64_reader.cpp) splits the FP64 values and passes them as four tiles of bit patterns,
// which are assembled into a triple-float here (common to both schemes):
//     hi  = c_0
//     mid = (c_4 - 12582912.0f) * s          (c_4 = 2^23 + 2^22 + b)
//     lo  = (c_1 -  8388608.0f) * s * 2^-7   (c_1 = 2^23 + c, s = c_2 = +-2^(elsb+7); elsb is the
//                                             exponent in split_f64)
// The lower two terms are received in magic-number form so that the reader does not have to search for
// the position of the leading bit.
// The normalization (a power of two) has already been folded into the exponent by the reader, so no
// multiplication by a per-row / per-column scale tile is needed here.
ALWI void tf_load_f64() {
    using namespace dst;
    copy_tile_to_dst_init_short(cb_tf_scale); copy_tile(cb_tf_scale, 0, term);   // s
    copy_tile_to_dst_init_short(cb_tf_mid); copy_tile(cb_tf_mid, 0, acc_mid);   // braw
    copy_tile_to_dst_init_short(cb_tf_lo);  copy_tile(cb_tf_lo, 0, acc_lo);     // craw
    copy_tile_to_dst_init_short(cb_tf_hi);  copy_tile(cb_tf_hi, 0, acc_hi);     // hi
    AddS(acc_mid, -12582912.0f); Mul(acc_mid, term, acc_mid);                    // mid
    AddS(acc_lo,  -8388608.0f); Mul(acc_lo, term, acc_lo); MulS(acc_lo, OZ_TF_LO_SCALE);   // lo
}

// The soft-float version of the reader delivers the three terms as they are (already normalized), so they
// only need to be copied.
ALWI void tf_load_plain() {
    copy_tile_to_dst_init_short(cb_tf_hi);  copy_tile(cb_tf_hi, 0, dst::acc_hi);
    copy_tile_to_dst_init_short(cb_tf_mid); copy_tile(cb_tf_mid, 0, dst::acc_mid);
    copy_tile_to_dst_init_short(cb_tf_lo);  copy_tile(cb_tf_lo, 0, dst::acc_lo);
}

// Carrying the residual. The contents of DST are not preserved across acquire/release, so in every
// iteration the residual is saved to the CBs and read back.
ALWI void tf_pop_residual() {
    cb_wait_front(cb_tf_res, 2);
    copy_tile_to_dst_init_short(cb_tf_res);
    copy_tile(cb_tf_res, 0, dst::acc_hi);
    copy_tile(cb_tf_res, 1, dst::acc_mid);
    cb_pop_front(cb_tf_res, 2);
    cb_wait_front(cb_tf_res2, 1);
    copy_tile_to_dst_init_short(cb_tf_res2);
    copy_tile(cb_tf_res2, 0, dst::acc_lo);
    cb_pop_front(cb_tf_res2, 1);
}
ALWI void tf_push_residual() {
    cb_reserve_back(cb_tf_res, 2);
    pack_reconfig_data_format(cb_tf_res);
    pack_tile(dst::acc_hi, cb_tf_res);
    pack_tile(dst::acc_mid, cb_tf_res);
    cb_push_back(cb_tf_res, 2);
    cb_reserve_back(cb_tf_res2, 1);
    pack_reconfig_data_format(cb_tf_res2);
    pack_tile(dst::acc_lo, cb_tf_res2);
    cb_push_back(cb_tf_res2, 1);
}
