// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// Ozaki scheme I: output step shared by the two compute kernels of the reconstruction (recon_tf_f32in.cpp,
// recon_tf_i32in.cpp)
// ---------------------------------------------------------------------------
// Packs the accumulator triple produced by the Horner evaluation (dst::acc_hi, acc_mid, acc_lo in DST) into
// the three CBs that the writer recon_f64_writer.cpp reads. The CB indices are in the order hi, lo, mid (see
// the top of args.h), which differs from the order in DST, so the packing is kept in this one place to avoid
// mixing them up (if only one of the kernels were changed, the order of the limbs would silently disagree
// depending on the input format).

#include "api/compute/pack.h"
#include "../../common/kernels/sfpu_tf_proc.h"   // names of the DST registers (namespace dst)
#include "args.h"

ALWI void recon_pack_triple() {
    constexpr auto cb_hi  = tt::CBIndex(oz1_args::CB_LIMB_HI);
    constexpr auto cb_lo  = tt::CBIndex(oz1_args::CB_LIMB_LO);
    constexpr auto cb_mid = tt::CBIndex(oz1_args::CB_LIMB_MID);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_hi, 1);
    cb_reserve_back(cb_mid, 1);
    cb_reserve_back(cb_lo, 1);
    pack_reconfig_data_format(cb_hi);  pack_tile(dst::acc_hi, cb_hi);
    pack_reconfig_data_format(cb_mid); pack_tile(dst::acc_mid, cb_mid);
    pack_reconfig_data_format(cb_lo);  pack_tile(dst::acc_lo, cb_lo);
    cb_push_back(cb_hi, 1);
    cb_push_back(cb_mid, 1);
    cb_push_back(cb_lo, 1);
    tile_regs_release();
}
