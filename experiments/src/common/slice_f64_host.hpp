// SPDX-License-Identifier: Apache-2.0
#pragma once
// FP64 のリーダ（common/kernels/slice_f64_reader.cpp）の引数をホストで詰める部品（尾崎スキーム I・II 共通）
// ---------------------------------------------------------------------------
// 並びと CB 番号は common/kernels/slice_f64_args.h の表で，カーネルも同じ表を名前で読む．両スキームのホスト
// （ozaki1_slice.hpp の slice_build・slice_run，ozaki2_main.cpp の build_slice）がこの 2 つの関数で詰めるので，
// 並びを変えるときは表とここだけを直す．
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include <vector>
#include "common/kernels/slice_f64_args.h"

// コンパイル時引数：FP64 のオペランドと指数差分の TensorAccessorArgs，続いて ReaderCT（soft-float 版か・後半を
// ライタに割らせるか）
inline std::vector<uint32_t> f64_reader_ct(const tt::tt_metal::distributed::MeshBuffer& f64,
                                           const tt::tt_metal::distributed::MeshBuffer& dlt, bool sf, bool split) {
    namespace sa = slice_f64_args;
    std::vector<uint32_t> ct;
    tt::tt_metal::TensorAccessorArgs(f64).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(dlt).append_to(ct);
    const size_t o = ct.size();
    ct.resize(o + sa::CT_COUNT);
    ct[o + sa::CT_SF] = sf ? 1u : 0u;
    ct[o + sa::CT_SPLIT] = split ? 1u : 0u;
    static_assert(sa::CT_COUNT == 2, "ReaderCT に欄を足したらここで詰める");
    return ct;
}

// 実行時引数（ReaderRT）．mode 0 = A（指数差分の番号はタイル番号 / dlt_stride），1 = B（タイル番号 % dlt_stride）
inline std::vector<uint32_t> f64_reader_rt(uint32_t f64_addr, uint32_t dlt_addr, uint32_t num, uint32_t start,
                                           uint32_t dlt_stride, uint32_t mode) {
    namespace sa = slice_f64_args;
    std::vector<uint32_t> rt(sa::R_COUNT);
    rt[sa::R_F64_ADDR] = f64_addr;
    rt[sa::R_DLT_ADDR] = dlt_addr;
    rt[sa::R_NUM] = num;
    rt[sa::R_START] = start;
    rt[sa::R_DLT_STRIDE] = dlt_stride;
    rt[sa::R_MODE] = mode;
    static_assert(sa::R_COUNT == 6, "ReaderRT に欄を足したらここで詰める");
    return rt;
}
