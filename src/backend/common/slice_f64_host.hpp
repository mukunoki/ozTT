// SPDX-License-Identifier: Apache-2.0
#pragma once
// Host-side argument builders of the FP64 reader of the slicing stage
// (common/kernels/slice_f64_reader.cpp), shared by both Ozaki schemes.
//
// The reader splits each FP64 tile of an operand into the triple-float form and applies the exponent
// delta of the scaling stage; the slicer of each scheme consumes its output. The layout of the arguments
// and the CB indices are the table common/kernels/slice_f64_args.h, which the kernel reads by the same
// names. The hosts of both schemes (ozaki1/slice.hpp, ozaki2/stages.hpp) fill the arguments with these
// two functions, so a change of the layout touches only the table and this file.

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <cstdint>
#include <vector>
#include "common/kernels/slice_f64_args.h"

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

// Compile-time arguments: the TensorAccessorArgs of the FP64 operand and of the exponent deltas, followed
// by ReaderCT (CT_SF selects the soft-float variant of the kernel, a diagnostic that the library never
// uses; split: lets the writer of the same core split the second half of each tile).
inline std::vector<uint32_t> f64_reader_ct(const tt::tt_metal::distributed::MeshBuffer& f64,
                                           const tt::tt_metal::distributed::MeshBuffer& dlt, bool split) {
    namespace sa = slice_f64_args;
    std::vector<uint32_t> ct;
    tt::tt_metal::TensorAccessorArgs(f64).append_to(ct);
    tt::tt_metal::TensorAccessorArgs(dlt).append_to(ct);
    const size_t o = ct.size();
    ct.resize(o + sa::CT_COUNT);
    ct[o + sa::CT_SF] = 0u;
    ct[o + sa::CT_SPLIT] = split ? 1u : 0u;
    static_assert(sa::CT_COUNT == 2, "fill the new ReaderCT entry here");
    return ct;
}

// Runtime arguments (ReaderRT) of one core: num input tiles starting at tile start. mode 0 = A (the index
// of the exponent delta is the tile index / dlt_stride), mode 1 = B (the tile index % dlt_stride).
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
    static_assert(sa::R_COUNT == 6, "fill the new ReaderRT entry here");
    return rt;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
