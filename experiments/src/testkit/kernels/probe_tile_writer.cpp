// SPDX-License-Identifier: Apache-2.0
//
// 探針のタイル運び（書き）：c_16 から 1 組につき nout 枚を受け取り，DRAM の平面へ並べる
// ---------------------------------------------------------------------------
// 組 g の k 枚目はタイル k*ntot + g．引数の並びは probe_args.h の probe_tile_args．

#include "api/dataflow/dataflow_api.h"
#include "probe_args.h"
using namespace probe_tile_args;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(W_OUT_ADDR);
    const uint32_t ntiles   = get_arg_val<uint32_t>(W_NTILES);
    const uint32_t ntot     = get_arg_val<uint32_t>(W_NTOT);
    const uint32_t nout     = get_arg_val<uint32_t>(W_NOUT);
    constexpr uint32_t cb_o = tt::CBIndex::c_16;
    constexpr auto oa = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(oa, out_addr);
    for (uint32_t g = 0; g < ntiles; ++g)
        for (uint32_t k = 0; k < nout; ++k) {
            cb_wait_front(cb_o, 1);
            noc_async_write_tile(k * ntot + g, out, get_read_ptr(cb_o));
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
}
