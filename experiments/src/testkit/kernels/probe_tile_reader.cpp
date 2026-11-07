// SPDX-License-Identifier: Apache-2.0
//
// 探針のタイル運び（読み）：DRAM の平面を並べたバッファから 1 組ずつ読み，平面ごとに決めた CB へ積む
// ---------------------------------------------------------------------------
// 組 g の平面 p はタイル p*ntot + g．引数の並びは probe_args.h の probe_tile_args．
// 1 枚ごとに読み終えてから積むので，計算カーネルは 1 組の枚数ぶん待てばよい（CB の深さはホストが決める）．

#include "api/dataflow/dataflow_api.h"
#include "probe_args.h"
using namespace probe_tile_args;

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(R_IN_ADDR);
    const uint32_t ntiles  = get_arg_val<uint32_t>(R_NTILES);
    const uint32_t ntot    = get_arg_val<uint32_t>(R_NTOT);
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, in_addr);
    constexpr uint32_t base = ia.next_compile_time_args_offset();
    constexpr uint32_t NPL = get_compile_time_arg_val(base + R_NPL);
    static_assert(NPL >= 1 && NPL <= R_MAX_PLANES, "平面の数は 1..R_MAX_PLANES");
    const uint32_t cb[R_MAX_PLANES] = {get_compile_time_arg_val(base + R_CB0), get_compile_time_arg_val(base + R_CB1),
                                       get_compile_time_arg_val(base + R_CB2), get_compile_time_arg_val(base + R_CB3)};
    for (uint32_t g = 0; g < ntiles; ++g)
        for (uint32_t p = 0; p < NPL; ++p) {
            cb_reserve_back(cb[p], 1);
            noc_async_read_tile(p * ntot + g, in, get_write_ptr(cb[p]));
            noc_async_read_barrier();
            cb_push_back(cb[p], 1);
        }
}
