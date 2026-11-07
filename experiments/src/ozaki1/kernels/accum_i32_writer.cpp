// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：Int32 累積の書き戻し．読み側と同じ線形順で Sbig へ返す．
// 実行時引数（並びは oz1_args.h の AccumWriterRT）: Sbig, 担当タイル数, S 側の先頭タイル
#include "api/dataflow/dataflow_api.h"
#include "oz1_args.h"
using namespace oz1_args;

void kernel_main() {
    uint32_t s_addr    = get_arg_val<uint32_t>(AW_S_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(AW_NUM);
    uint32_t start_id  = get_arg_val<uint32_t>(AW_START);

    constexpr uint32_t cb_out = CB_ACCUM_OUT;
    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto as = TensorAccessor(s_args, s_addr);

    for (uint32_t n = 0; n < num_tiles; ++n) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(start_id + n, as, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
