// tile_rate_probe のデータ移動カーネル：入力の CB（c_0・c_1）を 1 回だけ埋めて計算カーネルへ渡す．
// CT 引数：c_0 のタイル数，c_1 のタイル数，c_0 のバイト数，c_1 のバイト数（並びは testkit/kernels/probe_args.h の
// tile_rate_args::FillCT）
// 値は 1.0 付近の BF16 の対を並べたバイト列（形式によらず同じ．計時は値に依らないので中身は問わない）．
// 書き終えたら，CB を渡す前に書き込みが L1 に届くのを待つ（l1_writes_landed．RISC-V の store は fence だけでは
// L1 に届いた保証が無く，cb_push_back のカウンタの更新が先に処理されうる．受け側は c_0・c_1 がそろうと直ちに計時を
// 始めて読むので，届く前の値を読みうる）．同じ RISC-V の L1 への要求は順に処理されるので，最後に書いた番地を
// 1 度待てば c_0 の書き込みも含めて届いている
#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/l1_order.h"
#include "../../testkit/kernels/probe_args.h"
void kernel_main() {
    using namespace tile_rate_args;
    constexpr uint32_t n0 = get_compile_time_arg_val(F_N0), n1 = get_compile_time_arg_val(F_N1);
    constexpr uint32_t b0 = get_compile_time_arg_val(F_B0), b1 = get_compile_time_arg_val(F_B1);
    cb_reserve_back(tt::CBIndex::c_0, n0);
    cb_reserve_back(tt::CBIndex::c_1, n1);
    volatile uint32_t* p = (volatile uint32_t*)get_write_ptr(tt::CBIndex::c_0);
    uint32_t h = 12345;
    for (uint32_t i = 0; i < b0 / 4; i++) { h = h * 1103515245u + 12345u; p[i] = 0x3f803f80u | ((h >> 8) & 0x007f007fu); }
    p = (volatile uint32_t*)get_write_ptr(tt::CBIndex::c_1);
    for (uint32_t i = 0; i < b1 / 4; i++) { h = h * 1103515245u + 12345u; p[i] = 0x3f803f80u | ((h >> 8) & 0x007f007fu); }
    static_assert(b1 >= 4, "c_1 は 1 タイル以上");
    l1_writes_landed((uint32_t)(uintptr_t)&p[b1 / 4 - 1]);   // 最後に書いた語
    cb_push_back(tt::CBIndex::c_0, n0);
    cb_push_back(tt::CBIndex::c_1, n1);
}
