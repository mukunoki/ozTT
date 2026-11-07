// SPDX-License-Identifier: Apache-2.0
//
// 尾崎スキーム I：Int32 の部分和を読む，再構成のデータ移動カーネル（recon_tf_i32in.cpp と対）
// ---------------------------------------------------------------------------
// 出力タイル i ごとに，対角ごとの Int32 を d = 0..L の順に ndiag 枚 c_0 へ読み込む（recon_read.h．FP32 の部分積を読む
// recon_reader_f32in.cpp と共有）．
// あわせて分割用マスク 2 枚を c_1 へ積む．これは定数なので DRAM には置かず，
// このコアの L1 で 1 回だけ作る（ホストからも送らない．A・B を転送してから C が
// 戻るまで通信しないという仕様のため）．マスクが要るのは Int32 を無誤差な FP32
// 2 本へ復号するためである（recon_tf_i32in.cpp の decode を参照）．
//
// 上位マスクは Int32 の表現で決まる．コンパイル時引数の SIGNMAG で選ぶ：
//   SIGNMAG=0  2 の補数（typecast<Float32,Int32> が作る．遅延再構成の累積 S_d）
//              -> 0xFFFFFF00．符号ビット込みで切り出すので符号の復元が要らない
//   SIGNMAG=1  符号-絶対値（行列エンジンがパックした Int32．INT8 経路の P_d）
//              -> 0x7FFFFF00．符号は計算カーネル側で掛け戻す
// 取り違えると負の値が壊れる（相対誤差 1e4 の桁で落ちる）ので，計算カーネルと
// 同じ SIGNMAG をホストが両方へ渡すこと（ozaki1_recon.hpp の recon_build_prog）．
//
// 実行時引数（並びは oz1_args.h の ReaderRT の先頭 5 語）: 入力バッファ, 担当タイル数, 先頭タイル, ndiag,
//             面のタイル間隔（既定は Mt*Nt．行バンドではバンドの確保幅）
#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/l1_order.h"
#include "recon_read.h"
#include "oz1_args.h"
using namespace oz1_args;

constexpr uint32_t SIGNMAG = get_compile_time_arg_val(RI_CT_SIGNMAG);
constexpr uint32_t MHI = SIGNMAG ? 0x7FFFFF00u : 0xFFFFFF00u;
constexpr uint32_t MLO = 0x000000FFu;

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(RR_SRC_ADDR);
    uint32_t num_tiles = get_arg_val<uint32_t>(RR_NUM);
    uint32_t start_id = get_arg_val<uint32_t>(RR_START);
    uint32_t ndiag = get_arg_val<uint32_t>(RR_NDIAG);
    uint32_t slot_stride = get_arg_val<uint32_t>(RR_SLOT_STRIDE);
    // 累積入力（FP32 入力の経路の RR_ACC_*）は受け取らない：この経路は遅延再構成の
    // 最終段で，K 方向の累積は Int32 で済ませたあと（GEMM の中か accum_i32.cpp）1 回だけ走る．

    constexpr uint32_t cb_mask = CB_MASK;
    constexpr auto s_args = TensorAccessorArgs<RI_CT_COUNT>();
    const auto s = TensorAccessor(s_args, src_addr);
    const uint32_t tb = get_tile_size(CB_RECON_IN);

    // 分割用マスク 2 枚はこのコアの L1 で 1 回だけ作る．定数なので DRAM に置く必要が
    // 無く，ホストから送る必要も無い（A・B を転送してから C が戻るまで通信しない仕様）．
    // 出力タイルごとに DRAM から読み直すと，14080^2 で 1.59 GB になる．
    {
        cb_reserve_back(cb_mask, 2);
        volatile tt_l1_ptr uint32_t* mp = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_mask);
        for (uint32_t z = 0; z < 1024; ++z) { mp[z] = MHI; mp[1024 + z] = MLO; }
        l1_writes_landed((uint32_t)&mp[2047]);   // L1 に届いてから計算側へ渡す（l1_order.h）
        cb_push_back(cb_mask, 2);
    }
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i)
        recon_read_diags(s, i, ndiag, slot_stride, tb);   // 対角の順（d = 0..L）に ndiag 枚
}
