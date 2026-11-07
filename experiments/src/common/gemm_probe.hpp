// SPDX-License-Identifier: Apache-2.0
#pragma once
// GEMM 段の診断用の探針（OZ_GEMM_PROBE_*．docs/env_vars.md の C．答えは壊れる）
// ---------------------------------------------------------------------------
// 探針はデータ移動カーネル（common/kernels/mcast_split_dm.cpp）の define で出し分ける．どの探針で組むかは McastGemmSpec の
// probe が持ち，build_mcast_gemm（mcast_gemm.hpp）は環境を読まない（spec を見れば組んだ構成が分かる）．probe を
// oz_gemm_probe() で詰めるのは，立っていると CONFIG 行へ gemm_probe=・probe=set を残す尾崎の 2 つのドライバだけで，
// 基準実装と検査は既定（どれも 0 の本番の構成）のまま組む（立っていれば入口の oz_knobs::reject_unread が止める．
// oz_driver.hpp の reject_env_gemm）．
//   OZ_GEMM_PROBE_NODRAM=1   送信側の DRAM 読みを最初の 2 ブロックだけにする（mcast_split_dm.cpp の NODRAM）
//   OZ_GEMM_PROBE_NOMCAST=1  A/B の供給（DRAM 読み・マルチキャスト・同期）を外し，入力 CB の reserve/push だけを
//                            行う（同 NOMCAST）．出力の DRAM への書き込みとその完了待ち（融合ライタなら既存の S の
//                            先読みも）は残るので，残る時間は「計算パイプライン＋出力の転送」であって計算だけではない．
//                            名前の NOMCAST は「A/B のマルチキャスト（供給）なし」の意味で，NoC を一切使わないのではない
//   OZ_GEMM_PROBE_NOWRITE=1  出力の DRAM への書き込み（融合ライタなら既存の S の先読みも）を外し，c_16 を受け取って
//                            捨てるだけにする（同 NOWRITE）．NOMCAST と重ねると，供給も書き出しも無い GEMM 段の時間になる
#include <cstdint>
#include <string>
#include "common/oz_knobs.hpp"   // つまみの表（3 つとも 0 か 1．既定 0）

struct GemmProbe {
    uint32_t nodram = 0, nomcast = 0, nowrite = 0;
};

// 環境から読んだ探針（尾崎の 2 つのドライバが McastGemmSpec の probe に詰める）
inline GemmProbe oz_gemm_probe() {
    GemmProbe p;
    p.nodram = oz_knobs::u32(OZK("OZ_GEMM_PROBE_NODRAM"));
    p.nomcast = oz_knobs::u32(OZK("OZ_GEMM_PROBE_NOMCAST"));
    p.nowrite = oz_knobs::u32(OZK("OZ_GEMM_PROBE_NOWRITE"));
    return p;
}
// CONFIG 行の gemm_probe 欄：立っている探針の名前を '+' で繋ぐ（無ければ none）．
// 環境変数の有無ではなく読んだ値で決める（OZ_GEMM_PROBE_NODRAM=0 は既定と同じ構成なので none）．
inline std::string oz_gemm_probe_name(const GemmProbe& p) {
    std::string s;
    if (p.nodram) s += "nodram";
    if (p.nomcast) s += (s.empty() ? "" : "+") + std::string("nomcast");
    if (p.nowrite) s += (s.empty() ? "" : "+") + std::string("nowrite");
    return s.empty() ? std::string("none") : s;
}
// 環境から読み直した探針の名前．組んだ spec.probe（上の oz_gemm_probe で詰めたもの）と同じ環境から読むので今は一致するが，
// 欄の出どころは組んだ構成であるべきなので，呼び手（尾崎の 2 つのドライバの CONFIG 行）は組むのに使った GemmProbe を
// 上の版に渡す（呼び手が移ったらこの版を消す）
inline std::string oz_gemm_probe_name() { return oz_gemm_probe_name(oz_gemm_probe()); }
