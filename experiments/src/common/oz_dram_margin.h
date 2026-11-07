// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// 両尾崎スキームの DRAM の見積りに足す余裕（デバイス API に依存しない．<cstdint> だけ）
// ---------------------------------------------------------------------------
// 確保の前に「これから確保するもの」を DRAM の空きと比べる見積り（scheme I の行バンドと発行の計画 ozaki1_plan.hpp，
// scheme II の法の組分け ozaki2_plan.hpp の plan_mod_groups と，同じ式を使う段の時間のモデル ozaki2_awe.h・tools/awe_plan.cpp）
// は，見積りに入れない小さな器（範囲外の旗・limb を読まない構成の 1 タイルの器など）と割り付けの端数をこの余裕で受ける．
// 両スキームが別々に値を持つと，一方だけ直したときに組分け・行バンドの判断が食い違うので，ここに 1 つだけ置く．
// ホスト単体の道具（tools/）も scheme II の見積り（ozaki2_split.h）を通して同じ値を見る．
#include <cstdint>

inline constexpr uint64_t OZ_DRAM_MARGIN_BYTES = 512ull << 20;   // 0.5 GiB
