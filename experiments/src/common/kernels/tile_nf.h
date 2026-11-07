// SPDX-License-Identifier: Apache-2.0
#pragma once
//
// (行, 列) → タイル（32x32）の中の通し番号．タイルは 16x16 の面 4 つに分かれて格納されるので，その並びに合わせる．
// データ移動カーネルがタイルの要素を直に読み書きするとき（scale_f64_reader.cpp・expdelta.cpp）と，ホストがタイル順と
// 行優先を並べ替えるとき（common/ozaki_gemm_host.hpp が取り込む）の両方が使う．
#include <cstdint>

inline uint32_t nf(uint32_t r, uint32_t c) {
    return ((c < 16) ? 0u : 256u) + ((r < 16) ? 0u : 512u) + (r % 16) * 16u + (c % 16);
}
