// SPDX-License-Identifier: Apache-2.0
#pragma once
// BFP2_b のタイルの詰め・解き（tt-metalium の bfloat2.hpp と同じ宣言）
// ---------------------------------------------------------------------------
// 使っている tt-metal のビルド（build_Release/include）は bfloat2.hpp を入れていないが，
// libtt_metal.so は pack_as_bfp2_tiles<float>・unpack_bfp2_tiles_into_float_vec を公開している
// （tt_metal/api/tt-metalium/bfloat2.hpp の大域の宣言と同じ形）．ヘッダが入るまで宣言だけをここに置く．
#include <cstdint>
#include <optional>
#include <vector>
#include <tt-metalium/tile.hpp>
#include <tt_stl/span.hpp>

template <typename T>
std::vector<uint32_t> pack_as_bfp2_tiles(
    tt::stl::Span<const T> data, bool row_major_input, bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);

std::vector<float> unpack_bfp2_tiles_into_float_vec(
    tt::stl::Span<const uint32_t> bfp_tiles, bool row_major_output, bool is_exp_a,
    const std::optional<tt::tt_metal::Tile>& tile = std::nullopt);
