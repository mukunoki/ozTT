// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: assembly of one element at the exit of the reconstruction (integer arithmetic on
// the RISC-V of the data-movement kernels)
// ---------------------------------------------------------------------------
// Of the 1024 elements of a tile, the reader (recon_reader.cpp) assembles the first part and the
// writer (recon_writer.cpp) the second part (for the reason of the sharing see the head of
// recon_reader.cpp). Both call the same per-element procedure from this single place: if only one
// side were changed, the rounding of the first part and of the second part would silently disagree
// within the same C, possibly by an amount that the 53-bit criterion of the validation does not
// catch.
// How the L1 pointers are held (the writer reads through volatile) and the order of the handoff
// (semaphores, publication) stay in each kernel; here the pointer types are template parameters. The
// assembly procedures (a128_add_limbs, a128_frac_to_f64, assemble_frac_f64) are in
// common/kernels/f64_tf.h.
#pragma once
#include <cstdint>
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"

namespace recon_elem {

// First pass of the two-pass configuration (ROLE_TWO_FIRST): adds the 3 limbs of element j (the
// two-pass configuration uses 3 limbs in every pass; oz2_recon_limb3 in ozaki2/moduli.hpp) into
// 128-bit fixed point and places it in 4 words of the scratch area qa (w[0] is the least
// significant). PROBE is the diagnostic selector that removes the assembly (it breaks the answer;
// the output is 0)
template <uint32_t PROBE, class LP, class QP>
inline __attribute__((always_inline)) void pack_a128(LP l0, LP l1, LP l2, QP qa, uint32_t j) {
    const uint32_t lb[3] = {l0[j], l1[j], l2[j]};
    A128 a{{0, 0, 0, 0}};
    if constexpr (PROBE == 0) a128_add_limbs(a, lb, 3u);
    else                      { (void)lb; }
    qa[4 * j] = a.w[0]; qa[4 * j + 1] = a.w[1]; qa[4 * j + 2] = a.w[2]; qa[4 * j + 3] = a.w[3];
}

// Final pass, which assembles FP64: assembles FP64 from the limbs of element j (nlimb limbs; with
// 2 limbs the third is 0) and places it in 2 words of po (low word first). If ROLE is
// ROLE_TWO_FINAL, the own 3 limbs are added to the 128-bit fixed point of the first pass (4 words of
// qa) and the sum is folded. The delta tiles da and db hold sh-e, so the exponent to multiply back
// is 2sh-(dA+dB) (sh2 is 2sh). With PROBE the output is filled with a constant
template <uint32_t ROLE, uint32_t PROBE, class LP, class DP, class QP, class OP>
inline __attribute__((always_inline)) void assemble_f64(LP l0, LP l1, LP l2, uint32_t nlimb, DP da, DP db, QP qa, OP po,
                                                         uint32_t j, int32_t sh2, uint32_t Whi, uint32_t Wlo, int32_t qw) {
    uint32_t lb[3];
    lb[0] = l0[j]; lb[1] = l1[j]; lb[2] = (nlimb >= 3) ? l2[j] : 0u;
    const int32_t gg = sh2 - (da[j] + db[j]);
    uint64_t u = 0;
    if constexpr (PROBE == 0) {
        if constexpr (ROLE == recon_args::ROLE_TWO_FINAL) {
            A128 a{{qa[4 * j], qa[4 * j + 1], qa[4 * j + 2], qa[4 * j + 3]}};
            a128_add_limbs(a, lb, 3u);
            u = a128_frac_to_f64(a, gg, Whi, Wlo, qw);
        } else {
            u = assemble_frac_f64(lb, 3u, gg, Whi, Wlo, qw);
        }
    } else { (void)lb; (void)gg; }
    po[2 * j]     = (uint32_t)u;
    po[2 * j + 1] = (uint32_t)(u >> 32);
}

}  // namespace recon_elem
