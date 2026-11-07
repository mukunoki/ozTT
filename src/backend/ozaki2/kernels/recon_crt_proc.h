// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: per-modulus procedures of the reconstruction (recon_crt.cpp), independent of the
// device API
// ---------------------------------------------------------------------------
// kernel_main of recon_crt.cpp runs, for every modulus, "read r_i into tmp0 -> k_i =
// ModRound(r_i*y_i) into keep -> multiply k_i by the fragments of 1/m_i and add them into the
// triple-float -> drop the integer part". The three steps after the read are placed here. As in
// sfpu_tf_proc.h, the type W of the tile-operation wrapper is a template parameter, so that the same
// procedures can be instantiated with other hooks: the kernel instantiates them with the device
// wrapper SfpuDev; the checks described in docs/sfpu_lemmas.md instantiated them with a host-side
// model of the SFPU operations, which is not part of this distribution.
// The read (copy_tile for the residue input, the sign-magnitude decoding of the INT8 path) uses CBs
// and the type-conversion tile operations and therefore stays in the kernel. How the constants
// (RAWMASK, NSMALL_FRAG, the number of limbs) are decided is defined in one place on the host side,
// oz2_recon_ct in ozaki2/moduli.hpp.
#pragma once
#include <cstdint>
#include "../../common/kernels/sfpu_tf_proc.h"

namespace crt_proc {

// Places k_i = ModRound(r_i*y_i) (a representative of the residue class with |k_i| <= floor(m/2)+1;
// for the excess over the symmetric residue see the note on ModRound in recon_crt.cpp) into keep,
// from r_i in tmp0. If raw (a one-digit modulus; C_RAWMASK in recon_args.h), r_i is the GEMM output
// before the reduction modulo m, so it is reduced first (to |r_i| <= m/2 + 2, m <= 65)
template <class W>
ALWI void term_k(W& w, bool raw, float m, float rinv, const float& yi) {
    using namespace dst;
    if (raw) sfpu_proc::ModRound(w, tmp0, tmp0, tmp1, m, rinv);
    w.MulS(tmp0, yi);          // r_i * y_i (|r_i*y_i| < m^2/2 for |r_i| <= m/2; below 2^12 if raw)
    sfpu_proc::ModRound(w, tmp0, keep, tmp1, m, rinv);
}

// Multiplies k_i in keep by the fragment c.frag(t,i) of 1/m_i. The product with the last fragment
// of rp0/rp2 is allowed to round; the products with the other fragments are exact. The rounding of
// the lowest word remains in the multi-word accumulation as a whole. The products are added one at a
// time into the triple-float (two limbs if NLIMB=2). The trailing nsmall fragments skip the first
// stage and are added cheaply (for the condition and the error see tfAddSmall in sfpu_tf_proc.h)
template <uint32_t NLIMB, class W, class C>
ALWI void accumulate(W& w, uint32_t nrp, uint32_t nsmall, const C& c, uint32_t i) {
    using namespace dst;
    for (uint32_t t = 0; t < nrp; t++) {
        w.Cpy(keep, term); w.MulS(term, c.frag(t, i));
        if (t >= nrp - nsmall) {
            sfpu_proc::tfAddSmall<NLIMB>(w);
        } else {
            sfpu_proc::tfAddS<NLIMB>(w);
        }
    }
}

// Drops the integer part of a0 at every modulus (a0 -= round(a0); for the reason see kernel_main in
// recon_crt.cpp)
template <class W>
ALWI void drop_integer(W& w) {
    using namespace dst;
    w.Cpy(acc_hi, term); w.Rnd(term); w.Sub(acc_hi, term, acc_hi);
}

}  // namespace crt_proc
