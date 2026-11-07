// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: data-movement kernel that writes the modular slices to DRAM
// ---------------------------------------------------------------------------
// It drains c_16 (planes output tiles per input tile) and lays the tiles out contiguously as planes
// "independent matrices". Matrix i occupies the tile numbers
// [i*ntiles_total, (i+1)*ntiles_total).
// What planes stands for depends on the path: on the INT8 path it is one residue matrix per modulus
// (planes = the number of moduli), and on the digit-splitting path it is 1 to 5 digit matrices per
// modulus (1 for a one-digit modulus, n for balanced n digits; planes = nplane of the host), where the
// start of each modulus in the order is held by poff[] on the host side.
// The composition stage of the digit-splitting path (ozaki2/kernels/compose.cpp) uses the same writer.
// Its output is a single residue plane for one modulus, so it calls the writer with planes=1 and puts
// the position of the plane into start_id.
//
// The placement is selected by the compile-time argument LAYOUT (after the TensorAccessor. Layout in
// args.h):
//   LAYOUT_PLANES: matrix i is placed at tile i*ntot + g (slices of the INT8 path, and the composition
//                  stage)
//   LAYOUT_A_CAT : A of the digit-splitting path. The digit matrices of the moduli are concatenated
//                  along K. Digit matrix i of tile g = r*Ct + c (row r, K position c) is placed at
//                  r*(planes*Ct) + i*Ct + c (one matrix with planes*Ct columns)
//   LAYOUT_B_PERM: B of the digit-splitting path. Digit matrix i is placed at tile perm[i]*ntot + g
//                  (reversed order within a modulus. The matrices are stacked along K, so the result is
//                  one (planes*Kt) x Nt matrix)
// Layouts 1 and 2 concatenate the digit products of the same place value along K so that they are merged
// into a single GEMM (the same idea as the slice-concatenating writer of Ozaki scheme I.
// oz2_product_groups in split.hpp).
//
// Runtime arguments (the layout is WriterRT in args.h): dst_addr, num_tiles (the share of this core),
//   start_id, planes (number of output planes per input tile), ntiles_total,
//   then Ct for LAYOUT_A_CAT, or perm[0..planes-1] for LAYOUT_B_PERM

#include "api/dataflow/dataflow_api.h"
#include "args.h"
using namespace oz2_args;

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(SW_DST_ADDR);
    uint32_t num_tiles  = get_arg_val<uint32_t>(SW_NUM);
    uint32_t start_id   = get_arg_val<uint32_t>(SW_START);
    uint32_t planes     = get_arg_val<uint32_t>(SW_PLANES);   // number of output planes per input tile
    uint32_t ntot       = get_arg_val<uint32_t>(SW_NTOT);   // tiles per matrix (spacing of the planes)

    constexpr uint32_t cb_o = CB_PLANE_OUT;
    constexpr auto d_args = TensorAccessorArgs<0>();
    const auto d = TensorAccessor(d_args, dst_addr);
    constexpr uint32_t LAYOUT = get_compile_time_arg_val(d_args.next_compile_time_args_offset() + SW_CT_LAYOUT);
    const uint32_t Ct = (LAYOUT == LAYOUT_A_CAT) ? get_arg_val<uint32_t>(SW_A_KTILES) : 0u;

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t g = start_id; g < end_id; ++g) {
        for (uint32_t s = 0; s < planes; ++s) {
            cb_wait_front(cb_o, 1);
            uint32_t dst;
            if constexpr (LAYOUT == LAYOUT_A_CAT) {          // A: concatenate the digit matrices along K
                const uint32_t r = g / Ct, c = g - r * Ct;
                dst = r * (planes * Ct) + s * Ct + c;
            } else if constexpr (LAYOUT == LAYOUT_B_PERM) {  // B: permute the digit matrices, stack along K
                dst = get_arg_val<uint32_t>(sw_perm(s)) * ntot + g;
            } else {
                dst = s * ntot + g;                 // tile g of the residue matrix of modulus s
            }
            noc_async_write_tile(dst, d, get_read_ptr(cb_o));
            noc_async_write_barrier();
            cb_pop_front(cb_o, 1);
        }
    }
}
