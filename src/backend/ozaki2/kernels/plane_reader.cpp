// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II (the 7 digit-splitting paths): data-movement kernel that reads the input of the
// composition stage
// ---------------------------------------------------------------------------
// For each output tile g, the tiles s*ntot+g of the np GEMM outputs (planes of digit products) are read
// into CB_PLANE_IN in the order s = 0..np-1 (plane_fetch.h. np is the number of outputs of one modulus
// or, in a configuration that cuts them into chunks, the number of outputs of one chunk or of partial
// sums). The compute kernel of the composition stage, compose.cpp, receives them.
//
// Compile-time arguments: the TensorAccessorArgs of the input. Runtime arguments (the layout is
// PlaneReaderRT in args.h): in_addr, num_tiles (the share of this core), start_id (the first tile
// number, with the first plane of the input put into it), ntot (spacing of the planes), np (number of
// planes read at a time)
#include "api/dataflow/dataflow_api.h"
#include "args.h"
#include "plane_fetch.h"
using namespace oz2_args;

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(PR_IN_ADDR);
    const uint32_t num     = get_arg_val<uint32_t>(PR_NUM);
    const uint32_t start   = get_arg_val<uint32_t>(PR_START);
    const uint32_t ntot    = get_arg_val<uint32_t>(PR_NTOT);
    const uint32_t np      = get_arg_val<uint32_t>(PR_NPLANES);
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, in_addr);
    const uint32_t tb = get_tile_size(CB_PLANE_IN);
    for (uint32_t g = start; g < start + num; ++g) oz2_fetch_planes(CB_PLANE_IN, in, g, np, ntot, tb);
}
