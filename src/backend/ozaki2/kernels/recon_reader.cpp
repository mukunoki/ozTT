// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: data-movement kernel that reads the input of the reconstruction (the S per-modulus
// planes) and takes over half of the FP64 assembly in the final pass, or half of the conversion to
// 128-bit fixed point in the first pass of the two-pass configuration
// ---------------------------------------------------------------------------
// For every output tile g, tile s*ntot+g of plane s is read into c_0 in the order s = 0..S-1
// (plane_fetch.h; the input of the composition stage is read by another reader, plane_reader.cpp,
// with the same procedure).
//
// The reconstruction of the INT8 path receives C_i as sign-magnitude int32 and splits the magnitude
// into the upper 15 bits and the lower 16 bits before taking the modulus (it exceeds 2^24, up to
// which FP32 represents integers exactly). The 2 masks for this (0x7FFF0000, 0x0000FFFF) are sent to
// c_1 when mask=1. The masks are constants, so the compute kernel does not pop them, and they are
// built here only once before the loop.
//
// Why the FP64 assembly of the final pass is shared
//   If the assembly is done by the writer alone, the bottleneck of the reconstruction stage is not
//   the CRT (SFPU) but the scalar FP64 assembly of the writer (the reconstruction stage does not get
//   shorter when the number of moduli is reduced).
//   This core only reads S tiles and is idle most of the time, so it assembles the first part of the
//   1024 elements of a tile here and shares the work half and half with the writer (the per-element
//   procedure is recon_elem.h, the same as for the writer).
//   Both share the L1 of the same core, so the scratch area (c_19) of the writer is used as is.
//   Only the writer consumes c_16 (with two consumers on one CB things break silently, because each
//   RISC-V keeps its own read pointer); this side only looks at the L1 address that the writer hands
//   over. The handoff places two tags in semaphores and the address of the limbs in one word of a
//   shared scratch (c_26).
//   The tag increases monotonically from 1, so nothing is missed and nothing is taken twice.
//   The tags are kept in semaphores because only the semaphores return to their initial value at
//   every program launch: the contents of a CB persist across launches, so a tag placed there would
//   keep the previous value (1 if the core handles a single tile), and the next run would read stale
//   limbs without waiting.
//   The delta tiles (c_20/c_21) are used as read by the writer: reading them on both sides would
//   duplicate 2 tiles of DRAM reads per output tile.
//   The input is read one tile ahead so that the compute kernel does not stall (c_0 has depth 2S).
//   The two-pass configuration shares the work in the same form (the pass role ROLE is ReconRole of
//   recon_args.h, the same as for the writer).
//   The first pass adds its own 3 limbs into 128-bit fixed point and places it in the first part of
//   the scratch area c_22; the final pass adds its own 3 limbs to the fixed point of the first pass,
//   which the writer read into c_22, and folds it (both are the L1 of the same core, so the CB of the
//   writer is used as is).
//   In the final pass of two passes, assembling the 6 limbs by the writer alone would make it the
//   bottleneck of the reconstruction stage.
//
// Runtime arguments (the layout is R_* in ozaki2/kernels/recon_args.h):
//             in_addr, num_tiles, start_id, ntot, S, mask,
//             nlimb, Whi, Wlo, qw, sh2, jsplit, sem_pub, sem_done
//   When SHARE (compile-time argument CT_FINAL_OR_SHARE: whether the assembly is shared) is 0 (the
//   configuration without sharing; the library never uses it, the host always passes 1), only the
//   first 6 words are read.

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"
#include "recon_elem.h"
#include "plane_fetch.h"
#include "../../common/kernels/l1_order.h"
using namespace recon_args;

void kernel_main() {
    // The argument layout follows the single table in ozaki2/kernels/recon_args.h
    uint32_t in_addr   = get_arg_val<uint32_t>(R_IN_ADDR);
    uint32_t num       = get_arg_val<uint32_t>(R_NUM);
    uint32_t start     = get_arg_val<uint32_t>(R_START);
    uint32_t ntot      = get_arg_val<uint32_t>(R_NTOT);
    uint32_t S         = get_arg_val<uint32_t>(R_S);
    uint32_t mask      = get_arg_val<uint32_t>(R_MASK);   // 1: send the masks
    constexpr uint32_t cb_in   = CB_IN;
    constexpr uint32_t cb_mask = CB_MASK;
    constexpr uint32_t cb_out  = CB_LIMB;
    constexpr uint32_t cb_f64  = CB_F64;
    constexpr uint32_t cb_da   = CB_DLT_A;   // the deltas read by the writer are used as is
    constexpr uint32_t cb_db   = CB_DLT_B;
    constexpr uint32_t cb_sync = CB_SYNC;
    // Scratch area of the writer: one tile of 128-bit fixed point (OZ_A128_TILE_BYTES)
    constexpr uint32_t cb_q    = CB_A128;
    constexpr auto ia = TensorAccessorArgs<0>();
    const auto in = TensorAccessor(ia, in_addr);
    // Only with SHARE does this kernel take the first part of the assembly (in the first pass of the
    // two-pass configuration, of the conversion to 128-bit fixed point). The host passes 1 in every
    // pass of every configuration (recon_make_kernels in ozaki2/stages.hpp, append_rw_ct(ra, 1u, ..));
    // the branch for SHARE = 0 (reads only) is kept for completeness.
    constexpr uint32_t SHARE = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_FINAL_OR_SHARE);
    // Role of the pass (ReconRole of recon_args.h. ROLE_SINGLE = final pass of the one-pass
    // configuration (assembles the 3 limbs), ROLE_TWO_FINAL = final pass of the two-pass configuration
    // (adds the 3 limbs to the fixed point of the first pass and assembles), ROLE_TWO_FIRST = first
    // pass of the two-pass configuration (only adds the 3 limbs into fixed point))
    constexpr uint32_t ROLE  = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_ROLE);
    static_assert(ROLE == ROLE_SINGLE || ROLE == ROLE_TWO_FINAL || ROLE == ROLE_TWO_FIRST, "the pass role must be a ReconRole");
    // PROBE: diagnostic selector (not used by the library; always 0; a nonzero value breaks the
    // answer). 1 removes only assemble_frac_f64. It has the same index as on the writer side
    // (CT_PROBE), and the host passes the same value to both. With the default sharing this side
    // holds the first part of the assembly, so half of it remains unless the selector is applied on
    // both sides.
    constexpr uint32_t PROBE = get_compile_time_arg_val(ia.next_compile_time_args_offset() + CT_PROBE);

    // On the paths that do not send the masks, c_1 itself is not created, so it is not referenced
    // either. The masks are constants and are built in the L1 of this core: they are neither placed
    // in DRAM nor sent from the host (by specification there is no communication between the transfer
    // of A and B and the return of C).
    if (mask) {
        cb_reserve_back(cb_mask, 2);
        volatile tt_l1_ptr uint32_t* mp = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_mask);
        for (uint32_t z = 0; z < 1024; ++z) { mp[z] = 0x7FFF0000u; mp[1024 + z] = 0x0000FFFFu; }
        // Hand over to the compute side only after the writes have landed in L1
        // (common/kernels/l1_order.h)
        l1_writes_landed((uint32_t)&mp[2047]);
        cb_push_back(cb_mask, 2);
    }
    const uint32_t tb = get_tile_size(cb_in);
    // The S planes of one output tile are reserved together and read with a single barrier
    // (c_0 has depth 2S, so the read can run one tile ahead).
    auto fetch = [&](uint32_t g) { oz2_fetch_planes(cb_in, in, g, S, ntot, tb); };

    if constexpr (!SHARE) {
        for (uint32_t g = start; g < start + num; ++g) fetch(g);
        return;
    }

    const uint32_t nlimb = get_arg_val<uint32_t>(R_NLIMB);
    const uint32_t Whi = get_arg_val<uint32_t>(R_WHI);
    const uint32_t Wlo = get_arg_val<uint32_t>(R_WLO);
    const int32_t  qw  = (int32_t)get_arg_val<uint32_t>(R_QW);
    const int32_t  sh2 = (int32_t)get_arg_val<uint32_t>(R_SH2);
    // Split point of the sharing. This side also does the input reads, so somewhat less than half
    // balances the two.
    const uint32_t jsplit = get_arg_val<uint32_t>(R_JSPLIT);
    volatile tt_l1_ptr uint32_t* sy = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_sync);
    volatile tt_l1_ptr uint32_t* pub =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(R_SEM_PUB));
    volatile tt_l1_ptr uint32_t* done =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(R_SEM_DONE));
    // Scratch areas in the same L1 as the writer. The first pass (ROLE_TWO_FIRST) does not create
    // c_19 to c_21, so they are not referenced
    const uint32_t fp = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_f64);
    const uint32_t ap = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_da);
    const uint32_t bp = (ROLE == ROLE_TWO_FIRST) ? 0u : get_write_ptr(cb_db);
    const uint32_t qp = (ROLE == ROLE_SINGLE) ? 0u : get_write_ptr(cb_q);

    if (num) fetch(start);
    for (uint32_t g = start; g < start + num; ++g) {
        if (g + 1 < start + num) fetch(g + 1);   // keeps the compute kernel from stalling
        const uint32_t tag = g - start + 1;
        // Wait until the writer has placed the address of the limbs. The L0 data cache is not
        // invalidated by the writes of another RISC-V, so it is invalidated on every iteration before
        // L1 is read, and once more after the loop exits before the limbs and the deltas are read
        // (tt-isa-documentation, MemoryOrdering.md, "L0 data cache").
        while (*pub != tag) { invalidate_l1_cache(); }
        invalidate_l1_cache();
        const uint32_t p = sy[0], ts = get_tile_size(cb_out);
        const tt_l1_ptr uint32_t* l0 = (const tt_l1_ptr uint32_t*)p;
        const tt_l1_ptr uint32_t* l1p = (const tt_l1_ptr uint32_t*)(p + ts);
        const tt_l1_ptr uint32_t* l2 = (const tt_l1_ptr uint32_t*)(p + 2 * ts);
        const tt_l1_ptr int32_t*  da = (const tt_l1_ptr int32_t*)ap;
        const tt_l1_ptr int32_t*  db = (const tt_l1_ptr int32_t*)bp;
        tt_l1_ptr uint32_t* po = (tt_l1_ptr uint32_t*)fp;
        tt_l1_ptr uint32_t* qa4 = (tt_l1_ptr uint32_t*)qp;
        for (uint32_t j = 0; j < jsplit; ++j) {  // only the first part is handled here
            // ROLE_TWO_FIRST: only adds into fixed point and stores the result
            if constexpr (ROLE == ROLE_TWO_FIRST) recon_elem::pack_a128<PROBE>(l0, l1p, l2, qa4, j);
            else recon_elem::assemble_f64<ROLE, PROBE>(l0, l1p, l2, nlimb, da, db, qa4, po, j, sh2, Whi, Wlo, qw);
        }
        *done = tag;                             // the first part is done
    }
}
