// SPDX-License-Identifier: Apache-2.0
//
// Matmul kernel whose read-back of the K spill is error-free (shared by all paths)
// ---------------------------------------------------------------------------
// This is the standard bmm_large_block_zm of tt-metal, modified so that the spill through c_24 is
// error-free.
//
// What happens with the unmodified standard kernel
//   The standard kernel uses copy_tile_to_dst_init_short / mm_init_short, but these do not reconfigure
//   the data format of the unpacker. c_24 (Int32) would therefore be read back while the unpack
//   configuration for INT8 is still in place, and the 32-bit partial sums would be corrupted.
//
// What this kernel does about it
//   It uses the versions with the _with_dt suffix (copy_tile_to_dst_init_short_with_dt /
//   mm_init_short_with_dt) and switches the data format before and after the read-back.
//   The host has to give c_24 the format of the partial products (Int32 on the INT8 path, Float32
//   otherwise; acc_fmt of spill_gemm_spec in common/mcast_gemm.hpp) and has to set UnpackToDestFp32 in
//   unpack_to_dest_mode[c_24] (spill_gemm_spec does both).
//
//   Compile-time arguments 0-11 and the overall structure are the same as in the standard kernel, with
//   the calls that reconfigure the data format added. These are the places marked **difference from the
//   standard kernel** below (arguments 12 and above, and the L1 accumulation, FUSE and PAIRLOOP described
//   below, were added by this library).
//
// Accumulation across K blocks (L1 accumulation of the packer)
//   The packer adds the partial sums across K blocks in L1 (c_24) (pack_reconfig_l1_acc; the same as
//   PACKER_L1_ACC of tt-metal), and they are read back into DST only once, at the last block of a segment
//   (one pair with PAIRLOOP, the whole of K otherwise), which avoids a read-back and reconfiguration of
//   the unpacker at every block.
//   Exactness: the partial sums in DST are FP32 (Int32 on the INT8 path), and the L1 adder adds in the
//   same format. If all integer partial sums are at most 2^24 (FP32 represents 2^24 exactly), no addition
//   rounds. This is the same condition as for the accumulation inside DST (K_eff*d^2 <= 2^24 for one
//   segment; ozaki_max_digit in ozaki1/format.hpp), and the host already imposes it. The Int32 addition in
//   L1 is a sign-magnitude integer addition, verified on hardware to be exact up to just below 2^31
//   (2,147,092,480; including odd intermediate sums, negative values and block sums of opposite signs).
//   The bound that the host imposes on the partial sums of the INT8 path with the digits +-127
//   (K_eff*127^2 <= 2^31-1 with K a multiple of 32: ozaki_max_digit / ozaki_block_k in
//   ozaki1/format.hpp for scheme I, the modulus cap in ozaki2/moduli.hpp for scheme II) stays within
//   that verified range.
//
// Fusing the Int32 accumulation into the output path (FUSE)
//   In the deferred reconstruction of Ozaki scheme I, the partial sum P_d of diagonal d is added to the
//   accumulation plane S_d in DRAM. When a separate program does this, each update passes 4 planes
//   through DRAM (write P, read P, read S, write S); when it is fused into the final block of this
//   kernel, 2 planes suffice (read S, write S), because P no longer goes through DRAM.
//   With FUSE set, the output of the final block is converted to Int32, added to the existing S tiles
//   (c_2) that the writer has prefetched, and packed as Int32.
//   With fp32_dest_acc, DST holds only four 32-bit tiles and the output subblock uses all of them, so S
//   cannot be loaded at the same time. The matmul result is therefore first output to c_3, still in
//   FP32, and then reloaded two tiles at a time and added.
//   The Int32 representation is two's complement (typecast<Float32,Int32> and add_int_tile both use
//   two's complement), so the decoding in the final stage has to run recon_tf_i32in.cpp with SIGNMAG=0
//   (that is, as two's complement).
//
// Running the pairs inside one program (PAIRLOOP)
//   In the layout with concatenated slices, the (d+1) pairs of diagonal d are contiguous in the K
//   direction (pair a is the product of slice a of A and slice d-a of B, and when a advances by 1, both A
//   and B advance by exactly one slice). A single GEMM issued over the concatenation therefore runs the
//   (d+1) pairs as it is. Concatenating without PAIRLOOP is disadvantageous because the compute kernel
//   accumulates all pairs in one FP32 accumulator, so the exactness condition becomes K_eff=(d+1)K_c and
//   K has to be split.
//   PAIRLOOP folds the FP32 accumulator at each pair boundary (every bpp blocks), adds it to the Int32
//   plane in L1 (c_5), and restarts the accumulation for the next pair from 0. With this,
//   - the exactness condition only has to hold for K_eff=K_c of one pair, and the K blocking disappears
//   - the partial sums between pairs do not go through DRAM (S_d makes the round trip only once)
//   The accumulation plane in L1 is a ring of pm*pn + out_subblock tiles so that the region being written
//   does not overlap the region being read.
//
// Parts: kernel_main holds the loops over the K blocks and the subblocks (read-back, matmul, output and
// second pass, in this order; the synchronization points are the same as in the standard kernel) and calls
// pack_subblock, which outputs a subblock to its destination, and second_pass, the second pass. Both are
// always_inline, and their arguments and comments show which state they read and which they change (the
// loop state enable_reload and l1acc_on is held by kernel_main alone).

#include <cstdint>

#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/add_int_sfpu.h"
// Argument layout and CB indices (shared with common/mcast_gemm.hpp on the host)
#include "gemm_compute_args.h"
using namespace gemm_args;

// ---- Configuration (compile-time arguments; all are constants, and the parts below only read them) ----
// With FUSE set, the output of the final block is added to the existing S in Int32 (see the note above).
// F32 and I32 are the data formats for the typecast.
constexpr uint32_t FUSE = get_compile_time_arg_val(CA_FUSE);
constexpr uint32_t F32 = get_compile_time_arg_val(CA_F32);
constexpr uint32_t I32 = get_compile_time_arg_val(CA_I32);
// PAIRLOOP (compile-time argument CA_PAIRLOOP): with 0, all blocks are accumulated in a single FP32
// accumulator as in the standard kernel, and the runtime argument CR_BPP is not read. With 1, bpp from
// CR_BPP is the number of K blocks per pair; bpp must be at least 1 and must divide the number of K blocks
// (spill_check_run on the host stops otherwise. bpp=0 does not mean "turn the pair loop off"; it is an
// invalid value that makes the remainder below a division by zero. The pair loop is turned off with
// PAIRLOOP=0).
constexpr uint32_t PAIRLOOP = get_compile_time_arg_val(CA_PAIRLOOP);
// BLOCKMM: instead of matmul_tiles, which issues a subblock one tile at a time, use matmul_block, which
// takes the whole (rt,ct) outer product at once. The unpacker then knows about the reuse within the
// subblock, which can reduce the work per issue.
constexpr uint32_t BLOCKMM = get_compile_time_arg_val(CA_BLOCKMM);

constexpr auto cb_in0 = tt::CBIndex(CB_IN0);
constexpr auto cb_in1 = tt::CBIndex(CB_IN1);
constexpr auto cb_out = tt::CBIndex(CB_OUT);
constexpr auto cb_acc = tt::CBIndex(CB_PARTIAL);
constexpr auto cb_s   = tt::CBIndex(CB_S);      // fused: the existing S_d (Int32)
constexpr auto cb_tmp = tt::CBIndex(CB_TMP);    // fused: the matmul result (FP32)
constexpr auto cb_sl  = tt::CBIndex(CB_SL);     // PAIRLOOP: Int32 accumulation plane across pairs

// Shape of the output subblock (kernel_main builds it from the compile-time arguments; the parts only
// read it)
struct SubblockShape { uint32_t h, w, ntiles, in0_block_w; };

// Returns the unpacker (and the math unit) to the matmul configuration. old_cb is the CB that the unpacker
// was reading just before (the old format for _with_dt)
ALWI void mm_restore_with_dt(const SubblockShape& sb, uint32_t old_cb) {
    if constexpr (BLOCKMM)
        mm_block_init_short_with_dt(cb_in0, cb_in1, old_cb, 0, sb.w, sb.h, sb.in0_block_w);
    else
        mm_init_short_with_dt(cb_in0, cb_in1, old_cb);
}

// Outputs the subblock in DST to its destination (c_16, c_3 or c_5 for the last block of a segment, c_24
// otherwise).
// State read: pair_end, first_pair, last_out, segj, block (the position of this block within its segment),
// and first_sb (whether this is the first subblock of this block). State changed: out_num_tiles_to_wait
// (the number of tiles of c_16 reserved in the first block), the handover of each CB, and the format of the
// packer. In the typecast of the first pair, the configuration of the math unit is returned to the matmul
// (the unpacker is not touched)
ALWI void pack_subblock(const SubblockShape& sb, bool pair_end, bool first_pair, bool last_out, uint32_t segj,
                        uint32_t block, bool first_sb, uint32_t& out_num_tiles_to_wait) {
    const uint32_t out_subblock_num_tiles = sb.ntiles;
    if (pair_end && PAIRLOOP && first_pair) {
        // First pair: there is nothing to add to, so the result is converted to Int32 in place and
        // stored (no need to save it to c_3). Diagonal 0 has only one pair, so this is also the last
        // pair; in that case the result goes to the output instead of the accumulation plane.
        // All subblocks of this block go to the same cb_dst and no other pack comes in between (the
        // second pass is not taken for the first pair), so the format has to be reconfigured only for
        // the first subblock.
        const auto cb_dst = last_out ? cb_out : cb_sl;
        typecast_tile_init<F32, I32>();
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            typecast_tile<F32, I32>(i);
        cb_reserve_back(cb_dst, out_subblock_num_tiles);
        if (first_sb) pack_reconfig_data_format(cb_dst);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++)
            pack_tile(i, cb_dst);
        cb_push_back(cb_dst, out_subblock_num_tiles);
        // The typecast has changed the configuration of the math unit, so it is restored before the
        // matmul of the next pair starts (the unpacker has not been touched).
        if constexpr (BLOCKMM)
            mm_block_init_short(cb_in0, cb_in1, 0, sb.w, sb.h, sb.in0_block_w);
        else
            mm_init_short(cb_in0, cb_in1);
    } else if (pair_end && (PAIRLOOP || FUSE)) {
        // Save the matmul result temporarily to c_3, still in FP32.
        // Unless DST is released and the result reloaded, there is no room for the tiles to add.
        // The format has to be reconfigured for every subblock (the second pass in between outputs to
        // the Int32 CBs c_5 / c_16).
        cb_reserve_back(cb_tmp, out_subblock_num_tiles);
        pack_reconfig_data_format(cb_tmp);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_tmp);
        }
        cb_push_back(cb_tmp, out_subblock_num_tiles);
    } else if (last_out) {
        cb_reserve_back(cb_out, out_subblock_num_tiles);
        // **Difference from the standard kernel**: reconfigure the packer for the output CB. All
        // subblocks of the last block go to c_16 and no other pack comes in between, so once at the
        // first subblock is enough
        if (first_sb) pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_out);
        }
        cb_push_back(cb_out, out_subblock_num_tiles);
    } else {
        if (block == 0) {
            cb_reserve_back(cb_out, out_num_tiles_to_wait);
            out_num_tiles_to_wait += out_subblock_num_tiles;
        }
        cb_reserve_back(cb_acc, out_subblock_num_tiles);
        // **Difference from the standard kernel**: reconfigure the packer for c_24 (Int32).
        // Output goes to something other than c_24 only at the end of a segment, so the format has to
        // be reconfigured only at the first subblock of the first block of a segment (a
        // reconfiguration involves a STALL)
        if (segj == 0 && first_sb)
            pack_reconfig_data_format(cb_acc);
        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, cb_acc);
        }
        cb_push_back(cb_acc, out_subblock_num_tiles);
    }
}

// Second pass (at the last block of a segment when PAIRLOOP or FUSE is set; not for the first pair):
// converts the FP32 in c_3 to Int32, adds the addend that belongs to it, and outputs the sum to the
// destination.
//   intermediate pair of PAIRLOOP : the accumulation plane c_5 in L1 (the destination is also c_5)
//   last pair of PAIRLOOP         : add c_5 and output to c_16
//   fused (FUSE)                  : add the S from DRAM that the writer has prefetched (c_2) and output
//                                   to c_16 (if acc is 0, initialize without adding)
// There is at most one addend: the host makes fused and pair loop mutually exclusive (TT_FATAL in
// spill_gemm_spec in common/mcast_gemm.hpp). With fp32_dest_acc, DST holds only four 32-bit tiles, and n
// result tiles plus n addend tiles are needed, so the pass proceeds two tiles at a time.
// State read: last_out (whether this is the last pair) and acc. State changed: the handover of c_3, c_5,
// c_2 and c_16, and the configuration of the unpacker, the packer and the math unit (at the end the
// unpacker is returned to the matmul configuration). DST is acquired and released every two tiles in here
ALWI void second_pass(const SubblockShape& sb, bool last_out, uint32_t acc) {
    const uint32_t out_subblock_num_tiles = sb.ntiles;
    const bool use_sl  = PAIRLOOP;                 // whether c_5 is added
    const bool to_out  = !PAIRLOOP || last_out;    // whether the destination is c_16
    const bool use_dram = FUSE && acc && to_out;   // whether c_2 is added
    const bool add_any = use_sl || use_dram;
    const auto cb_dst = to_out ? cb_out : cb_sl;
    const auto cb_add = use_sl ? cb_sl : cb_s;     // CB of the addend
    cb_wait_front(cb_tmp, out_subblock_num_tiles);
    if (add_any) cb_wait_front(cb_add, out_subblock_num_tiles);
    cb_reserve_back(cb_dst, out_subblock_num_tiles);
    // The packer outputs only to cb_dst during this pass, so the format is reconfigured once
    // (inside the loop that steps by two tiles, a 4-tile subblock would reconfigure it twice)
    pack_reconfig_data_format(cb_dst);
    // The format of the unpacker is reconfigured starting from "the CB that was configured just before"
    // (_with_dt does nothing if the old and the new format are the same). Just before the first
    // iteration of the loop that is the matmul configuration (SrcA is cb_in1); from the second
    // iteration on it is cb_add or cb_tmp, whichever was read last in the previous iteration.
    // If the switch were always made from cb_in1, then in a configuration where cb_in1 has the same
    // format as c_3 the reconfiguration in the second iteration would be skipped, and the FP32 data of
    // c_3 would be read with the Int32 configuration (c_5 / c_2) still in place.
    // In I-FP32 (TF32) both the GEMM input cb_in1 and c_3 are Float32, so this really happens
    // (PAIRLOOP is the default on that path). Tracking the CB read last in prev_cb cannot be omitted.
    uint32_t prev_cb = cb_in1;
    for (uint32_t i = 0; i < out_subblock_num_tiles; i += 2) {
        uint32_t n = out_subblock_num_tiles - i;
        if (n > 2) n = 2;
        acquire_dst();
        copy_tile_to_dst_init_short_with_dt(prev_cb, cb_tmp);
        for (uint32_t j = 0; j < n; j++) copy_tile(cb_tmp, i + j, j);
        prev_cb = cb_tmp;
        typecast_tile_init<F32, I32>();
        for (uint32_t j = 0; j < n; j++) typecast_tile<F32, I32>(j);
        if (add_any) {   // the results are placed in 0..n-1 and the addends in n..2n-1
            copy_tile_to_dst_init_short_with_dt(cb_tmp, cb_add);
            for (uint32_t j = 0; j < n; j++) copy_tile(cb_add, i + j, n + j);
            prev_cb = cb_add;
            add_int_tile_init();
            for (uint32_t j = 0; j < n; j++)
                add_int_tile<DataFormat::Int32>(j, n + j, j);
        }
        for (uint32_t j = 0; j < n; j++) pack_tile(j, cb_dst);
        release_dst();
    }
    cb_push_back(cb_dst, out_subblock_num_tiles);
    cb_pop_front(cb_tmp, out_subblock_num_tiles);
    // The addend CB discards what has been read (the ring has a margin of out_subblock tiles, so the
    // reserve does not overlap the region being read)
    if (add_any) cb_pop_front(cb_add, out_subblock_num_tiles);
    // Return the unpacker to the matmul configuration (for the next subblock and the next pair)
    mm_restore_with_dt(sb, add_any ? cb_add : cb_tmp);
}

void kernel_main() {
    uint32_t in0_block_w = get_compile_time_arg_val(CA_IN0_BLOCK_W);
    uint32_t in0_num_subblocks = get_compile_time_arg_val(CA_IN0_NUM_SUBBLOCKS);
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(CA_IN0_BLOCK_NUM_TILES);
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(CA_IN0_SUBBLOCK_NUM_TILES);
    uint32_t in1_num_subblocks = get_compile_time_arg_val(CA_IN1_NUM_SUBBLOCKS);
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(CA_IN1_BLOCK_NUM_TILES);
    uint32_t in1_per_core_w = get_compile_time_arg_val(CA_IN1_PER_CORE_W);
    // num_blocks alone is not read from the compile-time argument CA_NUM_BLOCKS (index 7) but is the
    // runtime argument CR_NUM_BLOCKS: in the GEMM stage of the Ozaki schemes the length of K, (d+1)*K,
    // changes with the diagonal d, and this allows a single compiled program to be reused (argument 7
    // remains in order to share the layout with the standard kernel; see the top of
    // gemm_compute_args.h).
    uint32_t num_blocks = get_arg_val<uint32_t>(CR_NUM_BLOCKS);
    uint32_t out_subblock_h = get_compile_time_arg_val(CA_OUT_SUBBLOCK_H);
    uint32_t out_subblock_w = get_compile_time_arg_val(CA_OUT_SUBBLOCK_W);
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(CA_OUT_SUBBLOCK_NUM_TILES);
    uint32_t batch = get_compile_time_arg_val(CA_BATCH);
    const SubblockShape sb{out_subblock_h, out_subblock_w, out_subblock_num_tiles, in0_block_w};
    // acc is the runtime argument CR_ACC: 0 for the first write to that plane (initialize without
    // adding; read only when FUSE is set).
    const uint32_t acc = FUSE ? get_arg_val<uint32_t>(CR_ACC) : 0u;
    // 1 is used when PAIRLOOP is not set as well: with 0, constant folding would flag the remainder
    // below as a division by zero and the compilation would fail (the branch is evaluated even though
    // it is dead). With PAIRLOOP set, the host always passes a value of at least 1.
    const uint32_t bpp = PAIRLOOP ? get_arg_val<uint32_t>(CR_BPP) : 1u;
    // Number of blocks in one segment (the unit after which the accumulator restarts from 0)
    const uint32_t seglen = PAIRLOOP ? bpp : num_blocks;
    const uint32_t out_block_num_tiles = in0_num_subblocks * in1_num_subblocks * out_subblock_num_tiles;
    // Whether the L1 accumulation of the packer is currently on (a reconfiguration involves a STALL, so
    // the state is tracked)
    bool l1acc_on = false;

    if constexpr (BLOCKMM) mm_block_init(cb_in0, cb_in1, cb_out, 0,
                                         out_subblock_w, out_subblock_h, in0_block_w);
    else                    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t b = 0; b < batch; b++) {
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for (uint32_t block = 0; block < num_blocks; block++) {
            // With PAIRLOOP the accumulator is folded at every pair boundary.
            // pair_end is that boundary, and last_out is the boundary of the last pair of the whole
            // program.
            const bool last_out = block == (num_blocks - 1);
            const bool pair_end = PAIRLOOP ? ((block + 1) % bpp == 0) : last_out;
            const bool first_pair = PAIRLOOP ? (block < bpp) : true;
            const uint32_t segj = PAIRLOOP ? (block % bpp) : block;   // position within the segment
            // The last block of a segment outputs to something other than c_24 (c_16, c_3 or c_5), so
            // the L1 accumulation is turned off. The second and later blocks of a segment add to the
            // sum in c_24.
            const bool want_l1acc = !pair_end && segj > 0;
            if (want_l1acc != l1acc_on) { pack_reconfig_l1_acc(want_l1acc ? 1 : 0); l1acc_on = want_l1acc; }

            cb_wait_front(cb_in0, in0_block_num_tiles);
            cb_wait_front(cb_in1, in1_block_num_tiles);
            int in0_index_subblock_offset = 0;
            for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                int in1_index_subblock_offset = 0;
                for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                    // acquire_dst/release_dst are [[deprecated]] in tt-metal v0.73.1 (reg_api.h) but are
                    // kept: tile_regs_* would require moving the pack (between commit and wait) and would
                    // change the instruction sequence of the GEMM stage.
                    acquire_dst();

                    if (enable_reload) {
                        // **Difference from the standard kernel**: reconfigure the unpacker to read c_24
                        // (Int32 or Float32)
                        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_acc);
                        cb_wait_front(cb_acc, out_subblock_num_tiles);
                        for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                            copy_tile(cb_acc, i, i);
                        }
                        cb_pop_front(cb_acc, out_subblock_num_tiles);
                        // **Difference from the standard kernel**: return the unpacker to the input
                        // format (Int8 on the INT8 path)
                        mm_restore_with_dt(sb, cb_acc);
                    }

                    if constexpr (BLOCKMM) {
                        // For every inner K tile, the (out_subblock_h x out_subblock_w) outer product
                        // is added to DST at once. A advances one tile to the right, and B advances
                        // one row (in1_per_core_w) down.
                        uint32_t in0_index = (uint32_t)in0_index_subblock_offset;
                        uint32_t in1_index = (uint32_t)in1_index_subblock_offset;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                            matmul_block(cb_in0, cb_in1, in0_index, in1_index, 0, 0,
                                         out_subblock_w, out_subblock_h, in0_block_w);
                            in0_index++;
                            in1_index += in1_per_core_w;
                        }
                    } else {
                        int dst_index = 0;
                        int in0_index_h_offset = 0;
                        for (uint32_t h = 0; h < out_subblock_h; h++) {
                            for (uint32_t w = 0; w < out_subblock_w; w++) {
                                int in1_index_inner_dim_offset = 0;
                                for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                    int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                                    int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                                    matmul_tiles(cb_in0, cb_in1, in0_index, in1_index, dst_index);
                                    in1_index_inner_dim_offset += in1_per_core_w;
                                }
                                dst_index++;
                            }
                            in0_index_h_offset += in0_block_w;
                        }
                    }

                    // Whether this is the first subblock of this block (reconfiguring the format of the
                    // packer involves a STALL, so it is done once while the output keeps going to the
                    // same destination)
                    const bool first_sb = in0_subblock == 0 && in1_subblock == 0;
                    pack_subblock(sb, pair_end, first_pair, last_out, segj, block, first_sb, out_num_tiles_to_wait);

                    release_dst();

                    // At the end of the first pair (pair_end && PAIRLOOP && first_pair) the result has
                    // already been output by pack_subblock above, so there is no second pass and the
                    // unpacker can stay in the matmul configuration.
                    if (pair_end && (PAIRLOOP || FUSE) && !(PAIRLOOP && first_pair)) second_pass(sb, last_out, acc);
                    in1_index_subblock_offset += out_subblock_w;
                }
                in0_index_subblock_offset += in0_subblock_num_tiles;
            }

            // c_24 is a ring of exactly pm*pn tiles, so once the tiles written are discarded, the pack
            // of the next block adds to the same L1 location (= the running sum). Only the sum of the
            // second-to-last block of a segment is kept, and it is read back subblock by subblock in
            // the last block. When the segment changes (pair_end), the accumulator restarts from 0.
            // The discard is done once for a whole block (pm*pn tiles = the capacity of c_24) rather
            // than once per subblock, which costs measurable time. The number of tiles waited for is
            // exactly the pm*pn tiles packed in this block and does not exceed the capacity of c_24.
            if (pair_end) {
                enable_reload = false;
            } else if (segj + 2 < seglen) {
                cb_wait_front(cb_acc, out_block_num_tiles);
                cb_pop_front(cb_acc, out_block_num_tiles);
            } else {
                enable_reload = true;
            }

            cb_pop_front(cb_in0, in0_block_num_tiles);
            cb_pop_front(cb_in1, in1_block_num_tiles);
        }
    }
    if (l1acc_on) pack_reconfig_l1_acc(0);
}
