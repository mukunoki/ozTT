// SPDX-License-Identifier: Apache-2.0
//
// Ozaki scheme II: data-movement kernel that writes the result of the CRT reconstruction to DRAM
// ---------------------------------------------------------------------------
// The first pass of the two-pass configuration (ROLE_TWO_FIRST, FINAL=0) adds, for every output tile
// g, the 3 triple-float limbs of frac in c_16 into 128-bit fixed point (frac * 2^124; 4 words = 16 B
// per element, w[0] is the least significant) and writes it to out0[g] as a page of 16 KB per tile
// (the final pass reads it and adds its own 3 limbs; the value is bit-identical to adding the 6 limbs
// together; see a128_add_limbs in f64_tf.h).
// A pass with FINAL set (the one-pass configuration and the final pass of the two-pass configuration)
// assembles the FP64 C from frac and writes it to out64. The library sets FINAL for every role but
// ROLE_TWO_FIRST; a pass with FINAL=0 and another role (a diagnostic the library does not use) would
// write the nlimb FP32 limb tiles of frac as they are to out0[g], out1[g], out2[g].
// Only integer arithmetic is used: the RISC-V cores that do the data movement have no FP64 unit
// (only part of the single-precision F extension and no double-precision D extension; Blackhole ISA
// documentation, BabyRISCV/InstructionSet.md), so writing double would become soft-float (about
// 76 ns/op).
//
//   C = frac * KM * tau_A * tau_B
// where tau is a power of two and only requires an addition to the exponent (taken from the row and
// column exponent-delta tiles). KM is not a power of two, so it is received split into a 64-bit
// significand W and an exponent qw.
//
// The only difference from the writer of scheme I is the "width of the collection". The absolute
// precision of frac directly determines the attainable precision (about 2^-99 in the two-pass
// configuration) while frac itself is around 2^-8, so the collection needs a width of more than
// 90 bits and the 64 bits of scheme I are not enough. The limbs are therefore collected into
//   acc = frac * 2^124   (128-bit fixed point, 4 words)
// For a limb with exponent e the shift amount is e-26; a limb at the upper end 2^-1 does not exceed
// 2^124 and a limb at the lower end 2^-97 does not lose a single bit. The final pass of the two-pass
// configuration (ROLE_TWO_FINAL) reads the 128-bit fixed point that the first pass left in DRAM and
// adds its own 3 limbs to it. With one pass or with two, the collected value is folded while it is
// still in fixed point (values in (-1/2,1/2) are unchanged and the excess is returned by +-1; the
// image is [-1/2,1/2]; f64_tf.h). The compute kernel (recon_crt.cpp) does not fold; it only removes
// the integer part per modulus with a0 -= round(a0), so the sum of the limbs of each pass can leave
// [-1/2,1/2] by a1 + a2 (the magnitude of the tail). With two passes those sums are added to each
// other, so the excesses add up as well. In every case a single +-1 brings the value back (the sum is
// inside (-3/2,3/2); see assemble_frac_f64 in f64_tf.h). If the compute kernel folded with FP32 sums
// it would miss the excess at the edge of the window ("The fold of the fraction" in recon_crt.cpp),
// so the fold is done here.
// The per-element procedure is recon_elem.h, the same as for the first part done by the reader; the
// collecting procedure and the stage that multiplies by W and rounds are in common/kernels/f64_tf.h
// (assemble_frac_f64).
//
// Runtime arguments (the layout is W_* in ozaki2/kernels/recon_args.h):
//             out0_addr, num_tiles, start_id, out1_addr, out2_addr, nlimb,
//             out64_addr, dltA_addr, dltB_addr, Nt, Whi, Wlo, qw, sh2,
//             a128_addr (for ROLE_TWO_FINAL, the 128-bit fixed point of the first pass; 16 KB pages),
//             jlo (number of leading elements whose assembly is handed to the reader; 0 if not shared),
//             sem_pub, sem_done (semaphores of the handoff with the reader),
//             out_off (first tile of the band when the output is split into row bands; added to the
//                      tile index of o64 and of the exponent deltas)

#include "api/dataflow/dataflow_api.h"
#include "../../common/kernels/f64_tf.h"
#include "recon_args.h"
#include "recon_elem.h"
#include "../../common/kernels/l1_order.h"
using namespace recon_args;

constexpr uint32_t cb_out = CB_LIMB;
constexpr uint32_t cb_f64 = CB_F64;    // scratch: FP64 tile (OZ_F64_TILE_BYTES)
constexpr uint32_t cb_sync = CB_SYNC;  // handoff with the reader (one word, limb address; SYNC_PAGE B)
constexpr uint32_t cb_da  = CB_DLT_A;  // scratch: row exponent deltas
constexpr uint32_t cb_db  = CB_DLT_B;  // scratch: column exponent deltas
// Scratch: one tile of 128-bit fixed point (OZ_A128_TILE_BYTES). The first pass writes it; in the
// final pass it holds what was read
constexpr uint32_t cb_q   = CB_A128;

void kernel_main() {
    const uint32_t a0    = get_arg_val<uint32_t>(W_OUT0);
    const uint32_t num   = get_arg_val<uint32_t>(W_NUM);
    const uint32_t start = get_arg_val<uint32_t>(W_START);
    const uint32_t a1    = get_arg_val<uint32_t>(W_OUT1);
    const uint32_t a2    = get_arg_val<uint32_t>(W_OUT2);
    const uint32_t nlimb = get_arg_val<uint32_t>(W_NLIMB);
    const uint32_t o64   = get_arg_val<uint32_t>(W_O64);
    const uint32_t dAa   = get_arg_val<uint32_t>(W_DLT_A);
    const uint32_t dBa   = get_arg_val<uint32_t>(W_DLT_B);
    const uint32_t Nt    = get_arg_val<uint32_t>(W_NT);
    const uint32_t Whi   = get_arg_val<uint32_t>(W_WHI);
    const uint32_t Wlo   = get_arg_val<uint32_t>(W_WLO);
    const int32_t  qw    = (int32_t)get_arg_val<uint32_t>(W_QW);
    const int32_t  sh2   = (int32_t)get_arg_val<uint32_t>(W_SH2);   // 2*(b-(ND-1)*G)
    const uint32_t q0a   = get_arg_val<uint32_t>(W_A128);
    // Sharing of the assembly: the elements below jlo have been done by the reader (recon_reader.cpp).
    // Without sharing, jlo=0 and all 1024 elements are assembled here.
    const uint32_t jlo   = get_arg_val<uint32_t>(W_JLO);
    // First tile of the band: o64 and the row and column exponent deltas are addressed with the index
    // g + off over the whole C (the limbs and a128 are containers for a single band and keep g).
    // off is a multiple of Nt (the bands are cut along rows), so the column exponent delta index
    // (g + off) % Nt is the same as g % Nt
    const uint32_t off   = get_arg_val<uint32_t>(W_OUT_OFF);
    // The tags are kept in semaphores: only the semaphores return to their initial value at every
    // program launch, while the contents of a CB persist across launches (see the note in
    // recon_reader.cpp).
    volatile tt_l1_ptr uint32_t* pub =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(W_SEM_PUB));
    volatile tt_l1_ptr uint32_t* done =
        (volatile tt_l1_ptr uint32_t*)get_semaphore(get_arg_val<uint32_t>(W_SEM_DONE));

    constexpr auto t0a = TensorAccessorArgs<0>();
    const auto o0 = TensorAccessor(t0a, a0);
    constexpr auto t1a = TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    const auto o1 = TensorAccessor(t1a, a1);
    constexpr auto t2a = TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    const auto o2 = TensorAccessor(t2a, a2);
    // FINAL / ROLE are fixed at compile time (the program is a separate one per pass). ROLE is
    // ReconRole of recon_args.h (ROLE_SINGLE = one-pass configuration, ROLE_TWO_FINAL = final pass of
    // the two-pass configuration (reads the fixed point of the first pass and adds to it),
    // ROLE_TWO_FIRST = first pass of the two-pass configuration (adds its own 3 limbs into fixed point
    // and writes it; FINAL=0))
    constexpr uint32_t FINAL = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_FINAL_OR_SHARE);
    constexpr uint32_t ROLE  = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_ROLE);
    static_assert((ROLE == ROLE_SINGLE || ROLE == ROLE_TWO_FINAL || ROLE == ROLE_TWO_FIRST) && !(FINAL && ROLE == ROLE_TWO_FIRST),
                  "the pass role must be a ReconRole, and the first pass (ROLE_TWO_FIRST) is not FINAL");
    // PROBE: diagnostic selector (the library always passes 0). 1 replaces the FP64 assembly by a
    // constant output (the reads and the amount written stay the same; the answer breaks).
    constexpr uint32_t PROBE = get_compile_time_arg_val(t2a.next_compile_time_args_offset() + CT_PROBE);
    // The following TensorAccessor starts at CT_COUNT, after the 3 words above
    constexpr auto foa = TensorAccessorArgs<t2a.next_compile_time_args_offset() + CT_COUNT>();
    const auto of = TensorAccessor(foa, o64);
    constexpr auto daa = TensorAccessorArgs<foa.next_compile_time_args_offset()>();
    const auto ada = TensorAccessor(daa, dAa);
    constexpr auto dba = TensorAccessorArgs<daa.next_compile_time_args_offset()>();
    const auto adb = TensorAccessor(dba, dBa);
    constexpr auto qa = TensorAccessorArgs<dba.next_compile_time_args_offset()>();
    const auto aq0 = TensorAccessor(qa, q0a);

    const uint32_t fp = get_write_ptr(cb_f64);
    // Handoff when the work is shared with the reader (see the note in recon_reader.cpp)
    volatile tt_l1_ptr uint32_t* sy = (volatile tt_l1_ptr uint32_t*)get_write_ptr(cb_sync);
    const uint32_t ap = get_write_ptr(cb_da);
    const uint32_t bp = get_write_ptr(cb_db);
    const uint32_t qp = get_write_ptr(cb_q);

    for (uint32_t g = start; g < start + num; ++g) {
        cb_wait_front(cb_out, nlimb);
        const uint32_t p = get_read_ptr(cb_out), ts = get_tile_size(cb_out);
        const uint32_t tag = g - start + 1;
        // The limbs are not returned to DRAM in the library: the final pass writes FP64 and the first
        // pass of the two-pass configuration writes 128-bit fixed point, both reading the CB in L1
        // directly (l0/l1/l2 below), and nothing reads limb planes. Only the diagnostic combination
        // FINAL=0 with a role other than ROLE_TWO_FIRST (see the top) writes the 3 limb planes.
        if constexpr (!FINAL && ROLE != ROLE_TWO_FIRST) {
            noc_async_write_tile(g, o0, p);
            noc_async_write_tile(g, o1, p + ts);
            if (nlimb >= 3) noc_async_write_tile(g, o2, p + 2 * ts);
        }
        // First pass: its own 3 limbs go into fixed point (the first part is done by the reader)
        if constexpr (ROLE == ROLE_TWO_FIRST) {
            if (jlo) { sy[0] = p; *pub = tag; }
            volatile tt_l1_ptr uint32_t* l0 = (volatile tt_l1_ptr uint32_t*)p;
            volatile tt_l1_ptr uint32_t* l1 = (volatile tt_l1_ptr uint32_t*)(p + ts);
            volatile tt_l1_ptr uint32_t* l2 = (volatile tt_l1_ptr uint32_t*)(p + 2 * ts);
            volatile tt_l1_ptr uint32_t* pa = (volatile tt_l1_ptr uint32_t*)qp;
            for (uint32_t j = jlo; j < 1024; ++j) recon_elem::pack_a128<PROBE>(l0, l1, l2, pa, j);
            if (jlo) { while (*done != tag) { invalidate_l1_cache(); } }
            l1_writes_landed(qp + OZ_A128_TILE_BYTES - 4u);
            noc_async_write_tile(g, o0, qp);
        }
        if constexpr (FINAL) {
            noc_async_read_tile((g + off) / Nt, ada, ap);
            noc_async_read_tile((g + off) % Nt, adb, bp);
            if constexpr (ROLE == ROLE_TWO_FINAL) {   // read the fixed point of the first pass (16 KB)
                noc_async_read_tile(g, aq0, qp);
            }
            noc_async_read_barrier();
            // The delta tiles have been read at this point, so the reader can simply look at
            // c_20/c_21 (L1 of the same core). They are not read twice.
            if (jlo) { sy[0] = p; *pub = tag; }      // pass the limb address to the reader
            volatile tt_l1_ptr uint32_t* l0 = (volatile tt_l1_ptr uint32_t*)p;
            volatile tt_l1_ptr uint32_t* l1 = (volatile tt_l1_ptr uint32_t*)(p + ts);
            volatile tt_l1_ptr uint32_t* l2 = (volatile tt_l1_ptr uint32_t*)(p + 2 * ts);
            volatile tt_l1_ptr uint32_t* qa4 = (volatile tt_l1_ptr uint32_t*)qp;
            volatile tt_l1_ptr int32_t*  da = (volatile tt_l1_ptr int32_t*)ap;
            volatile tt_l1_ptr int32_t*  db = (volatile tt_l1_ptr int32_t*)bp;
            volatile tt_l1_ptr uint32_t* po = (volatile tt_l1_ptr uint32_t*)fp;
            for (uint32_t j = jlo; j < 1024; ++j)
                recon_elem::assemble_f64<ROLE, PROBE>(l0, l1, l2, nlimb, da, db, qa4, po, j, sh2, Whi, Wlo, qw);
            // Wait for the first part done by the reader (discarding L0 meanwhile; the reader writes
            // the first part and then the semaphore in the same L1, so they pass through the single
            // L1 access port in order, and once done is visible the first part is in L1 as well)
            if (jlo) { while (*done != tag) { invalidate_l1_cache(); } }
            // The NoC reads only after the own second part has landed in L1 (common/kernels/l1_order.h)
            l1_writes_landed(fp + OZ_F64_TILE_BYTES - 4u);
            noc_async_write_tile(g + off, of, fp);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, nlimb);
    }
}
