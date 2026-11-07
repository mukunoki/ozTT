// SPDX-License-Identifier: Apache-2.0
#pragma once
// The formats shared by both Ozaki schemes: the table of their properties, the selection of the format
// of a backend, and the rule for the MathFidelity.
//
// Every format-dependent value is looked up in a table, and each value is written in exactly one place.
// The tables are:
//   OzFormat (this header)                  the device data formats that the GEMM reads, the slicer writes
//                                           and the GEMM accumulates in, and the largest exactly
//                                           representable integer
//   Oz1Param (ozaki1/param.hpp)             per-format parameters of Ozaki scheme I (radix, ...)
//   OZ2_ROUTES (ozaki2/split.hpp)           the paths of Ozaki scheme II (bounds of the moduli and of the
//                                           digits, number of digits, default number of moduli)
// Each backend object is compiled with exactly one macro -DOZ_FMT_<format> (INT8 / FP8 / BF16 / FP16 /
// FP32 / BFP8 / BFP4 / BFP2; FP32 is the format published as TF32). The same macro is used for both
// schemes; this header provides the selected format OZ_FMT_ID and its properties OZ_FMT.
//
// "The largest integer that the format represents exactly" (exact_int_max) and "the bound of the digits
// actually used" are different quantities. The latter is also limited by the partition of the multiplier
// (32 with LoFi) and by the width of the accumulator, so it can be smaller (BF16: 256 versus 32). In
// Ozaki scheme I the digit bound follows from the radix (OZ1_DMAX in ozaki1/format.hpp, by the rule
// oz1_digit_max in ozaki1/param.hpp); in Ozaki scheme II it is digit_bound of the table of paths.

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/base_types.hpp>
#include <cstdint>

namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS {

enum class OzFmt { INT8, FP8, BF16, FP16, FP32, BFP8, BFP4, BFP2 };

// All formats in the order of the enumeration (for checks that go through the whole table).
inline constexpr OzFmt OZ_FMTS[] = {OzFmt::INT8, OzFmt::FP8,  OzFmt::BF16, OzFmt::FP16,
                                    OzFmt::FP32, OzFmt::BFP8, OzFmt::BFP4, OzFmt::BFP2};

// Properties of a format (the formats that the slicer writes, the GEMM reads and the GEMM accumulates in
// do not depend on the scheme). The tile of `in` and of `slice_out` has the same size; the block
// floating-point formats include 64 B of exponents per tile, one shared by every 16 elements.
struct OzFormat {
    tt::DataFormat in;         // format in which the GEMM reads the slices
    tt::DataFormat slice_out;  // format of the CB that the slicer writes (same tile size as `in`)
    tt::DataFormat out;        // format of the accumulation / partial products of the GEMM
    bool out_half;             // the slicer writes binary16 bit patterns as UInt16 (FP16)
    uint32_t exact_int_max;    // largest integer |a| that the format represents exactly (not the digit bound)
};
constexpr OzFormat oz_format(OzFmt f) {
    using DF = tt::DataFormat;
    switch (f) {
    // Carried as sign-magnitude UInt8 bytes, which the GEMM reads as Int8; the partial products are exact
    // Int32. The bound is the 7-bit magnitude.
    case OzFmt::INT8: return {DF::Int8, DF::UInt8, DF::Int32, false, 127};
    // Largest integer up to which e4m3 represents every integer exactly.
    case OzFmt::FP8: return {DF::Fp8_e4m3, DF::Fp8_e4m3, DF::Float32, false, 16};
    // 8-bit significand.
    case OzFmt::BF16: return {DF::Float16_b, DF::Float16_b, DF::Float32, false, 256};
    // IEEE binary16 with an 11-bit significand (including the implicit 1). Packing from the FP32
    // destination register directly into Float16 corrupts the values, so the slicer builds the binary16
    // bit pattern as an integer and writes it through a UInt16 CB; the GEMM reads the same bytes as
    // Float16 (both have 2048 B tiles). 2048 is the bound of the storage format. The bound up to which
    // consecutive integers are multiplied exactly is smaller: 32 with LoFi (width of the multiplier
    // partition).
    case OzFmt::FP16: return {DF::Float16, DF::UInt16, DF::Float32, true, 2048};
    // Stored as FP32 and given to the matrix engine as TF32 (the public name of this format is TF32).
    // TF32 does not exist as an L1 CB format: with fp32_dest_acc_en the unpacker shortens the significand
    // of an FP32 CB when it expands the tile. Every GEMM kernel of this library sets fp32_dest_acc_en, so
    // the multiplication is done in TF32 and the bound is counted with the 11-bit significand of TF32.
    case OzFmt::FP32: return {DF::Float32, DF::Float32, DF::Float32, false, 2048};
    // Block floating point 8. Even with an exponent shared by 16 elements, the integers 0..127 are exact
    // in every block (7-bit significand). 1.0625 B per element (1024 B of significands + 64 B of
    // exponents per tile).
    case OzFmt::BFP8: return {DF::Bfp8_b, DF::Bfp8_b, DF::Float32, false, 127};
    // Block floating point 4. 3-bit significand (the sign is separate), so the consecutive integers that
    // are exact under a shared exponent are [-7, 7]. 576 B per tile, 0.5625 B per element.
    case OzFmt::BFP4: return {DF::Bfp4_b, DF::Bfp4_b, DF::Float32, false, 7};
    // Block floating point 2. 1-bit significand (the sign is separate), so only the integers [-1, 1] are
    // exact. 320 B per tile, 0.3125 B per element: the narrowest input format of this hardware.
    case OzFmt::BFP2: return {DF::Bfp2_b, DF::Bfp2_b, DF::Float32, false, 1};
    }
    return {};
}

// The slicer writes the tiles that the GEMM reads, so the two formats must have the same tile size, and
// a slicer that writes binary16 bit patterns writes a UInt16 CB.
constexpr bool oz_format_table_ok() {
    for (OzFmt f : OZ_FMTS) {
        const OzFormat d = oz_format(f);
        if (tt::tile_size(d.in) != tt::tile_size(d.slice_out)) return false;
        if ((d.slice_out == tt::DataFormat::UInt16) != d.out_half) return false;
    }
    return true;
}
static_assert(oz_format_table_ok(), "in / slice_out / out_half of oz_format disagree");

#if (defined(OZ_FMT_INT8) + defined(OZ_FMT_FP8) + defined(OZ_FMT_BF16) + defined(OZ_FMT_FP16) + defined(OZ_FMT_FP32) + \
     defined(OZ_FMT_BFP8) + defined(OZ_FMT_BFP4) + defined(OZ_FMT_BFP2)) != 1
#error "define exactly one OZ_FMT_<format> macro (for example OZ_FMT_BF16); the Makefile passes it per backend object"
#endif

#if defined(OZ_FMT_INT8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::INT8;
#elif defined(OZ_FMT_FP8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP8;
#elif defined(OZ_FMT_BF16)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BF16;
#elif defined(OZ_FMT_FP16)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP16;
#elif defined(OZ_FMT_FP32)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::FP32;
#elif defined(OZ_FMT_BFP8)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP8;
#elif defined(OZ_FMT_BFP4)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP4;
#elif defined(OZ_FMT_BFP2)
inline constexpr OzFmt OZ_FMT_ID = OzFmt::BFP2;
#endif

inline constexpr OzFormat OZ_FMT = oz_format(OZ_FMT_ID);

// MathFidelity used by the GEMM stage of both schemes. The rule depends only on the bound of the digits
// that the multiplier actually receives:
//   INT8                     HiFi4. On the INT8 path the multiplier assigns only the 3 bits 7..5 to the
//                            upper partition of SrcA, so a low part remains with LoFi however small the
//                            digits are. The rule of the floating-point paths must not be reused.
//   otherwise, digits <= 32  LoFi. The LoFi multiplier reads 5 bits of SrcA, so integers with at most 5
//                            significant bits (0..31, and 32) are multiplied exactly.
//   otherwise                HiFi4.
// The digit bound follows from the radix in Ozaki scheme I (OZ1_FID in ozaki1/format.hpp) and is
// digit_bound of the table of paths in Ozaki scheme II (oz2_fidelity in ozaki2/format.hpp).
constexpr tt::tt_metal::MathFidelity oz_fidelity_rule(uint32_t digit_bound) {
    if (OZ_FMT_ID == OzFmt::INT8) return tt::tt_metal::MathFidelity::HiFi4;
    return (digit_bound <= 32) ? tt::tt_metal::MathFidelity::LoFi : tt::tt_metal::MathFidelity::HiFi4;
}

}  // namespace ozTTgemm::detail::OZTTGEMM_BACKEND_NS
