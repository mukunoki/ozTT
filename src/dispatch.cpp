// SPDX-License-Identifier: Apache-2.0
//
// Maps (format, scheme) to a backend. The build writes formats.inc with one
// DECLARE(Format, name) line per format selected by the Makefile's FORMATS setting.
#include "backend_interface.hpp"
#include <stdexcept>

namespace ozTTgemm::detail {
#define DECLARE(F, name)                     \
    namespace ozaki1_##name {                \
        Output run(Device&, const Problem&); \
    }                                        \
    namespace ozaki2_##name {                \
        Output run(Device&, const Problem&); \
    }
#include "formats.inc"
#undef DECLARE
Backend backend_for(Format format, Scheme scheme) {
    switch (format) {
#define DECLARE(F, name) \
    case Format::F: return scheme == Scheme::I ? ozaki1_##name::run : ozaki2_##name::run;
#include "formats.inc"
#undef DECLARE
    default: throw std::invalid_argument("requested format was not built");
    }
}
}  // namespace ozTTgemm::detail
namespace ozTTgemm {
bool is_available(Format format) noexcept {
    switch (format) {
#define DECLARE(F, name) \
    case Format::F: return true;
#include "formats.inc"
#undef DECLARE
    default: return false;
    }
}
const char* format_name(Format format) noexcept {
    switch (format) {
    case Format::BF16: return "BF16";
    case Format::FP16: return "FP16";
    case Format::TF32: return "TF32";
    case Format::BFP8: return "BFP8";
    case Format::FP8: return "FP8";
    case Format::INT8: return "INT8";
    case Format::BFP4: return "BFP4";
    case Format::BFP2: return "BFP2";
    default: return "unknown";
    }
}
}  // namespace ozTTgemm
