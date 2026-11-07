# SPDX-License-Identifier: Apache-2.0
# GNU Make build for the standalone C++20 library.
#   make -j2                  library and usage example
#   make test-host            API validation without opening a device
#   make test                 small device correctness cases
#   make PREFIX=/opt/ozTTgemm install
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

TT_METAL_HOME ?= /tt-metal
TT_BUILD ?= $(TT_METAL_HOME)/build_Release
BUILD_DIR ?= build
PREFIX ?= /usr/local
DESTDIR ?=
FORMATS ?= BF16 FP16 TF32 BFP8 FP8 INT8 BFP4 BFP2
CXXFLAGS ?= -O3 -DNDEBUG
# Metalium's aggregate types are initialized by designated initializers throughout.
WARNINGS ?= -Wall -Wextra -Wno-missing-field-initializers
# Single source of the version: the macros in the public header.
VERSION := $(shell sed -n 's/^\#define OZTTGEMM_VERSION_[A-Z]* //p' include/ozTTgemm/ozTTgemm.hpp | paste -sd.)

SUPPORTED := BF16 FP16 TF32 BFP8 FP8 INT8 BFP4 BFP2
ifeq ($(strip $(FORMATS)),)
$(error FORMATS must contain at least one format)
endif
ifneq ($(filter-out $(SUPPORTED),$(FORMATS)),)
$(error Unknown FORMATS: $(filter-out $(SUPPORTED),$(FORMATS)))
endif
ifneq ($(words $(FORMATS)),$(words $(sort $(FORMATS))))
$(error Duplicate entries in FORMATS: $(FORMATS))
endif

name_BF16 := bf16
name_FP16 := fp16
name_TF32 := tf32
name_BFP8 := bfp8
name_FP8  := fp8
name_INT8 := int8
name_BFP4 := bfp4
name_BFP2 := bfp2
macro_TF32 := FP32

METAL_CPPFLAGS := -isystem $(TT_BUILD)/include \
    -isystem $(TT_BUILD)/include/metalium-thirdparty \
    -isystem $(TT_BUILD)/libexec/tt-metalium \
    -isystem $(TT_BUILD)/libexec/tt-metalium/tt_metal/hostdevcommon/api \
    -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DNTEST \
    -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DTRACY_IMPORTS \
    -DTRACY_TIMER_FALLBACK -DTT_UMD_BUILD_SIMULATION
LOCAL_CPPFLAGS := -Iinclude -Isrc -Isrc/backend -I$(BUILD_DIR)
# Keep these after user flags: error-free transformations must retain their
# evaluation order, and the public interface requires C++20.
REQUIRED_CXXFLAGS := -std=gnu++20 -fno-fast-math -ffp-contract=off -fPIC -fopenmp -Wno-int-to-pointer-cast
COMPILE = $(CXX) $(CPPFLAGS) $(LOCAL_CPPFLAGS) $(METAL_CPPFLAGS) $(CXXFLAGS) $(WARNINGS) $(REQUIRED_CXXFLAGS)
METAL_LIBS := -L$(TT_BUILD)/lib -Wl,-rpath,$(abspath $(TT_BUILD)/lib) \
    -ltt_metal -ltracy -ldl -ltt-umd -ltt_stl -lspdlog -fopenmp -lquadmath
PUBLIC_CFLAGS := -std=c++20 -I$(abspath include)
PUBLIC_LIBS := -L$(abspath $(BUILD_DIR)) -lozTTgemm $(METAL_LIBS)

LIBRARY := $(BUILD_DIR)/libozTTgemm.a
BACKENDS := $(foreach f,$(FORMATS),$(foreach s,1 2,$(BUILD_DIR)/ozaki$(s)_$(name_$(f)).o))
OBJECTS := $(BUILD_DIR)/context.o $(BUILD_DIR)/dispatch.o $(BACKENDS)
CONFIG := $(BUILD_DIR)/.build-config
FORMAT_HEADER := $(BUILD_DIR)/formats.inc
# Installed as they are laid out here, so that the links between them keep working.
DOCS := LICENSE NOTICE README.md ORIGIN.md SOURCE_MAP.tsv VALIDATION.md
DOCS_SUBDIR := docs/sfpu_lemmas.md
# Quote values written to the build signature, including literal quotes in flags.
quote = '$(subst ','"'"',$(1))'

.PHONY: all library example evaluation test test-host install clean help print-cflags print-libs print-kernel-dir check-metal FORCE
all: library example
library: $(LIBRARY)
example: $(BUILD_DIR)/ozTTgemm_example
evaluation: $(BUILD_DIR)/ozTTgemm_evaluate

test-host: $(BUILD_DIR)/ozTTgemm_test
	"$(BUILD_DIR)/ozTTgemm_test" --host-only
test: $(BUILD_DIR)/ozTTgemm_test
	"$(BUILD_DIR)/ozTTgemm_test"

$(BUILD_DIR):
	mkdir -p "$@"

# Check flags/toolchain on every invocation, updating the signature only when
# its contents differ. make -n executes none of these recipes.
FORCE:
$(CONFIG): FORCE | $(BUILD_DIR)
	@{ printf '%s\n' $(call quote,$(COMPILE)) $(call quote,$(AR)) \
	    $(call quote,$(LDFLAGS) $(METAL_LIBS) $(LDLIBS)) \
	    $(call quote,$(abspath src/backend)) $(call quote,$(PREFIX)); \
	    $(CXX) --version | head -n 1; } > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

$(FORMAT_HEADER): FORCE | $(BUILD_DIR)
	@printf '%s\n' $(foreach f,$(FORMATS),'DECLARE($(f), $(name_$(f)))') > "$@.tmp"
	@if cmp -s "$@.tmp" "$@"; then rm -f "$@.tmp"; else mv "$@.tmp" "$@"; fi

check-metal:
	@test -f "$(TT_BUILD)/include/tt-metalium/host_api.hpp" || \
	    { echo 'TT-Metalium headers not found; set TT_METAL_HOME or TT_BUILD.' >&2; exit 2; }
	@test -f "$(TT_BUILD)/lib/libtt_metal.so" || \
	    { echo 'Built libtt_metal.so not found under TT_BUILD/lib.' >&2; exit 2; }

$(BUILD_DIR)/context.o: src/context.cpp $(CONFIG) | check-metal
	$(COMPILE) -DOZTTGEMM_SOURCE_KERNEL_DIR='"$(abspath src/backend)"' \
	    -DOZTTGEMM_INSTALL_KERNEL_DIR='"$(abspath $(PREFIX)/share/ozTTgemm/kernels)"' \
	    -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -c "$<" -o "$@"
$(BUILD_DIR)/dispatch.o: src/dispatch.cpp $(FORMAT_HEADER) $(CONFIG) | check-metal
	$(COMPILE) -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -c "$<" -o "$@"

define BACKEND_RULE
$(BUILD_DIR)/ozaki$(1)_$(name_$(2)).o: src/ozaki$(1).cpp $(CONFIG) | check-metal
	$(COMPILE) -DOZ_FMT_$(or $(macro_$(2)),$(2))=1 \
	    -DOZTTGEMM_BACKEND_NS=ozaki$(1)_$(name_$(2)) -DOZTTGEMM_PUBLIC_FORMAT=::ozTTgemm::Format::$(2) \
	    -MMD -MP -MF "$$(@:.o=.d)" -MT "$$@" -c "$$<" -o "$$@"
endef
$(foreach f,$(FORMATS),$(foreach s,1 2,$(eval $(call BACKEND_RULE,$(s),$(f)))))

# Rule/definition edits must also invalidate existing objects.
$(OBJECTS) $(BUILD_DIR)/example.o $(BUILD_DIR)/test.o $(BUILD_DIR)/evaluation.o: Makefile

# Recreate the archive so removing a format also removes its old members.
$(LIBRARY): $(OBJECTS) $(FORMAT_HEADER)
	rm -f "$@.tmp"
	$(AR) rcs "$@.tmp" $(OBJECTS)
	mv "$@.tmp" "$@"

$(BUILD_DIR)/example.o: examples/gemm.cpp $(CONFIG)
	$(COMPILE) -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -c "$<" -o "$@"
$(BUILD_DIR)/evaluation.o: evaluation/evaluate.cpp $(CONFIG)
	$(COMPILE) -march=native -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -c "$<" -o "$@"
$(BUILD_DIR)/ozTTgemm_evaluate: $(BUILD_DIR)/evaluation.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) "$<" $(LIBRARY) $(LDFLAGS) $(METAL_LIBS) $(LDLIBS) -o "$@"
$(BUILD_DIR)/test.o: tests/correctness.cpp $(CONFIG)
	$(COMPILE) -MMD -MP -MF "$(@:.o=.d)" -MT "$@" -c "$<" -o "$@"
$(BUILD_DIR)/ozTTgemm_example: $(BUILD_DIR)/example.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) "$<" $(LIBRARY) $(LDFLAGS) $(METAL_LIBS) $(LDLIBS) -o "$@"
$(BUILD_DIR)/ozTTgemm_test: $(BUILD_DIR)/test.o $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(REQUIRED_CXXFLAGS) "$<" $(LIBRARY) $(LDFLAGS) $(METAL_LIBS) $(LDLIBS) -o "$@"

# The .pc prefix is relative to the metadata file, so it follows a moved install.
$(BUILD_DIR)/ozTTgemm.pc: Makefile include/ozTTgemm/ozTTgemm.hpp $(CONFIG)
	@printf '%s\n' 'prefix=$${pcfiledir}/../..' 'libdir=$${prefix}/lib' \
	    'includedir=$${prefix}/include' 'kernelsdir=$${prefix}/share/ozTTgemm/kernels' \
	    '' 'Name: ozTTgemm' 'Description: Ozaki I/II FP64 GEMM for Tenstorrent Blackhole' \
	    'Version: $(VERSION)' 'Cflags: -std=c++20 -I$${includedir}' \
	    'Libs: -L$${libdir} -lozTTgemm $(METAL_LIBS)' > "$@"

install: $(LIBRARY) $(BUILD_DIR)/ozTTgemm.pc
	install -d "$(DESTDIR)$(PREFIX)/lib/pkgconfig" "$(DESTDIR)$(PREFIX)/include/ozTTgemm" \
	    "$(DESTDIR)$(PREFIX)/share/ozTTgemm/docs"
	install -m 644 $(LIBRARY) "$(DESTDIR)$(PREFIX)/lib/"
	install -m 644 $(BUILD_DIR)/ozTTgemm.pc "$(DESTDIR)$(PREFIX)/lib/pkgconfig/"
	install -m 644 include/ozTTgemm/ozTTgemm.hpp "$(DESTDIR)$(PREFIX)/include/ozTTgemm/"
	install -m 644 $(DOCS) "$(DESTDIR)$(PREFIX)/share/ozTTgemm/"
	install -m 644 $(DOCS_SUBDIR) "$(DESTDIR)$(PREFIX)/share/ozTTgemm/docs/"
	@for part in common ozaki1 ozaki2; do \
	    dir="$(DESTDIR)$(PREFIX)/share/ozTTgemm/kernels/$$part/kernels"; \
	    install -d "$$dir" && cp -R "src/backend/$$part/kernels/." "$$dir/" || exit; \
	done

print-cflags:
	@printf '%s\n' $(call quote,$(PUBLIC_CFLAGS))
print-libs:
	@printf '%s\n' $(call quote,$(PUBLIC_LIBS))
print-kernel-dir:
	@printf '%s\n' $(call quote,$(abspath src/backend))
help:
	@printf '%s\n' 'make [-j2]                   library and usage example' \
	    'make library                static library only' \
	    'make evaluation             optional paper evaluator' \
	    'make test-host              input validation without a device' \
	    'make test                   device correctness checks' \
	    'make FORMATS="BF16 BFP8"     select internal formats' \
	    'make PREFIX=/path install   library, headers, kernels, pkg-config metadata' \
	    'make print-cflags           compiler flags for a program using the library' \
	    'make print-libs             linker flags for such a program' \
	    'make clean                  remove generated build files'

clean:
	rm -rf -- "$(BUILD_DIR)"

-include $(OBJECTS:.o=.d) $(BUILD_DIR)/example.d $(BUILD_DIR)/test.d $(BUILD_DIR)/evaluation.d
