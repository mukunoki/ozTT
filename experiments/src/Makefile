# 全ターゲット用の素の GNU Makefile（CMake は使わない）
# ---------------------------------------------------------------------------
# tt-metalium コンテナの中で実行する（TT_METAL_HOME はそこで設定済み）：
#     make                # 全部の組をビルド -> build/<名前>（scripts/build_all.sh はこれを -B で呼ぶ）
#     make -j             # 並列ビルド
#     make release        # 本番の 16 経路と基準実装（論文の比較対象）だけ（DRIVERS・BASELINES．公開版を手で作るとき）
#     make probes checks host-tools check-tools    # 組ごと（drivers・baselines も同じ）
#     make ozaki1_int8    # 1 本
#     make print-host-flags   # ホストの道具のフラグ（別のイメージで作る tools/dgemmcheck.cpp 用．scripts/measure/cpu_dgemm_reference.sh）
#     make print-targets      # all が作るターゲットの名前（scripts/build_all.sh が build/ に余分な実行ファイルが無いかを見る）
#     make clean
#
# tt-metal にリンクするターゲット（DRIVERS・BASELINES・PROBES・CHECKS）は，ホスト側の .cpp 1 本を tt-metalium に
# 直接リンクするだけである．デバイスカーネルは実行時に JIT ビルドされるので，ここでは作らない
# （ホストは各組の kernels/ にあるカーネルを，このディレクトリの絶対パス OZ_SRC_DIR を基点にした
# common/oz_paths.hpp の置き場所で指す）．ホストの取り込みはこのディレクトリからの相対で書く（-I.．
# "common/host_eval.hpp" など）．出力は組によらず build/ に平らに置く（scripts/provenance.sh の bin_line と
# scripts の各所が src/build/<名前> の形を前提にしている）．
# 以下のフラグは tt-metalium の CMake パッケージが解決した内容を一度書き出して
# そのまま固定したものである．

TT_METAL_HOME ?= /tt-metal
TT_BR  := $(TT_METAL_HOME)/build_Release
TT_LIB := $(TT_BR)/lib

CXX := c++

# 先頭の -I. は src の取り込みの根（上の注記）で，tt-metalium のパッケージから写したものではない．tt-metal の行列積の例の
# ヘッダ（programming_examples/matmul/matmul_common の bmm_op.hpp）はどのソースも取り込まないので，その取り込みの根は置かない
# （例の判定は ozaki_gemm_host.hpp の example_max_block_dim に写してある．gemm_bf16 が使う例の計算カーネルは
# OVERRIDE_KERNEL_PREFIX から実行時に JIT で組む）
INC := -I. \
       -isystem $(TT_BR)/include \
       -isystem $(TT_BR)/libexec/tt-metalium/tt_metal/hostdevcommon/api \
       -isystem $(TT_BR)/libexec/tt-metalium \
       -isystem $(TT_BR)/include/metalium-thirdparty

DEF := -DENCHANTUM_ENABLE_MSVC_SPEEDUP=1 -DFMT_HEADER_ONLY=1 -DNTEST \
       -DSPDLOG_COMPILED_LIB -DSPDLOG_FMT_EXTERNAL -DTRACY_IMPORTS \
       -DTRACY_TIMER_FALLBACK -DTT_UMD_BUILD_SIMULATION \
       -DOVERRIDE_KERNEL_PREFIX='"tt_metal/programming_examples/"'

# デバイスカーネルの置き場所の基点（common/oz_paths.hpp）：このディレクトリの絶対パスと末尾の /．コンテナの中では
# scripts/run.sh がリポジトリを /home/user/tt にマウントするので /home/user/tt/src/ になる
SRCDIR_DEF := -DOZ_SRC_DIR='"$(CURDIR)/"'

# -march=native は参照計算（common/oz_reference.hpp の Dot2）の速さを左右する：これが無いと std::fma が
# ライブラリ呼び出しになり，j ループもベクトル化されない．
# -ffp-contract=off は必須：積和が勝手に融合されると誤差なし変換の残差が消える．
# -ffast-math は絶対に付けないこと（同じ理由）．
# tt-metal の指定初期化は省略したメンバを既定値にする．その省略だけを警告から外す．
CXXFLAGS := -O3 -DNDEBUG -std=gnu++20 -march=native -ffp-contract=off -fopenmp \
            -Wall -Wextra -Wno-missing-field-initializers -Wno-int-to-pointer-cast $(INC) $(DEF) $(SRCDIR_DEF)
LDFLAGS  := -Wl,-rpath,$(TT_LIB) -L$(TT_LIB)
LDLIBS   := -ltt_metal -ltracy -ldl -ltt-umd -ltt_stl -lspdlog -lquadmath

# ホストだけで動く道具（../tools/．tt-metal にリンクしない）のフラグ．-march=native・-ffp-contract=off は上と同じ理由．
# HOST_OPT は取り込みの根を含まない部分で，別のイメージで作る道具（scripts/measure/cpu_dgemm_reference.sh の tools/dgemmcheck.cpp）も
# print-host-flags でこれを受け取る（フラグの正本をここ 1 つにする）．取り込みの根はこのディレクトリ（-I.）
HOST_OPT      := -Wall -Wextra -O3 -std=gnu++20 -march=native -ffp-contract=off -fopenmp
HOST_CXXFLAGS := $(HOST_OPT) -I.
HOST_LDLIBS   := -lquadmath

B := build

# ---- 組（ターゲットの位置づけ）----------------------------------------------
# どのスクリプトが回し，論文・スライドの何に使うかで分ける．経路ごとの p・法の本数は ../scripts/routes.tsv，
# 性能と到達精度の値は ../docs/results.md，ターゲットごとの説明は README.md が正本で，ここには値を書かない．
#
# DRIVERS   本番：尾崎の 16 経路（scheme I・II × 8 形式）．論文・スライドの表と図の値を出す（remeasure_all.sh の
#           cool・acc・phi・pass・abl・ratio・split・ext・phi_nowin・whole などの段が measure/performance.sh・measure/accuracy.sh・
#           measure/exponent_sweep.sh などで回す）．ozaki1_bfp4・ozaki1_bfp2 は形式を狭めていったときの天井の式を確かめるための経路で，
#           全体では負ける（BFP2_b は入力に取れる最も狭い形式で桁が 1 bit しかなく，p が最も大きく部分積が千本を超える）．
# BASELINES 基準実装（baseline．尾崎の経路ではない論文の比較対象．どちらも論文の値を出す）：
#   gemm_bf16     素の BF16 行列積（基準性能）．構成は BF16 累積と FP32 累積（fp32_dest_acc=1．measure/performance.sh の
#                 表示名 BF16x2/PP・ログ名 cool_bf16x2_* は前からの名前で，2 成分の分割ではない）の 2 つ
#   gemm_bf16ext  BF16 の 2・3 成分（BF16x2・BF16x3）の素朴な無誤差分割と 3xTF32（精度の壁の実例）
# PROBES    探針：実機の性質を単一コアで測り，論文・スライドの地の文の値を出す．remeasure_all.sh の probes の段で
#           ../scripts/diagnose/probes.sh が回し，probe_checks.py が論文・スライドと照合する
#   fp8_probe       入力形式（bf16/fp16/fp32/fp8）x fidelity x 桁の上限で整数積が無誤差になるか，乱数入力の到達精度，
#                   出力形式を変えたパッカの丸め（probe_fp8_digit・probe_fp8_rand）
#   dst_probe       DST レジスタが FP32 で何本使えるか（probe_dst）
#   fp32acc_probe   Tenstorrent 提供の FP32 累算サンプルと同じ構成で，累算経路が保持する幅（probe_fp32acc）
#   fma_probe       SFPU の積和（SFPMAD）が 48 bit の積のうち何ビットを保つか（probe_fma．common/kernels/sfpu_tf.h の
#                   冒頭が TwoProd に Dekker 分割を使う根拠）
#   tile_rate_probe 1 コアの L1 に置いたブロックの行列積でタイル積 1 回のサイクル数（probe_tile_*・probe_tilecfg_*．
#                   冷ましてから回す）
# CHECKS    検査：正しさを確かめる
#   gemm_int8         素の INT8→INT32 行列積．INT8 の行列積が無誤差であること（復号した整数値が int64 の参照と全要素で
#                     一致すること）の検査として ../scripts/selftest.sh と ../scripts/tests/check_probes.sh が回す（論文の値は
#                     出さない．8197a0b までは基準実装の組 BASELINES に置いていた）
#   gemm_exact_check  尾崎の両スキームが使う GEMM 段（build_mcast_gemm＋bmm_int32_spill）と BF16 extended の計算カーネルの
#                     K 方向の累積が，敵対的な整数入力でも全要素で無誤差か．通しの selftest の段で回り
#                     （diagnose/probes.sh の probe_gemm_exact が ../scripts/diagnose/gemm_exact.sh を呼ぶ），論文の b_acc の出どころ
#   slice_split_check[_int8]  尾崎 I のスライス段の FP64 の分割を 2 つの RISC-V に分けた経路が分けない経路と全語で同じか
#                     （../scripts/tests/check_slice_split.sh．手で回す）
#   scale_check       スケール段の出力をホストの参照と全タイルで比べる（../scripts/tests/check_scale.sh．手で回す）
#   tfmul_probe       SFPU の模擬（checks/sfpu_tf_emu.hpp）と実機の照合．本番の関数を実機で呼んでビット単位で比べる
#                     （../scripts/selftest.sh が毎回回す．ホストの検査の模擬の前提）
# HOST_TOOLS  ホストだけで動く道具で，diagnose/probes.sh が論文・スライドの値を測るもの
#   dot2check     参照解 Dot2 の相対誤差を __float128 の参照和と比べる（論文「測定条件」の 2^-85）
#   awe_plan      尾崎 II の法の集合を段の時間のモデルで選ぶ探索のホスト版（スライドのモデルの見込みと所要）
#   splitsim      尾崎 II の分割だけを厳密演算で模擬した到達精度（スライドの幅 b の表の「分割のみ」の列）
#   reconsim      尾崎 II の再構成の最下位 limb の丸めの模擬（スライドの 2 パス化の頁）
#   twosum_probe  tfAddSmall の判定と第 2 段を戻したときの frac の最悪誤差（スライドの 26 bit の頁．selftest.sh も回す）
#   tfconv        triple-float と FP64 の変換の 8 つの検査（スライドの clz 化の頁．selftest.sh も回す）
# CHECK_TOOLS ホストだけで動く検査（論文の値は出さない）
#   modround_check  ModRound と剰余の桁分割の総当たり（selftest.sh）
#   tfmul_check     基数倍の最後の FastTwoSum の前提を本番の 3 本組で確かめる（selftest.sh）
#   crt_fold_check  分数形 CRT の小数部の畳み込みを窓の端で確かめる（selftest.sh）
#   aligned_accum   第 1 パスの累算が共通粒度 OZ2_RGRAN で無誤差か（selftest.sh．ozaki2_moduli.h の OZ2_RGRAN の根拠）
#   cmp_dump        OZ_DUMP_C の書き出し 2 つを要素ごとに比べる（../scripts/tests/check_nsmall.sh）
#   knobs_check     つまみの表（common/oz_knobs.hpp）が ../docs/env_vars.md の名前・分類と，src・tools が読むつまみの名前と
#                   食い違わないか（selftest.sh）
# どの組も build/ に置くので，build_all.sh のビルドの記録（BUILD_STAMP の BIN 行）に載り，測る前の確認（provenance_check）と
# 検査側の照合（BINARY 行の sha256）が同じ仕組みで効く（ホストで別に作ると，どのソースから作ったかがログに残らない）．
# tfmul_check・tfconv は実行時に ../scripts/routes.tsv を読むので，結果の出どころは BUILD_STAMP の sha256 だけでは決まらない
# （routes.tsv は scripts/ にあり，出どころの rev で決まる）．
FMTS        := bf16 fp16 fp32 fp8 int8 bfp8 bfp4 bfp2
DRIVERS     := $(addprefix ozaki1_,$(FMTS)) $(addprefix ozaki2_,$(FMTS))
BASELINES   := gemm_bf16 gemm_bf16ext
PROBES      := fp8_probe dst_probe fp32acc_probe fma_probe tile_rate_probe
CHECKS      := gemm_int8 gemm_exact_check slice_split_check slice_split_check_int8 scale_check tfmul_probe
HOST_TOOLS  := dot2check awe_plan splitsim reconsim twosum_probe tfconv
CHECK_TOOLS := modround_check tfmul_check crt_fold_check aligned_accum cmp_dump knobs_check routes_check accuracy_check
DEVICE      := $(DRIVERS) $(BASELINES) $(PROBES) $(CHECKS)    # tt-metal にリンクするもの
TOOLS       := $(HOST_TOOLS) $(CHECK_TOOLS)                    # ../tools/<名前>.cpp から作るもの

# ---- ターゲット -> ソース ----
$(foreach f,$(FMTS),$(eval SRC_ozaki1_$(f) := ozaki1/ozaki1_main.cpp)$(eval SRC_ozaki2_$(f) := ozaki2/ozaki2_main.cpp))
SRC_gemm_bf16              := baseline/gemm_bf16.cpp
SRC_gemm_bf16ext           := baseline/gemm_bf16ext.cpp
SRC_gemm_int8              := checks/gemm_int8.cpp
SRC_fp8_probe              := probes/fp8_probe.cpp
SRC_dst_probe              := probes/dst_probe.cpp
SRC_fp32acc_probe          := probes/fp32acc_probe.cpp
SRC_fma_probe              := probes/fma_probe.cpp
SRC_tile_rate_probe        := probes/tile_rate_probe.cpp
SRC_gemm_exact_check       := checks/gemm_exact_check.cpp
SRC_slice_split_check      := checks/slice_split_check.cpp
SRC_slice_split_check_int8 := checks/slice_split_check.cpp
SRC_scale_check            := checks/scale_check.cpp
SRC_tfmul_probe            := checks/tfmul_probe.cpp
$(foreach t,$(TOOLS),$(eval SRC_$(t) := ../tools/$(t).cpp))

# ---- ターゲット -> 出し分けのマクロ（各ターゲットにつまみは 1 つだけ）----
#      両スキームとも形式を -DOZ_FMT_<形式> の 1 つで選ぶ（INT8 も明示する）．
#      形式の性質は common/oz_format_host.h・oz_format_table.hpp，選択と MathFidelity の決め方は common/oz_format.hpp，スキームごとの値の表は
#      scheme I が ozaki1/ozaki1_param.h（Oz1Param），scheme II が ozaki2/ozaki2_split.h の経路の表 OZ2_ROUTES
#      （形式ごとの Oz2Route）にある．fidelity はここでは渡さない：両スキームとも桁の上限からコードが決める．
#      Ozaki-I が選ぶのは形式だけで，再構成は常に triple-float である．
DEF_ozaki1_bf16 := -DOZ_FMT_BF16
DEF_ozaki1_fp16 := -DOZ_FMT_FP16
DEF_ozaki1_fp32 := -DOZ_FMT_FP32
DEF_ozaki1_fp8  := -DOZ_FMT_FP8
DEF_ozaki1_int8 := -DOZ_FMT_INT8
DEF_ozaki1_bfp8 := -DOZ_FMT_BFP8
DEF_ozaki1_bfp4 := -DOZ_FMT_BFP4
DEF_ozaki1_bfp2 := -DOZ_FMT_BFP2
DEF_slice_split_check      := -DOZ_FMT_BF16
DEF_slice_split_check_int8 := -DOZ_FMT_INT8
DEF_ozaki2_int8 := -DOZ_FMT_INT8
DEF_ozaki2_fp8  := -DOZ_FMT_FP8
DEF_ozaki2_bfp8 := -DOZ_FMT_BFP8
DEF_ozaki2_bf16 := -DOZ_FMT_BF16
DEF_ozaki2_fp16 := -DOZ_FMT_FP16
DEF_ozaki2_fp32 := -DOZ_FMT_FP32
DEF_ozaki2_bfp4 := -DOZ_FMT_BFP4
DEF_ozaki2_bfp2 := -DOZ_FMT_BFP2

# ---- 組み立てのコマンド ----
# ターゲットごとにコンパイルとリンクを 1 回ずつ行う（ソースは各 1 本）．下の前提の印もこの文字列を使う．
# -MMD/-MP が「そのターゲットが取り込んだ src のヘッダ」を記録するので，common/*.hpp を編集すれば，それを使う
# ターゲットが作り直される．これが無いと共有ヘッダを直しても古いバイナリが黙って残り，それで測った値を報告してしまいかねない．
DEVCMD  = $(CXX) $(CXXFLAGS) $(DEF_$(1)) -MMD -MP -MT $(B)/$(1) -MF $(B)/$(1).d $(SRC_$(1)) -o $(B)/$(1) $(LDFLAGS) $(LDLIBS)
HOSTCMD = $(CXX) $(HOST_CXXFLAGS) -MMD -MP -MT $(B)/$(1) -MF $(B)/$(1).d $(SRC_$(1)) -o $(B)/$(1) $(HOST_LDLIBS)

# ---- ビルドの前提の印 ----------------------------------------------------
# -MMD はソースと src のヘッダしか記録しない．コンパイラ，フラグ（コマンド行や環境で上書きしたものを含む），
# -isystem の下の tt-metal のヘッダ，リンクする tt-metal のライブラリは記録しないので，それだけでは，イメージや
# tt-metal を入れ替えた後・フラグを変えた後にも古いバイナリが黙って残る．そこで各ターゲットに印 build/<名前>.cmd を
# 前提として持たせる．印の中身は，ツールチェーンの識別（下の TC_HOST・TC_DEV）とそのターゲットを作るコマンドそのもの．
# make を起動するたびに中身を作り，前の中身と違うときだけ書き換える（同じなら触らないので時刻も変わらない）．
# したがって同じなら作り直さず，どれかが変われば（DEF_・SRC_ を含む）そのターゲットだけを作り直す．
# Makefile 自身は前提に入れない（コメントを直しただけでは作り直さない．作り方を変える変更は印の中身に出る）．
# ツールチェーンの識別：
#   TC_HOST  コンパイラ・リンカ・glibc の版と，-march=native が解決した値（-march と，-Q --help=target の全部の sha256．
#            同じフラグでも計算機が違えば命令の集合が変わる）
#   TC_DEV   TC_HOST に tt-metal（TT_METAL_HOME の rev と，取り込むヘッダとリンクするライブラリの木 include・
#            libexec/tt-metalium・lib のファイルの一覧（パス・種類・大きさ・時刻）の sha256）を足したもの．
#            ファイルの中身ではなく一覧を見るのは毎回の起動を軽くするため（作り直し・入れ替えで時刻が変わる）．
#            したがって，同じパス・大きさ・時刻のまま中身だけを差し替えた tt-metal は見分けない（測定のビルドは
#            ../scripts/build_all.sh が make -B で必ず全部を作り直すので，この印に頼らない）
# 印を作らない目標（clean・print-host-flags・print-targets だけのとき）では，ツールチェーンを問い合わせない．
# 印を比べ・書くのは頼まれた目標が作るターゲットだけ（STAMP_FOR．make release なら DRIVERS・BASELINES だけ）．make -n
# （--dry-run・--just-print・--recon）では何も書かない：中身が違うターゲットは印を FORCE に依らせて作り直す予定として
# 表示し（印のレシピは空なので，-n でも -n でなくても印は触らない），build/ も作らない．
NOSTAMP_GOALS := clean print-host-flags print-targets
DRYRUN := $(findstring n,$(firstword -$(MAKEFLAGS)))
GOAL_all         := $(DEVICE) $(TOOLS)
GOAL_release     := $(DRIVERS) $(BASELINES)
GOAL_drivers     := $(DRIVERS)
GOAL_baselines   := $(BASELINES)
GOAL_probes      := $(PROBES)
GOAL_checks      := $(CHECKS)
GOAL_host-tools  := $(HOST_TOOLS)
GOAL_check-tools := $(CHECK_TOOLS)
# 目標 g が作るターゲット：組の名前ならその組，ターゲットの名前か build/<名前> ならそれ（ほかの目標は印を要らない）
goal_targets = $(if $(filter $(1),all release drivers baselines probes checks host-tools check-tools),$(GOAL_$(1)),$(filter $(1) $(patsubst $(B)/%,%,$(1)),$(DEVICE) $(TOOLS)))
STAMP_FOR := $(sort $(foreach g,$(or $(MAKECMDGOALS),all),$(call goal_targets,$(g))))
ifneq ($(filter-out $(NOSTAMP_GOALS),$(or $(MAKECMDGOALS),all)),)
TC_CXX    := $(shell $(CXX) --version 2>/dev/null | head -n1)
TC_LD     := $(shell $$($(CXX) -print-prog-name=ld) --version 2>/dev/null | head -n1)
TC_LIBC   := $(shell ldd --version 2>/dev/null | head -n1)
TC_MARCH  := $(shell $(CXX) -march=native -Q --help=target 2>/dev/null | awk '$$1 == "-march=" { print $$2; exit }')
TC_TARGET := $(shell $(CXX) -march=native -Q --help=target 2>/dev/null | sha256sum | cut -c1-16)
TC_MREV   := $(shell git -C $(TT_METAL_HOME) rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
TC_MFILES := $(shell if cd $(TT_BR) 2>/dev/null; then find include libexec/tt-metalium lib ! -type d -printf '%p %y %s %T@\n' 2>/dev/null \
                     | LC_ALL=C sort | sha256sum | cut -c1-16; else echo none; fi)
TC_HOST   := cxx=$(TC_CXX) | ld=$(TC_LD) | libc=$(TC_LIBC) | march=$(TC_MARCH) target=$(TC_TARGET)
TC_DEV    := $(TC_HOST) | metal=$(TC_MREV) metal_files=$(TC_MFILES)
define nl


endef
# STAMP 名前,中身   印 build/<名前>.cmd の中身が違えば書き換える（make -n では書かずに，作り直す予定として表示させる）
define STAMP
ifneq ($$(file <$(B)/$(1).cmd),$(2))
ifeq ($(DRYRUN),)
$$(file >$(B)/$(1).cmd,$(2))
else
$(B)/$(1).cmd: FORCE
endif
endif
endef
ifeq ($(DRYRUN),)
$(shell mkdir -p $(B))
endif
$(foreach t,$(filter $(STAMP_FOR),$(DEVICE)),$(eval $(call STAMP,$(t),$$(TC_DEV)$$(nl)$$(call DEVCMD,$(t)))))
$(foreach t,$(filter $(STAMP_FOR),$(TOOLS)),$(eval $(call STAMP,$(t),$$(TC_HOST)$$(nl)$$(call HOSTCMD,$(t)))))
endif

# ---- 規則 ----
.PHONY: all release drivers baselines probes checks host-tools check-tools clean print-host-flags print-targets FORCE $(DEVICE) $(TOOLS)
.DEFAULT_GOAL := all
FORCE:
all: $(DEVICE) $(TOOLS)
release: $(DRIVERS) $(BASELINES)
drivers: $(DRIVERS)
baselines: $(BASELINES)
probes: $(PROBES)
checks: $(CHECKS)
host-tools: $(HOST_TOOLS)
check-tools: $(CHECK_TOOLS)

# `make ozaki1_int8` と書けば build/ozaki1_int8 が作られるようにする
$(DEVICE) $(TOOLS): %: $(B)/%

$(B):
	mkdir -p $(B)
# 印が無い（make clean と同じ起動で作るなど）ときは，作り直す側に倒す
$(B)/%.cmd: ;

define DEVRULE
$(B)/$(1): $(SRC_$(1)) $(B)/$(1).cmd | $(B)
	$$(call DEVCMD,$(1))
endef
define HOSTRULE
$(B)/$(1): $(SRC_$(1)) $(B)/$(1).cmd | $(B)
	$$(call HOSTCMD,$(1))
endef
$(foreach t,$(DEVICE),$(eval $(call DEVRULE,$(t))))
$(foreach t,$(TOOLS),$(eval $(call HOSTRULE,$(t))))

# 前回のビルドの .d は，そのときのソースのパスを前提に持つ（-MP が空の規則を作るのはヘッダだけ）．ソースを移した
# ターゲットの古い .d を読むと「No rule to make target <古いパス>」で止まるので，記録されたソースが今の SRC_ と同じ .d だけを
# 読む（違うターゲットは印の中のコマンドが変わっているので，どのみち作り直す）
$(foreach t,$(DEVICE) $(TOOLS),$(if $(filter $(SRC_$(t)),$(word 2,$(file <$(B)/$(t).d))),$(eval -include $(B)/$(t).d)))

print-host-flags:
	@printf '%s\n' '$(HOST_OPT)'
print-targets:
	@printf '%s\n' $(DEVICE) $(TOOLS)

clean:
	rm -rf $(B)
