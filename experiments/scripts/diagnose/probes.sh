#!/bin/bash
# diagnose/probes.sh -- 論文・スライドの値のうち，表の測定（cool・pass・acc など）の外にある探針・検査・小さな実験を測り直す
# ---------------------------------------------------------------------------
#     ./scripts/diagnose/probes.sh                       # ホストから（tt-smi を使う）．既定の項目は設計の比較の群を含む
#                                                       # （所要時間は remeasure_all.sh の冒頭の実績の probes と design）
#     ONLY='^probe_tile_' ./scripts/diagnose/probes.sh    # ログ名の正規表現で項目を選ぶ（他の測定スクリプトと同じ）
#     PROBES_DESIGN=only ./scripts/diagnose/probes.sh     # 設計の比較の群（下の「設計の比較」）だけ
#     PROBES_DESIGN=0 PROBES_DOCS=0 ./scripts/diagnose/probes.sh  # 設計の比較の群と docs だけの項目を飛ばす（remeasure_all.sh の probes の段）
#     PROBES_EXTENDED=1 ./scripts/diagnose/probes.sh      # 既定の外の項目（下の「拡張」）も測る
#     PROBES_EXTENDED=only ./scripts/diagnose/probes.sh   # 拡張の項目だけ（remeasure_all.sh の probes_ext の段）
#     PROBES_DESIGN=only PROBES_DOCS=0 ./scripts/diagnose/probes.sh     # 設計の比較のうち原稿が引く項目だけ（design の段）
#     PROBES_DOCS=only ./scripts/diagnose/probes.sh                     # 全部の群のうち docs だけが引く項目（probes_docs の段）
#     ./scripts/diagnose/probes.sh --list                 # 測らずに，選んだ項目のログ名・形・コンテナへ渡すコマンドを 1 行ずつ出す
#                                                        # （上の変数で同じく選べる．排他も出どころの確認もしない．項目や引数を
#                                                        # 直す前と後で出力を比べ，流すコマンドと順が変わらないことを確かめる）
#
# なぜ要るか：論文とスライドには，単コアのタイル積のサイクル数・参照解 Dot2 の誤差・整列窓の 2 項実験・無誤差境界の
# 掃引・スライス数の決め方・L1 の空きなど，表の測定スクリプトが作らない値が載っている．その元は手で流した探針の出力で，
# 出どころ（どの rev・どのビルドで測ったか）も完了の記録も残っていなかった（棚卸しの記録 measure_logs/paper_update/old_values.md
# の (a)．git の外の作業の記録で，追跡していない）．
# ここで全部を同じビルドで流し，verify/check_paper_numbers.py が他の表と同じ出どころの規則（同じ rev・ビルドの記録どおりの
# バイナリ・EXIT,rc=0・性能なら冷却の記録）で照合できるログにする．remeasure_all.sh の probes の段がこれを呼ぶ．
#
# ログは 1 項目 1 本で measure_logs/probe_<項目>.log に置く（ホストのツールの結果も含め全部ここ．出どころの行の無い
# 2026-09 の記録（dot2_reference_check.log・fma_probe.log・pick_p.log・radix_odd.log など）は docs/notes/accuracy_logs_202609/ へ
# 移した）．1 項目を流す手続きは measure_lib.sh の run_item で，形は 3 通り．
#   性能の項目（perf）  measure_lib.sh の run_dram：冷ましてから OZ_NOVERIFY=1 と計時の方式（既定 OZ_TIMING=steady：段ごとに
#                 定常状態まで温めてから測る．TELEM 行に計時区間の AICLK）で 1 回実行（PROVENANCE・BINARY・AICLK・
#                 EXIT・COOLCOND）．単コアのタイル積も冷ましてから測る（1 コアでもチップの温度がクロックを決める．
#                 デバイスのサイクル数で測るので OZ_TIMING は効かず，CONFIG は timing=device_cycles）
#   精度・機能の項目（acc_item）  冷まさない．1 回のコンテナの実行で掃引を流し，先頭に PROVENANCE と使う全バイナリの
#                 BINARY 行，掃引の各回の前に「=== <説明>」，後に「RC,<終了コード>」を残し，最後に EXIT を置く．
#                 EXIT は掃引全体（コンテナの 1 回の実行）の終了コードで，掃引の各回の終了コードは RC 行にある
#                 （無誤差境界の掃引や基数を振る実験は，破綻する側の回が非零で終わるのが期待どおりのため）．
#                 各回の成否（期待どおりに通った／破綻したか）は verify/check_paper_numbers.py が RC 行と出力から判定する．
#                 ただし各回の終了コードのうち期待しうるものは項目ごとに決めて acc_item の 5 番目の引数に書き
#                 （既定 "0"．ホストの検算・探針・OZ_MIN_BITS=1 で値だけを残す到達精度の掃引など，どの回も通るはずの項目），
#                 1（精度不足・破綻・検証の不合格）を許すのは，破綻する回があることを前提に組んだ掃引（無誤差境界，
#                 桁・fidelity の破綻，誤った規則を当てる比較，無誤差条件を破る構成，53 bit を割る φ も記録する構成）だけ．
#                 許していない終了コード（0 だけの項目の 1，2 は入口の検査，126・127 は起動できない，128 以上はシグナル）の
#                 回が 1 つでもあれば EXIT,rc=3 にする（予期しない失敗を成功に数えない．SUB_DEF の sub・sub_done）
#   DRAM に収まらないことの確認（dram_item oom）  冷まさない．EXIT は実行の終了コードのまま（非零）で，DRAM の確保失敗の
#                 行があるときだけ期待どおりとする（diagnose/ablations.sh の abl_band_one_14080 と同じ扱い．検査側も同じ条件で読む）
#
# 項目（ログ名 → 論文・スライドのどの値か．P・A・S・C・E・F の番号は上の棚卸しの記録の通し番号で，照合（probe_checks.py）の
# 各節の注記も同じ番号を使う．番号・一行の説明・論文とスライドの場所・ログ名の表は docs/notes/paper_update/items.tsv．
# 値の出どころは各項目の説明の論文・スライドの節で追える）．項目を流す本体は群ごとのファイル scripts/lib/probes/<群>.sh で，
# 下の「群」の順に読み込む（この入口は選び方・項目の形の部品・群の並びだけを持つ）．
#   probe_tile_<形式>_<fid>_<mode>[_sub1x1]  単コアのタイル積（P1・E-02）．tile_rate_probe を格子 1x1・出力 8x8・K 4・
#                 サブブロック 2x2 で，ウォームアップ 1 回＋3 回の起動の平均・最小・最大（デバイスの壁時計のサイクル数）
#   probe_tilecfg_<ii|i|i12>_<形式>_<mode>  尾崎の GEMM 段のコアあたりの構成（出力ブロック・サブブロック・bw）を 1 コアで再現した
#                 タイル積（ii は 11x10・1x2，i は 8x8・2x2，i12 は 8x8・1x2．bw は経路の 14080^3 の既定．mmpack は K ブロックごとの
#                 パックとパッカの L1 累積を含み，unpack は展開だけ）．110 コアの GEMM 段のコアあたりのサイクル数（設計の比較の
#                 nodram・nomcast・nowrite・nmnw と並べる）と比べる．docs だけが引く形式の unpack と i12 は PROBES_DOCS の群
#   probe_modround      ModRound の総当たり（K=14080．全経路・節の判定と総件数）
#   probe_tfmul         SFPU の模型と実機のビット一致，非 2 冪の基数倍の丸め（per=4）
#   probe_dot2          参照解 Dot2 の __float128 に対する最悪相対誤差（P3・S5-01．ホストの dot2check）
#   probe_ext_k         無誤差分割（BF16x2）の到達精度の K 依存（P4・A8．gemm_bf16ext 3520 3520 K）
#   probe_micro         整列窓の 2 項実験と精度が失われる場所の箇条書き（P5．gemm_bf16ext の OZ_EXPDIFF・OZ_EXPK・OZ_FIDELITY，512^3）
#   probe_fp32acc       同じ 2 項実験の単一コア・K=32（P5．fp32acc_probe）
#   probe_fp8_digit     形式 x fidelity の無誤差境界（P7・P10・A3・C5・F-09．fp8_probe の整数 R の掃引）
#   probe_fp8_rand      乱数入力の到達精度（P7・P8・A6・A7・F-09・F-10・F-13．fp8_probe の rand 系）
#   probe_int8_fid      INT8 の fidelity 別の到達精度（A4・C3．基数 254・1280^3・p=8）
#   probe_int8_zero     INT8 で出力が厳密に 0 になる基数と fidelity（P9・C4．基数 31・15 と，桁 ±31・±15 の基数 62・30）
#   probe_digit_lofi    I-BF16 の桁 32／33 を LoFi で（P10．OZ_RADIX=64・66，3520^3）
#   probe_radix_odd     beta=2d+1 の破れ（P11・F-03．diagnose/radix_odd.sh）
#   probe_radix_d       beta=d のときの p と部分積（P12．OZ_RADIX=OZ_DMAX=d，512^3）
#   probe_pickp         スライス数 p の決め方（P13・C2・F-04．diagnose/select_slice_count.sh．採用値の前後 2 段）
#   probe_rdx3_14080    beta=3 の 14080^3 が DRAM に収まらないこと（P14．本文から外した文と docs だけが引く．PROBES_DOCS の群）
#   probe_l1cap         L1 の空き（P15・F-14．II-BFP4 7040^3 で OZ_L1_BUDGET_KB を二分探索し，失敗した回の CB の
#                       終端の番地から CB に使える L1 を出す）
#   probe_l1b_<予算>_<経路>_14080  L1 予算 1280 KiB と既定の性能（P15・F-15．I-BF16・II-BFP8）
#   probe_fma           SFPMAD の保持幅（P16・S5．fma_probe）
#   probe_dst           DST の FP32 の本数（A2．dst_probe）
#   probe_bf16acc       BF16 累積の単体 GEMM の到達精度（P17・S5-02．gemm_bf16 の verify=1・fp32_dest_acc=0）
#   probe_rprobe<r>_<N> スキーム I の再構成段を OZ_RECON_PROBE=0..3 で（P18・F-16．3520・14080）
#   probe_ndiag_p<p>_3520  組み立てを除いた再構成段の対角の本数への比例（P18・F-17）
#   probe_longk_oz1bf16_140800・probe_limit_*  長い K と立方の上限（P19・S5-16．Ozaki-I-BF16．スキーム II の桁分割の
#                       経路は K <= 16352 で止まるので測らない）
#   probe_psweep_1280   I-BF16 の p 掃引（S1・C1）
#   probe_int8_softfloat  最終スケールを double 定数で掛けた場合の I-INT8 の到達精度（S4）
#   probe_nonneg_7040   非負入力で無誤差条件を破る構成と守る構成（S8・P6．桁 63・HiFi2・7040^3）
#   probe_rowscale_7040 行・列スケールの入力の II-INT8（C8）
#   probe_awe           法の選び方の段の時間のモデルの見込み（C9．ホストの awe_plan）
#   probe_nopl_<経路>_14080・probe_kboff_oz1bfp8_14080  対ループ・K ブロック化の切り分け（S5-04・S5-06・S5-11．I-BF16 の
#                       K ブロックごとの Horner は diagnose/ablations.sh の abl_kb_off_14080．S5-06′・S5-07 の K ブロック化を強制した構成は，
#                       スライドの切り分けを設計の比較の r64hifi2・r126・r126nb で書き直した後どこも引かないので測らない）
#   probe_phi16         phi=16 の b の掃引と 1 パス／2 パス（S5-18・S5-19．4096^3 の II-INT8）
#   probe_2pass_14080   2 パス再構成の費用（S5-19）
#   probe_splitsim・probe_reconsim・probe_twosum・probe_tfconv  ホストの模擬（S5-20・S5-19・F-23・F-22）
#   probe_gemmfid_<fid>_<N>  素の BF16 GEMM の MathFidelity 別（E-09．3520・7040．14080 は measure/fidelity_comparison.sh の gemmfid_*）
#   probe_shape_<M>x<N>_LoFi  素の GEMM の形状の掃引（E-10．LoFi．HiFi4 の 2 本（E-12）はスライドが旧い GEMM 段の頁へ移して
#                       引かなくなったので測らない）
#   probe_bw11_oz1bfp8_14080  I-BFP8 を bw=11 に落とした性能（E-16）
#   probe_gemm_exact    GEMM 段の無誤差の境界（P6 の b_acc・E-07．diagnose/gemm_exact.sh の出力に出どころを付けたもの）．
#                       remeasure_all.sh では selftest の段がこの項目だけを流し，probes の段は PROBES_SKIP で飛ばす
#
# 設計の比較（PROBES_DESIGN．既定 1 で上の項目の後に流す．ログは measure_logs/probe_design_<項目>.log）
#   スライドの補足の頁が「工夫を入れた時点の構成」で測った比較（β=d と β=2d，matmul_tiles と matmul_block，K ブロックごとの
#   Horner，II の組み立ての分担，法の表の選び方，L1 予算，GEMM 段の探針，S と NSMALL の掃引，桁 63/31/15，HiFi2 で基数を
#   上げる選び分けと K ブロック化の切り分け，Int32 累積を別プログラムで足す構成，窓の過剰，桁 64 の破綻，長い K）を，
#   今の実装で代案をつまみ（OZ_RADIX・OZ_DMAX・OZ_FIDELITY・OZ_NOBLOCK・OZ_NO_BLOCKMM・OZ_NO_DEFER 系・OZ2_JSPLIT・
#   OZ_RECON_PROBE・OZ2_MODULI・OZ2_NSMALL・OZ_L1_BUDGET_KB・OZ_GEMM_PROBE_*）で作り，今の構成と同じ回に並べて測る．
#   効果が数 % の比較（matmul_block・法の選び方・L1 予算・GEMM 段の探針）は既定の側も隣に測る（別の回の表のログと
#   比べるとクロックの揺れ（第 5 部）が差に混ざるため）．項目の一覧と棚卸しの番号は lib/probes/design.sh の各節の注記
#
# 拡張（PROBES_EXTENDED=1 のときだけ．どの原稿も今は引いていない）
#   probe_sf_{off,on}_int8_4096（S3 の soft-float の倍率の代案．原稿は diagnose/implementation_ratios.sh の段別の倍率を引く）
#
# 群（scripts/lib/probes/ の下．この順に読み込む．--list の並びもこの順）
#   tile.sh    (1)〜(1c) 単コアのタイル積と GEMM 段のコアあたりの構成（性能．冷ましてから）
#   host.sh    (2) ホストだけの検査（コンテナの中で．デバイスは使わない）
#   accum.sh   (3) 累積側：整列窓の 2 項実験・無誤差分割の K 依存・DST
#   mult.sh    (4) 乗算側：形式 x fidelity の無誤差境界と乱数入力
#   bound.sh   (5) 尾崎の到達精度で見る境界（冷まさない）
#   limits.sh  (6) DRAM と L1 の上限
#   perf.sh    (7) 性能の項目（1 項目ずつ冷ましてから）
#   exact.sh   (8) GEMM 段の無誤差の境界（diagnose/gemm_exact.sh）
#   design.sh  (9) 設計の比較（PROBES_DESIGN．今の実装と代案を同じ回に並べて測る）
#   ext.sh     (拡張) どの原稿も今は引いていない項目（PROBES_EXTENDED）
#
#   ONLY         ログ名の正規表現．当たった項目だけを測る（remeasure_all.sh の REMEASURE_ONLY がここへ渡る）
#   PROBES_SKIP  ログ名の正規表現．当たった項目を飛ばす（remeasure_all.sh は selftest の段で測った probe_gemm_exact を飛ばす）．
#                ONLY と同じく，正規表現として読めなければ測り始める前に止める
#   PROBES_DESIGN=1  設計の比較の群も測る（既定）．0 なら飛ばし，only ならその群だけを測る（0・1・only のほかは止める）
#   PROBES_EXTENDED=1  上の拡張の項目も測る．only なら拡張の項目だけを測る（0・1・only のほかは止める）
#   PROBES_DOCS=1  論文・スライド・照合のどれも読まず docs/ の記述（と本文から外した文）だけが引く項目も測る（既定）．
#                群をまたいで印を付ける：設計の比較の群の φ を広げた入力で選び直した構成の probe_phiacc_*・probe_phiperf_*，
#                (1c) の 1 コアの値，probe_rdx3_14080．0 なら飛ばし，only ならそれだけを測る（0・1・only のほかは止める）．
#                remeasure_all.sh は probes・design の段（0）と probes_docs の段（only．失敗しても図と --fix を止めない）に
#                分けて流す（読まない測定の失敗で原稿の段を止めないため）
#   TMAX・TWAIT・CLK_DT は measure_lib.sh のとおり（冷却のしきい値は論文の条件 60 degC 以下）
set -u
cd "$(dirname "$0")/../.."
LIST=0
case "${1:-}" in
  '') ;;
  --list) LIST=1 ;;
  *) echo "ERROR: 引数 $1 は知らない（使えるのは --list だけ．項目は ONLY・PROBES_* で選ぶ）" >&2; exit 2 ;;
esac
OUT=measure_logs; [ "$LIST" = 1 ] || mkdir -p "$OUT"
# つまみの検査はデバイスの排他を取る前に行う（別の測定がデバイスを使っていると，排他を待った後で打ち間違いに気づく）
EXT="${PROBES_EXTENDED:-0}"
case $EXT in 0|1|only) ;; *) echo "ERROR: PROBES_EXTENDED=$EXT は 0・1・only のどれか" >&2; exit 2 ;; esac
IN_EXT=0             # 拡張の項目を流している間は 1（PROBES_EXTENDED=only は拡張の外の項目を飛ばす）
DES="${PROBES_DESIGN:-1}"
case $DES in 0|1|only) ;; *) echo "ERROR: PROBES_DESIGN=$DES は 0・1・only のどれか" >&2; exit 2 ;; esac
if [ "$DES" = only ] && [ "$EXT" = only ]; then
  echo "ERROR: PROBES_DESIGN=only と PROBES_EXTENDED=only は同時に与えない（どちらの群も流れない）" >&2; exit 2
fi
IN_DESIGN=0          # 設計の比較の群を流している間は 1
DOCS="${PROBES_DOCS:-1}"
case $DOCS in 0|1|only) ;; *) echo "ERROR: PROBES_DOCS=$DOCS は 0・1・only のどれか" >&2; exit 2 ;; esac
IN_DOCS=0            # docs だけが引く項目を流している間は 1
if [ -n "${PROBES_SKIP:-}" ]; then
  _r=0; printf 'x\n' | grep -Eq -- "$PROBES_SKIP" 2>/dev/null || _r=$?
  [ "$_r" -le 1 ] || { echo "ERROR: PROBES_SKIP='$PROBES_SKIP' は正規表現（grep -E）として読めない" >&2; exit 2; }
  unset _r
fi
if [ "$LIST" = 1 ]; then
  . ./scripts/measure_lib.sh --defs-only   # 一覧だけ：ビルドの記録の確認と排他は要らない（timing_env などの定義だけ）
else
  . ./scripts/measure_lib.sh   # provenance_check・デバイスの排他・wait_cool・run_dram・provenance_line・bin_line
fi
LABEL_W=40
fail=0
# DRAM の確保に失敗した行（lib/constants.tsv の OOM_RE．照合の probe_checks.py の OOM も同じ行を読む）
OOM_PAT=$(const_get OOM_RE) || exit 2
# 経路の既定の構成（routes.tsv の p・L・法の本数）．既定の構成に代案のつまみを足して測る項目（設計の比較の base・sbase・
# wbase と代案，L1 予算の対，再構成段の探針，長い K と上限，φ の選び直しなど）は，バイナリと引数をここから引く．
# 書き下すと，表の p・法の本数を変えたとき（I-BFP2 56→59，I-BFP4 15→16 など）に直し忘れた項目だけが既定でない構成で
# 黙って測られ，照合（probe_checks.py）もそれを経路の表と突き合わせない．p を振る・基数を変えるなど構成そのものを
# 変える項目は，その p を書き下したままにする
. ./scripts/routes.sh        # routes_each・routes_row_args
declare -A OZARGS            # ログ名の幹 → "バイナリ p L"（スキーム I）か "バイナリ s"（II）
_oz_row() { OZARGS[$r_log]=$(routes_row_args); }
routes_each _oz_row || exit 2
# oz_set ログ名の幹   その経路のバイナリを OZ_BIN に，既定の引数（I は p L，II は法の本数）を OZ_PA に置く．表に無ければ
#   ここで止める（値はコマンド置換で返さない：置換の中で止まると空の引数のまま流してしまうため）
oz_set() {
  [ -n "${OZARGS[$1]:-}" ] || { echo "ERROR: 経路の表 routes.tsv に log=$1 の行が無い" >&2; exit 2; }
  set -- ${OZARGS[$1]}
  OZ_BIN=$1; shift; OZ_PA="$*"
}
# 規則 (i)・(ii) で p を選び直す形式（diagnose/select_slice_count.sh と同じ選び方．probe_pickp の BINARY 行）
PICKP_BINS=""
_pk_row() { PICKP_BINS="$PICKP_BINS${PICKP_BINS:+ }$r_binary"; }
routes_each _pk_row scheme=I acc_as=- || exit 2
T0=$SECONDS

# ---- 群のファイル（scripts/lib/probes/*.sh）が項目ごとに呼ぶ部品 ----
# want ログ名   ONLY に当たり，PROBES_SKIP に当たらず，群の選択（PROBES_DESIGN・PROBES_EXTENDED・PROBES_DOCS）に入れば 0
want() {
  if [ "$EXT" = only ] && [ "$IN_EXT" = 0 ]; then return 1; fi
  if [ "$DES" = only ] && [ "$IN_DESIGN" = 0 ]; then return 1; fi
  if [ "$DES" = 0 ] && [ "$IN_DESIGN" = 1 ]; then return 1; fi
  if [ "$DOCS" = only ] && [ "$IN_DOCS" = 0 ]; then return 1; fi
  if [ "$DOCS" = 0 ] && [ "$IN_DOCS" = 1 ]; then return 1; fi
  if [ -n "${ONLY:-}" ] && ! printf '%s\n' "$1" | grep -Eq -- "$ONLY"; then return 1; fi
  if [ -n "${PROBES_SKIP:-}" ] && printf '%s\n' "$1" | grep -Eq -- "$PROBES_SKIP"; then return 1; fi
  return 0
}
# list_item ログ名 形 コマンド   --list の 1 行（ログ名・形・コンテナへ渡すコマンド．改行は \n と書く）
list_item() { local c=$3; printf '%s\t%s\t%s\n' "$1" "$2" "${c//$'\n'/\\n}"; }
# perf ラベル ログ名 環境変数… -- コマンド…   性能の 1 項目（run_dram．RUN_PAT は呼び出し側が置く）
perf() {
  want "$2" || return 0
  if [ "$LIST" = 1 ]; then   # run_dram が run.sh へ渡すのと同じ文字列（冷却・計時の方式・指標の行の拾い方は run_dram）
    local log=$2 env=""; shift 2
    while [ "$1" != "--" ]; do env="$env $1"; shift; done; shift
    list_item "$log" "perf(RUN_PAT=${RUN_PAT:-})" "cd /home/user/tt && OZ_NOVERIFY=1 $(timing_env) $env $*"
    return 0
  fi
  run_dram "$@"
}
# section 見出し   節の見出し．群を選んで流すとき（PROBES_DESIGN=only・PROBES_EXTENDED=only・PROBES_DOCS=0|only）は，流さない節の見出しを出さない
section() {
  [ "$LIST" = 1 ] && return 0
  if { [ "$DES" = only ] && [ "$IN_DESIGN" = 0 ]; } || { [ "$EXT" = only ] && [ "$IN_EXT" = 0 ]; } \
     || { [ "$DOCS" = only ] && [ "$IN_DOCS" = 0 ]; } || { [ "$DOCS" = 0 ] && [ "$IN_DOCS" = 1 ]; }; then return 0; fi
  printf '===== %s（%d 分経過）=====\n' "$*" $(( (SECONDS - T0) / 60 ))
}

# コンテナの中で掃引の各回を包む関数（acc_item の断片の先頭に置く）．sub 説明 -- コマンド…
# 見出しは「=== probe:」で始める（「=== fmt=」は measure/floating_decomposition.sh の区切りとして oz_logs.py が EXIT の数と突き合わせる）．
# 各回の終了コードが SUB_OK（その項目で期待しうる終了コードの空白区切り．acc_item が置く）に無ければ SUB_BAD に足し，
# 断片の後に acc_item が置く sub_done が，そのような回があれば理由を出して 3 を返す（コンテナの終了コード＝EXIT になる）．
# 断片の中で sub の外の準備に失敗したとき（probe_phi16 の法の集合を取る実行）も SUB_BAD に足す
SUB_DEF='sub() { local d="" r; while [ "$1" != "--" ]; do d="$d $1"; shift; done; shift; echo "=== probe:$d"; "$@"; r=$?; echo "RC,$r";
  case " ${SUB_OK:-0} " in *" $r "*) ;; *) SUB_BAD="${SUB_BAD:+$SUB_BAD }$r" ;; esac; };
  sub_done() { [ -z "${SUB_BAD:-}" ] && return 0; echo "ERROR: 掃引に予期しない終了コードの回がある（$SUB_BAD．この項目で期待しうるのは ${SUB_OK:-0}）" >&2; return 3; };'

# acc_item ログ名 ラベル "バイナリ…" 'コンテナの中で流す断片' [期待しうる終了コード（既定 "0"）]
#   精度・機能の項目（冷まさない）．断片は sub で各回を包む．表示は掃引の回数と所要時間．
#   5 番目の引数は掃引の各回に許す終了コード（冒頭の注記．破綻を測る掃引だけ "0 1"）．
#   コンテナへ渡すコマンドの文字列は 1 度だけ組み，--list と測るときに同じものを使う（SUB_DEF の中身も一覧に出る）
acc_item() {
  local log=$1 label=$2 bins=$3 body=$4 ok=${5:-0} cmd rc t=$SECONDS
  want "$log" || return 0
  case " $ok " in *[!0-9\ ]*) echo "ERROR: acc_item $log の期待しうる終了コード「$ok」が整数の並びでない" >&2; exit 2 ;; esac
  [ -n "$bins" ] || { echo "ERROR: acc_item $log の実行ファイルが空" >&2; exit 2; }
  cmd="cd /home/user/tt && SUB_OK='$ok'; $SUB_DEF $body
sub_done"
  if [ "$LIST" = 1 ]; then list_item "$log" "acc(bins=$bins)" "$cmd"; return 0; fi
  run_item --bins "$bins" "$OUT/$log.log" "$cmd"   # measure_lib.sh（流している途中で変わっていれば EXIT の代わりに INVALIDATED）
  rc=$RI_RC
  if [ "$rc" -ne 0 ]; then
    fail=1; printf "  %-${LABEL_W}s FAILED（exit %s．%s）\n" "$label" "$rc" "$OUT/$log.log"
  else
    printf "  %-${LABEL_W}s %d 回（%d s．%s）\n" "$label" "$(grep -c '^RC,' "$OUT/$log.log")" $((SECONDS - t)) "$OUT/$log.log"
  fi
}

# dram_item oom|limit ログ名 ラベル 環境変数… -- コマンド…   DRAM の上限の 1 項目（冷まさない．OZ_NOVERIFY=1）．
#   流し方と出どころの行・EXIT は共通で，期待どおりかの判定だけが違う：
#     oom    DRAM に収まらないことの確認．非零で終わり DRAM の確保失敗の行（OOM_PAT）があるときだけ期待どおり
#     limit  上限の確認（動くか DRAM 不足で止まるかを記録する）．0 で終わるか DRAM 不足で止まるかのどちらかなら期待の範囲で，
#            それ以外の止まり方だけを失敗にする
dram_item() {
  local kind=$1 log=$2 label=$3 env="" rc cmd; shift 3
  while [ "$1" != "--" ]; do env="$env $1"; shift; done; shift
  want "$log" || return 0
  cmd="cd /home/user/tt && OZ_NOVERIFY=1 $env $*"
  [ "$LIST" = 1 ] && { list_item "$log" "$kind" "$cmd"; return 0; }
  run_item "$OUT/$log.log" "$cmd"   # measure_lib.sh
  rc=$RI_RC
  if [ "$kind" = oom ]; then
    if [ "$rc" -ne 0 ] && grep -Eq -- "$OOM_PAT" "$OUT/$log.log"; then
      printf "  %-${LABEL_W}s 期待どおり DRAM 不足（exit %s）\n" "$label" "$rc"
    else
      fail=1; printf "  %-${LABEL_W}s FAILED（DRAM 不足で止まるはずが exit %s．%s）\n" "$label" "$rc" "$OUT/$log.log"
    fi
  elif [ "$rc" -eq 0 ]; then printf "  %-${LABEL_W}s 動いた（%s）\n" "$label" "$(grep -oE '[0-9.]+ TFLOP/s fp64-equiv' "$OUT/$log.log" | head -1)"
  elif grep -Eq -- "$OOM_PAT" "$OUT/$log.log"; then printf "  %-${LABEL_W}s DRAM 不足（exit %s）\n" "$label" "$rc"
  else fail=1; printf "  %-${LABEL_W}s FAILED（exit %s．DRAM 不足の行も無い．%s）\n" "$label" "$rc" "$OUT/$log.log"; fi
}

# ---- 群（冒頭の「群」．項目はこの順に流れる．群の選択と ONLY は各項目の want が見る）----
for _g in tile host accum mult bound limits perf exact design ext; do
  . "./scripts/lib/probes/$_g.sh" || { echo "ERROR: scripts/lib/probes/$_g.sh を読めない" >&2; exit 2; }
done
unset _g

[ "$LIST" = 1 ] && exit 0
printf '===== 終わり（%d 分）=====\n' $(( (SECONDS - T0) / 60 ))
echo "(全出力は $OUT/probe_*.log に残る)"
only_summary
exit $fail
