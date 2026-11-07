# tools/ — デバイスを使わないホスト単体の検査・模擬

尾崎の実装のうち，デバイスで回さなくても確かめられる部分（triple-float の変換，SFPU の FP32 演算列，法の選び方，
再構成の累算など）をホストの C++ で模擬し，厳密な参照（`__float128`・多倍長の固定小数・整数）と比べる道具を置く．
本番のドライバはここのソースを直接取り込まない．ただし `awe_plan emit-coef` が生成する係数表は本番の法の選択に使うので，その変更も本番に影響する．

どれもリポジトリ直下から回す（`scripts/routes.tsv` を読む道具がある．置き場所は `OZ_ROUTES_TSV` で変えられる）．
ビルドは `src/Makefile` のターゲットで行う（論文・スライドの値を測る HOST_TOOLS と検査の CHECK_TOOLS．コンテナの中で
`make -C src <名前>` が `src/build/<名前>` を作り，`build_all.sh` が作るものはビルドの記録 `BUILD_STAMP` に載る．フラグは
`HOST_CXXFLAGS`，取り込みの根は `src`）．`dgemmcheck.cpp` は別のイメージで `scripts/measure/cpu_dgemm_reference.sh` が同じフラグ（`make print-host-flags`）で作る．
`tools/study/` の 3 本（`oz2_quantsim`・`fp8_sparse_plan`・`tfrange`）はどのスクリプトも作らず回さない調査と記録の道具で，
手で作って回す（手順は各ファイルの冒頭）．`aligned_accum` は本番の定数 `OZ2_RGRAN` を判定する検査の道具なので `tools/` に置く．

## 道具ごとの位置づけ

「作る」「回す」は今のパイプラインのどこからか（「手」は誰も作らない・回さない）．「出どころ」はその道具の出力が
どの数値・どの注記の根拠になっているか．「確かめた版」の clean1 の段階 1 は `9bcd412`，段階 4 は `8197a0b` を指す．以下はその時点の検証記録であり，clean2 の検査結果は測定前検証の記録で区別する．

| 道具 | 何を見るか | 作る | 回す（引数） | 出どころ | 確かめた版 |
|---|---|---|---|---|---|
| `accuracy_check.cpp` | Dot2の非評価和に対する精度指標（相殺・零・極小・非有限・独立な113bit参照） | CHECK_TOOLS | `selftest.sh`（引数なし） | 精度検査の回帰確認。デバイスの測定値を生成しない | check7修正時に10021項目を確認 |
| `dot2check.cpp` | 参照解 Dot2 の相対誤差（`__float128` の参照和と比べる） | `src/Makefile` の HOST_TOOLS（`src/build/dot2check`） | `diagnose/probes.sh` の `probe_dot2`（引数なし．φ=0〜16・K=1024/14080 の 12 条件） | スライドの「参照和に対して 2^-85 以下」（`probe_checks.py` の `check_dot2`），`oz_reference.hpp` の `reference_gemm` の注記，`docs/accuracy.md` | 042dc00 の `measure_logs/probe_dot2.log`．rev `9bcd412` の整理で同じ出力を確かめた |
| `awe_plan.cpp` | 尾崎 II の法の集合を段の時間のモデルで選ぶ探索のホスト版（Phase A） | HOST_TOOLS | `diagnose/probes.sh` の `probe_awe`（`fp8 tf32`） | スライドの「見積り式による見込み」（`probe_checks.py` の `awe`），`docs/awe_moduli.md` の Phase A，`src/ozaki2/ozaki2_awe_coef.inc` の係数の表 `OZ2_AWE_COEF`（`emit-coef` が書き出す．`selftest.sh` が `verify-coef` で突き合わせる） | 042dc00 の `measure_logs/probe_awe.log`．rev `9bcd412` の整理で全経路の出力が同じことを確かめた |
| `splitsim.cpp` | 尾崎IIの量子化誤差の模型（CRTは含めず，量子化後の内積を`__float128`の補正和で計算） | HOST_TOOLS | `diagnose/probes.sh` の `probe_splitsim`（φ=0,2,4 × b=48..68，`1024 1024`） | スライドの「量子化ビット数 b に対する実効ビット数」の表の「量子化のみ」3 列（`probe_checks.py`），`docs/accuracy.md` の表 | rev `9bcd412` の整理．入力をドライバと同じ `gen_operands` にし，出力の区画の全要素で測るように直したので 042dc00 のログとは値が違う（`1024 1024` は実機の tab:phib と同じ量で，スライドの実機の列と b<=52 では 0.1 bit 以内，φ=2・4 は b=56 も同じ値） |
| `reconsim.cpp` | 再構成の 3 本組・4 本組の累算の最下位 limb の丸め（仮想の構成を含む） | HOST_TOOLS | `diagnose/probes.sh` の `probe_reconsim`（`16 3 1`・`16 6 1`） | スライドの「記録に残る epsilon_f は法の個数 16」（法の本数だけ照合），`ozaki2_moduli.h`・`recon_crt.cpp`・`docs/accuracy.md` の 2^-67.0 | 042dc00 の `measure_logs/probe_reconsim.log`．rev `9bcd412` の整理で 40 条件が同じことを確かめた |
| `twosum_probe.cpp` | `tfAddSmall` を入れても分数形 CRT の frac の最悪誤差が悪化しないか | HOST_TOOLS（`selftest.sh` も同じ `src/build/twosum_probe` を回す） | `selftest.sh`（`14080 11 20000`），`diagnose/probes.sh` の `probe_twosum`（seed 11・16 × 比較用のつまみ 4 通り） | スライドの tfAddSmall の表と「2 段全体の最悪絶対誤差」（`probe_checks.py`），`sfpu_tf.h` の `tfAddSmall` の注記 | 042dc00 の `measure_logs/probe_twosum.log`．rev `9bcd412` の整理で 9 通りの各行が同じことを確かめた（行の並びだけが変わる） |
| `tfconv.cpp` | FP64 と triple-float の変換（`f64_tf.h` の split・assemble）の 8 検査 | HOST_TOOLS（`selftest.sh` も同じものを回す） | `selftest.sh`，`diagnose/probes.sh` の `probe_tfconv`（引数なし） | スライドの FP64 の組み立ての頁（「8 検査もすべて PASS」）．照合は検査ごとの判定・件数・1 ulp を超える差まで見る | rev `9bcd412` の整理．検査 4・6 の定数を本番の値に替えた．ほかの検査は 042dc00 と同じ出力 |
| `modround_check.cpp` | `ModRound` を実際の法・係数の全域で総当たり（約 23.8 億通り） | CHECK_TOOLS | `selftest.sh` と `diagnose/probes.sh` の `probe_modround`（`14080`） | `sfpu_tf_proc.h` の `ModRound` の注記（m=3475 の例，はみ出し 8170 通り），`ozaki2_moduli.h` の `oz2_crt_k_bound` | rev `9bcd412` の整理．件数・値は 042dc00 と同じ．経路の行の名前だけが変わる |
| `tfmul_check.cpp` | 基数倍 `tfMulR`・`tfMulP2` の最後の FastTwoSum の前提（`sfpu_tf_proc.h` の補題 P・R） | CHECK_TOOLS | `selftest.sh`（`20000 1`） | `sfpu_tf_proc.h` の補題の注記（相殺しない入力の 尾/ulp の最大 1.500・1.432） | rev `9bcd412` の整理．尾崎 I の p・L を `routes.tsv` から読むように直した．BFP4_b の L=15・BFP2_b の L=58 で 2^-66 の判定を通る |
| `crt_fold_check.cpp` | 分数形 CRT の小数部の畳み込みを窓の端で確かめる | CHECK_TOOLS | `selftest.sh`（`400 7` と `const TF32 32 <16 本>` の終了コード 2） | `recon_crt.cpp` の畳み込みの注記（`table` の既定の法集合の余白の最小．BF16 で 1.5e-7 bit，K=12096） | rev `9bcd412` の整理．`400 7` は 042dc00 と同じ出力．`table` は手で回す |
| `cmp_dump.cpp` | `OZ_DUMP_C` の 2 つの書き出しを要素ごとに比べる | CHECK_TOOLS | `check_nsmall.sh` | `docs/nongemm_opt_results.md` の §3（数は表示だけで判定には使わない） | rev `9bcd412` の整理．合成した書き出しで確かめた |
| `dgemmcheck.cpp` | CPU の FP64 DGEMM（OpenBLAS）の到達精度を尾崎と同じ入力・参照・指標で | `scripts/measure/cpu_dgemm_reference.sh`（`scripts/Dockerfile.dgemmref` のイメージ） | `remeasure_all.sh` の dgemm の段 | 論文のネイティブ DGEMM の到達精度（`verify/check_paper_numbers.py`） | 042dc00 の `accuracy_logs/dgemm_ref.log`（clean1 の `8197a0b` では入口の検査と取り込みを整理．同じ入力・参照・指標かを `lib/peer_logs.py` が照合する） |
| `aligned_accum.cpp` | 2 パス構成の第 1 パス（共通粒度へ丸めた断片 rp1）の累算が無誤差か | CHECK_TOOLS | `selftest.sh`（引数なしで `OZ2_RGRAN` を判定し，粒度を掃引） | `ozaki2_moduli.h` の `OZ2_RGRAN=64` の根拠，`docs/accuracy.md`（全経路・3 サイズで g=68 まで無誤差，69 で崩れ始める） | rev `9bcd412` の整理．本番の法の集合・対称剰余・本番の足し込みで回すように書き直した |
| `routes_check.cpp` | `routes.tsv` の全経路と本番の桁・法の既定の構成 | CHECK_TOOLS | `selftest.sh` | 表の欠落・重複・構成のずれを測定前に検出 | rev `3bf2873` の selftest（`docs/release/clean2_premeasure_audit.tsv`） |
| `knobs_check.cpp` | つまみの表（`src/common/oz_knobs.hpp`）と `docs/env_vars.md` の節（A〜F・削除した）の名前と分類，`src`・`tools` のソースが読むつまみの名前の突き合わせ（`--dump` で表を TSV で出す） | CHECK_TOOLS | `selftest.sh`（`docs/env_vars.md src tools`） | `docs/env_vars.md` の冒頭（表が正本．食い違えば終了コード 1） | rev `8197a0b`で足した |
| `study/oz2_quantsim.cpp` | 尾崎 II の量子化だけの誤差を出力の全要素で（OpenBLAS） | 手（`dgemmcheck` と同じイメージ．手順は冒頭） | 手 | `docs/accuracy.md` の φ=16 の表，スライドの「b=60 で 49.52，b=71・80 で 49.95 bit」（照合も再実行もされない） | rev `9bcd412` の整理．512^3・φ=16 で前と同じ出力を確かめた．4096^3 は回していない |
| `study/fp8_sparse_plan.cpp` | FP8 の疎な e4m3 桁で法を選び直したときの見積り（Phase A の記録） | 手 | 手 | `docs/fp8_sparse_digits.md` の Phase A．今の `ozaki2_moduli.h` を取り込むので，その記録の版の再現にはならない | rev `9bcd412` の整理で動くことだけを確かめた |
| `study/tfrange.cpp` | FP64 を FP32 3 本へ分けたとき無誤差に運べる指数の下端（IEEE と FTZ） | 手 | 手 | `ozaki2_main.cpp` の 2^-97 の注記の前提 | rev `9bcd412` の整理で動くことだけを確かめた |

`awe_plan` の `coef`・`emit-coef`・`table`，`crt_fold_check` の `table` はどのスクリプトも回さない（`awe_plan` の `verify-coef` は
`selftest.sh` が回し，`crt_fold_check` の `const` は `selftest.sh` が積の溢れを断る 1 件だけ回す）．本番の係数の表は `emit-coef` の
出力そのもので（較正の基準か実測 `MEAS` を直したら `./src/build/awe_plan emit-coef > src/ozaki2/ozaki2_awe_coef.inc`），
`verify-coef` が書き出し直し忘れと手の直しを止める．

## 共有する部品

- `tools/oz_sim.h`：0/1 のつまみの読み方（`env_flag01`．中身は `src/common/oz_knobs.hpp` の表），`routes.tsv` のスキーム I の行（p・L）を読んで本番の基数・桁の上限と組にする
  `load_routes1`・`scheme1_configs`（基数・桁の上限は `ozaki1/ozaki1_param.h` の表から引く），スキーム II の行の法の本数が本番の既定と
  同じことを確かめる `check_routes2_def_s`（`routes_check` が回す），スキーム II の経路を表示名で引く `route2`，
  再構成 `ozaki2/kernels/recon_crt.cpp` の 1 要素（`crt_load`・`crt_term_k`・`crt_accumulate`・`crt_element`．法ごとの k_i・断片の
  足し込み・整数部の削除は本番と同じ `ozaki2/kernels/recon_crt_proc.h` のテンプレート `crt_proc::*`，その中の `tfAddS`・`tfAddSmall`・
  `ModRound` は本番と同じ `sfpu_tf_proc.h`）とカーネルへ渡す定数（`make_recon_kernels`．コンパイル時引数は本番と同じ
  `ozaki2_moduli.h` の `oz2_recon_ct`）．
- `src/checks/sfpu_tf_emu.hpp`：`common/kernels/sfpu_tf.h` の SFPU のタイル演算 1 つのビット単位の模型 `Dst`（非正規化数の
  扱い・比較用の版への切り替え・計装は `Dst` のメンバ）と，その上で組む triple-float の手続き．手続き（`TwoSum`・`tfMulR`・
  `tfMulP2`・`tfAddS`・`tfAddSmall`・`ModRound`）は本番と同じ `common/kernels/sfpu_tf_proc.h` のテンプレートで，写しではない．
  `src/checks/tfmul_probe.cpp` も取り込む．

本番のヘッダにある表・関数（形式の表 `common/oz_format_host.h`，スキーム I の `ozaki1/ozaki1_param.h` の `oz1_param`・
`oz1_default_digit_max`，スキーム II の `ozaki2/ozaki2_split.h` の経路の表 `OZ2_ROUTES`・`oz2_param`・`oz2_in_residue` と
桁分割の方式，`ozaki2_moduli.h` の法の選択・`oz2_choose_small_tail`・`oz2_crt_k_bound`・`oz2_recon_limb3`・`oz2_recon_ct`・`OZ2_TABLE_I8`・
`oz2_recon_rawmask`・`OZ2_B_DEFAULT`，`ozaki2_awe.h` の `OZ2_AWE_COEF`・`OZ2_AWE_SLICE_TILE`・`OZ2_AWE_ONE_RECON`・
`OZ2_AWE_DRAM_FREE`，`ozaki2_split.h` の合成段の平面と DRAM の見積り `oz2_compose_plan`・`oz2_dram_one_group`・`oz2_dram_groups`
（`awe_plan` の `dram_plan`），`f64_tf.h`，ビット変換の
`common/oz_bits.h`，`oz_input.hpp` の `gen_operands`・`host_eval.hpp` の `rt_double`，`oz_knobs.hpp` の `reject_unread_input`，`oz_env.hpp` の読み方）と，カーネルの手続きのうちデバイス API に
依存しない部分（triple-float と法の剰余の `common/kernels/sfpu_tf_proc.h`，再構成の法ごとの手続きの `ozaki2/kernels/recon_crt_proc.h`．
タイル演算のラッパの型を引数に取るテンプレートで，模擬の `Dst` の上で組む）はそのまま取り込んで使う（どれも tt-metal に依存しない）．

## 本番の写し（まだ残っているもの）

本番の手続きがカーネルの中にあってホストから呼べないため，次は写しのままである．
どれも，本番を直したら一緒に直す．写しが外れたときに止まる仕組みも書く．

| 写し | 元 | 外れたとき |
|---|---|---|
| `oz_sim.h` の `crt_load`・`crt_element`（入力の読み込みと，法のループの並び） | `ozaki2/kernels/recon_crt.cpp` の `load_residue_fp32`・`decode_signmag_residue`・`kernel_main` | 止まらない（法ごとの k_i・断片の足し込み・整数部の削除は `recon_crt_proc.h` を，その中の `tfAddS`・`tfAddSmall`・`ModRound` は `sfpu_tf_proc.h` を本番と共有し，コンパイル時引数 RAWMASK・NSMALL_FRAG・limb の本数は `ozaki2_moduli.h` の `oz2_recon_ct`・`oz2_recon_limb3` で決めるので，写しは読み込みと法のループだけ） |
| `sfpu_tf_emu.hpp` の `Dst`（SFPU のタイル演算 1 つの模型）と `load_f64`・`slice_step`（スライサの読み込みと 1 段） | `common/kernels/sfpu_tf.h` のタイル演算のラッパ，`slice_tf_core.h` の `tf_load_f64`，`slice_f64_reader.cpp` | `src/checks/tfmul_probe.cpp` が本番の関数（タイル演算のラッパ・`Rnd`・`tfMulRadix`・`tfMulR`・`tfAddS<3>`・`tf_load_f64`）を実機で呼んでビット単位で比べ，外れれば FAIL（`scripts/selftest.sh` の「SFPU emulation vs device」が毎回回す）．手続き（`tfMulR`・`tfAddS` など）は `common/kernels/sfpu_tf_proc.h` を本番と共有するので写しではない（実機と比べていない `tfAddSmall`・`ModRound` も，演算の模型が一致すれば一致する） |
| `aligned_accum.cpp` の `rp1_of`（粒度 g の断片の作り方） | `ozaki2_moduli.h` の `oz2_crt_constants` | g = `OZ2_RGRAN` で本番の `rp1` とビット単位で比べて外れれば FAIL |
| `modround_check.cpp` の FP32 の桁分割（`digits_fp32`） | `ozaki2/kernels/modslice_digit.cpp` | 写しのまま独立に持つ（本番と共通にすると模擬の独立性が無くなる） |

| `modround_check.cpp` の `modround`・`modround2` | `sfpu_tf_proc.h` の `ModRound` | `emu_compare` が共有手続きと照合（前者は全件，後者は 16 件ごと）．不一致は FAIL |
| `modround_check.cpp` の `sqs_fix_fp32` | `oz2_sparse_fix.h` の疎な桁の補正 | 独立な FP32 模型．modround_check が剰余類と桁の表現範囲を検算する．命令列の一致を保証する検査ではない |
| `tfmul_check.cpp` の桁を作るループ | `oz2_digits.h` の `oz2_make_digits` | FastTwoSum の前提と出力誤差を独立に検算．実機の桁分割は `check_concat.sh` が剰余・CRT の整数参照と照合する |
| `tfmul_check.cpp` の Int32 復号 | `recon_tf_i32in.cpp` | `check_nsmall.sh`・`check_exact_boundary.sh` で実機の復号結果と到達精度を確認する |
| `awe_plan.cpp` の集合探索 | `ozaki2_awe.h` の探索 | 調査用の費用探索で本番と同じ集合を保証しない．係数表は `verify-coef`，経路の構成は `routes_check` が照合する |

`recon_nsmall` の写しは廃止し，`tfmul_check` と本番が `ozaki1_param.h` の `oz1_recon_nsmall` を共有する．
上表の独立模型は，演算手続きそのものの共有・実機との命令列の一致とは区別する．

`crt_fold_check.cpp` の `recenter_fp32`（計算カーネルで FP32 のまま畳む方式）は本番に無い比較用の模型で，写しではない．

## つまみ

検査の比較用の版を有効にするつまみは 0/1 で読む（未設定と 0 は無効，1 は有効，ほかは終了コード 2）：
`TWOSUM_PROBE_ABS`・`TWOSUM_PROBE_FAST`・`TWOSUM_PROBE_ALL`（`twosum_probe`），`TFMUL_VERBOSE`（`tfmul_check`），
`AWE_DEBUG`（`awe_plan`）．経路の表の置き場所は `OZ_ROUTES_TSV`．一覧は `docs/env_vars.md` の F．
入力を作る `dot2check`・`splitsim` は尾崎の入力のつまみ（`OZ_PHI` を含む．φ は引数で渡す）・`OZ_MIN_BITS`・`OZ_DUMP_C` を
受け付けずに止まる（`oz_knobs.hpp` の `reject_unread_input`．つまみの表の入力・判定・書き出しの組）．
