# src — 全ターゲットを 1 つのビルドで

論文が測定する行列積プログラムと，探針・検査・ホストの道具はすべて，このディレクトリの素の `Makefile` 1 つから
ビルドされる（CMake は使わない）。出力は組によらず `build/<ターゲット>` に平らに置く。

```sh
make                     # tt-metalium コンテナの中で実行 -> build/<ターゲット>（全部の組）
make release             # 本番の 16 経路と基準実装（論文の比較対象 gemm_bf16・gemm_bf16ext）だけ（DRIVERS・BASELINES）
make probes checks       # 組ごと（drivers・baselines・host-tools・check-tools も同じ）
make -n release          # 作り直すものを表示するだけ（build/ には何も書かない）
./build/ozaki1_int8 1280 1280 1280 8 7 1 3
```

測定のパイプラインは `../scripts/build_all.sh`（`make -B` で全部を作り直してビルドの記録 `build/BUILD_STAMP` を書く）でビルドする。
測定に使うバイナリはこの全部の作り直しに頼り，下の前提の印には頼らない。
手で `make` するときのために，各ターゲットはソース・取り込んだ `src` のヘッダに加えて，前提の印 `build/<ターゲット>.cmd`（ツールチェーンの
識別と，そのターゲットを作るコマンド）を前提に持つ。印は頼まれた目標が作るターゲットについてだけ，中身が変わったときに書き換わる
（`make -n` では書かない）。印が見分けるのは，コンパイラ・リンカ・glibc の版，フラグ，`-march=native` の解決値，tt-metal の rev と，
取り込むヘッダとリンクするライブラリのファイルの一覧（パス・種類・大きさ・時刻）までで，ファイルの中身は見ない：同じ大きさ・時刻のまま
中身を差し替えた tt-metal では作り直さない（`Makefile` の「ビルドの前提の印」）。

## ディレクトリ

| 置き場所 | 置くもの |
|---|---|
| `ozaki1/`・`ozaki1/kernels/` | 尾崎 scheme I（DRIVERS の `ozaki1_*`）のホスト側（`ozaki1_main.cpp` と `ozaki1_*.hpp`，tt-metal に依存しない形式ごとの値の表 `ozaki1_param.h`）と，scheme I だけが使うデバイスカーネル（引数の表 `oz1_args.h` を含む） |
| `ozaki2/`・`ozaki2/kernels/` | 尾崎 scheme II（DRIVERS の `ozaki2_*`）のホスト側（`ozaki2_main.cpp`，経路の表と法 1 本の運び方の `ozaki2_split.h`，法の選択の `ozaki2_moduli.h`・`ozaki2_awe.h`）とデバイスカーネル（引数の表 `recon_args.h`・`oz2_args.h`，ホストと共有する上限 `oz2_limits.h` を含む） |
| `baseline/` | 基準実装（BASELINES．論文の比較対象）のホスト側 `gemm_bf16.cpp`・`gemm_bf16ext.cpp`（カーネルは `common/kernels/` と tt-metal 付属のもの） |
| `common/`・`common/kernels/` | 両スキームと基準実装が共有するホスト側とデバイスカーネル（GEMM 段の `bmm_int32_spill.cpp`・`bmm_l1acc.cpp` を含む）．カーネルの置き場所の定数 `oz_paths.hpp`，形式の表（`oz_format_host.h`・`oz_format_table.hpp`）と形式の選択（`oz_format.hpp`） |
| `probes/`・`probes/kernels/` | 探針（PROBES）とそのカーネル |
| `checks/`・`checks/kernels/` | 検査（CHECKS．INT8 の行列積の無誤差の検査 `gemm_int8.cpp` を含む）とそのカーネル，ホストの検査が使う SFPU の模擬 `sfpu_tf_emu.hpp` |
| `testkit/`・`testkit/kernels/` | 探針と検査が共有する部品（本番は取り込まない．下の「共通部分」） |
| `../tools/` | ホストだけで動く道具（HOST_TOOLS・CHECK_TOOLS のソース．置き場所の説明は `../tools/README.md`） |

取り込みとカーネルのパスの書き方：

- ホストの `#include` はこのディレクトリからの相対で書く（`Makefile` の `-I.`．`"common/mcast_gemm.hpp"`・`"ozaki2/ozaki2_moduli.h"` など．
  `../tools/` の道具も同じ根で作る）。ただし `common/host_eval.hpp` とその傘の下の 5 本（`oz_input.hpp`・`oz_reference.hpp`・
  `oz_accuracy.hpp`・`oz_timing.hpp`・`oz_dump.hpp`）・`common/telemetry.hpp` の中の取り込みは同じディレクトリからの相対
  （`"oz_input.hpp"`・`"telemetry.hpp"`・`"oz_env.hpp"` など）で書く（DGEMM 参照の照合 `../scripts/lib/peer_logs.py` が，測った rev どうしで
  `host_eval.hpp` からの取り込みの連鎖の定義・前処理指令を比べ，相対の取り込みを「取り込み元のディレクトリ → src/」の順にたどるため．
  これらはつまみの表 `oz_knobs.hpp` を取り込まない：表のつまみを 1 つ足しただけで DGEMM 参照を測り直すことにならないよう）。
- デバイスカーネルのパス（`CreateKernel` に渡す文字列）は，`common/oz_paths.hpp` の置き場所の定数（`OZ_COMMON_KDIR`・`OZ1_KDIR`・
  `OZ2_KDIR` など）にファイル名を続けて書く。基点はビルドした src の絶対パス（`Makefile` が `OZ_SRC_DIR` として渡す．コンテナの中では
  `/home/user/tt/src/`）なので，バイナリは作った木のカーネルを読む。
- カーネルどうしの `#include` は，取り込む側のファイルの置き場所からの相対で書く（`"../../common/kernels/sfpu_tf.h"` など．JIT は
  カーネルのファイルを絶対パスで取り込み，その置き場所も取り込みの根に加える）。
- ホストとカーネルの間で受け渡す引数（実行時引数・コンパイル時引数）の並びと CB 番号は，カーネルの置き場所に置いた
  デバイス API に依存しない表（`<cstdint>` だけ）で名前を付け，ホストは表の名前で添字を指定して詰め，カーネルは名前で読む
  （番号だけで対応づけると片側に 1 つ足したときに以降が黙ってずれる）。表は次のとおり：
  `common/kernels/mcast_args.h`（GEMM 段のデータ移動）・`gemm_compute_args.h`（GEMM 段の計算カーネルと CB）・
  `slice_f64_args.h`（両スキームの FP64 のリーダとスライサの入力．ホストが詰める関数は `common/slice_f64_host.hpp`）・
  `scale_args.h`（スケール段），`ozaki1/kernels/oz1_args.h`（scheme I のスライス・再構成・累積），
  `ozaki2/kernels/recon_args.h`（scheme II の再構成）・`oz2_args.h`（同，スライス・合成），`testkit/kernels/probe_args.h`（探針と検査）．
  片側だけ直すと黙って誤る値（scheme II の法の本数の上限・合成の係数の枠・桁の作り方の番号）は `ozaki2/kernels/oz2_limits.h`，
  FP64・128 bit 固定小数のタイルの大きさは `common/kernels/f64_tf.h` に置き，両側が取り込む。
- 形式ごとの値の表は tt-metal に依存しない形に分けてあり（`common/oz_format_host.h`・`ozaki1/ozaki1_param.h`・
  `ozaki2/ozaki2_split.h` の `OZ2_ROUTES`），ホスト単体の道具（`../tools/`）は本番と同じ表を取り込む（写しを持たない）。
- triple-float と法の剰余の手続き（`common/kernels/sfpu_tf_proc.h`）は，タイル演算のラッパの型を引数に取るテンプレートにしてあり，
  カーネルはデバイスのラッパ（`sfpu_tf.h` の `SfpuDev`），ホストの模擬（`checks/sfpu_tf_emu.hpp` の `Dst`）は同じものを組む。模擬が
  手続きを書き写すと，カーネルの手順を変えたときに検査が古い手順のまま通るためである（模擬と実機の照合は 1 演算の模型だけを見ればよい）。

## ターゲット

組（`Makefile` の「組（ターゲットの位置づけ）」）ごとに，どのスクリプトが回し，論文・スライドの何に使うかを書く。
経路ごとの p・法の本数は `../scripts/routes.tsv`，16 経路の性能と到達精度は `../docs/results.md` にある。

### DRIVERS（本番：尾崎の 16 経路．論文・スライドの表と図の値）

`remeasure_all.sh` の cool・acc・phi・pass・abl・ratio・split・ext・phi_nowin・whole などの段が，`measure/performance.sh`・`measure/accuracy.sh`・
`measure/exponent_sweep.sh` などで回す。

| ターゲット | ソース | 備考 |
|--------|--------|-------|
| `ozaki1_bf16` / `ozaki1_fp16` / `ozaki1_fp32` / `ozaki1_bfp8` / `ozaki1_fp8` / `ozaki1_int8` / `ozaki1_bfp4` / `ozaki1_bfp2` | `ozaki1/ozaki1_main.cpp` | 尾崎 scheme I．`-DOZ_FMT_<形式>` の 1 マクロで出し分け（scheme II と同じマクロ．形式の性質は `common/oz_format_host.h`・`oz_format_table.hpp` の `OzFormat`，scheme I の値の表は `ozaki1/ozaki1_param.h` の `Oz1Param`）．`fp32` は FP32 で格納し行列エンジンには TF32 として渡る経路で，論文では TF32 と呼ぶ．`fp16` はスライスを binary16 のビット列として書く（`sfpu_tf.h` の `HalfBits`） |
| `ozaki2_int8` / `ozaki2_fp8` / `ozaki2_bfp8` / `ozaki2_bf16` / `ozaki2_fp16` / `ozaki2_fp32` / `ozaki2_bfp4` / `ozaki2_bfp2` | `ozaki2/ozaki2_main.cpp` | 尾崎 scheme II（CRT）．8 形式すべてがこの 1 本から出る．scheme I と同じ `-DOZ_FMT_<形式>` で出し分け（`ozaki2_int8` も `-DOZ_FMT_INT8` を明示．INT8 以外は剰余の桁分割）．scheme II の経路ごとの値（法の上限・桁の上限・桁数・法の本数の既定・法の選び方）は `ozaki2/ozaki2_split.h` の経路の表 `OZ2_ROUTES`（形式と `Oz2Route`（`OZ2_ROUTE_*`）の組．`oz2_param` が形式から引き，`ozaki2_main.cpp` の `Oz2Param` はその別名）．`ozaki2_bfp2` だけは桁が ±1 しか取れないので平衡三進（`nrdig=5`，法は 3^5=243 まで．桁数は法ごとに 3〜5 の最小）を使う．BF16・FP16・BFP8_b・BFP4_b・BFP2_b は法の集合を段の時間のモデルで選ぶ（`ozaki2/ozaki2_awe.h`，`../docs/awe_moduli.md`．BF16・FP16・BFP8_b は 1 桁の小さい法も混ぜる）．TF32・FP8・INT8 は従来の表の選び方 |

### BASELINES（基準実装．尾崎の経路ではない論文の比較対象）

入口（位置引数の読み方・デバイスを開く前の検査の並び）・最後の行・CONFIG 行の組み方は尾崎の 2 つのドライバと同じ枠
（`common/oz_driver.hpp`・`common/oz_logline.hpp`）を使う。warmup は 0 も受ける（行列積 1 本を流す `run_mcast_gemm` が計時の前に
必ず 1 回流すので JIT は計時に入らない）。

| ターゲット | ソース | 備考 |
|--------|--------|-------|
| `gemm_bf16`    | `baseline/gemm_bf16.cpp`    | BF16 行列積（BF16 累積と FP32 累積の 2 構成．FP32 累積の `measure/performance.sh` の表示名 BF16x2/PP・ログ名 `cool_bf16x2_*` は前からの名前で，2 成分の分割ではない）。素の行列積性能の基準（論文の値を出す）。時間はデバイス上の行列積の実行時間だけ（CONFIG の `timing_scope=gemm_only`．パネルごとに別々にウォームアップして測った和）。構成は tt-metal の行列積の例の判定で選び L1 の予算を見ないので，`OZ_L1_BUDGET_KB` は受け付けない |
| `gemm_bf16ext` | `baseline/gemm_bf16ext.cpp` | 無誤差分割（BF16 の 2・3 成分の BF16x2・BF16x3．`OZ_EXT_FMT=tf32` で 3xTF32．`common/kernels/bmm_l1acc.cpp` を使う．論文の値を出す）。時間は部分積の行列積のデバイス実行時間の和で，ホストでの成分への分割・タイル化・パネルの切り出し・部分積の足し込みは含まない（`timing_scope=gemm_only`）。FP32 で有限に表せない入力はデバイスを開く前に止める |

### PROBES（探針．論文・スライドの地の文の値）

`remeasure_all.sh` の probes の段で `../scripts/diagnose/probes.sh` が回し（項目の名前は括弧の中），`../scripts/rules/probe_checks.py` が
論文・スライドと照合する。どれも単一コアで実機の性質を測り，最後に機械が読む `PROBE,<名前>,…` の行を出す。

| ターゲット | ソース | 備考 |
|--------|--------|-------|
| `fp8_probe`    | `probes/fp8_probe.cpp`    | 形式・fidelity ごとの無誤差整数積の境界，乱数入力の到達精度，出力形式を変えたパッカの丸め（`probe_fp8_digit`・`probe_fp8_rand`）。計算カーネルは tt-metal 付属の単一コア版 |
| `dst_probe`    | `probes/dst_probe.cpp`    | DST レジスタが FP32 で何枚使えるか（`probe_dst`．`probes/kernels/dst_probe.cpp`） |
| `fp32acc_probe` | `probes/fp32acc_probe.cpp` | Tenstorrent 提供の FP32 累算サンプルと同じ単一コア構成で、累算経路が保持する幅（`probe_fp32acc`．`../docs/fp32acc_check.md`） |
| `fma_probe`    | `probes/fma_probe.cpp`    | SFPU の積和（SFPMAD）が 48 bit の積のうち何ビットを保つか（`probe_fma`．判定は掃引の結果から出す．`probes/kernels/fma_probe.cpp`） |
| `tile_rate_probe` | `probes/tile_rate_probe.cpp` | 1 コア（格子 1x1）の L1 に置いたブロック（出力 8x8・K 4・サブブロック 2x2）の行列積を NoC 転送なしに繰り返し、タイル積 1 回のサイクル数を形式 × MathFidelity ×（積／演算のみ／展開のみ／mmpack）で測る（`probe_tile_*`・`probe_tilecfg_*`．冷ましてから回す）。mmpack はサブブロックごとのパックとパッカの L1 累積まで含め，出力ブロック・サブブロック・bw の引数で本番のコアあたりの構成（II は 11x10・1x2，I は 8x8・2x2）を 1 コアで再現する．論文の表 tab:fmt とスライドの単コアのタイル積のサイクル数は既定の構成（出力 8x8・K 4・サブブロック 2x2）の mm（出力をパックしない．`probe_tile_<形式>_<fidelity>_mm`）で，GEMM 段の効率の節の 1 コアの値は本番の構成を mmpack で測ったもの（`probe_tilecfg_*`）。`probes/kernels/tile_rate_probe.cpp`・`probes/kernels/tile_rate_fill.cpp`。値はデバイスの壁時計のサイクル数で、起動 3 回の平均・最小・最大 |

### CHECKS（検査．正しさを確かめる）

| ターゲット | ソース | 備考 |
|--------|--------|-------|
| `gemm_int8`    | `checks/gemm_int8.cpp`    | INT8 行列積。**論文の値は出さない**：INT8 の行列積が無誤差であることの検査（`../scripts/selftest.sh` の INT8 GEMM の 2 項目と `../scripts/tests/check_probes.sh` が回す．8197a0b までは `baseline/` の基準実装の組に置いていた）。判定は復号した整数値と int64 の参照の一致で，出力語のビット列の一致ではない。時間は行列積のデバイス実行時間だけで，mode 0 のホストでの K ブロックの切り出し・転送・int64 への足し込みは含まない（`timing_scope=gemm_only`）。入口・CONFIG 行の組み方は基準実装と同じ |
| `gemm_exact_check` | `checks/gemm_exact_check.cpp` | 尾崎の両スキームと同じ GEMM 段（`spill_gemm_spec`＋`build_mcast_gemm`）と BF16 extended の計算カーネル（`l1acc_gemm_spec`）を敵対的な整数入力で回し，全要素を int64 の参照と整数値で比べる（本番の 8 形式，素・対ループ・融合・付け替え・l1acc，同じプログラムの連続実行）。通しの selftest の段で回り（`diagnose/probes.sh` の `probe_gemm_exact` が `../scripts/diagnose/gemm_exact.sh` を呼ぶ），論文の b_acc の出どころ |
| `slice_split_check`（`_int8`） | `checks/slice_split_check.cpp` | 尾崎 I のスライス段の割る処理を 2 つの RISC-V に分けた経路と分けない経路のスライスを，入力を替えながら全語で比べる受け渡しの検査（`../scripts/tests/check_slice_split.sh`．手で回す） |
| `scale_check` | `checks/scale_check.cpp` | スケール段（`scale_build`・`expdelta_build`）の最大値タイルの有効 32 要素と指数差分タイルの全要素をホストの参照と比べる（コアあたり 2〜5 枚で 2 つの RISC-V の分担を踏む．`../scripts/tests/check_scale.sh`．手で回す） |
| `tfmul_probe` | `checks/tfmul_probe.cpp` | SFPU の模擬（`checks/sfpu_tf_emu.hpp`）と実機の照合．本番の関数（タイル演算のラッパ・`Rnd`・`tfMulRadix`・`tfMulR`・`tfAddS<3>`・`tf_load_f64`）を実機で呼んでビット単位で比べる（`checks/kernels/tfmul_probe.cpp`．`../scripts/selftest.sh` が毎回回す．ホストの検査が模擬の上で出す結論の前提） |

### HOST_TOOLS・CHECK_TOOLS（ホストだけで動く道具．tt-metal にリンクしない）

ソースは `../tools/<名前>.cpp`。ほかのターゲットと同じく `build/` に置くのでビルドの記録（`BUILD_STAMP`）に載り，ログの出どころを
同じ仕組みで照合できる。フラグは `Makefile` の `HOST_CXXFLAGS` 1 つ（別のイメージで作る `tools/dgemmcheck.cpp` も `make print-host-flags`
で同じフラグを受け取る．`../scripts/measure/cpu_dgemm_reference.sh`）。`tfmul_check`・`tfconv` は実行時に `../scripts/routes.tsv` を読むので，その結果の
出どころは `BUILD_STAMP` の sha256 だけでは決まらない（`routes.tsv` は `scripts/` にあり，出どころの rev で決まる）。

| 組 | ターゲット | 回すもの |
|---|---|---|
| HOST_TOOLS（論文・スライドの値） | `dot2check` / `awe_plan` / `splitsim` / `reconsim` / `twosum_probe` / `tfconv` | `../scripts/diagnose/probes.sh`（`twosum_probe`・`tfconv` は `selftest.sh` も） |
| CHECK_TOOLS（検査．`modround_check` はスライドの総当たりの件数も出す） | `modround_check` / `tfmul_check` / `crt_fold_check` / `aligned_accum` / `knobs_check` / `routes_check` | `../scripts/selftest.sh` の「host-side exhaustive checks」（`routes_check` は経路の表と本番の構成の一致，`knobs_check` はつまみの表と `../docs/env_vars.md`・ソースの読み取りの突き合わせ） |
| | `cmp_dump` | `../scripts/tests/check_nsmall.sh` |

`OZ_NOVERIFY` を立てると、尾崎の各経路と `gemm_bf16`・`gemm_bf16ext`・`gemm_int8` は参照解の計算と
精度（一致）の判定を省く（性能測定用。`common/oz_reference.hpp` の `verify_enabled`）。`gemm_bf16` は第 7 引数 `verify` でも切り替えられ，
`OZ_NOVERIFY` が立っていればそちらが優先する。検証を省いた実行の最後の行は `Test Passed` ではなく `Test Skipped`
（終了コードは 0）で，RESULT 行の精度の欄（`gemm_bf16` には無い）は `nan`，`OZ_DUMP_C` とは併用できない（デバイスを開く前に止まる）。`OZ_FIDELITY` は
`LoFi|HiFi2|HiFi3|HiFi4` 以外の値を受け付けずに止まる。尾崎の 16 経路の既定の MathFidelity は
どちらのスキームも同じ規則（`common/oz_format.hpp` の `oz_fidelity_rule`：INT8 は HiFi4，それ以外は
桁の上限が 32 以下なら LoFi，超えれば HiFi4）でコードが決め，Makefile では渡さない。

ソース中のコメントは論文の節を番号で引かない：投稿版 `paper/pdp2027_paper_jp.tex`（英訳 `pdp2027_paper_en.tex` も同じ `\label` を持つ）の
`\label` 名に節の名前を添えて書く（「論文の ssec:exact（無誤差条件）」の形．原稿を改訂して番号がずれても名前で辿れる）。投稿版に
載っていない内容（削った実験など）は論文を引かず，docs の場所を引く。

尾崎の両スキームの論文の値（全体の時間）は人向けの `DRAM->DRAM` 行と CONFIG の `d2d_ms`（デバイス上の段の和．スキーム I は段ごとの
`slice_ms`・`gemm_ms`・`recon_ms` も CONFIG にある）である。スキーム I の RESULT の時間欄と `Throughput:` 行の先頭の値は，1 回の構築・JIT と
読み戻しも含む tot（CONFIG の `time_in_result=tot_incl_build_and_readback`）で，基準実装の `End-to-end`（転送込みの 1 回の端から端）とも別物。

## 共通部分とターゲット固有部分

- **全ターゲット共通**：
  - `common/oz_env.hpp`：つまみと位置引数の読み方の 1 系統（有無・範囲つきの整数と実数・列挙．読めない値は理由を出して終了コード 2．
    tt-metal に依存しない）。
  - `common/oz_knobs.hpp`：つまみの表（名前・分類（`../docs/env_vars.md` の A〜F）・読むプログラム・範囲と既定・CONFIG 行の `knobs`・
    `diag` への記録）．スキームをまたぐ拒否・削除したつまみの拒否・入口の検査 `preflight` をこの表から作る（`../tools/knobs_check.cpp` が
    docs とソースとの一致を確かめる）。
  - `common/host_eval.hpp`：測定・評価の部品の傘．入力の生成 `oz_input.hpp`・参照 GEMM `oz_reference.hpp`・精度の指標 `oz_accuracy.hpp`・
    計時の手順とテレメトリの行 `oz_timing.hpp`（`telemetry.hpp` を取り込むのはここだけ）・結果の書き出し `oz_dump.hpp` を取り込み，
    部品をまたぐ入口の検査と CONFIG 行の欄の書き方を持つ。
  - `common/ozaki_gemm_host.hpp`：タイル並べ替え，符号-絶対値の符号化，DRAM の空きの照会，CB の大きさの数え方と行列積の構成探索，
    マルチキャストの引数，出力パネルの切り出し。
  - `common/mcast_gemm.hpp`：マルチキャスト行列積のプログラム構築 `build_mcast_gemm`．`gemm_bf16`・`gemm_bf16ext`・`gemm_int8`・
    `gemm_exact_check`・scheme I の `gemm_build`・scheme II の `run_partial_gemm` が共有し，違い（CB の形式と構成・ライタ・計算カーネル・
    引数の末尾・GEMM 段の探針 `gemm_probe.hpp`）だけを `McastGemmSpec` で渡す．共有の計算カーネル `bmm_int32_spill.cpp` を使うもの
    （`gemm_bf16`・`gemm_bf16ext` 以外）は CB と計算カーネルの仕様も `spill_gemm_spec` で，BF16 extended の計算カーネル
    `common/kernels/bmm_l1acc.cpp` を使う `gemm_bf16ext` と `gemm_exact_check`（l1acc の検査）はその仕様を `l1acc_gemm_spec` で共有する．
    `gemm_bf16` だけは計算カーネルに tt-metal 付属の `bmm_large_block_zm.cpp`（BF16 の c_24 へ退避して読み戻す）を使い続ける．
    基準実装の行列積 1 本（器の確保から計時と読み戻しまで）は `run_mcast_gemm`。
  - データ移動カーネル `common/kernels/mcast_split_dm.cpp`：A と B を 2 つの RISC に分け，行・列の両端の 2 コアから半分ずつ配る 1 通り
    （引数の並びは `common/kernels/mcast_args.h` の表をホストと共有する．tt-metal 付属の単一 RISC リーダ 4 種を使う従来の構成は採用後に
    削除した．`../docs/gemm_opt_results.md`）．尾崎の経路の K ブロック間の累積は `bmm_int32_spill.cpp` のパッカ L1 累積。
  - `common/oz_fidelity.hpp`：MathFidelity の上書き `OZ_FIDELITY`・表示名・パス数・理論ピーク `oz_peak_tops`（尾崎 I と基準実装が同じ式で
    求める．探針が位置引数の名前・番号から引くのも同じ表）。
  - 小さな部品：`common/oz_bits.h`（float・double・binary16 のビット列と最終スケールの 64 bit 仮数．ホスト単体の道具・模擬と共有）、
    `common/oz_normest.hpp`（自動決定 `OZ_FP64`・`OZ2_FP64` が共有する ||C||_F の見積り）、`common/oz_dram_margin.h`（確保の前の
    DRAM の見積りの余裕 `OZ_DRAM_MARGIN_BYTES`．両スキームで 1 つ）、`common/bfp2_decl.hpp`（BFP2_b のタイルの詰め・解きの宣言．
    使う tt-metal のビルドに `bfloat2.hpp` が無いため）。
  - デバイス側の小さな部品：`common/kernels/tile_nf.h`（タイルの中の要素の通し番号．カーネルとホストの並べ替えが共有）、
    `l1_order.h`（データ移動 RISC-V が L1 へ書いた値が届くまで待ってから相手に知らせる）、`slice_f64_split.h`（FP64 を triple-float へ
    割る仕事を 2 つのデータ移動 RISC-V で半分ずつ行う郵便受け．尾崎 I のスライス段と `slice_split_check`）。
  - `common/kernels/sfpu_tf.h`：SFPU のタイル演算ラッパ，TwoSum，triple-float の乗算・加算．尾崎の両スキームの計算カーネル（スライサ・
    再構成・合成）と SFPU を使う探針・検査が取り込む（GEMM 段の `bmm_int32_spill.cpp`・`bmm_l1acc.cpp` と `tile_rate_probe` は使わない）．
    手続きの本体は `common/kernels/sfpu_tf_proc.h` のテンプレートで，ホストの模擬 `checks/sfpu_tf_emu.hpp` も同じものを組む。
- **尾崎の 2 つのドライバの枠**（基準実装と `gemm_int8` も同じ枠を使う）：`common/oz_driver.hpp`（入口の検査の並び，位置引数の
  M N K・warmup・iters の読み方，CONFIG 行の共通の欄，WHOLE-PIPELINE 行，最後の判定の行）、`common/oz_logline.hpp`（CONFIG 行を欄の名前と値で足していく組み立て器．
  欄の重複と値の `,`・`=` を組み立ての誤りとして止める）、`common/oz_whole.hpp`（`OZ_TIME_WHOLE` の検査：番兵で埋めて 1 回流す
  手順と `WHOLE-CHECK` 行．投入の数え方と埋める器は各スキーム）。尾崎の 2 つの位置引数の warmup・iters はどちらも 1 以上（尾崎 I の GEMM の
  プログラムは最初の投入で JIT を済ませるので，0 回の温めは JIT を計時に入れる）．基準実装の warmup は 0 も受ける（上の BASELINES）。
- **探針と検査だけが共有する部分**（`testkit/`．本番のドライバは取り込まない）：`testkit/probe_common.hpp`（形式の名前から性質を引く
  `probe_format`，機械が読む結果の行 `PROBE,<名前>,<欄>=<値>,…`，単一コアの行列積，探針のタイル運びのカーネルの置き方）と
  `testkit/kernels/probe_args.h`（探針と検査のカーネルの引数の表）・`testkit/kernels/probe_tile_reader.cpp`・`testkit/kernels/probe_tile_writer.cpp`
  （`dst_probe` と `tfmul_probe` が使う）。どちらの組からも同じように取り込めるよう，探針（`probes/`）にも検査（`checks/`）にも置かない。
- **尾崎 scheme I** の段は `ozaki1/ozaki1_common.hpp`（次の 4 本とスケール段 `common/ozaki_scale.hpp` を取り込むだけのヘッダ）：
  `ozaki1_format.hpp`（形式ごとの値 `OZ1_PARAM`（表は tt-metal に依存しない `ozaki1_param.h`）、桁の上限・fidelity、無誤差条件）、
  `ozaki1_gemm.hpp`（L1 の見積り・構成探索、`gemm_build`/`gemm_run`）、
  `ozaki1_recon.hpp`（再構成 `recon_*`）、
  `ozaki1_slice.hpp`（`slice_build`/`slice_run`）。その上に載るもの（`ozaki1_main.cpp` が取り込む）：計画 `ozaki1_plan.hpp`
  （`RunCfg`・計画のつまみ `Oz1Knobs`・段の組み方を変える診断と測定モードのつまみ `Oz1Diag`（どちらも main の冒頭で 1 回だけ読み，
  段の奥では読まない．表示だけの `OZ_DBG` を除く）・`plan_issue`・`plan_panels`・`recon_scale`．デバイスに触らない）、`OZ_FP64` のモデル
  `ozaki1_autop.hpp`、段の資源と投入の手順 `ozaki1_stages.hpp`（段ごとの計時と全工程の流し直しが同じ手順で投入する）、
  `OZ_NO_FUSE_ACC` の別プログラムの累積 `ozaki1_accum.hpp`、`OZ_TIME_WHOLE` の流し直しと検査 `ozaki1_whole.hpp`。
  `ozaki1_main.cpp` は段を並べて計時し，結果を表示する。scheme I だけが使うデバイスカーネルは `ozaki1/kernels/`：
  スライサ `slice_tf_num.cpp`（INT8 以外の 7 形式．桁の値をそのまま出す）・`slice_tf_sm8.cpp`（INT8．符号-絶対値の UInt8）と
  その共通の本体 `slice_tf_body.h`，ライタ `slice_writer.cpp`，再構成のリーダ `recon_reader_f32in.cpp`（FP32 の部分積）・
  `recon_reader_i32in.cpp`（Int32 の部分和．INT8 と遅延再構成）と共通の読み込み `recon_read.h`，計算カーネル `recon_tf_f32in.cpp`・
  `recon_tf_i32in.cpp` と共通の出口 `recon_pack.h`，FP64 のライタ `recon_f64_writer.cpp`，Int32 累積 `accum_i32.cpp`・
  `accum_i32_reader.cpp`・`accum_i32_writer.cpp`（`OZ_NO_FUSE_ACC` だけ），引数の表 `oz1_args.h`。
- **scheme I と II の共通部分**：ホスト側は `common/ozaki_scale.hpp`（オペランドを
  FP64 のまま送る段と、行・列ごとの最大値をデバイスで求める段）、デバイス側は
  `common/kernels/` の `scale_f64_reader.cpp`（最大値を FP64 のビット列のまま 2 つのデータ移動 RISC-V で
  求めて書く）+ `expdelta.cpp`（指数へ足す差分）、
  `slice_f64_reader.cpp`（FP64 を読んで triple-float に割る）、
  `slice_tf_core.h`（スライサの CB 配置と残差の持ち回り）、`scale_bits.h`（スケール段の BF16 化と指数．`OZ_FP64` のモデルも同じ関数を使う）、
  `f64_tf.h`（FP64 ↔ triple-float のビット手続き。入口の分割と出口の丸めを両スキームが
  共有する。正しさは `tools/tfconv.cpp` が検査する）。
- **尾崎 scheme II** のホスト側は `ozaki2/` にある：`ozaki2_main.cpp`（段を組んで計時し，結果を表示する）、
  `ozaki2_format.hpp`（形式と fidelity），`ozaki2_plan.hpp`（つまみ・法・DRAM・パネルの計画），
  `ozaki2_autop.hpp`（自動精度のモデル），`ozaki2_stages.hpp`（スライス・GEMM・再構成の組み立て）、
  `ozaki2_split.h`（経路の表 `OZ2_ROUTES` と法 1 本の運び方．tt-metal に依存しない）、`ozaki2_moduli.h`（法の選択）、
  `ozaki2_awe.h`（段の時間のモデルで法の集合を選ぶ AWE）と，その係数の表 `ozaki2_awe_coef.inc`（生成物：`../tools/awe_plan.cpp` の
  `emit-coef` が書く．手で直さない）、`ozaki2_whole.hpp`（【診断】`OZ_TIME_WHOLE` の記録・流し直し・検査）、`ozaki2_concat_check.hpp`
  （【診断】`OZ2_CHECK_CONCAT` の連結 GEMM・剰余・CRT 復元の整数の参照との比べ．`../scripts/tests/check_concat.sh`）。
  デバイスカーネルは `ozaki2/kernels/` にある
  （`oz2_digits.h` が正規化・整数化・2^20 進 3〜5 桁への分解と法剰余を担い、
  `modslice_int8.cpp` と `modslice_digit.cpp` がそれを使う。スライスのライタは `modslice_writer.cpp`（法ごとの剰余・桁の行列を
  並べて書く）。桁分割の 7 経路（INT8 以外）は
  `oz2_compose.cpp` が桁積を合成して法剰余まで取る（入力は平面リーダ `oz2_plane_reader.cpp`）。再構成は
  `recon_reader.cpp`・`recon_crt.cpp`（法ごとの手続きは `recon_crt_proc.h`）・`recon_writer.cpp` で，FP64 の組み立ては
  リーダとライタが要素を半分ずつ `recon_elem.h` の同じ手続きで行う．平面を並べたバッファから n 枚を読む部品 `oz2_plane_fetch.h` は
  再構成と合成のリーダが共有する）。FP8 経路は e4m3 が無誤差に表せる疎な整数
  （|x| <= 16 か 32 以下の偶数）を桁に使い，奇数 q <= 49 の平方数の法（2401 まで）の桁を `oz2_sparse_fix.h`
  （本リポジトリで唯一の SFPI の関数）で手直しする（`../docs/fp8_sparse_digits.md`）。

## 到達精度と性能

16 経路の到達精度と性能の表（論文の tab:flops と同じ値）は `../docs/results.md` に置く（測るたびに `../scripts/verify/check_paper_numbers.py --fix` が書き戻すので，コードの木の外に置いている）。

## 対応する形状

M, N, K はタイル幅 32 の倍数でなければならない。以下，タイル数を Mt=M/32・Nt=N/32，コア格子の縦横を gy=10・gx=11 と書く。
マルチキャスト経路はコア格子の縦横に 2 コア以上を要求し，1 辺に並べられるのは格子の辺の数までなので，構成探索
（`common/ozaki_gemm_host.hpp` の `choose_cfg`）は行も列も割り切りを要求する：行のタイル数が 2 以上 gy 以下の約数を，
列のタイル数が 2 以上 gx 以下の約数を持ち，その分割でコアあたりのブロックが L1 に収まること。経路ごとの違いは
出力をパネルに切るかと，行方向の端数を許すかである（2026-09-27 のレビューで，行と列で条件が違うことを書き分けた）。

- **尾崎 scheme I**（`ozaki1/ozaki1_gemm.hpp`）：列方向は割り切りを要求し，Nt/pn（pn は 1・2・4・8 のうち Nt を割るもの）が
  2 以上 gx 以下の約数を持てばよい。行方向は端数のパネル行を許すので，Mt を割る pm と 2 <= cr <= min(gy, Mt/pm) で
  (Mt/pm) mod cr ≠ 1 となる cr があればよく，Mt >= 2 ならほぼ常に満たす（Mt <= gy なら cr = Mt/pm で取れる．例：Mt=13 は
  pm=1・cr=10 と残り 3 行）。外れるのは Mt が 3..gy のすべてで割って余り 1 になる場合（gy=10 なら Mt ≡ 1 mod 2520）だけである。
- **尾崎 scheme II**（`ozaki2/ozaki2_plan.hpp` の `plan_gemm_panels`）：出力のパネルの辺を Mt・Nt の 120 以下の最大の約数に取り，
  そのパネルの辺について行も列も割り切りを要求する（パネルの辺は探索しない．たとえば M=5056 は Mt=158=2×79 のパネルの辺が
  79 になり，縦 10 コアに分けられず止まる）。
- **`gemm_bf16`・`gemm_bf16ext`**（`common/mcast_gemm.hpp` の `choose_panels`）：パネルの辺（`gemm_bf16` は 120，`gemm_bf16ext` は
  80 以下）を Mt・Nt の約数の組から面積の大きい順に探し，構成が取れる（`gemm_bf16ext` は実際の CB が L1 の予算に収まる）最初の
  組を使う。先頭の候補は従来の「上限以下の最大の約数」の組なので従来通っていた形は同じパネルになり，M=5056 のような形も
  辺 2 のパネルで通る（2026-09-27 から．以前は最大の約数だけで選んで止まっていた）。
- **`gemm_int8`**：パネルに切らず，Mt・Nt そのものに上の割り切りを要求する。

たとえば N=416（Nt=13 は 11 を超える素数）はどの経路でも構成が見つからない。Nt=11（素数）は横 11 コアに並ぶので実行できる。
条件を満たさない場合は誤答ではなく、構成選択に失敗して止まる。
論文の測定サイズ（3520, 7040, 14080 の立方）はいずれも条件を満たす。

入力の値の範囲：スケール段は行（列）ごとの最大絶対値を FP64 のビット列のまま求めて BF16 にする。最大値が BF16 の範囲を超えた
行（列）は印 0x7F80 と旗のビット 0 を，非零の最大絶対値が 2^-126 を下回る行（列）は印 0xFF80 と旗の
ビット 1 を立て，C を読んだ後にホストが理由を表示して止める（両スキーム．`common/ozaki_scale.hpp` の `expdelta_check`）。
下端は 2026-09-27 から（以前は最大値が 0 になり，その行が黙って零として出ていた．A=2^-127・B=2^127 で 64 のはずが 0．
`../scripts/tests/check_scale.sh`・`selftest.sh` が確かめる）。全零の行（列）は範囲外ではない。
上端は経路で違い，BF16 化を切り上げる経路（尾崎 I の INT8・BFP2_b とスキーム II）は 2^128−2^120 を超えると，切り捨てる経路は
2^128 以上で止まる（エラー文に境目が出る）。
尾崎 I の BFP2_b は，行（列）の最大絶対値がちょうど 2 のべきのとき正規化の 2 のべきを 1 段上げる（目標 1 は桁 1 個で表せず，
最大値の要素の展開が毎段 桁 1・残差 1 のまま終わらないため．以前は全要素 1・±1・`OZ_EXPDIFF` で 51.0 bit．`ozaki1_main.cpp` の
`choose_radix_norm` の `tau_above`，`common/kernels/scale_f64_reader.cpp` の CEIL=2）。一様乱数の入力の結果がビット単位で変わらないことは，
16 経路の 7040³・14080³ の `CHECKSUM` 行の `chash`（結果 C のビット列のハッシュ）で確かめた（4 回目の点検の直し cd0bfdf の版．
32 構成すべて一致）。追跡しているログで辿れる形にすると：点検の前の測定（ca619f3 の `accuracy_logs/oz*_{7040,14080}.log`．09df796 で
測ったもの）と後の測定（8da4ac3 の同じログ．042dc00）で，p を変えなかった 28 構成の `chash` は同じである（`git show <rev>:<ログ>` の
CHECKSUM 行で比べられる）。残りの 4 構成は 5 回目の点検で p を変えた（b95ff4a．BFP4_b 15→16，BFP2_b 56→59）ので比べられず，この直しが効く
BFP2_b の p=56 の一致は点検の手元のログ（追跡していない）でだけ確かめた：直した版の `chash` は 7040³ が 7980691b19a886ee，
14080³ が a3d38f380c91782c で，ca619f3 の `oz1bfp2_{7040,14080}.log` と同じ。2 のべきの最大値の入力が 53 bit に届くことは `../scripts/tests/check_adversarial.sh` の
2 のべきの最大値の表が確かめる（別の検査で，ビット単位の一致は見ない）。最大値が 2 のべきのすぐ下（正規化後 1−δ）の行・列は，
スケールは上げずに先頭に約 log2(1/δ) 個の桁 1 が並ぶ展開になり，p=56 では 1−2^-53 の定数の入力で 51.4 bit に落ちたので，
既定の p を 59 に上げた（2026-10-21 の 5 回目の点検．`../scripts/diagnose/select_slice_count.sh` の規則に構造のある入力を足した）。

形式とサイズを振った全掃引は `../scripts/study/measure_paper.sh` で実行する。
リファクタリング後の動作確認は `../scripts/selftest.sh`。
