#!/bin/bash
# diagnose/select_slice_count.sh -- スキーム I のスライス数 p が，規則 (i)・(ii) をともに満たす最小値であることを採用値の前後で確かめ直す
# ---------------------------------------------------------------------------
#     ./scripts/run.sh "bash /home/user/tt/scripts/diagnose/select_slice_count.sh"
# 論文 7.1 節と付録「基数を 2d に採る理由」の p の決め方の根拠．規則は routes.tsv の p 列の注記と同じで，
# M=N=512・K=512 と 16384 の両方で見る．基準と入力は routes.sh の OZ_RULE_I_BITS・OZ_STRUCT_BITS・OZ_STRUCT_CONSTS・OZ_STRUCT_SIGNS
# （値の正本は lib/constants.tsv）．
#   (i)  一様乱数 [-1,1) の入力（既定の入力）で到達精度が OZ_RULE_I_BITS（53.9 bit）を上回る．54.0 にしないのは
#        BFP2_b の p=56 が 53.96 bit（表示は 54.0）であったためで，54.0 を厳密に要求すると 1 段多く選ぶ．
#   (ii) 構造のある入力（OZ_CONST の定数の行列．OZ_STRUCT_CONSTS を OZ_CONST_SIGN 無しと col で）のすべてで
#        OZ_STRUCT_BITS（53.05 bit．STRUCT_BITS で上書きできる）以上．53 ちょうどでは余裕が無いので，(i) が 53.9 で余裕を
#        取るのと同じく 53.05 にした（53 なら I-BFP4 が 15，I-BFP2 が 58 を選ぶ．routes.tsv の 16・59 はそれを落とした値）．
#        一様乱数では行・列の最大値が 2 のべきのすぐ下に来ることも誤差の符号がそろうことも無いので，(i) だけでは
#        見えない入力を補う（BFP2_b の 1-2^-53 は先頭に 53 個の桁 1 が並び，定数の行列は誤差が K に比例して足される）．
# 判定はどちらもプログラム自身の検証に任せる（OZ_MIN_BITS=<基準> を渡し，丸めずに比べた結果を終了コード・VERIFY 行・
# Test Passed で読む．lib/common.sh の accuracy_verdict）．RESULT 行の精度は小数第 2 位に丸めた表示で，基準の近くでは
# 判定と食い違うので表示にだけ使う（(i) の境は「以上」で判定する．ちょうど 53.9 は起きないとみなす）．
# 起動できない・クラッシュした・RESULT を出した後に落ちた回（error）は値を NA にして，終了コードを非零にする．
# 入口の検査がその構成を組めないとして止めた回（reject．INT8 の K=16384 で p が大きい構成など）は値を NA にして
# (i) を満たさないとするだけで，失敗には数えない（組めない構成は選べない，という規則どおりの結果なので）．
# 回す q は routes.tsv の採用値 p の前後 2 段（p-2..p+2）だけで，表の全部の p を探すのではない．(i) は全部の q で測り，
# (ii) は (i) を両方の K で満たす q だけを下から測って，両方の K で (ii) も満たした q より上は測らない．
# したがって PICK 行の 2 つの「最小」は，この範囲での最小である（p-2 でも通れば，それより下は確かめていない）．
# 出力（probe_checks.py が読む）：
#   PICKP,<形式>,<K>,<q>,<bit>                    (i) の 1 回．bit は RESULT の実効ビット数（error の回は NA）
#   PICKS,<形式>,<K>,<q>,<定数>,<符号>,<bit>,<pass|fail>   (ii) の 1 回（符号は none か col．error の回は fail）
#   PICK,<形式>,<(i) だけの最小>,<(i)・(ii) の最小>,<routes.tsv の p>,<same|DIFF>   形式ごとのまとめ（範囲に無ければ NA）
# 終了コード：0 は全形式で選んだ p が routes.tsv の p と一致し error の回が無い，1 は食い違う形式があるか範囲の中で選べない
# 形式があるか error の回がある，2 は routes.tsv を読めない．
set -u
cd "$(dirname "$0")/../.."
. ./scripts/lib/common.sh  # oz_env_clean・run_capture・result_bits・accuracy_verdict
# 固定の p で既定の入力を測るので，親の環境の尾崎のつまみ（自動決定 OZ_FP64，入力の作り方 OZ_PHI・OZ_CONST など，
# 検証を省く OZ_NOVERIFY，判定の基準 OZ_MIN_BITS）を引き継がない
oz_env_clean
# 回す形式と採用した p，規則の基準は routes.tsv・routes.sh から読む．
# FP16・TF32 は桁と基数が BF16 と同じなので回さない（表の acc_as が BF16 の経路）．
. ./scripts/routes.sh      # routes_each・OZ_RULE_I_BITS・OZ_STRUCT_BITS・OZ_STRUCT_CONSTS・OZ_STRUCT_SIGNS
STRUCT_BITS="${STRUCT_BITS:-$OZ_STRUCT_BITS}"   # (ii) の基準［bit］（プログラムに OZ_MIN_BITS として渡す）
fail=0
nerr=0
pick_row() {   # routes_each が r_fmt・r_binary・r_p を置いて呼ぶ
  local K q v c sg o rc vd ok ok1 qi=NA qb=NA
  for q in $(seq $((r_p - 2)) $((r_p + 2))); do
    ok1=1
    for K in 512 16384; do
      run_capture o rc -- env OZ_MIN_BITS=$OZ_RULE_I_BITS ./src/build/$r_binary 512 512 $K $q $((q-1)) 1 1
      vd=$(accuracy_verdict "$o" "$rc"); v=$(result_bits "$o")
      case $vd in
        reject) v=NA
                echo "注意: $r_binary 512 512 $K $q は入口の検査で止まった（終了コード 2．この構成は組めないので (i) を満たさない）" >&2 ;;
        error)  v=NA; nerr=$((nerr + 1))
                echo "ERROR: $r_binary 512 512 $K $q の (i) の実行が期待どおりに終わらない（終了コード $rc，VERIFY $(verify_state "$o")）" >&2 ;;
      esac
      printf 'PICKP,%s,%s,%s,%s\n' "$r_fmt" "$K" "$q" "${v:-NA}"
      [ "$vd" = pass ] || ok1=0
    done
    [ $ok1 = 1 ] || continue
    [ "$qi" = NA ] && qi=$q
    [ "$qb" = NA ] || continue      # 選んだ p より上の (ii) は測らない
    ok=1
    for K in 512 16384; do
      for c in $OZ_STRUCT_CONSTS; do
        for sg in $OZ_STRUCT_SIGNS; do
          if [ $sg = none ]; then run_capture o rc -- env OZ_CONST=$c OZ_MIN_BITS=$STRUCT_BITS ./src/build/$r_binary 512 512 $K $q $((q-1)) 1 1
          else run_capture o rc -- env OZ_CONST=$c OZ_CONST_SIGN=$sg OZ_MIN_BITS=$STRUCT_BITS ./src/build/$r_binary 512 512 $K $q $((q-1)) 1 1; fi
          vd=$(accuracy_verdict "$o" "$rc"); v=$(result_bits "$o")
          # (ii) は (i) を両方の K で通った q だけを測るので，組めない構成（reject）もここでは期待しない
          if [ "$vd" = error ] || [ "$vd" = reject ]; then
            nerr=$((nerr + 1))
            echo "ERROR: $r_binary 512 512 $K $q の (ii)（OZ_CONST=$c 符号 $sg）の実行が期待どおりに終わらない（終了コード $rc，VERIFY $(verify_state "$o")）" >&2
          fi
          if [ "$vd" = pass ]; then
            printf 'PICKS,%s,%s,%s,%s,%s,%s,pass\n' "$r_fmt" "$K" "$q" "$c" "$sg" "${v:-NA}"
          else
            printf 'PICKS,%s,%s,%s,%s,%s,%s,fail\n' "$r_fmt" "$K" "$q" "$c" "$sg" "${v:-NA}"; ok=0
          fi
        done
      done
    done
    [ $ok = 1 ] && qb=$q
  done
  local same=same
  [ "$qb" = "$r_p" ] || { same=DIFF; fail=1; }
  printf 'PICK,%s,%s,%s,%s,%s\n' "$r_fmt" "$qi" "$qb" "$r_p" "$same"
}
printf 'PICKP,impl,K,p,bit\n'
printf 'PICKS,impl,K,p,const,sign,bit,verdict\n'
printf 'PICK,impl,min_rule_i_in_p-2..p+2,min_rule_i_ii_in_p-2..p+2,p_routes,match\n'
routes_each pick_row scheme=I acc_as=- || exit 2
if [ "$nerr" -gt 0 ]; then
  echo "ERROR: 期待どおりに終わらない実行が $nerr 回あった（上の ERROR．その回の値は NA・fail にした）" >&2
  fail=1
fi
exit $fail
