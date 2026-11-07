// SPDX-License-Identifier: Apache-2.0
#pragma once
// デバイスカーネルのソースの置き場所（CreateKernel に渡すパスの基点）
// ---------------------------------------------------------------------------
// カーネルのパスはすべて，ここの置き場所の定数にファイル名を続けて書く（文字列の連結．`OZ2_KDIR "recon_crt.cpp"`）．
// 基点 OZ_SRC_DIR は src/Makefile が渡す，ビルドした src の絶対パス（末尾に /．コンテナの中では scripts/run.sh が
// リポジトリを /home/user/tt にマウントするので /home/user/tt/src/）．したがってバイナリは，作った木のカーネルを読む
// （別の worktree で作れば，その worktree のカーネル）．
// 絶対パスにするのは，tt-metal が相対パスを作業ディレクトリ・TT_METAL_KERNEL_PATH・TT_METAL_HOME の順に探すため
// （起動した場所によって別のファイルを読みうる）．
// カーネルどうしの取り込みは，取り込む側のファイルの置き場所からの相対で書く（JIT はカーネルのファイルを絶対パスで
// 取り込み，その置き場所を取り込みの根にも加えるので，引用符の #include はそこから解ける）．
#ifndef OZ_SRC_DIR
#error "OZ_SRC_DIR（ビルドした src の絶対パスと末尾の /）が無い．src/Makefile でビルドすること"
#endif

#define OZ_COMMON_KDIR  OZ_SRC_DIR "common/kernels/"    // 両スキームと基準実装（と検査）が共有するカーネル
#define OZ1_KDIR        OZ_SRC_DIR "ozaki1/kernels/"    // 尾崎 scheme I だけが使うカーネル
#define OZ2_KDIR        OZ_SRC_DIR "ozaki2/kernels/"    // 尾崎 scheme II のカーネル
// 本番のドライバは使わない置き場所（探針・検査．src/README.md の「ディレクトリ」）
#define OZ_PROBES_KDIR  OZ_SRC_DIR "probes/kernels/"
#define OZ_CHECKS_KDIR  OZ_SRC_DIR "checks/kernels/"
#define OZ_TESTKIT_KDIR OZ_SRC_DIR "testkit/kernels/"
