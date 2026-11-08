#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Checks that every kernel file of the working tree (src/backend/*/kernels/)
# differs from a git revision only in comments and whitespace (string and
# character literals are code), and is ASCII.
#   tools/check_kernels.sh [REV]      default REV is HEAD
# Prints OK/DIFF per file and a summary; exits 1 on any difference, 2 on a
# usage or git error.  Files added or removed relative to REV are reported
# as differences.
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
rev=${1:-HEAD}
if (( $# > 1 )) || [[ $rev == -* ]]; then echo "usage: $0 [REV]" >&2; exit 2; fi
cd "$root"
git rev-parse --verify --quiet "$rev^{commit}" >/dev/null ||
  { echo "check_kernels.sh: unknown revision '$rev'" >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
# Paths in the archive are relative to this directory (the repository root).
git archive --format=tar "$rev" src/backend | tar -x -C "$tmp"
ok=0 diff=0
list() { (cd -- "$1" && find src/backend -path '*/kernels/*' -type f | sort); }
# The union of both file lists, so that added and removed files show up.
while IFS= read -r f; do
  if [[ ! -f $tmp/$f ]]; then echo "DIFF $f (not in $rev)"; diff=$((diff + 1))
  elif [[ ! -f $root/$f ]]; then echo "DIFF $f (removed from the working tree)"; diff=$((diff + 1))
  elif out=$(python3 -I "$root/tools/codecheck.py" "$tmp/$f" "$root/$f" 2>&1 </dev/null); then echo "OK   $f"; ok=$((ok + 1))
  else echo "DIFF $f"; echo "$out" | sed 's/^/    /'; diff=$((diff + 1))
  fi
done < <({ list "$root"; list "$tmp"; } | sort -u)
echo "check_kernels: $ok OK, $diff DIFF (working tree vs $rev)"
[[ $diff == 0 ]]
