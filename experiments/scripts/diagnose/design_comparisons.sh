#!/usr/bin/env bash
# 論文・スライド向けの設計比較。
set -euo pipefail
cd "$(dirname "$0")/../.."
PROBES_DESIGN=only PROBES_DOCS=0 PROBES_EXTENDED=0 exec bash scripts/diagnose/probes.sh "$@"
