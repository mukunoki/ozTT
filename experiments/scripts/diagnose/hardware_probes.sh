#!/usr/bin/env bash
# 論文・スライド向けの基本プローブ。既存の一覧・測定機構を共有する。
set -euo pipefail
cd "$(dirname "$0")/../.."
PROBES_DESIGN=0 PROBES_DOCS=0 PROBES_EXTENDED=0 exec bash scripts/diagnose/probes.sh "$@"
