#!/usr/bin/env bash
# run_build.sh — reproducible index build wrapper
#
# IMPORTANT: Run this MANUALLY on the GPU server only. Never run autonomously.
# It wraps the read-only scripts/build_index.py with git-version and timestamp
# logging so each build artifact is attributable.
#
# Usage (from SectionB/, on GPU server):
#   bash scripts/run_build.sh
#
# The build takes ~5 hours on the GPU. Do not run locally.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECTION_B="$(dirname "$SCRIPT_DIR")"
cd "$SECTION_B"

# --------------------------------------------------------------------------
# 1. Code version
# --------------------------------------------------------------------------
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "nogit")
GIT_DIRTY=""
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    GIT_DIRTY="-dirty"
fi
GIT_VERSION="${GIT_SHA}${GIT_DIRTY}"

# --------------------------------------------------------------------------
# 2. Log filename
# --------------------------------------------------------------------------
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_NAME="logs/build_${GIT_VERSION}_${TIMESTAMP}.log"

# --------------------------------------------------------------------------
# 3. Header block → log
# --------------------------------------------------------------------------
{
    echo "=== run_build.sh ==="
    echo "timestamp:   $TIMESTAMP"
    echo "git_version: $GIT_VERSION"
    echo "host:        $(hostname 2>/dev/null || echo unknown)"
    echo "python:      $(python --version 2>&1)"
    echo "conda_env:   ${CONDA_DEFAULT_ENV:-none}"
    echo "==="
} | tee "$LOG_NAME"

# --------------------------------------------------------------------------
# 4. Run build
# --------------------------------------------------------------------------
echo "Starting build_index.py …"
python scripts/build_index.py 2>&1 | tee -a "$LOG_NAME"

# --------------------------------------------------------------------------
# 5. Record artifact fingerprint post-build
# --------------------------------------------------------------------------
META_FILE="artifacts/index_meta.json"
BM25_FILE="artifacts/bm25.json.gz"
{
    echo ""
    echo "=== post-build artifact summary ==="
    if [[ -f "$META_FILE" ]]; then
        N_VECTORS=$(python3 -c "import json; d=json.load(open('$META_FILE')); print(len(d.get('page_ids', [])))" 2>/dev/null || echo "?")
        META_MD5=$(md5sum "$META_FILE" 2>/dev/null | cut -c1-8 || echo "nomd5")
        echo "index_vectors: $N_VECTORS"
        echo "meta_md5:      $META_MD5"
    fi
    if [[ -f "$BM25_FILE" ]]; then
        echo "bm25.json.gz:  $(du -sh "$BM25_FILE" | cut -f1)"
    fi
    echo "==="
} | tee -a "$LOG_NAME"

echo ""
echo "Build log: $LOG_NAME"
