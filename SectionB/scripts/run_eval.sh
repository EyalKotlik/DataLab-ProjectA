#!/usr/bin/env bash
# run_eval.sh — reproducible eval wrapper
#
# Captures git SHA, every scoring env var, and index fingerprint into a
# deterministic log filename and header block.  Appends one CSV row to
# logs/results.csv so CHUNKING_PROGRESS.md is generated from data, not
# hand-typed (which caused the beta005 / 0.005 naming confusion).
#
# Usage (from SectionB/):
#   AGGREGATE_MODE=length_prior COUNT_BETA=0.05 USE_BM25=0 bash scripts/run_eval.sh
#   AGGREGATE_MODE=length_prior USE_BM25=1 BM25_MIN_IDF=7.0 bash scripts/run_eval.sh
#
# Never run scripts/build_index.py via this script — use run_build.sh (manually, on GPU server).

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
# 2. Scoring config (read from env with defaults matching retrieve.py)
# --------------------------------------------------------------------------
AGGREGATE_MODE="${AGGREGATE_MODE:-length_prior}"
COUNT_BETA="${COUNT_BETA:-0.05}"
USE_BM25="${USE_BM25:-1}"
BM25_MIN_IDF="${BM25_MIN_IDF:-7.0}"
BM25_WEIGHT="${BM25_WEIGHT:-1.0}"
RRF_K="${RRF_K:-60}"
LEAD_LAMBDA="${LEAD_LAMBDA:-0.2}"

# --------------------------------------------------------------------------
# 3. Index fingerprint (vector count + md5 of meta file)
# --------------------------------------------------------------------------
META_FILE="artifacts/index_meta.json"
BM25_FILE="artifacts/bm25.json.gz"
INDEX_FP="unknown"
if [[ -f "$META_FILE" ]]; then
    N_VECTORS=$(python3 -c "import json; d=json.load(open('$META_FILE')); print(len(d.get('page_ids', [])))" 2>/dev/null || echo "?")
    META_MD5=$(md5sum "$META_FILE" 2>/dev/null | cut -c1-8 || echo "nomd5")
    BM25_SIZE="absent"
    if [[ -f "$BM25_FILE" ]]; then
        BM25_SIZE=$(du -sh "$BM25_FILE" 2>/dev/null | cut -f1 || echo "?")
    fi
    INDEX_FP="${N_VECTORS}vecs_${META_MD5}_bm25${BM25_SIZE}"
fi

# --------------------------------------------------------------------------
# 4. Deterministic log filename
# --------------------------------------------------------------------------
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Sanitise beta: "0.05" → "0p05" to avoid dots in filenames
BETA_SAFE=$(echo "$COUNT_BETA" | tr '.' 'p')
IDF_SAFE=$(echo "$BM25_MIN_IDF" | tr '.' 'p')
W_SAFE=$(echo "$BM25_WEIGHT" | tr '.' 'p')
LOG_NAME="logs/eval_${AGGREGATE_MODE}_b${BETA_SAFE}_bm25${USE_BM25}_idf${IDF_SAFE}_w${W_SAFE}_${GIT_VERSION}_${TIMESTAMP}.log"

# --------------------------------------------------------------------------
# 5. Header block → log
# --------------------------------------------------------------------------
{
    echo "=== run_eval.sh ==="
    echo "timestamp:       $TIMESTAMP"
    echo "git_version:     $GIT_VERSION"
    echo "index_fp:        $INDEX_FP"
    echo "AGGREGATE_MODE:  $AGGREGATE_MODE"
    echo "COUNT_BETA:      $COUNT_BETA"
    echo "LEAD_LAMBDA:     $LEAD_LAMBDA"
    echo "USE_BM25:        $USE_BM25"
    echo "BM25_MIN_IDF:    $BM25_MIN_IDF"
    echo "BM25_WEIGHT:     $BM25_WEIGHT"
    echo "RRF_K:           $RRF_K"
    echo "==="
} | tee "$LOG_NAME"

# --------------------------------------------------------------------------
# 6. Run eval (tee output to log; preserve exit code)
# --------------------------------------------------------------------------
export AGGREGATE_MODE COUNT_BETA LEAD_LAMBDA USE_BM25 BM25_MIN_IDF BM25_WEIGHT RRF_K
EVAL_OUTPUT=$(python scripts/eval_public.py 2>&1 | tee -a "$LOG_NAME")
echo "$EVAL_OUTPUT"

# --------------------------------------------------------------------------
# 7. Append CSV row to logs/results.csv
# --------------------------------------------------------------------------
NDCG=$(echo "$EVAL_OUTPUT" | grep "mean_ndcg" | sed 's/.*=//')
QPT=$(echo "$EVAL_OUTPUT" | grep "query_phase_time" | sed 's/.*=//')

CSV_FILE="logs/results.csv"
if [[ ! -f "$CSV_FILE" ]]; then
    echo "timestamp,git_version,index_fp,AGGREGATE_MODE,COUNT_BETA,USE_BM25,BM25_MIN_IDF,BM25_WEIGHT,RRF_K,mean_ndcg,query_phase_time,log" > "$CSV_FILE"
fi
echo "${TIMESTAMP},${GIT_VERSION},${INDEX_FP},${AGGREGATE_MODE},${COUNT_BETA},${USE_BM25},${BM25_MIN_IDF},${BM25_WEIGHT},${RRF_K},${NDCG},${QPT},${LOG_NAME}" >> "$CSV_FILE"

echo ""
echo "Log:    $LOG_NAME"
echo "CSV:    $CSV_FILE  (last row appended)"
