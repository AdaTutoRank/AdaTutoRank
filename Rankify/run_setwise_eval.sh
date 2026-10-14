#!/usr/bin/env bash
# SetwiseBench: rerank -> RAG generation -> rubric-based document-set judge.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash run_setwise_eval.sh adatutorank-8b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/rag_pipeline.py"
JUDGE="${SCRIPT_DIR}/setwise_judge.py"

INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/data/setwise_eval_kit/SetwiseEvalKit_short.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/setwise_bench}"
NUM_ENTRIES="${NUM_ENTRIES:-all}"
GENERATOR="${GENERATOR:-meta-llama/Llama-3.1-8B-Instruct}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-5}"
CONDA_ENV="${CONDA_ENV-adatutorank}"

read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS=${#GPU_IDS[@]}

RERANKERS=(
  bm25
  bge-reranker-large
  monot5-3b
  rankt5-3b
  rankllama-7b
  rankvicuna-7b
  rankzephyr-7b
  rearank-7b
  reasonrank-7b
  rank4gen-8b
  setr-8b
  rubricranker-8b
  adatutorank-8b
)
if [ "$#" -gt 0 ]; then
  RERANKERS=("$@")
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_shards() {
  local log_prefix="$1"; shift
  local pids=() failed=0
  for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPU_IDS[$shard_id]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" \
      --gpu "$gpu" --shard-id "$shard_id" --num-shards "$NUM_SHARDS" \
      > "${log_prefix}_shard${shard_id}.log" 2>&1 &
    pids+=($!)
  done
  for shard_id in "${!pids[@]}"; do
    if ! wait "${pids[$shard_id]}"; then
      log "shard ${shard_id} failed; see ${log_prefix}_shard${shard_id}.log"
      failed=1
    fi
  done
  return "$failed"
}

if [ -n "$CONDA_ENV" ]; then
  if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$CONDA_ENV"
fi

[ -f "$INPUT_FILE" ] || { echo "missing SetwiseBench input: $INPUT_FILE" >&2; exit 1; }
[ -n "${DEEPSEEK_API_KEY:-${RM_API_KEY:-}}" ] || {
  echo "set DEEPSEEK_API_KEY for the rubric judge" >&2
  exit 1
}

mkdir -p "${OUTPUT_DIR}/logs"
python "$PIPELINE" download --generator "$GENERATOR"
GEN_DIR="${GENERATOR//\//_}"

for reranker in "${RERANKERS[@]}"; do
  log "==================== ${reranker} ===================="
  python "$PIPELINE" download --reranker "$reranker"
  RANKER_DIR="${OUTPUT_DIR}/${reranker}"
  LOG_DIR="${OUTPUT_DIR}/logs/${reranker}"
  mkdir -p "$LOG_DIR"
  COMMON=(
    --reranker "$reranker"
    --input "$INPUT_FILE"
    --output-dir "$OUTPUT_DIR"
    --num-entries "$NUM_ENTRIES"
  )

  if [ ! -f "${RANKER_DIR}/ranker_cache.jsonl" ]; then
    log "ranking on ${NUM_SHARDS} GPUs"
    run_shards "${LOG_DIR}/rank" python "$PIPELINE" rank "${COMMON[@]}"
    python "$PIPELINE" merge-shards "${COMMON[@]}" --num-shards "$NUM_SHARDS" \
      > "${LOG_DIR}/rank_merge.log" 2>&1
  fi

  if [ ! -f "${RANKER_DIR}/${GEN_DIR}/output.jsonl" ]; then
    log "generating answers on ${NUM_SHARDS} GPUs"
    run_shards "${LOG_DIR}/generate" python "$PIPELINE" generate \
      "${COMMON[@]}" --generator "$GENERATOR"
    python "$PIPELINE" merge-gen-shards \
      --reranker "$reranker" --output-dir "$OUTPUT_DIR" \
      --generator "$GENERATOR" --num-shards "$NUM_SHARDS" \
      > "${LOG_DIR}/generate_merge.log" 2>&1
  fi
done

python "$PIPELINE" summarize \
  --input "$INPUT_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --num-entries "$NUM_ENTRIES" \
  --rerankers "${RERANKERS[@]}" \
  --generators "$GENERATOR"

RANKER_CSV="$(IFS=,; echo "${RERANKERS[*]}")"
python "$JUDGE" \
  "${OUTPUT_DIR}/all_rankers_summary.jsonl" \
  "${OUTPUT_DIR}/setwise_judged.jsonl" \
  --rankers "$RANKER_CSV" \
  --concurrency "$JUDGE_CONCURRENCY"

log "SetwiseBench complete: ${OUTPUT_DIR}"
