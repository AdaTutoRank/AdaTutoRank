#!/bin/bash
###############################################################################
# RAG evaluation on 5 benchmarks: NQ / TriviaQA / PopQA / 2WikiMultiHopQA / Bamboogle
#
# Every reranker reranks the BM25 top-50 candidates of each benchmark, then
# Llama-3.1-8B-Instruct answers from the passages it keeps (top-5 for ranking
# models, the whole selection for set-selection models).
#
#   bash run_rag_eval.sh                                       # all rerankers on all benchmarks
#   DATASETS="nq popqa" bash run_rag_eval.sh adatutorank-8b    # chosen rerankers / benchmarks
#
# Missing weights are downloaded automatically into RANKIFY_MODEL_DIR (default <repo>/models).
# Environment overrides: DATASETS, DATA_DIR, OUTPUT_DIR, GPU_IDS, NUM_ENTRIES, CONDA_ENV,
#                        HTTP_PROXY_URL, RANKIFY_MODEL_DIR
#
# Output:
#   <OUTPUT_DIR>/<benchmark>/<reranker>/ranker_cache.jsonl
#   <OUTPUT_DIR>/<benchmark>/<reranker>/<generator>/output.jsonl (+ output_eval_summary.json)
#   <OUTPUT_DIR>/logs/<benchmark>/<reranker>/*.log
# Finished steps are skipped, so the script can simply be re-run after a failure.
###############################################################################

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="${SCRIPT_DIR}/rag_pipeline.py"

# ---------------------------- configuration ---------------------------------
# Candidate files: ${DATA_DIR}/<benchmark>/<split>.jsonl
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/bm25_top50}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/bm25_top50}"
NUM_ENTRIES="${NUM_ENTRIES:-all}"

# One data-parallel shard per GPU, for both ranking and generation.
read -r -a GPU_IDS <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
NUM_SHARDS=${#GPU_IDS[@]}

CONDA_ENV="${CONDA_ENV-adatutorank}"     # set CONDA_ENV= to use the current environment
HTTP_PROXY_URL="${HTTP_PROXY_URL:-}"    # e.g. http://proxy:3128 if downloads need a proxy

read -r -a DATASETS <<< "${DATASETS:-nq triviaqa popqa 2wikimultihopqa bamboogle}"
declare -A DATASET_SPLITS=(
    [nq]=test [triviaqa]=test [popqa]=test [2wikimultihopqa]=dev [bamboogle]=test
)
for ds in "${DATASETS[@]}"; do
    [ -n "${DATASET_SPLITS[$ds]}" ] || { echo "unknown benchmark ${ds} (choose from: ${!DATASET_SPLITS[*]})"; exit 1; }
done

# Names from configs/rerankers.yaml (order = row order of the result table)
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

# ------------------------------- helpers ------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

input_file() { echo "${DATA_DIR}/$1/${DATASET_SPLITS[$1]}.jsonl"; }

# run_shards <log prefix> <command...>: one process per GPU with --gpu/--shard-id/--num-shards appended
run_shards() {
    local log_prefix="$1"; shift
    local pids=() failed=0
    for shard_id in $(seq 0 $((NUM_SHARDS - 1))); do
        local gpu="${GPU_IDS[$shard_id]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$@" --gpu "$gpu" --shard-id "$shard_id" --num-shards "$NUM_SHARDS" \
            > "${log_prefix}_shard${shard_id}.log" 2>&1 &
        pids+=($!)
    done
    for shard_id in "${!pids[@]}"; do
        if ! wait "${pids[$shard_id]}"; then
            log "  shard ${shard_id} failed, see ${log_prefix}_shard${shard_id}.log"
            failed=1
        fi
    done
    return $failed
}

# ------------------------------ environment ---------------------------------
if [ -n "$CONDA_ENV" ]; then
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook)"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    fi
    conda activate "$CONDA_ENV" || { echo "cannot activate conda env ${CONDA_ENV}"; exit 1; }
fi
if [ -n "$HTTP_PROXY_URL" ]; then
    export http_proxy="$HTTP_PROXY_URL" https_proxy="$HTTP_PROXY_URL"
fi

# ------------------------------ input data ----------------------------------
for ds in "${DATASETS[@]}"; do
    [ -f "$(input_file "$ds")" ] || { log "missing input file $(input_file "$ds")"; exit 1; }
done

echo "###############################################################################"
echo "  input:      ${DATA_DIR}"
for ds in "${DATASETS[@]}"; do
    printf "              %-18s %6s queries\n" "$ds" "$(wc -l < "$(input_file "$ds")")"
done
echo "  output:     ${OUTPUT_DIR}"
echo "  rerankers:  ${RERANKERS[*]}"
echo "  GPUs:       ${GPU_IDS[*]}"
echo "###############################################################################"

# ------------------------------- pipeline -----------------------------------
mkdir -p "${OUTPUT_DIR}/logs"
python "$PIPELINE" download --generator meta-llama/Llama-3.1-8B-Instruct || exit 1
GEN_DIR="meta-llama_Llama-3.1-8B-Instruct"

for reranker in "${RERANKERS[@]}"; do
    log "==================== ${reranker} ===================="
    python "$PIPELINE" download --reranker "$reranker" || { log "cannot resolve weights of ${reranker}"; continue; }

    for ds in "${DATASETS[@]}"; do
        INPUT="$(input_file "$ds")"
        DS_OUT="${OUTPUT_DIR}/${ds}"
        LOGS="${OUTPUT_DIR}/logs/${ds}/${reranker}"
        mkdir -p "$LOGS"
        COMMON=(--reranker "$reranker" --input "$INPUT" --output-dir "$DS_OUT" --num-entries "$NUM_ENTRIES")

        if [ ! -f "${DS_OUT}/${reranker}/ranker_cache.jsonl" ]; then
            log "[${ds}] ranking on ${NUM_SHARDS} GPUs"
            run_shards "${LOGS}/rank" python "$PIPELINE" rank "${COMMON[@]}" || continue
            python "$PIPELINE" merge-shards "${COMMON[@]}" --num-shards "$NUM_SHARDS" \
                > "${LOGS}/rank_merge.log" 2>&1 || { log "[${ds}] merge failed, see ${LOGS}/rank_merge.log"; continue; }
        fi

        if [ ! -f "${DS_OUT}/${reranker}/${GEN_DIR}/output.jsonl" ]; then
            log "[${ds}] generation on ${NUM_SHARDS} GPUs"
            run_shards "${LOGS}/gen" python "$PIPELINE" generate "${COMMON[@]}" || continue
            python "$PIPELINE" merge-gen-shards --reranker "$reranker" --output-dir "$DS_OUT" \
                --num-shards "$NUM_SHARDS" > "${LOGS}/gen_merge.log" 2>&1 \
                || { log "[${ds}] merge failed, see ${LOGS}/gen_merge.log"; continue; }
        fi
        log "[${ds}] done: $(tr -d '\n ' < "${DS_OUT}/${reranker}/${GEN_DIR}/output_eval_summary.json" | head -c 300)"
    done
done
