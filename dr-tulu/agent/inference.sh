#!/bin/bash
# Run DR Tulu with one or more rerankers, then score each output with scripts/evaluate.py.
#
#   bash inference.sh --reranker adatutorank_8b
#   bash inference.sh --reranker no_rerank,monot5_3b --tasks healthbench --num-examples 10
#
# Before running, start the vLLM servers this run talks to, either with
# `vllm serve` or with serve.sh:
#   search agent  :30001  --served-model-name rl-research/DR-Tulu-8B
#   browse agent  :30002  --served-model-name Qwen/Qwen3-8B
#   reranker      :30003  --served-model-name from configs/rerankers.yaml (vLLM rerankers only)
# The MCP server (:8000) is started by the workflow if it is not already running.
# Rerankers: configs/rerankers.yaml (`python dr_agent/run_config.py list`).
#
# Options:
#   --reranker NAMES      comma-separated reranker names (required)
#   --tasks NAMES         comma-separated tasks (default: healthbench,deep_research_bench,researchqa,webwalker)
#   --num-examples N      first N examples of each task (default: all)
#   --num-docs N          search results fetched per query (default: 30)
#   --topn N              documents returned to the agent by ranking rerankers (default: 5)
#   --hf-device DEV       GPU of the in-process HF rerankers (default: cuda:0)
#   --output DIR          output folder (default: eval_output)

set -euo pipefail
cd "$(dirname "$0")"
PYTHON_EXEC="${PYTHON_EXEC:-python}"

RERANKERS=""
TASKS="healthbench,deep_research_bench,researchqa,webwalker"
NUM_EXAMPLES=""
NUM_DOCS=30
TOPN=5
HF_DEVICE="cuda:0"
SAVE_FOLDER="eval_output"
while [ $# -gt 0 ]; do
    case "$1" in
        --reranker)     RERANKERS="$2";    shift 2 ;;
        --tasks)        TASKS="$2";        shift 2 ;;
        --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
        --num-docs)     NUM_DOCS="$2";     shift 2 ;;
        --topn)         TOPN="$2";         shift 2 ;;
        --hf-device)    HF_DEVICE="$2";    shift 2 ;;
        --output)       SAVE_FOLDER="$2";  shift 2 ;;
        -h|--help)      sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[ -n "$RERANKERS" ] || { echo "--reranker is required" >&2; exit 1; }

MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
MAX_CONCURRENT=10
export MCP_MAX_CONCURRENT_CALLS=30
export MAX_CONCURRENT_REQUESTS=20

COMMON="use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=serper,mcp_host=127.0.0.1,number_documents_to_search=$NUM_DOCS,return_topn=$TOPN,reranker_doc_max_words=200,reranker_skip_browse=false,reranker_append_page_content_to_snippet=true,reranker_hf_device=$HF_DEVICE"
NUM_EXAMPLES_ARG=()
[ -n "$NUM_EXAMPLES" ] && NUM_EXAMPLES_ARG=(--num-examples "$NUM_EXAMPLES")

IFS=',' read -r -a RERANKER_LIST <<< "$RERANKERS"
IFS=',' read -r -a TASK_LIST <<< "$TASKS"
for reranker in "${RERANKER_LIST[@]}"; do
    overrides="$COMMON,$("$PYTHON_EXEC" dr_agent/run_config.py overrides "$reranker")"
    for task in "${TASK_LIST[@]}"; do
        echo "===== $MODEL | task: $task | reranker: $reranker ====="
        output_file="$SAVE_FOLDER/$MODEL/$task/reranker-$reranker.jsonl"
        mkdir -p "$(dirname "$output_file")"

        "$PYTHON_EXEC" -m workflows.$MODEL \
            generate-dataset "$task" \
            ${NUM_EXAMPLES_ARG[@]+"${NUM_EXAMPLES_ARG[@]}"} \
            --max-concurrent $MAX_CONCURRENT \
            --batch-size $MAX_CONCURRENT \
            --use-cache \
            --config $YAML_CONFIG \
            --config-overrides "$overrides" \
            --output "$output_file"

        "$PYTHON_EXEC" scripts/evaluate.py "$task" "$output_file"
    done
done
