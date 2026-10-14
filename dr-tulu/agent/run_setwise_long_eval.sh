#!/usr/bin/env bash
# Long-form SetwiseEvalKit: DR-Tulu rollout -> cross-round doc union -> rubric judge.
#
# Usage:
#   bash run_setwise_long_eval.sh --reranker adatutorank_8b [options]
#
# Options:
#   --reranker NAME      reranker from configs/rerankers.yaml (required)
#   --num-examples N     evaluate the first N queries
#   --num-docs N         web results fetched per search round (default: 30)
#   --topn N             documents kept by ranking rerankers (default: 5)
#   --hf-device DEVICE   device for in-process rerankers (default: cuda:0)
#   --input FILE         SetwiseEvalKit_long.jsonl path
#   --output DIR         output root (default: eval_output/setwise_long)

set -euo pipefail
cd "$(dirname "$0")"

PYTHON_EXEC="${PYTHON_EXEC:-python}"
INPUT="${INPUT_FILE:-$PWD/data/setwise_eval_kit/SetwiseEvalKit_long.jsonl}"
OUTPUT_ROOT="${OUTPUT_DIR:-$PWD/eval_output/setwise_long}"
RERANKER=""
NUM_EXAMPLES=""
NUM_DOCS=30
TOPN=5
HF_DEVICE="cuda:0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --reranker) RERANKER="$2"; shift 2 ;;
    --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
    --num-docs) NUM_DOCS="$2"; shift 2 ;;
    --topn) TOPN="$2"; shift 2 ;;
    --hf-device) HF_DEVICE="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,26p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

[ -n "$RERANKER" ] || { echo "--reranker is required" >&2; exit 1; }
[ -f "$INPUT" ] || { echo "missing SetwiseEvalKit long data: $INPUT" >&2; exit 1; }
[ -n "${DEEPSEEK_API_KEY:-${RM_API_KEY:-}}" ] || {
  echo "set DEEPSEEK_API_KEY for the Setwise judge" >&2
  exit 1
}

RUN_DIR="${OUTPUT_ROOT}/${RERANKER}"
TRAJECTORY="${RUN_DIR}/trajectory.jsonl"
SUMMARY="${RUN_DIR}/all_rankers_summary.jsonl"
JUDGED="${RUN_DIR}/setwise_judged.jsonl"
mkdir -p "$RUN_DIR"

COMMON="use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=serper,mcp_host=127.0.0.1,number_documents_to_search=${NUM_DOCS},return_topn=${TOPN},reranker_doc_max_words=200,reranker_skip_browse=false,reranker_append_page_content_to_snippet=true,reranker_hf_device=${HF_DEVICE}"
OVERRIDES="${COMMON},$("$PYTHON_EXEC" dr_agent/run_config.py overrides "$RERANKER")"
NUM_ARGS=()
[ -n "$NUM_EXAMPLES" ] && NUM_ARGS=(--num-examples "$NUM_EXAMPLES")

"$PYTHON_EXEC" -m workflows.auto_search_sft \
  generate-dataset setwise_eval_long \
  --local-path "$INPUT" \
  "${NUM_ARGS[@]}" \
  --max-concurrent 10 \
  --batch-size 10 \
  --use-cache \
  --config workflows/auto_search_sft.yaml \
  --config-overrides "$OVERRIDES" \
  --output "$TRAJECTORY"

"$PYTHON_EXEC" scripts/build_setwise_long_summary.py \
  "$TRAJECTORY" "$SUMMARY" --ranker "$RERANKER"

RANKIFY_DIR="$PWD/../../Rankify"
"$PYTHON_EXEC" "$RANKIFY_DIR/setwise_judge.py" \
  "$SUMMARY" "$JUDGED" --rankers "$RERANKER"

echo "SetwiseEvalKit long evaluation complete: $RUN_DIR"
