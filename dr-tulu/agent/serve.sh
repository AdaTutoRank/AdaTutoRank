#!/bin/bash
# Start the vLLM server of an agent or a reranker; weights are downloaded on first use.
#
#   bash serve.sh search_agent    --gpus 0,1    # DR Tulu        :30001
#   bash serve.sh browse_agent    --gpus 2      # Qwen3-8B       :30002
#   bash serve.sh adatutorank_8b  --gpus 3      # one reranker   :30003
#
# Names, ports and vLLM flags are in configs/rerankers.yaml (`python dr_agent/run_config.py list`).
# Extra arguments after the options are passed to `vllm serve`.

set -euo pipefail
cd "$(dirname "$0")"
PYTHON_EXEC="${PYTHON_EXEC:-python}"

NAME="${1:?usage: bash serve.sh <name> [--gpus 0,1] [vllm args...]}"
shift
GPUS=""
if [ "${1:-}" = "--gpus" ]; then
    GPUS="$2"
    shift 2
fi

CMD=$("$PYTHON_EXEC" dr_agent/run_config.py serve "$NAME")
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    CMD="$CMD --tensor-parallel-size $(echo "$GPUS" | tr ',' '\n' | wc -l)"
fi
echo "$CMD $*"
eval "$CMD" '"$@"'
