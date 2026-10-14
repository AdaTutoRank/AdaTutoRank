#!/usr/bin/env bash
set -euo pipefail

# AdaTutoRank-ATO training. The method choices from the paper are intentionally
# fixed here; extra arguments are forwarded as Hydra overrides.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/AdaTutoRank-ATO-Data-Train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/AdaTutoRank-ATO-Data-Val.parquet}"
MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-8}"
REWARD_PARALLEL_WORKERS="${REWARD_PARALLEL_WORKERS:-256}"

export RM_BASE_URL="${RM_BASE_URL:-https://api.deepseek.com}"
export RM_API_KEY="${RM_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export RM_MODEL="${RM_MODEL:-deepseek-flash}"
export RM_THINKING="${RM_THINKING:-false}"
export RM_MAX_CONCURRENCY="${RM_MAX_CONCURRENCY:-256}"
export REWARD_PARALLEL_WORKERS
export RUBRIC_FMT_MODE=wo_cot
export TASK=adatutorank-ato
export TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/tensorboard"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must point to the Setwise-SFT checkpoint." >&2
  exit 1
fi
if [[ -z "${RM_API_KEY}" ]]; then
  echo "Set DEEPSEEK_API_KEY (or RM_API_KEY) for the reward judge." >&2
  exit 1
fi
for path in "${TRAIN_DATA}" "${VAL_DATA}" "${MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done
python3 "${SCRIPT_DIR}/validate_data.py" "${TRAIN_DATA}" "${VAL_DATA}"

RUN_ID="${RUN_ID:-ato-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}/rollouts" "${RUN_DIR}/tutoring"

HINT_PROMPT="${SCRIPT_DIR}/prompt/adaptive_hints.yaml"
SELECTOR_PROMPT="${SCRIPT_DIR}/prompt/self_selector.yaml"
REFLECTOR_PROMPT="${SCRIPT_DIR}/prompt/self_reflector.yaml"

DATA=(
  "data.train_files=${TRAIN_DATA}"
  "data.val_files=${VAL_DATA}"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.max_prompt_length=16000"
  "data.max_response_length=64"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "+data.apply_chat_template_kwargs={enable_thinking: false}"
  "custom_reward_function.path=experiments/adatutorank_ato/reward/rubric_judge.py"
  "custom_reward_function.name=compute_score"
  "+val_reward_function.path=experiments/adatutorank_ato/reward/rubric_judge.py"
  "+val_reward_function.name=compute_score_val"
)

REWARD=(
  "reward_model.use_reward_loop=False"
  "reward_manager.source=importlib"
  "reward_manager.name=ParallelRewardManager"
  "reward_manager.module.path=experiments/adatutorank_ato/parallel_reward_manager.py"
)

MODEL=(
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.strategy=fsdp"
  "actor_rollout_ref.actor.optim.lr=5e-6"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.optim.lr_scheduler_type=cosine"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.ref.fsdp_config.param_offload=False"
  'actor_rollout_ref.actor.checkpoint.save_contents=["model","hf_model"]'
)

ATO=(
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=False"
  "actor_rollout_ref.actor.self_distillation.joint_grpo_distill=True"
  "actor_rollout_ref.actor.self_distillation.opsd_combine_mode=advantage"
  "actor_rollout_ref.actor.self_distillation.opsd_combine_form=convex"
  "actor_rollout_ref.actor.self_distillation.opsd_convex_lambda=0.7"
  "actor_rollout_ref.actor.self_distillation.alpha=1.0"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0"
  "actor_rollout_ref.actor.self_distillation.joint_exclude_incorrect_format=True"
  'actor_rollout_ref.actor.self_distillation.hint_thresholds=[3.0,6.0]'
  'actor_rollout_ref.actor.self_distillation.hint_bucket_types=["demonstration","diff","rubric"]'
  "actor_rollout_ref.actor.self_distillation.rubric_template_path=${HINT_PROMPT}"
  "actor_rollout_ref.actor.self_distillation.max_reprompt_len=20000"
  "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
  "actor_rollout_ref.actor.self_distillation.use_rubric_set_hint=True"
  "actor_rollout_ref.actor.self_distillation.rubric_set_source=policy"
  "actor_rollout_ref.actor.self_distillation.rubric_set_max_response_length=64"
  "actor_rollout_ref.actor.self_distillation.rubric_set_select_template_path=${SELECTOR_PROMPT}"
  "actor_rollout_ref.actor.self_distillation.diff_prompt_template_path=${REFLECTOR_PROMPT}"
  "actor_rollout_ref.actor.self_distillation.diff_max_response_length=256"
  "actor_rollout_ref.actor.self_distillation.rubric_set_gate_relative=True"
  "actor_rollout_ref.actor.self_distillation.rubric_set_gate_margin=0.0"
)

ROLLOUT=(
  "actor_rollout_ref.rollout.name=sglang"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
)

TRAINER=(
  "max_model_len=20128"
  "algorithm.norm_adv_by_std_in_grpo=True"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
  "trainer.use_legacy_worker_impl=enable"
  "trainer.total_epochs=1"
  "trainer.test_freq=5"
  "trainer.save_freq=15"
  "trainer.val_before_train=True"
  "trainer.project_name=AdaTutoRank-ATO"
  "trainer.experiment_name=${RUN_ID}"
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "trainer.rollout_data_dir=${RUN_DIR}/rollouts"
  "+trainer.validation_data_dir=${RUN_DIR}/rollouts"
  "trainer.distill_data_dir=${RUN_DIR}/tutoring"
  'trainer.logger=["console","tensorboard"]'
)

echo "AdaTutoRank-ATO: current-policy selector/reflector, frozen SFT teacher, lambda=0.7"
echo "train=${TRAIN_DATA} val=${VAL_DATA} model=${MODEL_PATH} output=${RUN_DIR}"

RUNTIME_ENV="$(mktemp /tmp/ato-runtime-env.XXXXXX.yaml)"
trap 'rm -f "${RUNTIME_ENV}"' EXIT
python3 -c 'import os,sys; print(os.path.expandvars(sys.stdin.read()))' \
  < "${ROOT_DIR}/runtime_env.yaml" > "${RUNTIME_ENV}"

ray job submit \
  --runtime-env="${RUNTIME_ENV}" \
  --working-dir="${ROOT_DIR}" \
  -- \
  python -m verl.trainer.main_ppo --config-name ato \
  "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ATO[@]}" \
  "${ROLLOUT[@]}" "${TRAINER[@]}" "$@"
