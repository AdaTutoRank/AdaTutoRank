# AdaTutoRank-ATO training

This directory contains the pruned FSDP + SGLang implementation of
AdaTutoRank's Adaptive Tutoring Optimization (ATO). The only training entry
point is `experiments/adatutorank_ato/train_ato.sh`.

## Data

By default, the launcher reads:

```text
data/AdaTutoRank-ATO-Data-Train.parquet
data/AdaTutoRank-ATO-Data-Val.parquet
```

The train split contains 3,557 rows and the validation split contains 100 rows.
The static files do not contain precomputed sibling sets: the self-selector and
self-reflector are generated online by the current rollout policy.

## ATO

For every query, the rollout policy samples eight subsets and the rubric judge
scores valid outputs on nine hierarchical dimensions. Invalid identifier
format receives `-1` and is masked from adaptive tutoring.

The tutoring route is:

- `r <= 3`: current-policy sibling set;
- `3 < r < 6`: current-policy corrective self-reflection;
- `r >= 6`: query-specific rubrics.

A sibling set or reflection is used only when the sibling set scores strictly
higher than the original rollout. The frozen Setwise-SFT checkpoint evaluates
the same rollout tokens under the selected hint, and training optimizes:

```text
A_ATO = 0.7 * A_GRPO + 0.3 * m * (log pi_teacher - log pi_old)
```

No explicit KL loss or response regeneration is used.

## Prompts

```text
experiments/adatutorank_ato/prompt/
├── adaptive_hints.yaml
├── self_selector.yaml
├── self_reflector.yaml
└── reward_judge/
    ├── rag.yaml
    └── deep_research.yaml
```

These files preserve the original selector, reflector, reward task, and JSON
output contracts.

## DeepSeek reward judge

The original experiments used a locally deployed DeepSeek-V4-Flash model,
which is not included in this release. The client now defaults to DeepSeek's
official OpenAI-compatible API:

```text
base_url = https://api.deepseek.com
model    = deepseek-flash
```

Create an API key by following the
[DeepSeek API documentation](https://api-docs.deepseek.com/zh-cn/). The reward
judge disables thinking by default to preserve the original no-reason JSON
output. Set `RM_THINKING=true` only when intentionally testing reasoning mode.

## Run

```bash
pip install -e ".[sglang]"

export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
export MODEL_PATH=/path/to/setwise-sft-checkpoint

bash experiments/adatutorank_ato/train_ato.sh
```

Optional infrastructure overrides include `TRAIN_DATA`, `VAL_DATA`,
`OUTPUT_DIR`, `NNODES`, `GPUS_PER_NODE`, `ROLLOUT_TP`,
`REWARD_PARALLEL_WORKERS`, `RM_BASE_URL`, and `RM_MODEL`.
