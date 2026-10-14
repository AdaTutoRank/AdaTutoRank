<h1 align="center"> AdaTutoRank: RAG Evaluation of Document Rerankers </h1>

<div align="center">
<a href="https://arxiv.org/abs/2609.32472" target="_blank"><img src=https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://huggingface.co/papers/2609.32472" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-ffc107.svg></a>
<a href="https://adatutorank.github.io/" target="_blank"><img src=https://img.shields.io/badge/Project-Page-4b8bbe.svg?logo=googlechrome&logoColor=white></a>
<a href="https://huggingface.co/datasets/kailinjiang/AdaTutoRank-Train-Data" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset-ff9d00.svg></a>
<a href="https://huggingface.co/kailinjiang/AdaTutoRank-8B" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-ff9d00.svg></a>
<a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg"></a>
</div>

This code reproduces the RAG comparison of the AdaTutoRank paper: every reranker reorders
(or selects from) the **BM25 top-50** candidates of five QA benchmarks, then
Llama-3.1-8B-Instruct answers from the passages it keeps. It is a trimmed fork of
[Rankify](https://github.com/DataScienceUIBK/rankify) that keeps only the rerankers of the paper
table, with every model's weights and prompt in plain config files.

## 📋 Table of Contents

- [1. How to run the RAG evaluation](#-1-how-to-run-the-rag-evaluation)
  - [1.1 Environment and Preparation](#️-11-environment-and-preparation)
  - [1.2 Evaluation](#-12-evaluation)
- [2. Rerankers](#-2-rerankers)
- [3. Configuration](#️-3-configuration)
- [4. Outputs and Metrics](#-4-outputs-and-metrics)

## ⚡ 1. How to run the RAG evaluation

### 🛠️ 1.1 Environment and Preparation

Set up the conda environment and install the package:

```bash
cd Rankify/
conda create -n adatutorank python=3.10 -y && conda activate adatutorank
pip install -r requirements.txt
pip install -e .
```

Download the BM25 top-50 candidate files of the five benchmarks (1.5 GB in total) from
[kailinjiang/rag-bm25-top50](https://huggingface.co/datasets/kailinjiang/rag-bm25-top50) into `data/bm25_top50/`:

```bash
huggingface-cli download kailinjiang/rag-bm25-top50 --repo-type dataset --local-dir data/bm25_top50
```

```
data/bm25_top50/
├── meta.json
├── nq/test.jsonl                   # 3,610 queries
├── triviaqa/test.jsonl             # 11,313 queries
├── popqa/test.jsonl                # 14,267 queries
├── 2wikimultihopqa/dev.jsonl       # 12,576 queries
└── bamboogle/test.jsonl            # 125 queries
```

Each line is one query with its 50 BM25 passages:

```json
{"id": "nq_test_0", "question": "...", "answer": "...", "answers_all": ["...", "..."],
 "docs": [{"original_id": 1, "doc": "passage text", "title": "...", "bm25_score": 16.3}, ...]}
```

### 📊 1.2 Evaluation

Run the evaluation. Choose the rerankers as arguments (see [Rerankers](#-2-rerankers)) and the
benchmarks with `DATASETS` (`nq triviaqa popqa 2wikimultihopqa bamboogle`); leaving either out
runs all of them. Missing weights are downloaded automatically into `models/` (see
[Configuration](#️-3-configuration)), the data is split over the GPUs in `GPU_IDS`, and finished
steps are skipped when the script is re-run:

```bash
GPU_IDS="0 1 2 3" DATASETS="nq popqa" bash run_rag_eval.sh adatutorank-8b setr-8b
```

Running one step by hand:

```bash
python rag_pipeline.py rank     --reranker adatutorank-8b --gpu 0 \
    --input data/bm25_top50/nq/test.jsonl --output-dir outputs/bm25_top50/nq
python rag_pipeline.py generate --reranker adatutorank-8b --gpu 0 \
    --input data/bm25_top50/nq/test.jsonl --output-dir outputs/bm25_top50/nq
```

To split a step over N processes, add `--shard-id i --num-shards N` to each of them, then run
`merge-shards` (after `rank`) or `merge-gen-shards` (after `generate`).

From Python:

```python
from rankify.models.reranking import Reranking
reranker = Reranking.from_config("adatutorank-8b")    # weights + prompt resolved from configs/
reranked = reranker.rank(documents)                   # list[rankify.dataset.dataset.Document]
```

## 🏆 2. Rerankers

| Name (`--reranker`) | Paper row | Type | Weights |
|---|---|---|---|
| `bm25` | Initial Retrieval (BM25) | keeps BM25 order | – |
| `bge-reranker-large` | BGE-Reranker-L | pointwise | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |
| `monot5-3b` | MonoT5 (3B) | pointwise | [castorini/monot5-3b-msmarco-10k](https://huggingface.co/castorini/monot5-3b-msmarco-10k) |
| `rankt5-3b` | RankT5 (3B) | pointwise | [Soyoung97/RankT5-3b](https://huggingface.co/Soyoung97/RankT5-3b) |
| `rankllama-7b` | RankLlama (7B) | pointwise | [castorini/rankllama-v1-7b-lora-passage](https://huggingface.co/castorini/rankllama-v1-7b-lora-passage) (+ [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)) |
| `rankvicuna-7b` | RankVicuna (7B) | listwise | [castorini/rank_vicuna_7b_v1](https://huggingface.co/castorini/rank_vicuna_7b_v1) |
| `rankzephyr-7b` | RankZephyr (7B) | listwise | [castorini/rank_zephyr_7b_v1_full](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) |
| `rearank-7b` | Rearank (7B) | listwise, reasoning | [le723z/Rearank-7B](https://huggingface.co/le723z/Rearank-7B) |
| `reasonrank-7b` | ReasonRank (7B) | listwise, reasoning | [liuwenhan/reasonrank-7B](https://huggingface.co/liuwenhan/reasonrank-7B) |
| `rank4gen-8b` | Rank4Gen (8B) | set selection | [JohnnyFan/Rank4Gen-DPO-Qwen3-8B](https://modelscope.cn/models/JohnnyFan/Rank4Gen-DPO-Qwen3-8B) (ModelScope) |
| `setr-8b` | SetR (8B) | set selection | [JohnnyFan/SETR-Qwen3-8B](https://modelscope.cn/models/JohnnyFan/SETR-Qwen3-8B) (ModelScope) |
| `rubricranker-8b` | RubricRanker (8B) | set selection | [lwhlwh/rubricranker_sft_rl](https://modelscope.cn/models/lwhlwh/rubricranker_sft_rl) (ModelScope) |
| `adatutorank-8b` | **AdaTutoRank (8B)** | set selection | [kailinjiang/AdaTutoRank-8B](https://huggingface.co/kailinjiang/AdaTutoRank-8B) |

The downstream generator is [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
Ranking models give it their **top-5** passages; set-selection models give it **their whole
selection**, so the number of passages varies per query (reported as "Psg").

## ⚙️ 3. Configuration

```
configs/
├── rerankers.yaml        name -> display name, method, prompt file, runtime parameters
├── weights.yaml          name -> repo id + local folder
├── generator.yaml        generator model, vLLM settings, sampling, prompt
└── prompts/<name>.yaml   one prompt file per reranker (Rank4Gen also uses rank4gen_descriptions.json)
rankify/                  reranker implementations, generator, metrics, config loading
rag_pipeline.py           download / rank / generate / merge shards for one reranker on one benchmark
run_rag_eval.sh           full evaluation on N GPUs
```

**Weights.** All weights live in one folder, `model_dir` in `configs/weights.yaml` (default
`models/`, or the env var `RANKIFY_MODEL_DIR`), one sub-folder per model:

```
models/
├── bge-reranker-large/
├── monot5-3b-msmarco-10k/
├── RankT5-3b/
├── rankllama-v1-7b-lora-passage/
├── Llama-2-7b-hf/                  # RankLlama base model
├── rank_vicuna_7b_v1/
├── rank_zephyr_7b_v1_full/
├── Rearank-7B/
├── reasonrank-7B/
├── Rank4Gen-DPO-Qwen3-8B/          # ModelScope
├── SETR-Qwen3-8B/                  # ModelScope
├── rubricranker_sft_rl/            # ModelScope
├── AdaTutoRank-8B/
└── Llama-3.1-8B-Instruct/          # generator
```

Each entry of `configs/weights.yaml` gives the repo first and the folder second:

```yaml
model_dir: models
rerankers:
  adatutorank-8b:
    repo_id: kailinjiang/AdaTutoRank-8B
    local_path: AdaTutoRank-8B                 # -> models/AdaTutoRank-8B
  setr-8b:
    repo_id: JohnnyFan/SETR-Qwen3-8B
    hub: modelscope
    local_path: /data/weights/SETR-Qwen3-8B    # absolute paths are used as is
```

If the folder already holds the weights, they are loaded from there. Otherwise the repo is
downloaded into exactly that folder (`hub: modelscope` downloads from ModelScope; a failed
Hugging Face download is retried there).

**Prompts.** Each file in `configs/prompts/` holds the exact strings sent to one model; its
header documents the placeholders (`{query}`, `{num}`, `{context}`, ...) and how the passage
block is formatted. They are byte-identical to the prompts used for the paper numbers. The
trained rerankers (Rearank, ReasonRank, Rank4Gen, SetR, RubricRanker, AdaTutoRank) are sensitive
to any change.

## 📈 4. Outputs and Metrics

```
outputs/bm25_top50/
└── <benchmark>/<reranker>/
    ├── ranker_cache.jsonl                                         reranked / selected passages
    ├── ranker_eval.json
    └── meta-llama_Llama-3.1-8B-Instruct/
        ├── output.jsonl                                           generated answers + scores
        └── output_eval_summary.json                               benchmark averages
```

Answers are scored against every alias in `answers_all`, keeping the best match. The **"EM"
column of the paper table is `contains_match`** (the normalized gold answer appears in the
prediction); strict `exact_match` and token F1 are in `output_eval_summary.json` as well.

## 📄 Citation

```bibtex
@article{adatutorank2026,
  title   = {AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research},
  journal = {arXiv preprint arXiv:2609.32472},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.32472}
}
```

This code builds on Rankify; please also cite it (see `CITATION.cff`).
