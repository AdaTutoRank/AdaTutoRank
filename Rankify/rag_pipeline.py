"""
Rerank -> RAG generation -> evaluation on BM25 candidate files.

Rerankers are defined in configs/rerankers.yaml (weights in configs/weights.yaml,
prompts in configs/prompts/), the generator in configs/generator.yaml.

Output layout (one input file = one dataset):
    <output-dir>/<reranker>/ranker_cache.jsonl          reranked / selected passages
    <output-dir>/<reranker>/ranker_eval.json            answer-string relevance metrics
    <output-dir>/<reranker>/<generator>/output.jsonl    generated answers + per-sample scores
    <output-dir>/<reranker>/<generator>/output_eval_summary.json

Sub-commands:
    download      resolve (and download if needed) the weights of a reranker / the generator
    rank          rerank one shard (or everything) on the given GPU(s)
    merge-shards  merge rank shards and evaluate
    generate      generate answers for one shard (or everything)
    merge-gen-shards  merge generation shards and evaluate
    summarize     combine reranker doc sets with the original rubric data
    list          print the configured rerankers

Examples:
    python rag_pipeline.py rank --reranker adatutorank-8b --gpu 0 \
        --input data/bm25_top50/nq/test.jsonl --output-dir outputs/nq
    python rag_pipeline.py generate --reranker adatutorank-8b --gpu 0 \
        --input data/bm25_top50/nq/test.jsonl --output-dir outputs/nq
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional


def _early_set_gpu():
    """CUDA_VISIBLE_DEVICES must be set before torch / vLLM are imported."""
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            gpu_id = sys.argv[i + 1]
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            if "," in gpu_id:
                os.environ.setdefault("NCCL_P2P_DISABLE", "1")
                os.environ.setdefault("NCCL_IB_DISABLE", "1")
                os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
            return


_early_set_gpu()
os.environ.setdefault("VLLM_USE_V1", "0")

import yaml  # noqa: E402

from rankify.config import CONFIG_DIR, get_reranker_spec, list_rerankers, resolve_generator, resolve_weights  # noqa: E402
from rankify.dataset.dataset import Answer, Context, Document, Question  # noqa: E402
from rankify.metrics import generation_metrics  # noqa: E402

TOP_K_FOR_EVAL = [1, 3, 5, 10]
NDCG_CUTS = [3, 5, 10]


# ============================================================
# I/O helpers
# ============================================================
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def load_entries(input_file: str, num_entries: str) -> List[Dict[str, Any]]:
    rows = load_jsonl(input_file)
    return rows if num_entries == "all" else rows[: int(num_entries)]


def shard_slice(n: int, shard_id: Optional[int], num_shards: Optional[int]):
    if shard_id is None or num_shards is None:
        return 0, n
    size = math.ceil(n / num_shards)
    start = shard_id * size
    return start, min(start + size, n)


def gold_answers(entry: Dict[str, Any]) -> List[str]:
    """All gold aliases (answers_all), falling back to the single answer."""
    aliases = entry.get("answers_all")
    if isinstance(aliases, str):
        aliases = [aliases]
    cleaned = [str(a) for a in (aliases or []) if a is not None and str(a) != ""]
    return cleaned or [entry.get("answer", "")]


def entry_to_document(entry: Dict[str, Any]) -> Document:
    """Candidates keep their 1-based BM25 position (original_id) as id; titles are not used."""
    contexts = [
        Context(id=str(d["original_id"]), title="", text=d["doc"], score=0.0)
        for d in entry["docs"]
    ]
    return Document(question=Question(entry["question"]), answers=Answer(gold_answers(entry)), contexts=contexts)


def cleanup_gpu():
    gc.collect()
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def load_generator_config() -> Dict[str, Any]:
    with open(os.path.join(CONFIG_DIR, "generator.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generator_dir_name(model: str) -> str:
    return model.replace("/", "_")


# ============================================================
# Ranker evaluation (answer-string relevance)
# ============================================================
def relevance_map(entry: Dict[str, Any]) -> Dict[int, bool]:
    """A candidate counts as relevant if it contains the answer string, or (for answers
    longer than 3 chars) any answer word longer than 3 chars."""
    answer = (entry.get("answer") or "").lower()
    rel = {}
    for d in entry["docs"]:
        text = d["doc"].lower()
        hit = False
        if answer:
            if answer in text:
                hit = True
            elif len(answer) > 3:
                hit = any(len(w) > 3 and w in text for w in answer.split())
        rel[d["original_id"]] = hit
    return rel


def evaluate_ranker(entries: List[Dict[str, Any]], cache: List[Dict[str, Any]], set_selection: bool,
                    method: str) -> Dict[str, Any]:
    """Returns the corpus summary; set-selection rows also get a per-query ``ranker_evaluation``."""
    n = min(len(entries), len(cache))
    doc_counts = [len(c.get("context_for_generator", [])) for c in cache]
    avg_doc_count = sum(doc_counts) / len(doc_counts) if doc_counts else 0

    if set_selection:
        sums = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": 0.0, "num_selected": 0.0}
        for entry, row in zip(entries[:n], cache[:n]):
            rel = relevance_map(entry)
            selected = {int(d["original_id"]) for d in row.get("selected_docs", []) if d.get("original_id") is not None}
            relevant = {i for i, r in rel.items() if r}
            tp = len(selected & relevant)
            precision = tp / len(selected) if selected else 0.0
            recall = tp / len(relevant) if relevant else 1.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            row["ranker_evaluation"] = {
                "method_type": "set_selection", "num_selected": len(selected), "num_relevant": len(relevant),
                "precision": precision, "recall": recall, "f1": f1, "hit": tp > 0,
            }
            for k, v in (("precision", precision), ("recall", recall), ("f1", f1),
                         ("hit", float(tp > 0)), ("num_selected", len(selected))):
                sums[k] += v
        d = n or 1
        summary = {
            "method": method, "method_type": "set_selection",
            "avg_precision": sums["precision"] / d, "avg_recall": sums["recall"] / d,
            "avg_f1": sums["f1"] / d, "hit_rate": sums["hit"] / d,
            "avg_num_selected": sums["num_selected"] / d, "avg_doc_count": avg_doc_count,
        }
    else:
        rels = [relevance_map(e) for e in entries[:n]]
        ndcg = {}
        for k in NDCG_CUTS:
            scores = []
            for entry, row, rel in zip(entries[:n], cache[:n], rels):
                dcg = sum((1.0 if rel.get(d.get("original_id"), False) else 0.0) / math.log2(pos + 2)
                          for pos, d in enumerate(row.get("ranked_docs", [])[:k]))
                ideal = sorted((d["original_id"] for d in entry["docs"]), key=lambda x: rel.get(x, False), reverse=True)
                idcg = sum((1.0 if rel.get(i, False) else 0.0) / math.log2(pos + 2) for pos, i in enumerate(ideal[:k]))
                scores.append(dcg / idcg if idcg > 0 else 0.0)
            ndcg[f"ndcg@{k}"] = sum(scores) / len(scores) if scores else 0
        top_k = {}
        for k in TOP_K_FOR_EVAL:
            hits = sum(any(rel.get(d.get("original_id"), False) for d in row.get("ranked_docs", [])[:k])
                       for row, rel in zip(cache[:n], rels))
            top_k[f"top_{k}"] = hits / n * 100 if n else 0
        summary = {"method": method, "method_type": "ranking", "ndcg": ndcg,
                   "top_k_accuracy": top_k, "avg_doc_count": avg_doc_count}

    print(f"  [EVAL] {json.dumps(summary)}")
    return summary


# ============================================================
# download / list
# ============================================================
def cmd_download(args):
    if args.reranker:
        spec = get_reranker_spec(args.reranker)
        if spec.weights:
            print(resolve_weights(spec.weights))
            if spec.weights.get("base_model"):
                print(resolve_weights(spec.weights["base_model"]))
    if args.generator:
        print(resolve_generator(args.generator))


def cmd_list(_args):
    for name in list_rerankers():
        spec = get_reranker_spec(name)
        kind = "set-selection" if spec.set_selection else "ranking"
        prompt = os.path.basename(spec.prompt_file) if spec.prompt_file else "-"
        print(f"{name:<20} {spec.display_name:<26} {spec.method:<20} {kind:<14} {prompt}")


# ============================================================
# rank
# ============================================================
def ranked_rows(contexts) -> List[Dict[str, Any]]:
    return [
        {"rank": i + 1, "original_id": int(c.id) if c.id is not None else None, "score": c.score}
        for i, c in enumerate(contexts)
    ]


def cmd_rank(args):
    spec = get_reranker_spec(args.reranker)
    gen_cfg = load_generator_config()
    top_k = args.top_k if args.top_k is not None else gen_cfg.get("top_k", 5)

    ranker_dir = os.path.join(args.output_dir, spec.name)
    sharded = args.shard_id is not None and args.num_shards is not None
    cache_file = os.path.join(ranker_dir, f"ranker_cache_shard_{args.shard_id}.jsonl" if sharded else "ranker_cache.jsonl")
    print(f"\n[RANK] {spec.name} ({spec.display_name}) | GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} | {cache_file}")
    if os.path.exists(cache_file):
        print(f"  [SKIP] {cache_file} exists (delete it to re-run)")
        return

    entries = load_entries(args.input, args.num_entries)
    start, end = shard_slice(len(entries), args.shard_id, args.num_shards)
    entries = entries[start:end]
    print(f"  entries [{start}:{end}] = {len(entries)}")
    if not entries:
        save_jsonl([], cache_file)
        return

    if spec.method == "bm25":
        cache = []
        for e in entries:
            ranked = [{"rank": i + 1, "original_id": d.get("original_id"), "score": d.get("bm25_score", 0)}
                      for i, d in enumerate(e["docs"])]
            cache.append({
                "id": e.get("id"), "data_source": e.get("data_source"),
                "question": e["question"], "answer": e.get("answer", ""), "ranked_docs": ranked,
                "context_for_generator": [{"original_id": d["original_id"], "doc": d["doc"]} for d in e["docs"][:top_k]],
            })
    else:
        from rankify.models.reranking import Reranking

        overrides = {"num_gpus": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))}
        if args.max_passages and "max_passages" in spec.params:
            overrides["max_passages"] = args.max_passages
        reranker = Reranking.from_config(spec.name, **overrides)

        documents = [entry_to_document(e) for e in entries]
        t0 = time.time()
        documents = reranker.rank(documents)
        print(f"  reranked {len(documents)} queries in {time.time() - t0:.1f}s")
        del reranker
        cleanup_gpu()

        cache = []
        for e, doc in zip(entries, documents):
            ctxs = doc.reorder_contexts if doc.reorder_contexts else doc.contexts
            rows = ranked_rows(ctxs)
            row = {
                "id": e.get("id"), "data_source": e.get("data_source"),
                "question": e["question"], "answer": e.get("answer", ""),
            }
            row["selected_docs" if spec.set_selection else "ranked_docs"] = rows
            chosen = ctxs if spec.set_selection else ctxs[:top_k]
            row["context_for_generator"] = [
                {"original_id": int(c.id) if c.id is not None else None, "doc": c.text} for c in chosen
            ]
            if getattr(doc, "ranker_raw_outputs", None):
                row["ranker_raw_outputs"] = doc.ranker_raw_outputs
            cache.append(row)

    if not sharded:
        summary = evaluate_ranker(entries, cache, spec.set_selection, spec.method)
        with open(os.path.join(ranker_dir, "ranker_eval.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    save_jsonl(cache, cache_file)
    print(f"  saved {len(cache)} rows -> {cache_file}")


def cmd_merge_shards(args):
    spec = get_reranker_spec(args.reranker)
    ranker_dir = os.path.join(args.output_dir, spec.name)
    out_file = os.path.join(ranker_dir, "ranker_cache.jsonl")
    if os.path.exists(out_file):
        print(f"  [SKIP] {out_file} exists")
        return
    shard_files = [os.path.join(ranker_dir, f"ranker_cache_shard_{i}.jsonl") for i in range(args.num_shards)]
    missing = [f for f in shard_files if not os.path.exists(f)]
    if missing:
        sys.exit(f"[ERROR] missing shards: {missing}")
    cache = [row for f in shard_files for row in load_jsonl(f)]

    entries = load_entries(args.input, args.num_entries)
    summary = evaluate_ranker(entries, cache, spec.set_selection, spec.method)
    with open(os.path.join(ranker_dir, "ranker_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    save_jsonl(cache, out_file)
    for f in shard_files:
        os.remove(f)
    print(f"  merged {len(cache)} rows -> {out_file}")


# ============================================================
# generate
# ============================================================
def write_generation_eval(rows: List[Dict[str, Any]], output_file: str):
    summary, per_sample = generation_metrics([r.get("generated_answer", "") for r in rows],
                                             [gold_answers(r) for r in rows])
    for row, scores in zip(rows, per_sample):
        row["evaluation_scores"] = scores
    with open(output_file.replace(".jsonl", "_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total_samples": len(rows), "evaluation_metrics": summary}, f, ensure_ascii=False, indent=2)
    print("  " + "  ".join(f"{k}={v:.2f}" for k, v in summary.items()))


def cmd_generate(args):
    spec = get_reranker_spec(args.reranker)
    gen_cfg = load_generator_config()
    model = args.generator or gen_cfg["model"]

    ranker_dir = os.path.join(args.output_dir, spec.name)
    cache_file = os.path.join(ranker_dir, "ranker_cache.jsonl")
    out_dir = os.path.join(ranker_dir, generator_dir_name(model))
    sharded = args.shard_id is not None and args.num_shards is not None
    output_file = os.path.join(out_dir, f"output_shard_{args.shard_id}.jsonl" if sharded else "output.jsonl")
    print(f"\n[GEN] {spec.name} + {model} | GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} | {output_file}")
    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} exists")
        return
    if not os.path.exists(cache_file):
        sys.exit(f"[ERROR] ranker cache not found: {cache_file} (run `rank` first)")

    inputs = load_entries(args.input, args.num_entries)
    cache = load_jsonl(cache_file)[: len(inputs)]
    entries = []
    for e, c in zip(inputs, cache):
        if c.get("id") is not None and c["id"] != e.get("id"):
            sys.exit(f"[ERROR] {cache_file} does not match {args.input} (id {c['id']} vs {e.get('id')})")
        # output row: query fields of the input (without the candidate passages) + reranker result
        row = {k: e[k] for k in ("id", "data_source", "question", "answer", "answers_all") if k in e}
        for key in ("ranked_docs", "selected_docs", "ranker_raw_outputs", "ranker_evaluation", "context_for_generator"):
            if key in c:
                row[key] = c[key]
        entries.append(row)
    start, end = shard_slice(len(entries), args.shard_id, args.num_shards)
    entries = entries[start:end]
    if not entries:
        save_jsonl([], output_file)
        return

    from rankify.generator import Generator

    vllm_kwargs = dict(gen_cfg.get("vllm") or {})
    if args.gpu_memory_utilization is not None:
        vllm_kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        vllm_kwargs["max_model_len"] = args.max_model_len
    generator = Generator(resolve_generator(model), gen_cfg["prompt"], **vllm_kwargs)

    questions = [Question(e["question"]).question for e in entries]
    contexts = [[d["doc"] for d in e.get("context_for_generator", [])] for e in entries]
    t0 = time.time()
    answers = generator.generate(questions, contexts, dict(gen_cfg["sampling"]))
    print(f"  generated {len(answers)} answers in {time.time() - t0:.1f}s")
    del generator
    cleanup_gpu()

    for e, a in zip(entries, answers):
        e["generated_answer"] = a
    if not sharded:
        write_generation_eval(entries, output_file)
    save_jsonl(entries, output_file)
    print(f"  saved {len(entries)} rows -> {output_file}")


def cmd_merge_gen_shards(args):
    spec = get_reranker_spec(args.reranker)
    model = args.generator or load_generator_config()["model"]
    out_dir = os.path.join(args.output_dir, spec.name, generator_dir_name(model))
    output_file = os.path.join(out_dir, "output.jsonl")
    if os.path.exists(output_file):
        print(f"  [SKIP] {output_file} exists")
        return
    shard_files = [os.path.join(out_dir, f"output_shard_{i}.jsonl") for i in range(args.num_shards)]
    missing = [f for f in shard_files if not os.path.exists(f)]
    if missing:
        sys.exit(f"[ERROR] missing shards: {missing}")
    rows = [row for f in shard_files for row in load_jsonl(f)]
    write_generation_eval(rows, output_file)
    save_jsonl(rows, output_file)
    for f in shard_files:
        os.remove(f)
    print(f"  merged {len(rows)} rows -> {output_file}")


# ============================================================
# SetwiseBench summary
# ============================================================
def entry_keys(entry: Dict[str, Any]):
    """Stable join keys for input, cache, and generation rows."""
    keys = []
    if entry.get("id") is not None:
        keys.append(("id", str(entry["id"])))
    answer = entry.get("answer", "")
    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False, sort_keys=True)
    qa = ("qa", str(entry.get("question", "")), str(answer))
    if entry.get("data_source") is not None:
        keys.append(qa + (str(entry["data_source"]),))
    keys.append(qa)
    return keys


def cmd_summarize(args):
    """Build the multi-reranker input consumed by the rubric judge."""
    rows = load_entries(args.input, args.num_entries)
    by_key = {key: i for i, row in enumerate(rows) for key in entry_keys(row)}
    for row in rows:
        row["selected_docs"] = {}
        row["ranker_doc_counts"] = {}
        row["ranker_generations"] = {}
        row["ranker_scores"] = {}

    found = []
    for reranker in args.rerankers:
        cache_file = os.path.join(args.output_dir, reranker, "ranker_cache.jsonl")
        if not os.path.exists(cache_file):
            print(f"  [SKIP] {reranker}: missing {cache_file}")
            continue
        found.append(reranker)
        matched = 0
        for cached in load_jsonl(cache_file):
            idx = next((by_key[key] for key in entry_keys(cached) if key in by_key), None)
            if idx is None:
                continue
            matched += 1
            selected = cached.get("selected_docs") or cached.get("context_for_generator") or []
            doc_ids = [doc.get("original_id") for doc in selected if doc.get("original_id") is not None]
            rows[idx]["selected_docs"][reranker] = doc_ids
            rows[idx]["ranker_doc_counts"][reranker] = len(doc_ids)
        print(f"  [OK] {reranker}: matched {matched} cache rows")

        for generator in args.generators:
            output_file = os.path.join(
                args.output_dir, reranker, generator_dir_name(generator), "output.jsonl"
            )
            if not os.path.exists(output_file):
                continue
            generator_name = generator.split("/")[-1]
            for generated in load_jsonl(output_file):
                idx = next((by_key[key] for key in entry_keys(generated) if key in by_key), None)
                if idx is None:
                    continue
                rows[idx]["ranker_generations"].setdefault(reranker, {})[
                    generator_name
                ] = generated.get("generated_answer", "")
                if generated.get("evaluation_scores"):
                    rows[idx]["ranker_scores"].setdefault(reranker, {})[
                        generator_name
                    ] = generated["evaluation_scores"]

    missing_rubrics = sum(not isinstance(row.get(args.rubric_field), dict) for row in rows)
    if missing_rubrics:
        sys.exit(
            f"[ERROR] {missing_rubrics}/{len(rows)} rows lack a mapping in "
            f"'{args.rubric_field}'"
        )
    output_file = os.path.join(args.output_dir, "all_rankers_summary.jsonl")
    save_jsonl(rows, output_file)
    print(f"  summarized {len(rows)} rows for {found} -> {output_file}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, needs_input=True, gpu=False, shards=False):
        p.add_argument("--reranker", required=True, help="name from configs/rerankers.yaml")
        p.add_argument("--output-dir", required=True, help="output folder of this dataset")
        if needs_input:
            p.add_argument("--input", required=True, help="BM25 candidate jsonl")
            p.add_argument("--num-entries", default="all", help="'all' or the first N rows")
        if gpu:
            p.add_argument("--gpu", required=True, help="GPU id(s), e.g. 0 or 0,1")
        if shards:
            p.add_argument("--shard-id", type=int, default=None)
            p.add_argument("--num-shards", type=int, default=None)

    p = sub.add_parser("download", help="resolve / download weights")
    p.add_argument("--reranker", default=None)
    p.add_argument("--generator", default=None)

    sub.add_parser("list", help="list configured rerankers")

    p = sub.add_parser("rank", help="rerank (one shard)")
    common(p, gpu=True, shards=True)
    p.add_argument("--top-k", type=int, default=None, help="passages passed to the generator by ranking rerankers")
    p.add_argument("--max-passages", type=int, default=None,
                   help="override max_passages of set-selection rerankers")

    p = sub.add_parser("merge-shards", help="merge rank shards + evaluate")
    common(p)
    p.add_argument("--num-shards", type=int, required=True)

    p = sub.add_parser("generate", help="generate answers (one shard)")
    common(p, gpu=True, shards=True)
    p.add_argument("--generator", default=None, help="default: model in configs/generator.yaml")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--max-model-len", type=int, default=None)

    p = sub.add_parser("merge-gen-shards", help="merge generation shards + evaluate")
    common(p, needs_input=False)
    p.add_argument("--generator", default=None)
    p.add_argument("--num-shards", type=int, required=True)

    p = sub.add_parser("summarize", help="build SetwiseBench rubric-judge input")
    p.add_argument("--input", required=True, help="candidate JSONL carrying query rubrics")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-entries", default="all")
    p.add_argument("--rerankers", nargs="+", required=True)
    p.add_argument("--generators", nargs="*", default=[])
    p.add_argument("--rubric-field", default="hybrid_rubrics")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {
        "download": cmd_download,
        "list": cmd_list,
        "rank": cmd_rank,
        "merge-shards": cmd_merge_shards,
        "generate": cmd_generate,
        "merge-gen-shards": cmd_merge_gen_shards,
        "summarize": cmd_summarize,
    }[args.command](args)
