#!/usr/bin/env python3
"""Score reranked document sets with query-specific SetwiseBench rubrics."""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "setwise_judge.yaml")

LEVELS = {
    "Doc-Level": ["Relevance", "Authenticity", "Quality"],
    "Set-Level": ["Complementarity", "Redundancy", "Conflict"],
    "Global-Level": ["Completeness", "Density", "Reachability"],
}
DOC_DIMS = set(LEVELS["Doc-Level"])
ALL_DIMS = [dim for dims in LEVELS.values() for dim in dims]
VALID_SCORES = {0, 1, 2, 3, 4}


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def make_key(entry: dict):
    if entry.get("id") is not None:
        return ("id", str(entry["id"]))
    answer = entry.get("answer", "")
    if isinstance(answer, (dict, list)):
        answer = json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return (
        "qa",
        str(entry.get("question", "")),
        str(answer),
        str(entry.get("data_source", "")),
    )


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("user_message"), str):
        raise ValueError(f"{path} must contain a string user_message")
    return data["user_message"]


def rubric_list(block: dict, dim: str) -> List[str]:
    value = block.get(dim)
    if isinstance(value, dict):
        value = value.get("rubrics", [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def format_rubrics(block: dict):
    dimensions: Dict[str, List[str]] = {}
    lines = []
    for level, dims in LEVELS.items():
        for dim in dims:
            rubrics = rubric_list(block, dim)
            dimensions[dim] = rubrics
            if not rubrics:
                continue
            granularity = (
                "score each document (per_doc)"
                if dim in DOC_DIMS
                else "one overall score for the whole doc set (set_level)"
            )
            lines.append(f"[{dim}] ({level}, {granularity})")
            lines.extend(f"  {dim} #{i}: {text}" for i, text in enumerate(rubrics, 1))
            lines.append("")
    return "\n".join(lines).strip(), dimensions


def select_docs(item: dict, ranker: str, docs_field: str, selected_field: str):
    docs = {
        int(doc["original_id"]): doc
        for doc in item.get(docs_field, [])
        if isinstance(doc, dict) and doc.get("original_id") is not None
    }
    selected = item.get(selected_field, {})
    ids = selected.get(ranker, []) if isinstance(selected, dict) else []
    result = []
    for doc_id in ids:
        try:
            doc = docs.get(int(doc_id))
        except (TypeError, ValueError):
            doc = None
        if doc is not None:
            result.append(doc)
    return result


def format_docs(docs: List[dict], max_docs: Optional[int]):
    used = docs[:max_docs] if max_docs is not None else docs
    text = "\n\n".join(
        f"[{doc.get('original_id', i)}] {str(doc.get('doc', '')).strip()}"
        for i, doc in enumerate(used, 1)
    )
    return text, used


def parse_score_obj(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        text = obj.group(0)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    score = data.get("score") if isinstance(data.get("score"), dict) else data
    return score if isinstance(score, dict) else None


def to_score(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = int(round(value))
    else:
        match = re.search(r"[0-4]", str(value))
        if not match:
            return None
        score = int(match.group())
    return score if score in VALID_SCORES else None


def round1(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value + 1e-9, 1)


def compute_scores(score_obj: dict, sent_rubrics: Dict[str, List[str]]) -> Dict:
    result = {}
    for dim in ALL_DIMS:
        rubrics = sent_rubrics.get(dim, [])
        level = "doc" if dim in DOC_DIMS else "set"
        if not rubrics:
            result[dim] = {"level": level, "rubrics": [], "dimension_score": None}
            continue
        raw_dim = score_obj.get(dim) if isinstance(score_obj, dict) else None
        returned = raw_dim.get("rubrics", []) if isinstance(raw_dim, dict) else []
        if isinstance(raw_dim, list):
            returned = raw_dim
        output_rubrics, averages = [], []
        for index, text in enumerate(rubrics):
            raw_rubric = returned[index] if index < len(returned) and isinstance(returned[index], dict) else {}
            if level == "doc":
                per_doc, valid = [], []
                for raw_doc in raw_rubric.get("per_doc", []) if isinstance(raw_rubric.get("per_doc"), list) else []:
                    if not isinstance(raw_doc, dict):
                        continue
                    score = to_score(raw_doc.get("score"))
                    per_doc.append({
                        "doc_id": raw_doc.get("doc_id"),
                        "evidence": raw_doc.get("evidence", ""),
                        "score": score,
                        "reason": raw_doc.get("reason", ""),
                    })
                    if score is not None:
                        valid.append(score)
                avg = round1(sum(valid) / len(valid)) if valid else None
                output_rubrics.append({"rubric": text, "per_doc": per_doc, "rubric_avg": avg})
            else:
                raw_set = raw_rubric.get("set_level", {})
                raw_set = raw_set if isinstance(raw_set, dict) else {}
                score = to_score(raw_set.get("score"))
                if score is None:
                    score = to_score(raw_rubric.get("score"))
                avg = float(score) if score is not None else None
                output_rubrics.append({
                    "rubric": text,
                    "set_level": {
                        "evidence": raw_set.get("evidence", ""),
                        "score": score,
                        "reason": raw_set.get("reason", ""),
                    },
                    "rubric_avg": round1(avg),
                })
            if avg is not None:
                averages.append(avg)
        result[dim] = {
            "level": level,
            "rubrics": output_rubrics,
            "dimension_score": round1(sum(averages) / len(averages)) if averages else None,
        }

    if sent_rubrics.get("Conflict") and "Conflict" in score_obj and score_obj["Conflict"] is None:
        result["Conflict"] = {
            "level": "set", "rubrics": [], "dimension_score": None,
            "not_applicable": "fewer than two comparable statements about the same fact",
        }

    relevance = [
        item.get("score")
        for rubric in result.get("Relevance", {}).get("rubrics", [])
        for item in rubric.get("per_doc", [])
        if item.get("score") is not None
    ]
    if relevance and max(relevance) == 0:
        for dim in ALL_DIMS:
            if dim != "Relevance":
                result[dim] = {
                    "level": "doc" if dim in DOC_DIMS else "set",
                    "rubrics": [], "dimension_score": None,
                    "skipped": "all docs irrelevant (Relevance=0)",
                }
    return result


def compute_overall(scores: Dict[str, Dict]) -> Dict:
    overall, all_values = {}, []
    for level, dims in LEVELS.items():
        values = [
            scores[dim]["dimension_score"]
            for dim in dims
            if scores.get(dim, {}).get("dimension_score") is not None
        ]
        overall[level] = round1(sum(values) / len(values)) if values else None
        all_values.extend(values)
    overall["all_dims_mean"] = round1(sum(all_values) / len(all_values)) if all_values else None
    return overall


def missing_dimensions(block: dict, rubrics: Dict[str, List[str]]) -> List[str]:
    scores = block.get("rubric_scores", {}) if isinstance(block, dict) else {}
    if not isinstance(scores, dict):
        return [dim for dim in ALL_DIMS if rubrics.get(dim)]
    gated = bool(block.get("low_relevance_skip"))
    missing = []
    for dim in ALL_DIMS:
        if not rubrics.get(dim) or (gated and dim != "Relevance"):
            continue
        value = scores.get(dim)
        if not isinstance(value, dict):
            missing.append(dim)
        elif not value.get("skipped") and not value.get("not_applicable") and value.get("dimension_score") is None:
            missing.append(dim)
    return missing


class DeepSeekJudge:
    def __init__(self, api_key: str, config: dict):
        self.api_key = api_key
        self.config = config
        self.local = threading.local()

    def _client(self):
        if not hasattr(self.local, "client"):
            from openai import OpenAI
            self.local.client = OpenAI(
                api_key=self.api_key,
                base_url=self.config["base_url"],
                timeout=self.config["timeout"],
                max_retries=0,
            )
        return self.local.client

    def call(self, prompt: str):
        kwargs = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config["max_tokens"],
            "stream": False,
        }
        if self.config.get("thinking"):
            kwargs["reasoning_effort"] = "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        error = None
        for attempt in range(self.config["max_retries"]):
            try:
                response = self._client().chat.completions.create(**kwargs)
                choice = response.choices[0]
                return choice.message.content or "", choice.finish_reason
            except Exception as exc:
                error = exc
                if attempt + 1 < self.config["max_retries"]:
                    time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"DeepSeek request failed: {error}")


def call_and_parse(client: DeepSeekJudge, prompt: str, parse_retries: int):
    raw, finish = "", None
    for attempt in range(parse_retries):
        raw, finish = client.call(prompt)
        parsed = parse_score_obj(raw)
        if parsed is not None:
            return parsed, raw, finish
        if attempt + 1 < parse_retries:
            time.sleep(1)
    return None, raw, finish


def score_ranker(item: dict, ranker: str, rubrics: Dict[str, List[str]],
                 rubrics_text: str, prompt_template: str, client: DeepSeekJudge,
                 docs_field: str, selected_field: str, max_docs: Optional[int],
                 parse_retries: int):
    docs_text, docs = format_docs(
        select_docs(item, ranker, docs_field, selected_field), max_docs
    )
    block = {"scored_doc_ids": [doc.get("original_id") for doc in docs]}
    if not docs:
        scores = compute_scores({}, rubrics)
        return {
            **block, "rubric_scores": scores, "overall": compute_overall(scores),
            "scoring_error": f"ranker={ranker} selected no docs",
        }
    prompt = prompt_template.format(docs_str=docs_text, rubrics_str=rubrics_text)
    parsed, raw, finish = call_and_parse(client, prompt, parse_retries)
    if parsed is None:
        scores = compute_scores({}, rubrics)
        return {
            **block, "rubric_scores": scores, "overall": compute_overall(scores),
            "scoring_error": f"parse failed (finish_reason={finish})",
            "scoring_raw": raw,
        }
    scores = compute_scores(parsed, rubrics)
    block.update({"rubric_scores": scores, "overall": compute_overall(scores)})
    if any(scores.get(dim, {}).get("skipped") for dim in ALL_DIMS if dim != "Relevance"):
        block["low_relevance_skip"] = True
    return block


def process_entry(item: dict, existing: Optional[dict], rankers: Optional[List[str]],
                  rubric_field: str, docs_field: str, selected_field: str,
                  max_docs: Optional[int], prompt_template: str,
                  client: DeepSeekJudge, parse_retries: int):
    result = dict(item)
    rubric_block = item.get(rubric_field) or {}
    rubrics_text, rubrics = format_rubrics(rubric_block)
    if not any(rubrics.values()):
        result["rubric_scores_by_ranker"] = {}
        result["scoring_error"] = "no rubric"
        return result

    selected = item.get(selected_field) or {}
    available = [name for name in selected if rankers is None or name in rankers]
    existing_blocks = {}
    if isinstance(item.get("rubric_scores_by_ranker"), dict):
        existing_blocks.update(item["rubric_scores_by_ranker"])
    if isinstance(existing, dict) and isinstance(existing.get("rubric_scores_by_ranker"), dict):
        existing_blocks.update(existing["rubric_scores_by_ranker"])

    blocks = {}
    for ranker in available:
        old = existing_blocks.get(ranker)
        missing = (
            list(ALL_DIMS)
            if not isinstance(old, dict) or old.get("scoring_error")
            else missing_dimensions(old, rubrics)
        )
        missing = [dim for dim in missing if rubrics.get(dim)]
        if not missing:
            blocks[ranker] = old
            continue
        if "Relevance" in missing or not isinstance(old, dict):
            try:
                blocks[ranker] = score_ranker(
                    item, ranker, rubrics, rubrics_text, prompt_template, client,
                    docs_field, selected_field, max_docs, parse_retries,
                )
            except Exception as exc:
                empty = compute_scores({}, rubrics)
                blocks[ranker] = {
                    "scored_doc_ids": [],
                    "rubric_scores": empty,
                    "overall": compute_overall(empty),
                    "scoring_error": f"ERROR: {exc}",
                }
            continue

        partial_block = {dim: rubric_block.get(dim) for dim in missing}
        partial_text, partial_rubrics = format_rubrics(partial_block)
        try:
            repaired = score_ranker(
                item, ranker, partial_rubrics, partial_text, prompt_template, client,
                docs_field, selected_field, max_docs, parse_retries,
            )
        except Exception:
            blocks[ranker] = old
            continue
        if repaired.get("scoring_error"):
            blocks[ranker] = old
            continue
        merged = dict(old)
        merged_scores = dict(old.get("rubric_scores", {}))
        merged_scores.update({
            dim: repaired["rubric_scores"][dim]
            for dim in missing if dim in repaired.get("rubric_scores", {})
        })
        merged["rubric_scores"] = merged_scores
        merged["overall"] = compute_overall(merged_scores)
        merged.pop("scoring_error", None)
        merged.pop("scoring_raw", None)
        blocks[ranker] = merged

    result["rubric_scores_by_ranker"] = blocks
    if not available:
        result["scoring_error"] = "no ranker selected"
    else:
        result.pop("scoring_error", None)
    return result


def compact_result(row: dict) -> dict:
    """Public result: identifiers plus dimension, level, and overall scores."""
    compact = {
        key: row[key]
        for key in ("id", "question", "answer", "data_source")
        if key in row
    }
    compact["scores_by_ranker"] = {}
    for ranker, block in (row.get("rubric_scores_by_ranker") or {}).items():
        rubric_scores = block.get("rubric_scores") or {}
        overall = block.get("overall") or {}
        compact["scores_by_ranker"][ranker] = {
            "dimensions": {
                dim: rubric_scores.get(dim, {}).get("dimension_score")
                for dim in ALL_DIMS
            },
            "levels": {
                level: overall.get(level)
                for level in LEVELS
            },
            "overall": overall.get("all_dims_mean"),
        }
    if row.get("scoring_error"):
        compact["scoring_error"] = row["scoring_error"]
    return compact


def write_stats(output_file: str, stats_file: Optional[str] = None):
    from openpyxl import Workbook
    from openpyxl.styles import Font

    rows = load_jsonl(output_file)
    values: Dict[str, Dict[str, List[float]]] = {}
    passages: Dict[str, List[int]] = {}
    for row in rows:
        for ranker, block in (row.get("rubric_scores_by_ranker") or {}).items():
            passages.setdefault(ranker, []).append(len(block.get("scored_doc_ids") or []))
            for dim in ALL_DIMS:
                value = block.get("rubric_scores", {}).get(dim, {}).get("dimension_score")
                if value is not None:
                    values.setdefault(ranker, {}).setdefault(dim, []).append(float(value))

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "rubric_scores"
    headers = ["Ranker", "#Passages", *ALL_DIMS, "Overall"]
    sheet.append(headers)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
    for ranker in sorted(set(values) | set(passages)):
        dimension_means = {
            dim: (sum(values.get(ranker, {}).get(dim, [])) / len(values[ranker][dim])
                  if values.get(ranker, {}).get(dim) else None)
            for dim in ALL_DIMS
        }
        valid = [value for value in dimension_means.values() if value is not None]
        avg_passages = passages.get(ranker, [])
        sheet.append([
            ranker,
            round(sum(avg_passages) / len(avg_passages), 1) if avg_passages else None,
            *[round(dimension_means[dim] * 25, 2) if dimension_means[dim] is not None else None for dim in ALL_DIMS],
            round(sum(valid) / len(valid) * 25, 2) if valid else None,
        ])
    sheet.freeze_panes = "A2"
    stats_file = stats_file or os.path.splitext(output_file)[0] + "_stats.xlsx"
    workbook.save(stats_file)
    return stats_file


def str_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--rankers", default=None, help="comma-separated; default: all present")
    parser.add_argument("--rubric-field", default=None)
    parser.add_argument("--docs-field", default=None)
    parser.add_argument("--selected-field", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    api_cfg, judge_cfg = dict(config["api"]), dict(config["judge"])
    api_cfg["base_url"] = args.base_url or os.getenv("RM_BASE_URL") or api_cfg["base_url"]
    api_cfg["model"] = args.model or os.getenv("RM_MODEL") or api_cfg["model"]
    api_cfg["thinking"] = (
        args.thinking if args.thinking is not None
        else str_bool(os.getenv("RM_THINKING", api_cfg.get("thinking", False)))
    )
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("RM_API_KEY")
    if not api_key:
        raise SystemExit("[ERROR] Set DEEPSEEK_API_KEY (or RM_API_KEY).")

    prompt_file = args.prompt_file or judge_cfg["prompt_file"]
    if not os.path.isabs(prompt_file):
        prompt_file = os.path.join(ROOT, "configs", prompt_file)
    prompt_template = load_prompt(prompt_file)
    rubric_field = args.rubric_field or judge_cfg["rubric_field"]
    docs_field = args.docs_field or judge_cfg["docs_field"]
    selected_field = args.selected_field or judge_cfg["selected_field"]
    concurrency = args.concurrency or judge_cfg["concurrency"]
    rankers = [name.strip() for name in args.rankers.split(",") if name.strip()] if args.rankers else None

    items = load_jsonl(args.input_file)
    end = len(items) if args.limit is None else min(len(items), args.offset + args.limit)
    selected_items = items[args.offset:end]
    details_file = os.path.splitext(args.output_file)[0] + "_details.jsonl"
    existing_rows = load_jsonl(details_file) if os.path.exists(details_file) else []
    existing = {make_key(row): row for row in existing_rows}
    results = dict(existing)
    client = DeepSeekJudge(api_key, api_cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    write_lock = threading.Lock()
    append_file = open(details_file, "a", encoding="utf-8")

    def run(item):
        return process_entry(
            item, existing.get(make_key(item)), rankers, rubric_field, docs_field,
            selected_field, args.max_docs, prompt_template, client,
            judge_cfg["parse_retries"],
        )

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run, item): item for item in selected_items}
            for future in tqdm(as_completed(futures), total=len(futures), desc="setwise judge"):
                result = future.result()
                key = make_key(result)
                results[key] = result
                with write_lock:
                    append_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    append_file.flush()
    finally:
        append_file.close()

    ordered = []
    seen = set()
    for row in existing_rows + selected_items:
        key = make_key(row)
        if key not in seen and key in results:
            ordered.append(results[key])
            seen.add(key)
    save_jsonl(ordered, details_file)
    save_jsonl([compact_result(row) for row in ordered], args.output_file)
    stats_file = os.path.splitext(args.output_file)[0] + "_stats.xlsx"
    print(f"Saved {len(ordered)} concise rows -> {args.output_file}")
    print(f"Saved resumable details -> {details_file}")
    print(f"Saved summary -> {write_stats(details_file, stats_file)}")


if __name__ == "__main__":
    main()
