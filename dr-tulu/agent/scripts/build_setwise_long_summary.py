#!/usr/bin/env python3
"""Build SetwiseBench judge input from multi-round DR-Tulu trajectories."""

import argparse
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit


SEARCH_TOOLS = {"snippet_search", "google_search"}


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_search_calls(value) -> Iterable[dict]:
    """Yield chronological search calls, including answer-agent nested traces."""
    if isinstance(value, dict):
        if value.get("tool_name") in SEARCH_TOOLS and isinstance(value.get("documents"), list):
            yield value
            return
        for key, child in value.items():
            if key in {"tool_calls", "full_traces"}:
                yield from iter_search_calls(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_search_calls(child)


def normalize_url(url: str) -> str:
    try:
        parts = urlsplit(url.strip())
        path = re.sub(r"/+$", "", parts.path)
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))
    except Exception:
        return url.strip().lower()


def document_text(document: dict) -> str:
    parts = []
    if document.get("title"):
        parts.append(f"Title: {document['title']}")
    if document.get("url"):
        parts.append(f"URL: {document['url']}")
    content = (
        document.get("summary")
        or document.get("text")
        or document.get("snippet")
        or ""
    )
    if content:
        parts.append(str(content))
    return "\n".join(parts).strip()


def document_key(document: dict) -> str:
    if document.get("url"):
        return "url:" + normalize_url(str(document["url"]))
    identity = "\n".join([
        str(document.get("title", "")).strip().lower(),
        document_text(document).strip().lower(),
    ])
    return "text:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ordered_document_union(full_traces: dict) -> List[dict]:
    """Union post-rerank documents from every search round in first-seen order."""
    documents: List[dict] = []
    positions: Dict[str, int] = {}
    for call in iter_search_calls(full_traces):
        for document in call.get("documents") or []:
            if not isinstance(document, dict):
                continue
            text = document_text(document)
            if not text:
                continue
            key = document_key(document)
            if key in positions:
                # Keep ordering stable but retain a richer later representation.
                index = positions[key]
                if len(text) > len(documents[index]["doc"]):
                    documents[index]["doc"] = text
                continue
            positions[key] = len(documents)
            documents.append({
                "original_id": len(documents) + 1,
                "doc": text,
                "title": document.get("title", ""),
                "url": document.get("url", ""),
            })
    return documents


def build_summary(row: dict, ranker: str) -> dict:
    original = row.get("original_data") or {}
    documents = ordered_document_union(row.get("full_traces") or {})
    example_id = str(row.get("example_id") or original.get("id") or "")
    return {
        "id": example_id,
        "data_source": original.get("data_source", "setwise_eval_long"),
        "question": original.get("question") or row.get("problem", ""),
        "answer": original.get("reference_answer") or original.get("answer", ""),
        "docs": documents,
        "hybrid_rubrics": original.get("hybrid_rubrics", {}),
        "selected_docs": {
            ranker: [document["original_id"] for document in documents]
        },
        "ranker_doc_counts": {ranker: len(documents)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_file")
    parser.add_argument("output_file")
    parser.add_argument("--ranker", required=True)
    args = parser.parse_args()

    rows = [build_summary(row, args.ranker) for row in load_jsonl(args.trajectory_file)]
    missing = [row["id"] for row in rows if not row["hybrid_rubrics"]]
    if missing:
        raise SystemExit(
            f"[ERROR] {len(missing)} rows lack hybrid_rubrics; examples: {missing[:3]}"
        )
    empty = [row["id"] for row in rows if not row["docs"]]
    if empty:
        print(f"[WARN] {len(empty)} queries have no selected documents: {empty[:3]}")
    save_jsonl(rows, args.output_file)
    print(f"Saved {len(rows)} queries -> {args.output_file}")


if __name__ == "__main__":
    main()
