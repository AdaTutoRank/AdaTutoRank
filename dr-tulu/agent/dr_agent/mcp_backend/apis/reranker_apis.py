from typing import Any, Dict, List, Optional, Tuple

import cohere
import requests
from pydantic import BaseModel
import json
import os
import re
import threading

import yaml

from dr_agent.run_config import reranker_prompt_file, truncation_tokenizer_path

from ..cache import cached

# Hard-coded token-length limits (kept simple, not user-configurable).
_BGE_TOTAL_MAX_TOKENS = 510            # BGE max_length=512, reserve 2 for [CLS]/[SEP]
_BGE_QUERY_MAX_TOKENS = 96             # cap query first, give remainder to doc
_BGE_DOC_MAX_TOKENS = 400              # hard cap on doc tokens (also bounded by remainder)
_HF_POINTWISE_DOC_MAX_TOKENS = 400     # monoT5/RankT5 max_length=512, leave room for query+template
_LISTWISE_PROMPT_MAX_TOKENS = 3800     # RankVicuna/RankZephyr max_model_len=4096, leave 256 for output
_LISTWISE_DOC_MIN_TOKENS = 16          # don't shrink each doc below this when balancing

_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_tokenizer_cached(model_path: str):
    """Process-level tokenizer cache (used purely for length truncation)."""
    tok = _TOKENIZER_CACHE.get(model_path)
    if tok is not None:
        return tok
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    _TOKENIZER_CACHE[model_path] = tok
    return tok


def _truncate_text_by_tokens(text: str, tokenizer, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


class Document(BaseModel):
    """Document model containing the text content."""

    text: str


class RerankResultItem(BaseModel):
    """Reranker result item containing index, relevance score, and document."""

    index: int
    relevance_score: float
    document: Document


class RerankerResult(BaseModel):
    """Reranker result containing a list of RerankResultItem objects."""

    method: str
    model_name: str
    results: List[RerankResultItem]


def vllm_hosted_pointwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    tokenizer_path: Optional[str] = None,
) -> RerankerResult:
    """VLLM-hosted pointwise reranker (Cohere rerank API protocol, e.g. BGE).

    tokenizer_path: tokenizer of the served model, used to cap the input at
    512 tokens; defaults to model_name.
    """
    if top_n == -1:
        top_n = len(documents)

    # Truncate both query and each doc so that query+doc+specials <= 512 (BGE max_length).
    # Critical: if the combined length exceeds 512, vLLM's CUDA kernel will assert and
    # poison the whole CUDA context (the server then needs a restart).
    truncated_query = query
    try:
        tok = _get_tokenizer_cached(tokenizer_path or model_name)
        # 1) cap query tokens
        truncated_query = _truncate_text_by_tokens(query, tok, _BGE_QUERY_MAX_TOKENS)
        q_len = len(tok.encode(truncated_query, add_special_tokens=False))
        # 2) doc gets the remaining budget (also bounded by hard cap)
        doc_budget = max(64, min(_BGE_DOC_MAX_TOKENS, _BGE_TOTAL_MAX_TOKENS - q_len))
        documents_for_api = [
            _truncate_text_by_tokens(d, tok, doc_budget) for d in documents
        ]
    except Exception as e:
        print(f"[vllm_hosted_pointwise_reranker] tokenizer load failed: {e}; using raw docs")
        documents_for_api = documents

    try:
        client = cohere.ClientV2("sk-fake-key", base_url=api_url)
        rerank_result = client.rerank(model=model_name, query=truncated_query, documents=documents_for_api)
        sorted_results = sorted(
            rerank_result.results, key=lambda x: x.relevance_score, reverse=True
        )
        top_results = [
            RerankResultItem(
                index=result.index,
                relevance_score=result.relevance_score,
                document=Document(text=result.document["text"]),
            )
            for result in sorted_results[:top_n]
        ]
    except Exception as e:
        # Fallback: original order, equal scores
        print(f"[vllm_hosted_pointwise_reranker] failed: {e}; returning original order")
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=1.0 - i * 1e-3,
                document=Document(text=doc),
            )
            for i, doc in enumerate(documents[:top_n])
        ]

    return RerankerResult(
        method="vllm_hosted_pointwise",
        model_name=model_name,
        results=top_results,
    )


# Backward-compat alias
vllm_hosted_reranker = vllm_hosted_pointwise_reranker


# ==================== Listwise Reranker ====================

# The vLLM listwise variant keeps its original inline prompt (below, inside
# ``vllm_hosted_listwise_reranker``) for backward compatibility.
_VLLM_LISTWISE_RANKING_PROMPT = """I will provide you with {num_docs} passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}

{formatted_docs}

Search Query: {query}

Rank the {num_docs} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [] > ... e.g., [4] > [2] > [5] > [1] > [3]. Only respond with the ranking results, do not say any word or explain."""


class ListwiseRerankerResult(BaseModel):
    """Result from listwise LLM reranker, containing raw ranking output."""

    method: str
    model_name: str
    raw_ranking_output: str  # Raw LLM output, e.g. "[2] > [5] > [1] > ..."
    # Optional fields populated only by the rubrics-guided variant. Kept
    # Optional so existing callers / serialized payloads remain compatible.
    reference_answer: Optional[str] = None
    rubrics_raw: Optional[str] = None


def vllm_hosted_listwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 1024,
) -> ListwiseRerankerResult:
    """
    Listwise reranker using a vLLM-hosted LLM to rank documents.

    The LLM receives all documents at once and outputs a ranked list of doc indices.

    Args:
        query: The query string to rank documents against
        documents: List of document strings to rerank
        top_n: Number of top documents to return (passed for client-side use)
        model_name: Name of the LLM model
        api_url: Base URL for the vLLM API (e.g. "http://localhost:30003/v1")
        max_tokens: Max tokens for LLM generation

    Returns:
        ListwiseRerankerResult with raw LLM ranking output
    """
    # Format documents with numbered identifiers
    formatted_docs = "\n\n".join(
        f"[{i + 1}] {doc}" for i, doc in enumerate(documents)
    )

    prompt = _VLLM_LISTWISE_RANKING_PROMPT.format(
        num_docs=len(documents),
        query=query,
        formatted_docs=formatted_docs,
    )
    # Disable thinking mode for models like Qwen3 to get clean ranking output
    # Use a long timeout since listwise ranking prompts can be very large
    response = requests.post(
        f"{api_url}/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            # "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=600,
    )
    response.raise_for_status()
    result = response.json()
    raw_output = result["choices"][0]["message"]["content"].strip()

    return ListwiseRerankerResult(
        method="vllm_hosted_listwise",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )




# ==================== HF Pointwise Reranker (monoT5 / RankT5 / RankLLaMA) ====================

# Process-level cache so the model is loaded once per MCP server.
_HF_RERANKER_CACHE: Dict[str, Any] = {}
_HF_RERANKER_LOCK = threading.Lock()


# Tokenizer used to length-bound documents passed to rerankers. We use
# Qwen3-8B's tokenizer regardless of the actual reranker model because we
# want a *consistent*, language-aware token count across all reranker
# variants (BGE / monoT5 / RankT5 / RankVicuna / RubricRanker / ...).
# Plain ``str.split()`` was previously used, but it silently no-ops on CJK
# text (no whitespace -> 1 "word"), so configured ``doc_max_words`` had no
# effect on Chinese pages. The tokenizer is `truncation_tokenizer` in
# configs/models.yaml.


def _qwen3_tokenizer_path():
    """Resolve the truncation tokenizer path, or None if unavailable."""
    try:
        return truncation_tokenizer_path()
    except Exception as e:
        print(f"[_qwen3_tokenizer_path] could not resolve configs/models.yaml: {e}")
        return None


def _get_qwen3_truncation_tokenizer():
    """Return the cached Qwen3-8B tokenizer, or None if loading fails."""
    path = _qwen3_tokenizer_path()
    if not path:
        return None
    tok = _TOKENIZER_CACHE.get(path)
    if tok is not None:
        return tok if tok is not False else None
    try:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(path, use_fast=False)
        _TOKENIZER_CACHE[path] = tok
        return tok
    except Exception as e:
        print(
            f"[_get_qwen3_truncation_tokenizer] failed to load "
            f"{path}: {e}; falling back to whitespace split"
        )
        # Sentinel so we don't retry on every call.
        _TOKENIZER_CACHE[path] = False
        return None


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate ``text`` to at most ``max_words`` Qwen3-8B tokens.

    The parameter is still named ``max_words`` for backward compatibility
    with the existing call sites and the ``doc_max_words`` config knob;
    the unit semantically is *Qwen3-8B tokens* (which is a much better
    proxy for prompt length than whitespace-delimited words, and which
    correctly bounds CJK text).

    Falls back to whitespace truncation if the tokenizer cannot be loaded
    or fails on this particular input.
    """
    if max_words is None or max_words <= 0 or not text:
        return text
    tok = _get_qwen3_truncation_tokenizer()
    if tok is not None:
        try:
            return _truncate_text_by_tokens(text, tok, max_words)
        except Exception as e:
            print(f"[_truncate_words] tokenizer encode/decode failed: {e}; falling back to whitespace split")
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def _get_or_load_hf_reranker(
    model_path: str, model_kind: str, device: str, base_model_path: Optional[str] = None
):
    """Load (model, tokenizer) once per process, keyed by model_path."""
    with _HF_RERANKER_LOCK:
        cached = _HF_RERANKER_CACHE.get(model_path)
        if cached is not None:
            return cached

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            T5ForConditionalGeneration,
            T5Tokenizer,
        )

        kind = (model_kind or "").lower()
        if kind == "monot5":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        elif kind == "rankt5":
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device).eval()
            # rankt5 ckpt typically lacks a usable tokenizer; reuse monoT5's spiece if available,
            # otherwise fall back to the same path.
            try:
                tokenizer = T5Tokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        elif kind == "rankllama":
            # RankLLaMA ships as a LoRA adapter over LLaMA-2-7B with a
            # single-logit classification head. The adapter is merged in so
            # scoring runs at plain-model speed.
            from peft import PeftConfig, PeftModel

            # The adapter's PEFT config names the base model by hub id
            # ("meta-llama/Llama-2-7b-hf"), which would need network access
            # and a gated-repo token. Prefer the local copy when given.
            base_model_name = base_model_path
            if not base_model_name:
                peft_config = PeftConfig.from_pretrained(model_path)
                base_model_name = peft_config.base_model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
            # Move the base to the target device before merging. Merging on CPU
            # in float16 walks every layer through unoptimised CPU fp16 kernels
            # and takes minutes for a 7B, all while holding
            # _HF_RERANKER_LOCK, which stalls every other tool call in the MCP
            # server until the client's timeout fires.
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=1, torch_dtype=dtype
            ).to(device)
            model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
            model.config.pad_token_id = tokenizer.pad_token_id
            model = model.eval()
        else:
            raise ValueError(f"Unsupported HF reranker model_kind: {model_kind}")

        _HF_RERANKER_CACHE[model_path] = (model, tokenizer, kind)
        return _HF_RERANKER_CACHE[model_path]


def _score_rankllama(
    query: str,
    documents: List[str],
    model,
    tokenizer,
    device: str,
    doc_max_words: int,
    batch_size: int,
    max_length: int,
) -> List[float]:
    """Score (query, doc) pairs with RankLLaMA's single-logit head.

    The pair is fed as two sequences so the tokenizer's ``longest_first``
    truncation keeps the query intact when a document overflows
    ``max_length``.
    """
    import torch

    query_texts = [f"query: {query}"] * len(documents)
    doc_texts = [f"document: {_truncate_words(d, doc_max_words)}" for d in documents]

    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(query_texts), batch_size):
            tokenized = tokenizer(
                query_texts[start:start + batch_size],
                doc_texts[start:start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            logits = model(**tokenized).logits  # (batch, 1)
            scores.extend(logits.squeeze(-1).float().cpu().tolist())
    return scores


def hf_pointwise_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_path: str,
    model_kind: str,
    device: str = "cuda:0",
    doc_max_words: int = 0,
    batch_size: int = 8,
    max_length: int = 512,
    base_model_path: Optional[str] = None,
) -> RerankerResult:
    """HF in-process pointwise reranker for monoT5 / RankT5 / RankLLaMA."""
    if top_n == -1 or top_n is None:
        top_n = len(documents)

    try:
        import torch

        model, tokenizer, kind = _get_or_load_hf_reranker(
            model_path, model_kind, device, base_model_path
        )

        if kind == "rankllama":
            scores = _score_rankllama(
                query=query,
                documents=documents,
                model=model,
                tokenizer=tokenizer,
                device=device,
                doc_max_words=doc_max_words,
                batch_size=batch_size,
                max_length=max_length,
            )
            order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_n]
            return RerankerResult(
                method="hf_pointwise_rankllama",
                model_name=model_path,
                results=[
                    RerankResultItem(
                        index=i,
                        relevance_score=float(scores[i]),
                        document=Document(text=documents[i]),
                    )
                    for i in order
                ],
            )

        q = _truncate_text_by_tokens(query, tokenizer, _BGE_QUERY_MAX_TOKENS)
        docs_trunc = [
            _truncate_text_by_tokens(
                _truncate_words(d, doc_max_words), tokenizer, _HF_POINTWISE_DOC_MAX_TOKENS
            )
            for d in documents
        ]

        if kind == "monot5":
            patterns = [f"Query: {q} Document: {d} Relevant:" for d in docs_trunc]
        else:  # rankt5
            patterns = [f"Query: {q} Document: {d}" for d in docs_trunc]

        scores: List[float] = []
        with torch.no_grad():
            for i in range(0, len(patterns), batch_size):
                batch = patterns[i:i + batch_size]
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding="longest",
                    max_length=max_length,
                    truncation=True,
                ).to(device)
                outputs = model.generate(
                    **inputs,
                    max_length=2,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                first_step_logits = outputs.scores[0]  # (B, V)
                if kind == "monot5":
                    # token ids for "true" / "false"
                    s = torch.nn.functional.log_softmax(
                        first_step_logits[:, [1176, 6136]], dim=1
                    )[:, 0]
                else:  # rankt5: <extra_id_10> -> 32089
                    s = first_step_logits[:, 32089]
                scores.extend(s.float().cpu().tolist())

        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_n]
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=float(scores[i]),
                document=Document(text=documents[i]),
            )
            for i in order
        ]
        method = f"hf_pointwise_{kind}"
    except Exception as e:
        print(f"[hf_pointwise_reranker] failed: {e}; returning original order")
        top_results = [
            RerankResultItem(
                index=i,
                relevance_score=1.0 - i * 1e-3,
                document=Document(text=doc),
            )
            for i, doc in enumerate(documents[:top_n])
        ]
        method = f"hf_pointwise_{model_kind}_fallback"

    return RerankerResult(
        method=method,
        model_name=model_path,
        results=top_results,
    )


# ==================== Listwise Sliding-Window Reranker (RankVicuna / RankZephyr) ====================

_LISTWISE_PROMPT_CACHE: Dict[str, Dict[str, str]] = {}


def _load_listwise_prompt() -> Dict[str, str]:
    if "tpl" in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE["tpl"]
    path = reranker_prompt_file("listwise_reranker.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE["tpl"] = cfg
    return cfg


def _build_vicuna_prompt(system_message: str, user_content: str) -> str:
    # fastchat vicuna_v1.1 template (matches castorini/rank_vicuna_7b_v1)
    sys = system_message or ""
    return (
        f"{sys}\n\n"
        f"USER: {user_content}\n"
        f"ASSISTANT:"
    )


def _build_zephyr_prompt(system_message: str, user_content: str) -> str:
    # zephyr/mistral chat template
    sys = system_message or ""
    return (
        f"<|system|>\n{sys}</s>\n"
        f"<|user|>\n{user_content}</s>\n"
        f"<|assistant|>\n"
    )


def _build_listwise_user_content(query: str, documents: List[str], doc_max_words: int) -> Tuple[str, int]:
    cfg = _load_listwise_prompt()
    prefix = cfg.get("prefix", "").format(num=len(documents), query=query)
    body_tpl = cfg.get("body", "[{rank}] {candidate}\n")
    suffix = cfg.get("suffix", "").format(num=len(documents), query=query)

    parts = [prefix]
    for i, d in enumerate(documents):
        d_trunc = _truncate_words(d, doc_max_words)
        parts.append(body_tpl.format(rank=i + 1, candidate=d_trunc))
    parts.append(suffix)
    return "".join(parts), len(documents)


def _build_listwise_prompt_with_token_limit(
    query: str,
    window_docs: List[str],
    doc_max_words: int,
    system_message: str,
    model_kind: str,
    tokenizer,
) -> str:
    """Build a sliding-window prompt and shrink each doc until it fits in the model context.

    On any failure (tokenizer issues, cannot shrink enough), returns the un-shrunk prompt.
    """
    def _assemble(docs_for_window: List[str]) -> str:
        user_content, _ = _build_listwise_user_content(query, docs_for_window, 0)
        if (model_kind or "").lower() == "zephyr":
            return _build_zephyr_prompt(system_message, user_content)
        return _build_vicuna_prompt(system_message, user_content)

    # Word-level pre-truncate.
    docs = [_truncate_words(d, doc_max_words) for d in window_docs]

    if tokenizer is None:
        return _assemble(docs)

    try:
        prompt = _assemble(docs)
        n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if n_tokens <= _LISTWISE_PROMPT_MAX_TOKENS:
            return prompt

        # Iteratively halve each doc's per-doc token budget until prompt fits.
        # Start budget = average current doc tokens.
        doc_token_lists = [tokenizer.encode(d, add_special_tokens=False) for d in docs]
        budget = max(
            _LISTWISE_DOC_MIN_TOKENS,
            int(sum(len(t) for t in doc_token_lists) / max(1, len(docs))),
        )
        for _ in range(8):
            budget = max(_LISTWISE_DOC_MIN_TOKENS, budget // 2 if budget > _LISTWISE_DOC_MIN_TOKENS else budget)
            shrunk = [
                tokenizer.decode(t[:budget], skip_special_tokens=True) for t in doc_token_lists
            ]
            prompt = _assemble(shrunk)
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if n_tokens <= _LISTWISE_PROMPT_MAX_TOKENS:
                return prompt
            if budget <= _LISTWISE_DOC_MIN_TOKENS:
                break
        return prompt
    except Exception as e:
        print(f"[_build_listwise_prompt_with_token_limit] tokenizer error: {e}; using raw prompt")
        return _assemble(docs)


def _parse_permutation(raw: str) -> List[int]:
    """Parse "[2] > [5] > [1] > ..." into 0-based indices."""
    if not raw:
        return []
    cleaned = "".join(c if c.isdigit() else " " for c in raw)
    tokens = cleaned.split()
    out: List[int] = []
    seen = set()
    for t in tokens:
        try:
            v = int(t) - 1
        except ValueError:
            continue
        if v >= 0 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _apply_permutation_window(order: List[int], perm: List[int], start: int, end: int) -> List[int]:
    """Reorder order[start:end] according to perm (0-based, relative to window)."""
    window = order[start:end]
    n = len(window)
    perm = [p for p in perm if 0 <= p < n]
    seen = set(perm)
    perm = perm + [i for i in range(n) if i not in seen]
    new_window = [window[p] for p in perm]
    return order[:start] + new_window + order[end:]


def vllm_listwise_sliding_window_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    model_kind: str = "vicuna",
    window_size: int = 20,
    step: int = 10,
    max_tokens: int = 200,
    doc_max_words: int = 0,
    timeout: int = 600,
    tokenizer_path: Optional[str] = None,
) -> ListwiseRerankerResult:
    """Sliding-window listwise reranker for RankVicuna / RankZephyr.

    Calls vLLM's /v1/completions (raw text) with a model-specific chat template
    assembled in-process. Slides from the tail toward the head, step by step.
    """
    n = len(documents)
    if n == 0:
        return ListwiseRerankerResult(
            method="vllm_listwise_sliding_window",
            model_name=model_name,
            raw_ranking_output="",
        )

    cfg = _load_listwise_prompt()
    system_message = cfg.get("system_message", "")

    base_url = api_url.rstrip("/")
    completions_url = f"{base_url}/completions"

    order = list(range(n))
    last_raw = ""

    # Tokenizer used only for length-bounding the prompt.
    try:
        tokenizer_for_len = _get_tokenizer_cached(tokenizer_path or model_name)
    except Exception as e:
        print(f"[vllm_listwise_sliding_window_reranker] tokenizer load failed: {e}; skipping length cap")
        tokenizer_for_len = None

    try:
        windows_end = n
        windows_start = max(0, n - window_size)
        # iterate backwards
        while windows_end > 0:
            ws = max(0, windows_start)
            we = windows_end
            window_docs = [documents[order[i]] for i in range(ws, we)]
            prompt = _build_listwise_prompt_with_token_limit(
                query=query,
                window_docs=window_docs,
                doc_max_words=doc_max_words,
                system_message=system_message,
                model_kind=model_kind,
                tokenizer=tokenizer_for_len,
            )

            resp = requests.post(
                completions_url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"]
            last_raw = text
            perm = _parse_permutation(text)
            order = _apply_permutation_window(order, perm, ws, we)

            if ws == 0:
                break
            windows_end -= step
            windows_start -= step

        raw_output = " > ".join(f"[{i + 1}]" for i in order)
    except Exception as e:
        print(f"[vllm_listwise_sliding_window_reranker] failed: {e}; returning original order")
        raw_output = ""

    return ListwiseRerankerResult(
        method=f"vllm_listwise_sliding_window_{model_kind}",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Setwise Rank4Gen Reranker ====================

def _load_rank4gen_prompt() -> Dict[str, str]:
    cache_key = "rank4gen"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("rank4gen.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def vllm_setwise_rank4gen_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    downstream_model: str = "default",
    downstream_desc: str = "default",
    max_tokens: int = 4096,
    doc_max_words: int = 0,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Setwise reranker for Rank4Gen. Uses chat completions; English prompt only."""
    cfg = _load_rank4gen_prompt()
    system_message = cfg.get("system_message", "").format(
        downstream_model=downstream_model,
        description=downstream_desc,
    )
    user_tpl = cfg.get("user_message", "")
    mode_tag = cfg.get("mode_tag", "/index")

    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    context = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))
    user_content = user_tpl.format(num=len(documents), question=query, context=context) + "\n" + mode_tag

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_rank4gen_reranker] failed: {e}; returning empty output")
        raw_output = ""
        
    return ListwiseRerankerResult(
        method="vllm_setwise_rank4gen",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Setwise SETR Reranker ====================

def _load_setr_prompt() -> str:
    cache_key = "setr"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]["prompt"]
    path = reranker_prompt_file("setr.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg["prompt"]


def vllm_setwise_setr_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 4096,
    doc_max_words: int = 0,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Setwise reranker for SETR. English prompt only."""
    tpl = _load_setr_prompt()
    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))
    prompt = tpl.format(num=len(documents), question=query, context=context)

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_setr_reranker] failed: {e}; returning empty output")
        raw_output = ""


    return ListwiseRerankerResult(
        method="vllm_setwise_setr",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )

# ==================== Setwise RubricRanker Reranker ====================

def _load_rubricranker_prompt() -> Dict[str, str]:
    """Load (and cache) the RubricRanker prompt template from rubricranker.yaml."""
    cache_key = "rubricranker"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("rubricranker.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg

def vllm_setwise_rubricranker_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    search_intent: str = "",
    max_tokens: int = 1024,
    doc_max_words: int = 0,
    timeout: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> ListwiseRerankerResult:
    """Setwise reranker for the student RubricRanker model.

    The prompt is formatted with {num, question, query_intent, context}; the
    query intent is the agent's reasoning before it issued the query.

    The student model has the 5 rubrics baked into its system message and
    outputs a selection like "[3] [7] [9] ...". Sampling defaults follow
    Qwen-style chat: temperature=0.7, top_p=0.8, top_k=20, min_p=0.
    """
    cfg = _load_rubricranker_prompt()
    system_message = cfg.get("system_message", "")
    user_tpl = cfg.get("user_message", "")

    docs_trunc = [_truncate_words(d, 300) for d in documents]
    context = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))

    user_content = user_tpl.format(
        num=len(documents),
        question=query,
        query_intent=search_intent or "",
        context=context,
    )

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "extra_body": {
                    "top_k": top_k,
                    "min_p": min_p,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_rubricranker_reranker] failed: {e}; returning empty output")
        raw_output = ""
        
    return ListwiseRerankerResult(
        method="vllm_setwise_rubricranker",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )

# ==================== Listwise Single-Pass Reranker (RankQwen-style) ====================

def _load_listwise_singlepass_prompt() -> Dict[str, str]:
    """Load (and cache) the RankZephyr-style listwise prompt from
    ``configs/prompts/listwise_reranker.yaml``.
    Returns dict with keys: system_message, prefix, body, suffix.
    """
    cache_key = "listwise_singlepass::default"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("listwise_reranker.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def vllm_listwise_singlepass_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    max_tokens: int = 1024,
    doc_max_words: int = 0,
    timeout: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> ListwiseRerankerResult:
    """Single-pass listwise reranker (RankQwen-style).

    Uses the RankZephyr training prompt (``listwise_reranker.yaml``) but
    sends it as system+user via chat-completions (no sliding window).
    Output format: ``"[2] > [5] > [1] > ..."``; client-side parses and
    truncates to top_n.
    """
    cfg = _load_listwise_singlepass_prompt()
    system_message = cfg.get("system_message", "")
    prefix_tpl = cfg.get("prefix", "")
    body_tpl = cfg.get("body", "")
    suffix_tpl = cfg.get("suffix", "")

    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    num = len(docs_trunc)
    parts = [prefix_tpl.format(num=num, query=query)]
    for i, d in enumerate(docs_trunc):
        parts.append(body_tpl.format(rank=i + 1, candidate=d))
    parts.append(suffix_tpl.format(num=num, query=query))
    user_content = "".join(parts)

    base_url = api_url.rstrip("/")
    chat_url = f"{base_url}/chat/completions"

    raw_output = ""
    try:
        resp = requests.post(
            chat_url,
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "extra_body": {
                    "top_k": top_k,
                    "min_p": min_p,
                },
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_listwise_singlepass_reranker] failed: {e}; returning empty output")
        raw_output = ""

    return ListwiseRerankerResult(
        method="vllm_listwise_singlepass",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Reasoning Listwise Rerankers (Rearank / ReasonRank) ====================

# RankVicuna/RankZephyr above hit /v1/completions with a hand-assembled chat
# template. Rearank and ReasonRank instead post ``messages`` to
# /v1/chat/completions and let the server apply the model's own template.
# Both emit a <think> block before the ranking, so the answer has to be
# carved out before the identifiers are parsed -- feeding the whole response
# to _parse_permutation would pick up digits from the reasoning.

_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_ANSWER_RE = re.compile(
    r"<think>.*?</think>\s*<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
)


def _sliding_window_chat_rerank(
    documents: List[str],
    build_messages,
    extract_answer,
    model_name: str,
    api_url: str,
    window_size: int,
    step: int,
    payload_extra: Dict[str, Any],
    timeout: int,
    log_prefix: str,
) -> str:
    """Slide a chat-based listwise reranker from the tail of the list to the head.

    ``build_messages(window_docs)`` returns the chat messages for one window
    and ``extract_answer(raw)`` narrows a raw completion down to the text
    holding the identifiers. Returns the final order as ``"[2] > [5] > [1]"``,
    or an empty string if any window call failed (callers treat that as an
    error and fall back to the original search order).
    """
    n = len(documents)
    if n == 0:
        return ""

    chat_url = f"{api_url.rstrip('/')}/chat/completions"
    order = list(range(n))

    try:
        windows_end = n
        windows_start = max(0, n - window_size)
        # iterate backwards
        while windows_end > 0:
            ws = max(0, windows_start)
            we = windows_end
            window_docs = [documents[order[i]] for i in range(ws, we)]

            resp = requests.post(
                chat_url,
                json={
                    "model": model_name,
                    "messages": build_messages(window_docs),
                    **payload_extra,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]

            perm = _parse_permutation(extract_answer(raw))
            order = _apply_permutation_window(order, perm, ws, we)

            if ws == 0:
                break
            windows_end -= step
            windows_start -= step
    except Exception as e:
        print(f"[{log_prefix}] failed: {e}; returning original order")
        return ""

    return " > ".join(f"[{i + 1}]" for i in order)


# ==================== Rearank ====================

def _load_rearank_prompt() -> Dict[str, str]:
    cache_key = "rearank"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("rearank.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def _truncate_chars(text: str, max_chars: int) -> str:
    """Collapse whitespace and hard-truncate to ``max_chars`` characters.

    Rearank budgets its candidates in characters rather than tokens, so this
    deliberately bypasses the shared token-based ``_truncate_words``.
    """
    text = " ".join((text or "").split())
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


def _extract_rearank_answer(raw: str) -> str:
    """Narrow a Rearank completion down to the ranking text."""
    match = _ANSWER_BLOCK_RE.search(raw or "")
    if match:
        return match.group(1).strip()
    if "<think>" in (raw or ""):
        return raw.split("</think>")[-1]
    return raw or ""


def vllm_listwise_rearank_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    window_size: int = 20,
    step: int = 10,
    max_tokens: int = 2048,
    max_passage_chars: int = 400,
    enable_thinking: Optional[bool] = None,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Reasoning listwise sliding-window reranker for Rearank.

    Rearank hands the candidates over one per turn, each acknowledged by a
    synthetic assistant reply, and only then asks for the ranking; the turn
    sequence is described in ``rearank.yaml``. ``enable_thinking`` is only
    forwarded when set, since the released 7B checkpoint is Qwen2.5-based
    and its template does not accept the flag.
    """
    cfg = _load_rearank_prompt()

    def build_messages(window_docs: List[str]) -> List[Dict[str, str]]:
        num = len(window_docs)
        messages = [
            {"role": "system", "content": cfg.get("system_message", "")},
            {"role": "user", "content": cfg.get("instruction", "").format(query=query, num=num)},
            {"role": "assistant", "content": cfg.get("instruction_ack", "")},
        ]
        for rank, doc in enumerate(window_docs, 1):
            candidate = _truncate_chars(
                doc.replace("Title: Content: ", "").strip(), max_passage_chars
            )
            messages.append(
                {"role": "user", "content": cfg.get("passage", "").format(rank=rank, candidate=candidate)}
            )
            messages.append(
                {"role": "assistant", "content": cfg.get("passage_ack", "").format(rank=rank)}
            )
        messages.append(
            {"role": "user", "content": cfg.get("final_request", "").format(query=query)}
        )
        return messages

    payload_extra: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_tokens": max_tokens,
    }
    if enable_thinking is not None:
        payload_extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    raw_output = _sliding_window_chat_rerank(
        documents=documents,
        build_messages=build_messages,
        extract_answer=_extract_rearank_answer,
        model_name=model_name,
        api_url=api_url,
        window_size=window_size,
        step=step,
        payload_extra=payload_extra,
        timeout=timeout,
        log_prefix="vllm_listwise_rearank_reranker",
    )

    return ListwiseRerankerResult(
        method="vllm_listwise_rearank",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== ReasonRank ====================

def _load_reasonrank_prompt() -> Dict[str, str]:
    cache_key = "reasonrank"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("reasonrank.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def _replace_bracketed_numbers(text: str) -> str:
    """Rewrite ``[7]`` as ``(7)`` inside candidate text.

    Web pages are full of bracketed citation markers, which would otherwise
    be indistinguishable from the ``[rank]`` identifiers the model must
    return.
    """
    return re.sub(r"\[(\d+)\]", r"(\1)", text or "")


def _extract_reasonrank_answer(raw: str) -> str:
    """Narrow a ReasonRank completion down to the ranking text."""
    raw = raw or ""
    match = _THINK_ANSWER_RE.search(raw)
    if match:
        return match.group(1).strip()
    if "<answer>" in raw.lower():
        # Reasoning ran long and the <think> block was never closed.
        tail = re.split(r"<answer>", raw, flags=re.IGNORECASE)[-1]
        return re.split(r"</answer>", tail, flags=re.IGNORECASE)[0]
    return raw


def vllm_listwise_reasonrank_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    window_size: int = 20,
    step: int = 10,
    max_tokens: int = 3172,
    doc_max_words: int = 100,
    timeout: int = 600,
) -> ListwiseRerankerResult:
    """Reasoning listwise sliding-window reranker for ReasonRank.

    Single system+user turn per window (unlike Rearank), and the reasoning
    budget is large: the model spends most of ``max_tokens`` inside <think>
    before emitting <answer>[2] > [1]</answer>.
    """
    cfg = _load_reasonrank_prompt()
    prefix_tpl = cfg.get("prefix", "")
    body_tpl = cfg.get("body", "")
    suffix_tpl = cfg.get("suffix", "")
    clean_query = _replace_bracketed_numbers(query).strip()

    def build_messages(window_docs: List[str]) -> List[Dict[str, str]]:
        num = len(window_docs)
        parts = [prefix_tpl.format(num=num, query=clean_query), "\n"]
        for rank, doc in enumerate(window_docs, 1):
            candidate = _replace_bracketed_numbers(_truncate_words(doc, doc_max_words))
            parts.append(body_tpl.format(rank=rank, candidate=candidate))
        parts.append(suffix_tpl.format(num=num, query=clean_query))
        return [
            {"role": "system", "content": cfg.get("system_message", "")},
            {"role": "user", "content": "".join(parts)},
        ]

    raw_output = _sliding_window_chat_rerank(
        documents=documents,
        build_messages=build_messages,
        extract_answer=_extract_reasonrank_answer,
        model_name=model_name,
        api_url=api_url,
        window_size=window_size,
        step=step,
        payload_extra={"temperature": 0.0, "max_tokens": max_tokens},
        timeout=timeout,
        log_prefix="vllm_listwise_reasonrank_reranker",
    )

    return ListwiseRerankerResult(
        method="vllm_listwise_reasonrank",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )


# ==================== Setwise Ada TutoRank Reranker ====================

def _load_adatutorank_prompt() -> Dict[str, str]:
    """Load (and cache) the Ada TutoRank prompt template from adatutorank.yaml."""
    cache_key = "adatutorank"
    if cache_key in _LISTWISE_PROMPT_CACHE:
        return _LISTWISE_PROMPT_CACHE[cache_key]
    path = reranker_prompt_file("adatutorank.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _LISTWISE_PROMPT_CACHE[cache_key] = cfg
    return cfg


def vllm_setwise_adatutorank_reranker(
    query: str,
    documents: List[str],
    top_n: int,
    model_name: str,
    api_url: str,
    search_intent: str = "",
    max_tokens: int = 1024,
    doc_max_words: int = 300,
    timeout: int = 600,
    temperature: float = 0.0,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
) -> ListwiseRerankerResult:
    """Setwise reranker for the Ada TutoRank student model.

    Same shape as ``vllm_setwise_rubricranker_reranker`` -- the model picks a
    subset of at most 10 identifiers and orders them -- but it judges the set
    against 9 rubric dimensions instead of 5, and its prompts are byte-aligned
    with the RL training data builder rather than with the serving YAML of the
    earlier student. The prompt is formatted with
    {num, question, query_intent, context}, the long-horizon template the
    checkpoint was trained on.

    Decoding defaults to greedy, matching the offline benchmark; the training
    rollouts used sampling, so pass ``temperature > 0`` to reproduce those.
    """
    cfg = _load_adatutorank_prompt()
    system_message = cfg.get("system_message", "")
    user_tpl = cfg.get("user_message", "")

    # Single-newline separator and verbatim document text: the training
    # builder joined passages this way, and reflowing them here would be a
    # train/serve mismatch.
    docs_trunc = [_truncate_words(d, doc_max_words) for d in documents]
    context = "\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs_trunc))

    # The training builder emitted "[QUERY INTENT]\n {thinking} \n\n", so the
    # intent carries one space on each side. Applied here rather than in the
    # YAML so an editor stripping trailing whitespace can't break it.
    user_content = user_tpl.format(
        num=len(documents),
        question=query,
        query_intent=f" {search_intent or ''} ",
        context=context,
    )

    chat_url = f"{api_url.rstrip('/')}/chat/completions"

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temperature > 0.0:
        # Greedy decoding ignores these, and some vLLM builds reject them
        # alongside temperature=0.
        payload.update({"top_p": top_p, "top_k": top_k, "min_p": min_p})

    raw_output = ""
    try:
        resp = requests.post(chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw_output = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[vllm_setwise_adatutorank_reranker] failed: {e}; returning empty output")
        raw_output = ""

    return ListwiseRerankerResult(
        method="vllm_setwise_adatutorank",
        model_name=model_name,
        raw_ranking_output=raw_output,
    )
