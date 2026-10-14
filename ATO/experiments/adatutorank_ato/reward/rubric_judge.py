"""AdaTutoRank-ATO rubric reward.

Invalid identifier output receives -1. Valid selections are judged on the nine
hierarchical rubric dimensions by an OpenAI-compatible reward-model endpoint.
RAG and deep-research use separate prompts; deep-research additionally receives
the query intent stored in ``extra_info.thinking``.

verl integration:
    custom_reward_function.path="experiments/adatutorank_ato/reward/rubric_judge.py"
    custom_reward_function.name="compute_score"
    +val_reward_function.path="experiments/adatutorank_ato/reward/rubric_judge.py"
    +val_reward_function.name="compute_score_val"

Required endpoint variables are RM_BASE_URL, RM_API_KEY, and RM_MODEL.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------- Paths ----------------
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_ATO_DIR = os.path.dirname(_SELF_DIR)
_PROMPT_DIR = os.environ.get("RUBRIC_PROMPT_DIR", os.path.join(_ATO_DIR, "prompt"))
_FORMAT_PATH = os.environ.get("RUBRIC_FORMAT_PATH", os.path.join(_SELF_DIR, "format.py"))

# The OpenAI-compatible reward client is kept separate from the judge logic.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_ATO_DIR))
_REWARD_CLIENT_DIR = os.environ.get(
    "REWARD_CLIENT_DIR",
    os.path.join(_PROJECT_ROOT, "scripts", "reward_model"),
)
_RM_CONFIG_PATH = os.environ.get("RUBRIC_RM_CONFIG", os.path.join(_REWARD_CLIENT_DIR, "config.yaml"))

JUDGE_CALL_TIMEOUT = int(os.environ.get("RUBRIC_JUDGE_CALL_TIMEOUT", "900"))
FORMAT_FAIL_REWARD = -1.0
FAIL_SCORE = float(os.environ.get("RUBRIC_JUDGE_FAIL_SCORE", "0.0"))


MIN_SELECTED = int(os.environ.get("RUBRIC_MIN_SELECTED", "1"))
MAX_SELECTED = int(os.environ.get("RUBRIC_MAX_SELECTED", "10"))


LEVELS = {
    "Doc-Level":    ["Relevance", "Authenticity", "Quality"],
    "Set-Level":    ["Complementarity", "Redundancy", "Conflict"],
    "Global-Level": ["Completeness", "Density", "Reachability"],
}
DOC_DIMS = set(LEVELS["Doc-Level"])
ALL_DIMS = [d for dims in LEVELS.values() for d in dims]
_TIER_KEYS = ("doc_level_rubrics", "set_level_rubrics", "global_level_rubrics")
VALID_SCORES = set(range(0, 11))
DEFAULT_WEIGHT = 3


def is_doc_level(dim: str) -> bool:
    return dim in DOC_DIMS




def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_fmt_mod = None
_load_lock = threading.Lock()


def _check_format(text: str, num_passages: int, mode: str):
    global _fmt_mod
    if _fmt_mod is None:
        with _load_lock:
            if _fmt_mod is None:
                _fmt_mod = _load_module(_FORMAT_PATH, "rubric_format")
    return _fmt_mod.check_format(text, num_passages, mode)


# ---------------- RM prompt loading ----------------

_PROMPT_CACHE: dict = {}


def _load_prompt_template(deepresearch: bool = False) -> str:
    """Load the fixed no-reason judge prompt used by ATO."""
    key = "deepresearch" if deepresearch else "rag"
    if key in _PROMPT_CACHE:
        return _PROMPT_CACHE[key]
    import yaml
    name = "deep_research.yaml" if deepresearch else "rag.yaml"
    path = os.path.join(_PROMPT_DIR, "reward_judge", name)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "user_message" not in data:
        raise ValueError(f"{path} must contain a user_message key")
    tmpl = data["user_message"]
    required = ["{query}", "{docs_str}", "{rubrics_str}"]
    if deepresearch:
        required.append("{query_intent}")
    for ph in required:
        if ph not in tmpl:
            raise ValueError(f"Prompt template {path} is missing placeholder {ph}")
    _PROMPT_CACHE[key] = tmpl
    return tmpl




def _dim_entries(rubrics: dict, dim: str) -> list:
    if not isinstance(rubrics, dict):
        return []
    for tier in _TIER_KEYS:
        sec = rubrics.get(tier)
        if isinstance(sec, dict) and isinstance(sec.get(dim), list):
            return sec[dim]
    v = rubrics.get(dim)
    if isinstance(v, list):
        return v
    if isinstance(v, (str, dict)):
        return [v]
    return []


def _entry_text(e) -> str:
    if isinstance(e, dict):
        t = e.get("description") or e.get("rubric") or e.get("question") or e.get("text") or ""
    else:
        t = str(e)
    return t.strip()


def _entry_weight(e) -> float:
    w = e.get("weight") if isinstance(e, dict) else None
    try:
        w = float(w)
    except (TypeError, ValueError):
        w = None
    return w if (w is not None and w > 0) else DEFAULT_WEIGHT


def format_rubrics(rubrics: dict) -> str:
    """Render the nine rubric dimensions for the judge prompt."""
    lines = []
    for level, dims in LEVELS.items():
        for dim in dims:
            entries = _dim_entries(rubrics, dim)
            texts = [t for t in (_entry_text(e) for e in entries) if t]
            if not texts:
                continue
            gran = ("score each document (per_doc)" if is_doc_level(dim)
                    else "one overall score for the whole doc set (set_level)")
            lines.append(f"[{dim}] ({level}, {gran})")
            for j, t in enumerate(texts, 1):
                lines.append(f"  {dim} #{j}: {t}")
            lines.append("")
    return "\n".join(lines).strip()


def _rubric_weights(rubrics: dict, dim: str) -> List[float]:
    return [_entry_weight(e) for e in _dim_entries(rubrics, dim)]




def _doc_content(item) -> str:
    if isinstance(item, dict):
        for k in ("content", "doc", "text"):
            v = item.get(k)
            if v not in (None, ""):
                return str(v)
        return ""
    return str(item)


def format_docs(initial_list: list, selected_idxs: List[int]) -> str:
    """Render selected zero-based document indices as one-based labels."""
    lines = []
    for idx in selected_idxs:
        if idx < 0 or idx >= len(initial_list):
            continue
        lines.append(f"[{idx + 1}] {_doc_content(initial_list[idx]).strip()}")
    return "\n\n".join(lines)


# ---------------- Reward-model client ----------------

_judge = None
_judge_lock = threading.Lock()


def _apply_rm_env_overrides(cfg) -> None:
    """Apply the portable OpenAI-compatible RM endpoint settings."""

    def _get(name: str) -> str:
        v = os.environ.get(name, "")
        return v.strip() if isinstance(v, str) else ""

    cfg.backend.type = "openai"
    client = cfg.backend.openai
    client.base_url = _get("RM_BASE_URL") or client.get("base_url")
    client.api_key = _get("RM_API_KEY") or client.get("api_key", "")
    client.model = _get("RM_MODEL") or client.get("model")
    client.thinking = _get("RM_THINKING").lower() in {"1", "true", "yes", "on"}
    cfg.server.max_concurrent_llm_calls = int(
        _get("RM_MAX_CONCURRENCY") or cfg.server.get("max_concurrent_llm_calls", 256)
    )


def _load_rm_config():
    import sys
    if _REWARD_CLIENT_DIR not in sys.path:
        sys.path.insert(0, _REWARD_CLIENT_DIR)
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_RM_CONFIG_PATH)
    _apply_rm_env_overrides(cfg)
    return cfg


def _create_llm(cfg):
    from models.openai_client import OpenAIRewardClient

    return OpenAIRewardClient(cfg.backend.openai)


class JudgeBackend:
    """Thread-safe synchronous facade over the async RM client."""

    def __init__(self):
        import asyncio

        self.cfg = _load_rm_config()
        bconf = self.cfg.backend.openai
        self.thinking = bool(bconf.get("thinking", False))
        self.max_concurrent = int(self.cfg.server.get("max_concurrent_llm_calls", 256))
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._init(), self._loop)
        fut.result()

    def _run_loop(self):
        import asyncio

        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init(self):
        import asyncio

        self.model = _create_llm(self.cfg)
        self.sem = asyncio.Semaphore(self.max_concurrent)

    async def _score_coro(self, prompt: str):
        async with self.sem:
            return await self.model.call(prompt, thinking=self.thinking)

    def score(self, prompt: str) -> dict:
        import asyncio

        fut = asyncio.run_coroutine_threadsafe(self._score_coro(prompt), self._loop)
        return fut.result(timeout=JUDGE_CALL_TIMEOUT)


def _get_judge() -> JudgeBackend:
    global _judge
    if _judge is None:
        with _judge_lock:
            if _judge is None:
                _judge = JudgeBackend()
    return _judge




def _to_score(x) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        v = int(round(x))
    else:
        m = re.search(r"\d+", str(x))
        if not m:
            return None
        v = int(m.group(0))
    return v if v in VALID_SCORES else None


def extract_score_obj(resp: dict):
    """Extract the optional reasoning and score object from a parsed response."""
    if not isinstance(resp, dict) or resp.get("parse_error"):
        return None, None
    reason = resp.get("reason")
    base = resp.get("score")
    score = dict(base) if isinstance(base, dict) else {}
    for k, v in resp.items():
        if k in ALL_DIMS and k not in score:
            score[k] = v
    score = score or None
    if not isinstance(reason, str):
        reason = "" if reason is None else json.dumps(reason, ensure_ascii=False)
    return reason, score


def _rubric_score_10(dim: str, rub: dict) -> Optional[float]:
    if not isinstance(rub, dict):
        return None
    if is_doc_level(dim):
        vals = []
        for pd in rub.get("per_doc", []) or []:
            if isinstance(pd, dict):
                s = _to_score(pd.get("score"))
                if s is not None:
                    vals.append(s)
        return sum(vals) / len(vals) if vals else None
    sl = rub.get("set_level")
    if isinstance(sl, dict):
        return _to_score(sl.get("score"))
    if "score" in rub:
        return _to_score(rub.get("score"))
    return None


def weighted_reward(score_obj: dict, rubrics: dict) -> Optional[float]:
    """Aggregate non-null rubric scores into a weighted 0–10 reward."""
    if not isinstance(score_obj, dict):
        return None
    num = den = 0.0
    for dim in ALL_DIMS:
        entry = score_obj.get(dim)
        if not isinstance(entry, dict):
            continue
        rubs = entry.get("rubrics", []) or []
        weights = _rubric_weights(rubrics, dim)
        for k, rub in enumerate(rubs):
            s = _rubric_score_10(dim, rub)
            if s is None:
                continue
            w = weights[k] if k < len(weights) else DEFAULT_WEIGHT
            num += w * s
            den += w
    if den <= 0:
        return None
    return round(num / den, 3)


def rubric_breakdown(score_obj: dict, rubrics: dict) -> dict:
    """Return per-rubric scores and weights for rollout diagnostics."""
    out: dict = {}
    if not isinstance(score_obj, dict):
        return out
    for dim in ALL_DIMS:
        entry = score_obj.get(dim)
        if not isinstance(entry, dict):
            continue
        rubs = entry.get("rubrics", []) or []
        if not rubs:
            continue
        weights = _rubric_weights(rubrics, dim)
        items = []
        for k, rub in enumerate(rubs):
            w = weights[k] if k < len(weights) else DEFAULT_WEIGHT
            items.append({"score": _rubric_score_10(dim, rub), "weight": w})
        out[dim] = items
    return out




def _fmt_mode() -> str:
    return os.environ.get("RUBRIC_FMT_MODE", "wo_cot")


def _score_impl(solution_str: str, extra_info: dict | None, ground_truth, val: bool) -> dict:
    extra_info = extra_info or {}
    initial_list = extra_info.get("initial_list") or []
    rubrics = extra_info.get("rubrics") or {}
    query = extra_info.get("query") or extra_info.get("question") or ""

    result = {
        
        
        
        "score": FORMAT_FAIL_REWARD, "format_valid": 0, "incorrect_format": 1,
        "reward_raw": None,
        "len(pred)": 0, "reason": "", "query": query,
        "scenario": extra_info.get("scenario", ""),
        
        
        "rubric_scores": "",
    }
    if val:
        result["eval_system"] = "rubric_judge"

    num_docs = len(initial_list)
    if num_docs == 0:
        result["reason"] = "missing initial_list"
        return result

    
    is_valid, idx0, freason = _check_format(solution_str, num_docs, _fmt_mode())
    if not is_valid:
        result["reason"] = f"format_invalid:{freason}"
        return result
    result["format_valid"] = 1
    result["incorrect_format"] = 0
    selected = idx0 or []
    result["len(pred)"] = len(selected)

    if not selected:
        result["score"] = 0.0
        result["reason"] = "empty_selection"
        return result
    if len(selected) < MIN_SELECTED or len(selected) > MAX_SELECTED:
        result["score"] = 0.0
        result["reason"] = f"pred length not in [{MIN_SELECTED},{MAX_SELECTED}]"
        return result

    rubrics_str = format_rubrics(rubrics)
    if not rubrics_str:
        result["score"] = FAIL_SCORE
        result["reason"] = "no_rubrics"
        return result

    docs_str = format_docs(initial_list, selected)
    
    
    is_deepresearch = str(extra_info.get("scenario", "")).lower() == "deepresearch"
    query_intent = str(extra_info.get("thinking") or "").strip()
    if is_deepresearch and not query_intent:
        
        is_deepresearch = False
    template = _load_prompt_template(deepresearch=is_deepresearch)
    if is_deepresearch:
        prompt = template.format(query=query, query_intent=query_intent,
                                 docs_str=docs_str, rubrics_str=rubrics_str)
    else:
        prompt = template.format(query=query, docs_str=docs_str, rubrics_str=rubrics_str)

    
    try:
        resp = _get_judge().score(prompt)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"judge call failed: {e}")
        resp = {"parse_error": True}
    reason, score_obj = extract_score_obj(resp)

    if score_obj is None:
        result["score"] = FAIL_SCORE
        result["reason"] = "judge_parse_failed"
        return result

    r = weighted_reward(score_obj, rubrics)
    result["reward_raw"] = r
    result["score"] = float(r) if r is not None else FAIL_SCORE
    result["reason"] = reason or "rubric_judge_scored"
    
    try:
        result["rubric_scores"] = json.dumps(rubric_breakdown(score_obj, rubrics), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        result["rubric_scores"] = ""
    return result




def compute_score(data_source: str, solution_str: str, ground_truth=None, extra_info: dict | None = None) -> dict:
    return _score_impl(solution_str, extra_info, ground_truth, val=False)


def compute_score_val(data_source: str, solution_str: str, ground_truth=None, extra_info: dict | None = None) -> dict:
    return _score_impl(solution_str, extra_info, ground_truth, val=True)
