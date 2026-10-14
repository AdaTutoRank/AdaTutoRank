"""
Loading configs/rerankers.yaml, configs/weights.yaml and configs/prompts/.

The config folder defaults to ``<repo>/configs`` and can be moved with
``RANKIFY_CONFIG_DIR``.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

from rankify.utils.model_downloader import resolve_model_path

CONFIG_DIR = os.environ.get(
    "RANKIFY_CONFIG_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs"),
)
PROMPT_DIR = os.path.join(CONFIG_DIR, "prompts")


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> Dict[str, Any]:
    path = os.path.join(CONFIG_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class RerankerSpec:
    name: str
    display_name: str
    method: str
    set_selection: bool
    prompt_file: Optional[str]
    params: Dict[str, Any] = field(default_factory=dict)
    weights: Optional[Dict[str, Any]] = None


def list_rerankers() -> List[str]:
    return list(_load_yaml("rerankers.yaml"))


def get_reranker_spec(name: str) -> RerankerSpec:
    rerankers = _load_yaml("rerankers.yaml")
    if name not in rerankers:
        raise KeyError(f"Unknown reranker '{name}'. Available: {list(rerankers)}")
    cfg = rerankers[name]

    prompt_file = None
    if cfg.get("prompt"):
        prompt_file = os.path.join(PROMPT_DIR, cfg["prompt"])
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"Prompt file for '{name}' not found: {prompt_file}")

    weights = _load_yaml("weights.yaml").get("rerankers", {}).get(name)
    if cfg["method"] != "bm25" and weights is None:
        raise KeyError(f"configs/weights.yaml has no entry for reranker '{name}'")

    return RerankerSpec(
        name=name,
        display_name=cfg.get("display_name", name),
        method=cfg["method"],
        set_selection=bool(cfg.get("set_selection", False)),
        prompt_file=prompt_file,
        params=dict(cfg.get("params") or {}),
        weights=weights,
    )


def resolve_weights(entry: Dict[str, Any]) -> str:
    """Local folder for a weights.yaml entry (downloads into <model_dir>/... when needed)."""
    return resolve_model_path(
        entry.get("repo_id"),
        local_path=entry.get("local_path"),
        hub=entry.get("hub", "huggingface"),
        model_dir=os.environ.get("RANKIFY_MODEL_DIR") or _load_yaml("weights.yaml").get("model_dir"),
    )


def resolve_generator(name: str) -> str:
    """Local folder for a generator listed in weights.yaml; other names are downloaded from HF."""
    entry = _load_yaml("weights.yaml").get("generators", {}).get(name)
    if entry is None:
        if os.path.isdir(name):
            return name
        entry = {"repo_id": name}
    return resolve_weights(entry)


__all__ = [
    "CONFIG_DIR",
    "PROMPT_DIR",
    "RerankerSpec",
    "list_rerankers",
    "get_reranker_spec",
    "resolve_weights",
    "resolve_generator",
]
