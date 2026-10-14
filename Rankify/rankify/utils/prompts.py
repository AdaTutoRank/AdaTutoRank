"""Loading per-reranker prompt files (``configs/prompts/<name>.yaml``)."""

import os
from typing import Any, Dict, Iterable, Optional

import yaml


def load_prompt(prompt_file: Optional[str], required_keys: Iterable[str] = ()) -> Dict[str, Any]:
    """Read a prompt yaml and check that ``required_keys`` are present."""
    if not prompt_file:
        raise ValueError(
            "prompt_file is required: every reranker reads its templates from "
            "configs/prompts/<name>.yaml (see configs/rerankers.yaml)"
        )
    if not os.path.isfile(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f) or {}
    missing = [k for k in required_keys if k not in prompt]
    if missing:
        raise KeyError(f"{prompt_file} is missing keys: {missing}")
    prompt["_path"] = prompt_file
    return prompt


def resolve_relative(prompt: Dict[str, Any], rel_path: str) -> str:
    """Resolve a path written in a prompt yaml relative to that yaml's folder."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(os.path.dirname(prompt["_path"]), rel_path)
