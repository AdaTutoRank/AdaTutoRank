"""Strict output-format validation for AdaTutoRank-ATO rollouts."""

from __future__ import annotations

import re
from typing import Optional

FORMAT_FAIL_REWARD = -1.0
_TOKEN = r"[1-9]\d*"
_STRICT_SELECTION = re.compile(r"^\[" + _TOKEN + r"\](?: \[" + _TOKEN + r"\])*$")


def check_format(
    text: str,
    num_passages: int,
    mode: str = "wo_cot",
) -> tuple[bool, Optional[list[int]], str]:
    """Validate a plain one-based identifier sequence such as ``[1] [16]``."""
    if mode != "wo_cot":
        raise ValueError("AdaTutoRank-ATO supports only direct, no-CoT identifier output")

    selection = text.strip()
    if not selection:
        return False, None, "empty_output"
    if not _STRICT_SELECTION.fullmatch(selection):
        return False, None, "invalid_identifier_sequence"

    identifiers = [int(value) for value in re.findall(_TOKEN, selection)]
    if any(value > num_passages for value in identifiers):
        return False, None, "index_out_of_range"
    if len(set(identifiers)) != len(identifiers):
        return False, None, "duplicate_index"
    return True, [value - 1 for value in identifiers], "ok"


def format_reward(text: str, num_passages: int, mode: str = "wo_cot") -> float:
    """Return zero for valid output and ``-1`` for invalid output."""
    valid, _, _ = check_format(text, num_passages, mode)
    return 0.0 if valid else FORMAT_FAIL_REWARD
