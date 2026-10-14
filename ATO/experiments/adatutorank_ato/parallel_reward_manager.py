"""Parallel reward evaluation for the IO-bound rubric judge."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager.abstract import AbstractRewardManager


class ParallelRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_workers = int(os.environ.get("REWARD_PARALLEL_WORKERS", "256"))

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        cached = self._extract_reward_from_rm_scores(data, return_dict)
        if cached is not None:
            return cached

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        jobs: list[dict[str, Any]] = []

        for index in range(len(data)):
            item = data[index]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = dict(item.non_tensor_batch.get("extra_info", {}) or {})
            extra_info["num_turns"] = item.non_tensor_batch.get("__num_turns__")
            extra_info["rollout_reward_scores"] = item.non_tensor_batch.get("reward_scores", {})
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()

            jobs.append(
                {
                    "index": index,
                    "valid_response_length": valid_response_length,
                    "prompt": self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                    "response": self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                    "ground_truth": item.non_tensor_batch["reward_model"]["ground_truth"],
                    "data_source": item.non_tensor_batch[self.reward_fn_key],
                    "extra_info": extra_info,
                }
            )

        def score_one(job: dict[str, Any]) -> tuple[int, Any]:
            result = self.compute_score(
                data_source=job["data_source"],
                solution_str=job["response"],
                ground_truth=job["ground_truth"],
                extra_info=job["extra_info"],
            )
            return job["index"], result

        started = time.monotonic()
        scores: list[Any] = [None] * len(jobs)
        workers = min(self.num_workers, max(len(jobs), 1))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(score_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    index, result = future.result()
                    scores[index] = result
                except Exception as exc:  # noqa: BLE001
                    scores[job["index"]] = {
                        "score": 0.0,
                        "reason": f"reward_exception:{exc}",
                        "reward_invalid": 1,
                    }

        all_keys = {
            key
            for score in scores
            if isinstance(score, dict)
            for key in score
        }
        reward_extra_info: defaultdict[str, list[Any]] = defaultdict(list)
        examined_sources: defaultdict[str, int] = defaultdict(int)

        for job, result in zip(jobs, scores, strict=True):
            if isinstance(result, dict):
                reward = float(result.get("score", 0.0))
                for key in all_keys:
                    value = result.get(key)
                    reward_extra_info[key].append("" if value is None else value)
            else:
                reward = float(result or 0.0)
                for key in all_keys:
                    reward_extra_info[key].append(reward if key == "score" else "")

            response_index = max(job["valid_response_length"] - 1, 0)
            reward_tensor[job["index"], response_index] = reward

            source = str(job["data_source"])
            if examined_sources[source] < self.num_examine:
                examined_sources[source] += 1
                print("[prompt]", job["prompt"])
                print("[response]", job["response"])
                print("[ground_truth]", job["ground_truth"])
                print("[reward]", result)

        print(
            f"[ParallelRewardManager] scored {len(jobs)} rollouts with "
            f"{workers} workers in {time.monotonic() - started:.1f}s"
        )
        output = {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return output if return_dict else reward_tensor
