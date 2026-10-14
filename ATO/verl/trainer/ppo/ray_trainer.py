# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


# --------------------------------------------------------------------------- #
#  Rubric hint helpers (set-selection / rubric-ranker task)
#
#  The rubric-ranker RL data stores ``extra_info["rubrics"]`` as a *dict*:
#      {"global_level_rubrics": {dim: [{"description": str, "weight": int}, ...]},
#       "set_level_rubrics": {dim: [...]},
#       "doc_level_rubrics": {dim: [...]}}
#  (three tiers x three dimensions = the full nine-dimension rubric)
#  The user prompt ends with a ranking instruction, e.g.
#      EN: "Select the best subset of the 20 documents above ..."

#  For the "rubric-only hint" training we serialize that dict into a readable
#  hint block and inject it right *before* the final ranking instruction.
# --------------------------------------------------------------------------- #

# Substrings that mark the start of the final ranking instruction in the user
# prompt. Matched by ``rfind`` so an occurrence inside a document never wins over
# the trailing instruction sentence.
RANK_INSTRUCTION_MARKERS = (
    "Select the best subset of the",  # English (deepresearch / rag)
    "Select the best subset",
)


def format_rubric_hint(rubric) -> str:
    """Serialize the rubric (dict or str) into a readable hint block.

    A string rubric is returned as-is (legacy behavior). A dict rubric following
    the rubric-ranker schema is flattened into weighted bullet lines grouped by
    set-level / doc-level. Empty dimensions are skipped. Returns "" when nothing
    usable is present.
    """
    if rubric is None:
        return ""
    if isinstance(rubric, str):
        return rubric.strip()
    if not isinstance(rubric, dict):
        return ""

    def _emit(section_title: str, section) -> list[str]:
        if not isinstance(section, dict):
            return []
        body: list[str] = []
        for dim, items in section.items():
            if not items:
                continue
            for it in items:
                if isinstance(it, dict):
                    desc = str(it.get("description", "")).strip()
                    weight = it.get("weight", None)
                else:
                    desc = str(it).strip()
                    weight = None
                if not desc:
                    continue
                if weight is not None:
                    body.append(f"- {dim} (weight {weight}): {desc}")
                else:
                    body.append(f"- {dim}: {desc}")
        if body:
            return [section_title, *body]
        return []

    lines: list[str] = []
    lines += _emit("[Global-level rubrics]", rubric.get("global_level_rubrics"))
    lines += _emit("[Set-level rubrics]", rubric.get("set_level_rubrics"))
    lines += _emit("[Doc-level rubrics]", rubric.get("doc_level_rubrics"))
    return "\n".join(lines).strip()


def find_rank_instruction_index(text: str) -> int:
    """Return the index of the final ranking instruction, or -1 if not found."""
    best = -1
    for marker in RANK_INSTRUCTION_MARKERS:
        idx = text.rfind(marker)
        if idx > best:
            best = idx
    return best


def inject_hint_before_instruction(prompt_text: str, hint_block: str) -> str:
    """Insert ``hint_block`` right before the final ranking instruction.

    The trailing instruction sentence is preserved so the model still receives
    the original task specification. Falls back to appending at the end when no
    instruction marker is found.
    """
    if not hint_block:
        return prompt_text
    idx = find_rank_instruction_index(prompt_text)
    if idx != -1:
        return prompt_text[:idx] + hint_block + "\n\n" + prompt_text[idx:]
    return prompt_text + "\n\n" + hint_block


def inject_hint_generic(prompt_text: str, hint_block: str) -> str:
    """Insert ``hint_block`` into a prompt in a format-agnostic way.

    Tries the final ranking instruction marker first (English/Chinese rubric-ranker
    format), then the legacy ``### Output`` marker, otherwise appends at the end. The
    trailing instruction / output section is always preserved.
    """
    if not hint_block:
        return prompt_text
    idx = find_rank_instruction_index(prompt_text)
    if idx != -1:
        return prompt_text[:idx] + hint_block + "\n\n" + prompt_text[idx:]
    marker = "### Output"
    j = prompt_text.rfind(marker)
    if j != -1:
        return prompt_text[:j] + hint_block + "\n\n" + prompt_text[j:]
    return prompt_text + "\n\n" + hint_block


# Anchors bounding the GENERIC rubric block inside the ranker system message. In the
# rubric-ranker training prompts (see rubricranker_reranker.py: RAG/DEEPRESEARCH/WO_COT/
# W_COT system messages) the block always starts with "Use these N rubrics to judge a
# candidate document set:" and ends right before "Output format (STRICT):". The framework
# intro (rag vs deep-research) and the cot/wo_cot output-format section live OUTSIDE this
# block, so replacing only this block swaps the generic rubric for the query-specific one
# while leaving the scenario intro and output format byte-identical to training.
RUBRIC_BLOCK_START_RE = re.compile(r"Use these \d+ rubrics? to judge a candidate document set:")
RUBRIC_BLOCK_END_MARKER = "Output format (STRICT):"
DEFAULT_RUBRIC_SELECT_TEMPLATE = (
    "Use the following query-specific rubrics to judge a candidate document set. "
    "They specify the concrete scoring points that the optimal document subset for this "
    "query should cover, and a higher weight indicates a more important point.\n\n{rubric}"
)


def replace_rubric_block(text: str, replacement_block: str) -> tuple[str, bool]:
    """Replace the generic rubric block in ``text`` with ``replacement_block``.

    Returns ``(new_text, replaced)``. The scenario intro (before the block) and the output
    format section (from ``Output format (STRICT):`` onward) are preserved verbatim, so the
    same replacement works for rag/deepresearch and w_cot/wo_cot without enumerating variants.
    """
    m = RUBRIC_BLOCK_START_RE.search(text)
    if not m:
        return text, False
    start = m.start()
    end_idx = text.find(RUBRIC_BLOCK_END_MARKER, m.end())
    if end_idx == -1:
        return text[:start] + replacement_block, True
    return text[:start] + replacement_block + "\n\n" + text[end_idx:], True


def _doc_content_text(item) -> str:
    """Extract the text of a single candidate document (dict or str)."""
    if isinstance(item, dict):
        for k in ("content", "doc", "text"):
            v = item.get(k)
            if v not in (None, ""):
                return str(v)
        return ""
    return str(item)


def render_all_candidate_docs(initial_list: list) -> str:
    """Render every candidate document as ``[id] text`` lines (id = position+1).

    Used to build the diff-generation prompt so the model sees the full candidate
    pool when comparing the student subset against the rubric-guided subset.
    """
    if not initial_list:
        return ""
    lines = []
    for k, d in enumerate(initial_list):
        lines.append(f"[{k + 1}] {_doc_content_text(d).strip()}")
    return "\n\n".join(lines)


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations_fine(
        self,
        inputs: list[str],
        outputs: list[str],
        gts: list,
        reward_info: dict,
        dump_path: str,
        validate: bool = False,
        rollout_n: int = 1,
        uids: list[str] | None = None,
        sample_indices: list | None = None,
    ):
        """Dump fine-grained rollout/validation samples as JSONL.

        Each line contains: step, sample_index, uid, rollout_index, validate,
        prompt_text, response_text, ground_truth_text, reward (dict of all reward fields).

        Args:
            sample_indices: Per-sample index from extra_info["Index"]. When present,
                used as the primary sample identifier and for rollout grouping
                (instead of uid).
            uids: Per-sample uuid. Used as fallback grouping key when sample_indices
                is not available.

        Note: After balance_batch or other reordering operations, the positional order
        of samples in the batch may no longer follow the original interleave pattern.
        We use sample_index (or uid) to group rollouts belonging to the same sample,
        and assign rollout_index based on occurrence order within each group.
        """
        os.makedirs(dump_path, exist_ok=True)
        if validate:
            filename = os.path.join(dump_path, f"{self.global_steps}_val.jsonl")
        else:
            filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)

        # Determine the grouping key: prefer sample_indices over uids
        if sample_indices is not None:
            group_keys = [str(sample_indices[i]) for i in range(n)]
        elif uids is not None:
            group_keys = [str(uids[i]) for i in range(n)]
        else:
            group_keys = None

        # Track rollout index per group key (same key = same original sample)
        group_rollout_counter: dict[str, int] = {}

        lines = []
        for i in range(n):
            # Build reward dict: include score and all extra reward fields
            reward_dict = {}
            for k, v in reward_info.items():
                if isinstance(v, list) and len(v) == n:
                    reward_dict[k] = v[i]

            if group_keys is not None:
                gk = group_keys[i]
                rollout_index = group_rollout_counter.get(gk, 0)
                group_rollout_counter[gk] = rollout_index + 1
            else:
                rollout_index = i % rollout_n if rollout_n > 0 else 0

            entry = {
                "step": self.global_steps,
                "sample_index": sample_indices[i] if sample_indices is not None else None,
                "uid": uids[i] if uids is not None else None,
                "rollout_index": rollout_index,
                "validate": validate,
                "prompt_text": inputs[i] if rollout_index == 0 else "",
                "response_text": outputs[i],
                "ground_truth_text": gts[i] if i < len(gts) else None,
                "reward": reward_dict,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped fine generations to {filename}")

    def _maybe_add_rollout_uid(self, batch):
        """Attach a stable, row-unique `rollout_uid` to batch.non_tensor_batch.

        The id is derived deterministically as ``{step}#{group_key}#{rollout_index}`` where the
        (group_key, rollout_index) enumeration matches the exact logic used by both the distill and
        rollout jsonl dumps, so lines across all dump files line up on the same rollout. non_tensor
        columns ride with their rows through balance/union/split, so alignment holds even after
        reordering inside the actor update.
        """
        ntb = batch.non_tensor_batch
        if "rollout_uid" in ntb:
            return
        try:
            n = batch.batch.batch_size[0]
        except Exception:  # noqa: BLE001
            n = len(next(iter(ntb.values()))) if ntb else 0
        if n == 0:
            return

        sample_indices = ntb.get("index")
        uids = ntb.get("uid")
        if sample_indices is not None:
            group_keys = [str(sample_indices[i]) for i in range(n)]
        elif uids is not None:
            group_keys = [str(uids[i]) for i in range(n)]
        else:
            group_keys = None

        counter: dict[str, int] = {}
        vals = []
        for i in range(n):
            if group_keys is not None:
                gk = group_keys[i]
                ri = counter.get(gk, 0)
                counter[gk] = ri + 1
            else:
                gk = ""
                ri = i
            vals.append(f"{self.global_steps}#{gk}#{ri}")
        ntb["rollout_uid"] = np.array(vals, dtype=object)

    def _maybe_arm_opsd_adv_dump(self, batch):
        """If OPSD advantage dumping is enabled for this step, stamp the request onto
        ``batch.meta_info`` (delivered to every actor rank) and return (dump_dir, token_n).
        Returns (None, 0) when disabled so the caller skips the post-update merge.
        """
        tcfg = self.config.trainer
        if not tcfg.get("opsd_adv_dump", False):
            return None, 0
        freq = int(tcfg.get("opsd_adv_dump_freq", 1) or 0)
        if freq <= 0 or (self.global_steps % freq != 0):
            return None, 0
        dump_dir = tcfg.get("opsd_adv_dump_dir", None) or tcfg.get("distill_data_dir", None)
        if not dump_dir:
            return None, 0
        token_n = int(tcfg.get("opsd_adv_dump_token_n", 4))
        batch.meta_info["opsd_adv_dump"] = True
        batch.meta_info["opsd_adv_dump_dir"] = dump_dir
        batch.meta_info["opsd_adv_dump_step"] = int(self.global_steps)
        batch.meta_info["opsd_adv_dump_token_n"] = token_n
        return dump_dir, token_n

    def _merge_opsd_adv_dump_shards(self, dump_dir: str, step: int):
        """Merge the per-rank ``{step}_{kind}.part{rank}.jsonl`` shards written by the actor
        workers into single ``{step}_{kind}.jsonl`` files, de-duplicating rows that are replicated
        across ulysses sequence-parallel ranks (keyed by (rollout_uid, token_idx))."""
        import glob

        for kind in ("distill_adv", "distill_token"):
            parts = sorted(glob.glob(os.path.join(dump_dir, f"{step}_{kind}.part*.jsonl")))
            if not parts:
                continue
            seen = set()
            lines = []
            for p in parts:
                try:
                    with open(p) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:  # noqa: BLE001
                                continue
                            key = (obj.get("rollout_uid"), obj.get("token_idx"))
                            if key in seen:
                                continue
                            seen.add(key)
                            lines.append(line)
                except OSError:
                    continue
            if not lines:
                continue
            out = os.path.join(dump_dir, f"{step}_{kind}.jsonl")
            with open(out, "w") as f:
                f.write("\n".join(lines) + "\n")
            for p in parts:
                try:
                    os.remove(p)
                except OSError:
                    pass
            print(f"[opsd_adv_dump] merged {len(parts)} shards -> {out} ({len(lines)} rows)")

    def _dump_distill_details(
        self,
        *,
        batch,
        batch_size: int,
        seq_scores_np,
        hint_types,
        rubric_strs,
        rubric_set_text,
        rubric_set_reward,
        diff_text,
        response_texts,
        messages,
        resolve_fn,
        dump_path: str,
    ):
        """Write per-rollout adaptive tutoring diagnostics as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}_distill.jsonl")

        ntb = batch.non_tensor_batch
        uids = ntb.get("uid")
        sample_indices = ntb.get("index")
        reward_model = ntb.get("reward_model")
        rollout_uids = ntb.get("rollout_uid")

        if sample_indices is not None:
            group_keys = [str(sample_indices[i]) for i in range(batch_size)]
        elif uids is not None:
            group_keys = [str(uids[i]) for i in range(batch_size)]
        else:
            group_keys = None
        group_rollout_counter: dict[str, int] = {}

        bucket_label = {
            "demonstration": "low", "diff": "medium", "rubric": "high",
            "self_rollout": "self", "best_rollout": "group_best", "none": "none",
        }

        def _scalar(x):
            return x.item() if hasattr(x, "item") else x

        lines = []
        for i in range(batch_size):
            if group_keys is not None:
                gk = group_keys[i]
                rollout_index = group_rollout_counter.get(gk, 0)
                group_rollout_counter[gk] = rollout_index + 1
            else:
                rollout_index = i

            content, hint_reward, _reflection, resolved_kind = resolve_fn(i)

            try:
                teacher_reprompt = messages[i][-1]["content"] if messages and messages[i] else ""
            except Exception:  # noqa: BLE001
                teacher_reprompt = ""

            gold = None
            if reward_model is not None:
                try:
                    gold = reward_model[i].get("ground_truth")
                except Exception:  # noqa: BLE001
                    gold = None

            assigned = hint_types[i] if hint_types is not None else None
            r_set_v = None
            if rubric_set_reward is not None and rubric_set_reward[i] is not None:
                r_set_v = float(rubric_set_reward[i])

            entry = {
                "step": self.global_steps,
                "sample_index": _scalar(sample_indices[i]) if sample_indices is not None else None,
                "uid": str(uids[i]) if uids is not None else None,
                "rollout_index": rollout_index,
                "rollout_uid": str(rollout_uids[i]) if rollout_uids is not None else None,
                "r_i": float(seq_scores_np[i]),
                "assigned_hint_type": assigned,
                "assigned_bucket_label": bucket_label.get(assigned, assigned),
                "resolved_hint_type": resolved_kind,
                "resolved_bucket_label": bucket_label.get(resolved_kind, resolved_kind),
                "rubric_text": rubric_strs[i] if rubric_strs is not None else None,
                "rubric_set_text": rubric_set_text[i] if rubric_set_text is not None else None,
                "r_set": r_set_v,
                "diff_text": diff_text[i] if diff_text is not None else None,
                "hint_content": content,
                "student_set": response_texts[i] if response_texts is not None else None,
                "gold_set": gold,
                "teacher_reprompt_text": teacher_reprompt,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped distill details to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ) -> dict:
        """Log rollout data to disk and return extra metrics for tensorboard.

        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data

        Returns:
            dict: Extra metrics to be logged to tensorboard (e.g. reward_extra/ stats).
        """
        extra_metrics = {}
        # Keys to exclude from rollout dump (logged as tensorboard metrics instead)
        _METRIC_ONLY_KEYS = {"len(pred)", "comprehensive", "conflict", "redundancy", "richness", "relevance", "truthfulness", "quality", "authority", "timeliness"}
        # Keys already represented in 'score' or internal-only, avoid duplication in dump
        _SKIP_KEYS = {"score", "reward_invalid"}

        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            fine_dump_path = rollout_data_dir
            rollout_n = self.config.actor_rollout_ref.rollout.n
            reward_info = {"score": scores}
            for k, v in reward_extra_infos_to_dump.items():
                if k in _SKIP_KEYS or k in _METRIC_ONLY_KEYS:
                    continue
                if isinstance(v, list) and len(v) == len(inputs):
                    reward_info[k] = v
                elif isinstance(v, np.ndarray) and len(v) == len(inputs):
                    reward_info[k] = v.tolist()

            
            
            
            if "rubric_scores" in reward_info:
                _parsed = []
                for _rs in reward_info["rubric_scores"]:
                    if isinstance(_rs, str) and _rs:
                        try:
                            _parsed.append(json.loads(_rs))
                        except Exception:  # noqa: BLE001
                            _parsed.append(_rs)
                    else:
                        _parsed.append(_rs if _rs not in ("", None) else {})
                reward_info["rubric_scores"] = _parsed

            # Collect metric-only fields (e.g. len(pred)) as tensorboard metrics
            # Build reward_invalid set for filtering metric-only fields
            _reward_invalid_indices = set()
            _ri_raw = reward_extra_infos_to_dump.get("reward_invalid", None)
            if _ri_raw is not None:
                _ri_list = _ri_raw if isinstance(_ri_raw, list) else _ri_raw.tolist() if hasattr(_ri_raw, "tolist") else list(_ri_raw)
                _reward_invalid_indices = {i for i, f in enumerate(_ri_list) if f}

            for k in _METRIC_ONLY_KEYS:
                raw = reward_extra_infos_to_dump.get(k, None)
                if raw is None:
                    continue
                vals = raw if isinstance(raw, list) else raw.tolist() if hasattr(raw, "tolist") else list(raw)
                # Filter out sentinel values (e.g. -1 means not available) and reward_invalid samples
                numeric_vals = [v for i, v in enumerate(vals) if isinstance(v, (int, float)) and v >= 0 and i not in _reward_invalid_indices]
                if numeric_vals:
                    extra_metrics[f"reward_extra/{k}/mean"] = sum(numeric_vals) / len(numeric_vals)
                    # extra_metrics[f"reward_extra/{k}/max"] = max(numeric_vals)
                    # extra_metrics[f"reward_extra/{k}/min"] = min(numeric_vals)

            # Extract uids for correct sample grouping (order may be shuffled by balance_batch)
            sample_uids = batch.non_tensor_batch.get("uid", None)
            uid_list = sample_uids.tolist() if sample_uids is not None else None

            # Extract sample_index from non_tensor_batch["index"] if available
            raw_indices = batch.non_tensor_batch.get("index", None)
            if raw_indices is not None:
                sample_index_list = raw_indices.tolist() if hasattr(raw_indices, "tolist") else list(raw_indices)
            else:
                sample_index_list = None

            self._dump_generations_fine(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                reward_info=reward_info,
                dump_path=fine_dump_path,
                validate=False,
                rollout_n=rollout_n,
                uids=uid_list,
                sample_indices=sample_index_list,
            )

        return extra_metrics

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float, use_mean_as_threshold: bool = False) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]

        if use_mean_as_threshold:
            
            uid_to_indices: dict[Any, list[int]] = defaultdict(list)
            for idx, uid in enumerate(uids):
                uid_to_indices[uid].append(idx)
            success_by_uid: dict[Any, list[int]] = defaultdict(list)
            for uid, indices in uid_to_indices.items():
                group_mean = sum(seq_scores[i] for i in indices) / len(indices)
                for idx in indices:
                    if seq_scores[idx] >= group_mean:
                        success_by_uid[uid].append(idx)
        else:
            success_by_uid: dict[Any, list[int]] = defaultdict(list)
            for idx, uid in enumerate(uids):
                if seq_scores[idx] >= success_reward_threshold:
                    success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove <think>...</think> tags and their content from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
        seq_scores: Optional[np.ndarray] = None,
    ) -> tuple[Optional[str], Optional[float]]:
        """Returns (solution_str, solution_reward)."""
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None, None
        if seq_scores is not None:
            
            solution_idx = max(solution_idxs, key=lambda j: seq_scores[j])
            solution_reward = float(seq_scores[solution_idx])
        else:
            solution_idx = solution_idxs[0]
            solution_reward = None
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str, solution_reward

    @staticmethod
    def _assign_hint_types(seq_scores: np.ndarray, self_distillation_cfg) -> list[str]:
        """Assign one of the 3 hint types to each rollout based on its reward bucket.

        Buckets are defined by ``hint_thresholds`` (ascending cut points). With N thresholds
        there are N+1 buckets, and ``hint_bucket_types`` gives the hint type per bucket.
        Returns a list of hint-type strings (one per rollout), each in
        {"rubric", "self_rollout", "best_rollout", "none"}.
        """
        thresholds = self_distillation_cfg.get("hint_thresholds", None) or []
        bucket_types = self_distillation_cfg.get("hint_bucket_types", None) or ["self_rollout"]
        thresholds = list(thresholds)

        def _bucket_index(score: float) -> int:
            # ATO paper buckets: low 0<=r<=3, medium 3<r<6,
            # high r>=6. The two boundaries intentionally have different
            # inclusivity, so a generic bisect cannot express them exactly.
            if thresholds == [3.0, 6.0] and len(bucket_types) == 3:
                if score <= 3.0:
                    return 0
                if score < 6.0:
                    return 1
                return 2
            # bucket i = number of thresholds strictly <= score is not used;
            # we use: score < thresholds[0] -> 0, thresholds[k-1] <= score < thresholds[k] -> k, ...
            idx = 0
            for t in thresholds:
                if score >= t:
                    idx += 1
                else:
                    break
            return min(idx, len(bucket_types) - 1)

        return [bucket_types[_bucket_index(float(s))] for s in seq_scores]

    @staticmethod
    def _winsorize_robust(scores: list, trim_n: int = 1, max_deviation: float = 3.0) -> list:
        """Dual-layer truncation to handle extreme reward values.

        Layer 1: Standard Winsorize (handles single extreme outliers)
        Layer 2: Median ± max_deviation clip (handles multiple extreme outliers)

        Args:
            scores: reward scores for rollouts of the same query
            trim_n: number of extremes to trim in standard Winsorize
            max_deviation: maximum allowed deviation from median
        """
        n = len(scores)

        # Layer 1: Standard Winsorize
        if n > 2 * trim_n:
            sorted_s = sorted(scores)
            lb = sorted_s[trim_n]
            ub = sorted_s[-trim_n - 1]
            scores = [max(lb, min(ub, s)) for s in scores]

        # Layer 2: Median ± max_deviation clip
        median = float(np.median(scores))
        lb = median - max_deviation
        ub = median + max_deviation
        scores = [max(lb, min(ub, s)) for s in scores]

        return scores

    def _winsorize_rewards_by_uid(
        self, batch: DataProto, reward_key: str = "token_level_rewards",
        trim_n: int = 1, max_deviation: float = 3.0,
        variance_threshold: float = 0.0,
    ) -> DataProto:
        """Apply robust winsorization to per-uid reward scores before advantage computation.

        Modifies batch.batch[reward_key] in-place so that per-uid
        sequence-level scores are clipped to remove extreme outliers.

        Args:
            reward_key: Which batch tensor to winsorize (e.g. "token_level_rewards").
            trim_n: Number of extremes to trim in standard Winsorize layer.
            max_deviation: Maximum allowed deviation from median.
            variance_threshold: Only apply winsorization when group variance exceeds this value.
                               Set to 0 to always apply.
        """
        token_level_rewards = batch.batch[reward_key]
        uids = batch.non_tensor_batch["uid"]
        seq_scores = token_level_rewards.sum(dim=-1)  # (bs,)

        # Group indices by uid
        uid2indices = defaultdict(list)
        for i, uid in enumerate(uids):
            uid2indices[uid].append(i)

        # Apply winsorize per group and rescale token_level_rewards
        for uid, indices in uid2indices.items():
            if len(indices) <= 1:
                continue
            original_scores = [seq_scores[i].item() for i in indices]
            # Skip groups whose variance is below the threshold
            if variance_threshold > 0.0:
                group_var = float(np.var(original_scores))
                if group_var <= variance_threshold:
                    continue
            clipped_scores = self._winsorize_robust(original_scores, trim_n=trim_n, max_deviation=max_deviation)
            for idx, (orig, clipped) in zip(indices, zip(original_scores, clipped_scores)):
                if orig != clipped and orig != 0.0:
                    # Scale the token-level rewards proportionally
                    scale = clipped / orig
                    token_level_rewards[idx] = token_level_rewards[idx] * scale
                elif orig == 0.0 and clipped != 0.0:
                    # Edge case: original score is 0 but clipped is not
                    # Find the last valid token position and place the diff there
                    response_mask = batch.batch["response_mask"][idx]
                    last_pos = response_mask.sum().long() - 1
                    if last_pos >= 0:
                        token_level_rewards[idx, last_pos] = clipped

        return batch


    def _run_policy_generation(
        self, messages_list: list[list[dict]], max_new_tokens: Optional[int] = None
    ) -> DataProto:
        """Generate one response per message list using the CURRENT rollout policy.

        Builds a minimal generation DataProto carrying only ``raw_prompt`` (+ a dummy
        tensor to keep the batch non-empty) and dispatches it to the same rollout engine
        used for the primary rollouts, so rubric_set / diff are produced by the very model
        that generated the 8 rollouts. Returns 1 response per input row, order preserved.

        ``max_new_tokens`` optionally overrides the response-length budget for THIS call only
        (used by diff generation, whose prose needs far more tokens than the tiny id-list
        ``rollout.response_length`` used for the main rollout). The override applies to both the
        generation cap and the response padding width in the agent loop, so it does not affect
        the main rollout's tensor shapes.
        """
        n = len(messages_list)
        raw = np.empty(n, dtype=object)
        for k, m in enumerate(messages_list):
            raw[k] = m
        meta_info = {"global_steps": getattr(self, "global_steps", 0)}
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            meta_info["response_length"] = int(max_new_tokens)
        gen = DataProto.from_dict(
            tensors={"dummy_tensor": torch.zeros((n, 1), dtype=torch.uint8)},
            non_tensors={"raw_prompt": raw},
            meta_info=meta_info,
        )
        # generate_sequences chunks the batch across workers/ranks and requires the batch
        # size to be divisible by that count, so pad to the divisor and unpad the output
        # (mirrors the validation path).
        size_divisor = (
            self.config.actor_rollout_ref.rollout.agent.num_workers
            if self.async_rollout_mode
            else self.actor_rollout_wg.world_size
        )
        size_divisor = max(int(size_divisor), 1)
        gen_padded, pad_size = pad_dataproto_to_divisor(gen, size_divisor)
        if self.async_rollout_mode:
            out_padded = self.async_rollout_manager.generate_sequences(gen_padded)
        else:
            out_padded = self.actor_rollout_wg.generate_sequences(gen_padded)
        out = unpad_dataproto(out_padded, pad_size=pad_size)
        return out

    def _build_rubric_select_messages(
        self, raw_prompt, rubric_text: str, select_template: str
    ) -> tuple[list[dict], bool]:
        """Build the "re-select a subset using this query's concrete rubric" messages.

        The training prompt is reused verbatim except for the GENERIC rubric block inside the
        system message, which is swapped for ``select_template`` filled with this query's
        concrete rubric (see ``replace_rubric_block``). Everything else — the rag/deepresearch
        intro, the w_cot/wo_cot output format, and the full user query with its candidate
        documents — stays byte-identical to the training distribution, so the policy answers in
        exactly the same format as the main rollout.

        Returns ``(messages, replaced)``; ``replaced`` is False when no generic rubric block was
        found and the block had to be injected before the final ranking instruction instead.
        """
        block = select_template.format(rubric=rubric_text)
        msgs = [dict(m) for m in raw_prompt]
        for m in msgs:
            new_text, replaced = replace_rubric_block(str(m.get("content", "")), block)
            if replaced:
                m["content"] = new_text
                return msgs, True
        # No generic rubric block (e.g. legacy Chinese setwise prompts): fall back to injecting
        # the concrete rubric right before the final ranking instruction of the user message.
        msgs[-1]["content"] = inject_hint_before_instruction(str(msgs[-1].get("content", "")), block)
        return msgs, False

    def _score_sets_with_rm(
        self, batch: DataProto, indices: list[int], set_texts: list[str]
    ) -> dict[int, tuple[Optional[float], bool]]:
        """RM-score policy-selected subsets with the SAME reward function as the rollouts.

        Calls ``self.reward_fn.compute_score`` (the custom rubric reward) once per subset, in a
        thread pool because each call is a blocking judge HTTP request. Returns
        ``{index: (score, format_ok)}``; entries missing from the result failed outright.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        score_fn = getattr(self.reward_fn, "compute_score", None)
        if score_fn is None or not indices:
            if score_fn is None:
                print("[rubric_set] WARN: reward_fn has no compute_score; cannot score rubric_set")
            return {}

        extra_info_arr = batch.non_tensor_batch.get("extra_info", None)
        data_source_arr = batch.non_tensor_batch.get("data_source", None)
        reward_model_arr = batch.non_tensor_batch.get("reward_model", None)

        def _args(i: int, text: str) -> dict:
            ei = extra_info_arr[i] if extra_info_arr is not None else {}
            ei = dict(ei) if isinstance(ei, dict) else {}
            gt = None
            if reward_model_arr is not None:
                try:
                    gt = reward_model_arr[i].get("ground_truth")
                except Exception:  # noqa: BLE001
                    gt = None
            return {
                "data_source": data_source_arr[i] if data_source_arr is not None else "",
                "solution_str": text,
                "ground_truth": gt,
                "extra_info": ei,
            }

        n_workers = min(
            int(os.environ.get("REWARD_PARALLEL_WORKERS", "32")), max(len(indices), 1)
        )
        results: dict[int, tuple[Optional[float], bool]] = {}

        def _one(i: int, text: str):
            return i, score_fn(**_args(i, text))

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_one, i, t) for i, t in zip(indices, set_texts)]
            for fut in as_completed(futures):
                try:
                    i, res = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[rubric_set] WARN: RM scoring of rubric_set failed: {e}")
                    continue
                if isinstance(res, dict):
                    # incorrect_format==1 means the selection did not even parse as a valid
                    # subset, so it is useless both as a demonstration and as a gate value.
                    fmt_ok = int(res.get("incorrect_format", 0) or 0) == 0
                    try:
                        score = float(res.get("score"))
                    except (TypeError, ValueError):
                        score = None
                else:
                    fmt_ok = True
                    try:
                        score = float(res)
                    except (TypeError, ValueError):
                        score = None
                results[i] = (score, fmt_ok)
        return results

    def _select_rubric_set_with_policy(
        self,
        batch: DataProto,
        rubric_strs: list[Optional[str]],
        self_distillation_cfg,
    ) -> tuple[list[Optional[str]], list[Optional[float]], dict[str, float]]:
        """``rubric_set_source=policy``: have the CURRENT policy re-select the subset.

        One generation per QUERY (not per rollout): rollouts sharing a ``uid`` share the same
        rubric and candidate documents, so the selection is made once from the group's first
        rollout and broadcast to the whole group. The result is RM-scored to obtain r_set, the
        validation gate used downstream. Selections that fail the format check are dropped
        (left as None) so those rollouts fall back to the abstract rubric hint.
        """
        batch_size = len(rubric_strs)
        rubric_set_text: list[Optional[str]] = [None] * batch_size
        rubric_set_reward: list[Optional[float]] = [None] * batch_size
        metrics: dict[str, float] = {}

        select_tmpl = (
            self_distillation_cfg.get("rubric_set_select_template", "")
            or DEFAULT_RUBRIC_SELECT_TEMPLATE
        )
        remove_think = bool(self_distillation_cfg.get("rubric_set_remove_thinking", True))
        raw_prompts = batch.non_tensor_batch["raw_prompt"]
        uids = batch.non_tensor_batch["uid"]

        uid_members: dict[Any, list[int]] = defaultdict(list)
        uid_to_rep: dict[Any, int] = {}
        for i in range(batch_size):
            uid_members[uids[i]].append(i)
            if rubric_strs[i] and uids[i] not in uid_to_rep:
                uid_to_rep[uids[i]] = i

        metrics["rubric_set/policy_num_queries"] = float(len(uid_to_rep))
        if not uid_to_rep:
            print("[rubric_set] WARN: no rubric available in this batch; skipping policy selection")
            return rubric_set_text, rubric_set_reward, metrics

        rep_indices: list[int] = []
        messages_list: list[list[dict]] = []
        n_block_replaced = 0
        for i in uid_to_rep.values():
            try:
                msgs, replaced = self._build_rubric_select_messages(
                    raw_prompts[i], rubric_strs[i], select_tmpl
                )
            except Exception as e:  # noqa: BLE001
                print(f"[rubric_set] WARN: select prompt build failed at idx={i}: {e}")
                continue
            n_block_replaced += int(replaced)
            rep_indices.append(i)
            messages_list.append(msgs)

        if not messages_list:
            return rubric_set_text, rubric_set_reward, metrics
        metrics["rubric_set/policy_rubric_block_replaced_fraction"] = (
            n_block_replaced / len(rep_indices)
        )

        max_len = int(self_distillation_cfg.get("rubric_set_max_response_length", 0) or 0)
        t0 = time.time()
        out = self._run_policy_generation(
            messages_list, max_new_tokens=max_len if max_len > 0 else None
        )
        raw_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in out.batch["responses"]
        ]
        metrics["rubric_set/policy_select_gen_seconds"] = time.time() - t0

        # Score the RAW generation: the reward function's format check is mode-aware and under
        # w_cot it requires the full <think>...</think><answer>...</answer> structure, so a
        # thinking-stripped text would be rejected as malformed. Thinking is only stripped for
        # the text we keep as the reference subset (demonstration hint / diff prompt input).
        t1 = time.time()
        scored = self._score_sets_with_rm(batch, rep_indices, raw_texts)
        metrics["rubric_set/policy_select_rm_seconds"] = time.time() - t1
        sel_texts = [
            (self._remove_thinking_trace(t) if remove_think else t).strip() for t in raw_texts
        ]

        n_ok = 0
        n_rejected = 0
        for k, i in enumerate(rep_indices):
            score, fmt_ok = scored.get(i, (None, False))
            if not sel_texts[k] or not fmt_ok or score is None:
                n_rejected += 1
                continue
            # Broadcast the query-level selection + its r_set to every rollout of this query.
            for j in uid_members[uids[i]]:
                rubric_set_text[j] = sel_texts[k]
                rubric_set_reward[j] = float(score)
            n_ok += 1

        metrics["rubric_set/policy_selected_query_fraction"] = n_ok / len(rep_indices)
        metrics["rubric_set/policy_rejected_query_fraction"] = n_rejected / len(rep_indices)
        print(
            f"[rubric_set] policy selection: {n_ok}/{len(rep_indices)} queries usable "
            f"({n_rejected} rejected by format/scoring), "
            f"gen={metrics['rubric_set/policy_select_gen_seconds']:.1f}s "
            f"rm={metrics['rubric_set/policy_select_rm_seconds']:.1f}s"
        )
        return rubric_set_text, rubric_set_reward, metrics

    def _rubric_set_from_data(
        self, batch: DataProto, batch_size: int
    ) -> tuple[list[Optional[str]], list[Optional[float]], dict[str, float]]:
        """``rubric_set_source=data``: read the precomputed columns straight from the data.

        Instead of re-selecting a subset with the policy and RM-scoring it, the v2/v3 training
        data ships two per-sample columns that we use verbatim:
          rubric_set        : the rubric-guided reference subset, e.g. "[2] [12]"
          rubric_set_reward : its precomputed RM score (0-10), used as the r_set validation gate
        Columns may live at the top level of non_tensor_batch (v2 data) or nested in extra_info
        (v3 data). Absent columns (older data) leave the arrays as None -> graceful fallback to
        the abstract rubric hint.
        """
        rubric_set_text: list[Optional[str]] = [None] * batch_size
        rubric_set_reward: list[Optional[float]] = [None] * batch_size
        extra_info_arr = batch.non_tensor_batch.get("extra_info", None)
        rubric_set_arr = batch.non_tensor_batch.get("rubric_set", None)
        rubric_set_reward_arr = batch.non_tensor_batch.get("rubric_set_reward", None)

        def _field(i: int, arr, key: str):
            if arr is not None:
                v = arr[i]
                if v is not None:
                    return v
            if extra_info_arr is not None and isinstance(extra_info_arr[i], dict):
                return extra_info_arr[i].get(key)
            return None

        n_with_rset = 0
        for i in range(batch_size):
            rs = _field(i, rubric_set_arr, "rubric_set")
            if rs is None or (isinstance(rs, str) and not rs.strip()):
                continue
            rubric_set_text[i] = rs if isinstance(rs, str) else str(rs)
            n_with_rset += 1
            rr = _field(i, rubric_set_reward_arr, "rubric_set_reward")
            if rr is not None:
                try:
                    rubric_set_reward[i] = float(rr)
                except (TypeError, ValueError):
                    rubric_set_reward[i] = None

        metrics = {"rubric_set/data_provided_fraction": n_with_rset / max(batch_size, 1)}
        return rubric_set_text, rubric_set_reward, metrics

    def _generate_rubric_set_and_diff(
        self,
        batch: DataProto,
        response_texts: list[str],
        seq_scores_np: np.ndarray,
        hint_types: list[str],
        rubric_strs: list[Optional[str]],
        self_distillation_cfg,
    ) -> tuple[list[Optional[str]], list[Optional[float]], list[Optional[str]], dict[str, float]]:
        """rubric_set branch generation.

        Step A obtains the rubric-guided reference subset ``rubric_set`` (e.g. "[2] [12]") and
        its RM score ``r_set`` (the validation gate). Two interchangeable sources, selected by
        ``self_distillation.rubric_set_source``:
          "data"   : read the precomputed ``rubric_set`` / ``rubric_set_reward`` columns shipped
                     by the v2/v3 training data (no extra generation, no extra RM call).
          "policy" : let the CURRENT rollout policy re-select the subset from this query's
                     concrete rubric (1 generation per query) and RM-score it.
        Step B then lets the current policy write a diff(student_set, rubric_set) revision
        direction for each "medium"-bucket rollout that passes the medium gate.

        Returns per-sample arrays (length == batch_size):
          rubric_set_text[i]   : rubric_set reference subset for sample i (or None)
          rubric_set_reward[i] : r_set for sample i (or None)
          diff_text[i]         : diff revision direction for sample i (medium bucket only, or None)
        plus a metrics dict.
        """
        batch_size = len(response_texts)
        rubric_set_text: list[Optional[str]] = [None] * batch_size
        rubric_set_reward: list[Optional[float]] = [None] * batch_size
        diff_text: list[Optional[str]] = [None] * batch_size
        metrics: dict[str, float] = {}

        extra_info_arr = batch.non_tensor_batch.get("extra_info", None)
        remove_think = self_distillation_cfg.get("rubric_set_remove_thinking", True)
        bar_medium = float(self_distillation_cfg.get("rubric_set_bar_medium", 6.0))
        gate_relative = bool(self_distillation_cfg.get("rubric_set_gate_relative", True))
        gate_margin = float(self_distillation_cfg.get("rubric_set_gate_margin", 0.0))

        def _strip(text: str) -> str:
            return self._remove_thinking_trace(text) if remove_think else text

        # ---- Step A: obtain rubric_set (+ r_set) from the configured source ----
        # Both branches produce the same two arrays, so everything downstream (diff generation,
        # validation gates, bucket resolution, logging) is identical between them.
        source = str(self_distillation_cfg.get("rubric_set_source", "data") or "data")
        if source == "policy":
            rubric_set_text, rubric_set_reward, source_metrics = self._select_rubric_set_with_policy(
                batch, rubric_strs, self_distillation_cfg
            )
        else:
            rubric_set_text, rubric_set_reward, source_metrics = self._rubric_set_from_data(
                batch, batch_size
            )
        metrics.update(source_metrics)
        metrics["rubric_set/source_is_policy"] = 1.0 if source == "policy" else 0.0

        n_with_rset = sum(1 for t in rubric_set_text if t is not None)
        rset_rewards = [float(r) for r in rubric_set_reward if r is not None]
        metrics["rubric_set/available_fraction"] = n_with_rset / max(batch_size, 1)
        if rset_rewards:
            metrics["rubric_set/r_set_mean"] = float(np.mean(rset_rewards))
            metrics["rubric_set/r_set_pass_medium_fraction"] = float(
                np.mean([1.0 if r >= bar_medium else 0.0 for r in rset_rewards])
            )

        if n_with_rset == 0:
            return rubric_set_text, rubric_set_reward, diff_text, metrics

        # ---- Step B: generate diff(student_set, rubric_set) for medium-bucket rollouts ----
        # Only for samples routed to "diff" whose query passed the medium gate; otherwise the
        # sample falls back to the abstract rubric (handled later in hint resolution).
        diff_tmpl = self_distillation_cfg.get("diff_prompt_template", "")
        if diff_tmpl:
            diff_indices: list[int] = []
            diff_messages: list[list[dict]] = []
            for i in range(batch_size):
                if hint_types[i] != "diff":
                    continue
                if rubric_set_text[i] is None or rubric_set_reward[i] is None:
                    continue
                # Medium gate: relative mode -> only generate the diff if the rubric-guided
                # reference subset actually scores higher than THIS rollout's own reward r_i
                # (r_set > r_i + margin); absolute mode -> r_set >= bar_medium. Failing the gate
                # means the reference isn't better, so the sample falls back to the abstract rubric.
                if gate_relative:
                    if rubric_set_reward[i] <= float(seq_scores_np[i]) + gate_margin:
                        continue
                elif rubric_set_reward[i] < bar_medium:
                    continue  # gate failed -> will fall back to abstract rubric
                ei = extra_info_arr[i] if extra_info_arr is not None else {}
                ei = ei if isinstance(ei, dict) else {}
                query = ei.get("query") or ei.get("question") or ""
                initial_list = ei.get("initial_list") or []
                docs_str = render_all_candidate_docs(initial_list)
                try:
                    user_content = diff_tmpl.format(
                        query=query,
                        rubrics=rubric_strs[i] or "",
                        docs=docs_str,
                        student_set=_strip(response_texts[i]),
                        rubric_set=_strip(rubric_set_text[i]),
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"[rubric_set] WARN: diff prompt format failed at idx={i}: {e}")
                    continue
                # diff is a standalone task: user_content already carries query, rubrics, docs,
                # and both selected subsets. Do NOT reuse the original ranking system prompt
                # (raw_prompts[i][:-1]) — it instructs the model to emit a ranked "[id] [id]" list
                # and biases the wo_cot policy into that degenerate format. Use a dedicated,
                # prose-oriented system prompt instead.
                diff_system = self_distillation_cfg.get("diff_system_prompt", "") or (
                    "You are a careful analyst of document selection for search queries. "
                    "You explain your reasoning in clear natural-language prose and never "
                    "answer with a bare list of ids."
                )
                diff_indices.append(i)
                diff_messages.append(
                    [{"role": "system", "content": diff_system}, {"role": "user", "content": user_content}]
                )

            if diff_messages:
                diff_max_len = int(self_distillation_cfg.get("diff_max_response_length", 0) or 0)
                diff_out = self._run_policy_generation(
                    diff_messages, max_new_tokens=diff_max_len if diff_max_len > 0 else None
                )
                diff_resp = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in diff_out.batch["responses"]]
                # Guard against the policy relapsing into its ranked-id-list output format
                # (a bare "[1] [2] [3]" instead of a prose corrective idea). Such degenerate
                # outputs are useless as a teacher hint, so discard them (-> None) and let hint
                # resolution fall back to the abstract rubric for that rollout.
                _idlist_re = re.compile(r"^(?:\s*\[\d+\]\s*)+$")
                n_degenerate = 0
                for k, i in enumerate(diff_indices):
                    dt = _strip(diff_resp[k]).strip()
                    if not dt or _idlist_re.match(dt):
                        diff_text[i] = None
                        n_degenerate += 1
                    else:
                        diff_text[i] = dt
                metrics["rubric_set/diff_generated_fraction"] = len(diff_indices) / batch_size
                metrics["rubric_set/diff_degenerate_idlist_fraction"] = (
                    n_degenerate / max(len(diff_indices), 1)
                )

        return rubric_set_text, rubric_set_reward, diff_text, metrics

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode != "sdpo":
            return None

        # If a rubric_template_path is configured, load it from yaml ONCE and override
        # `rubric_template`. This lets users keep a multi-line template with '#'/'{}'
        # in a yaml file, without fighting shell/Hydra CLI escaping.
        if not getattr(self, "_rubric_template_loaded_from_path", False):
            import yaml

            def _load_yaml_key(path: str, key: str, target_attr: str):
                """Load ``key`` from the yaml at ``path`` and override ``target_attr`` on the
                self_distillation config. Silently keeps the default on any failure."""
                if not path:
                    return
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = yaml.safe_load(f) or {}
                    tmpl = loaded.get(key)
                    if isinstance(tmpl, str) and tmpl.strip():
                        setattr(self_distillation_cfg, target_attr, tmpl)
                        print(f"[self_distillation] {target_attr} loaded from {path} "
                              f"(key='{key}', {len(tmpl)} chars)")
                    else:
                        print(f"[self_distillation] WARN: {path} has no valid '{key}' key; "
                              f"keeping default {target_attr}")
                except Exception as e:  # noqa: BLE001
                    print(f"[self_distillation] WARN: failed to load {target_attr} from "
                          f"{path}: {e}; keeping default {target_attr}")

            tmpl_path = self_distillation_cfg.get("rubric_template_path", None)
            # The hint yaml (rubric_template_path) holds the teacher-context wrappers:
            #   rubric_hint_template / diff_hint_template / demonstration_hint_template
            _load_yaml_key(tmpl_path, "rubric_hint_template", "rubric_template")
            if self_distillation_cfg.get("use_rubric_set_hint", False):
                _load_yaml_key(tmpl_path, "diff_hint_template", "diff_hint_template")
                _load_yaml_key(tmpl_path, "demonstration_hint_template", "demonstration_hint_template")
                # The rubric-select / diff-generation prompts live in their own yaml files.
                _load_yaml_key(
                    self_distillation_cfg.get("rubric_set_select_template_path", None),
                    "rubric_select_template", "rubric_set_select_template",
                )
                _load_yaml_key(
                    self_distillation_cfg.get("diff_prompt_template_path", None),
                    "diff_prompt_template", "diff_prompt_template",
                )
            self._rubric_template_loaded_from_path = True

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        distill_only = self_distillation_cfg.get("distill_only", False)
        sample_route = self_distillation_cfg.get("sample_route", False)
        joint_mode = self_distillation_cfg.get("joint_grpo_distill", False)
        # In joint mode the rubric is one of the 3 hint types (assigned per bucket), so we
        # need rubric extraction regardless of the legacy ``use_rubric_as_demonstration`` flag.
        use_rubric = self_distillation_cfg.get("use_rubric_as_demonstration", False) or joint_mode
        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold, use_mean_as_threshold=self_distillation_cfg.get("use_mean_as_threshold", False))
        seq_scores_np = reward_tensor.sum(dim=-1).detach().cpu().numpy()

        solution_results = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                seq_scores=seq_scores_np,
            )
            for i in range(batch_size)
        ]
        solution_strs = [r[0] for r in solution_results]
        solution_rewards = [r[1] for r in solution_results]

        # Extract rubrics from extra_info if use_rubric_as_demonstration is enabled
        rubric_strs: list[Optional[str]] = [None] * batch_size
        num_rubric_available = 0
        if use_rubric:
            for i in range(batch_size):
                ei = batch.non_tensor_batch.get("extra_info", None)
                if ei is not None:
                    rubric = ei[i].get("rubrics", None) if isinstance(ei[i], dict) else None
                    # ``rubrics`` may be a plain string (legacy) or the rubric-ranker
                    # dict schema. ``format_rubric_hint`` normalizes both to text.
                    rubric_text = format_rubric_hint(rubric)
                    if rubric_text:
                        rubric_strs[i] = rubric_text
                        num_rubric_available += 1

        # For distill_only / sample_route: fall back to ground truth for groups with no successful rollout
        num_gt_fallback = 0
        gt_fallback_flags = [False] * batch_size
        if distill_only or sample_route:
            for i in range(batch_size):
                if solution_strs[i] is None:
                    gt = batch.non_tensor_batch["reward_model"][i].get("ground_truth", None)
                    if gt is not None:
                        solution_strs[i] = str(gt)
                        num_gt_fallback += 1
                        gt_fallback_flags[i] = True

        # ------------------------------------------------------------------
        # Joint mode (L = L_RL + lambda_opsd * L_OPSD): assign one of 3 hint
        # types per rollout by reward bucket and resolve its content.
        # ------------------------------------------------------------------
        hint_types: Optional[list[str]] = None
        group_best_str: list[Optional[str]] = [None] * batch_size
        group_best_reward: list[Optional[float]] = [None] * batch_size
        if joint_mode:
            hint_types = self._assign_hint_types(seq_scores_np, self_distillation_cfg)
            # Per-uid best rollout (highest reward), independent of success threshold, so the
            # "best_rollout" hint is always resolvable within a group.
            uids_arr = batch.non_tensor_batch["uid"]
            uid_to_best: dict[Any, int] = {}
            for i in range(batch_size):
                uid = uids_arr[i]
                if uid not in uid_to_best or seq_scores_np[i] > seq_scores_np[uid_to_best[uid]]:
                    uid_to_best[uid] = i
            for i in range(batch_size):
                best_idx = uid_to_best[uids_arr[i]]
                group_best_str[i] = response_texts[best_idx]
                group_best_reward[i] = float(seq_scores_np[best_idx])

        # ------------------------------------------------------------------
        # rubric_set adaptive-hint branch: generate rubric_set (+ r_set) and the
        # per-rollout diff with the CURRENT rollout policy. Populated only when
        # use_rubric_set_hint is enabled; failures degrade gracefully to the
        # abstract rubric hint (arrays stay None).
        # ------------------------------------------------------------------
        use_rubric_set_hint = joint_mode and self_distillation_cfg.get("use_rubric_set_hint", False)
        rubric_set_text: list[Optional[str]] = [None] * batch_size
        rubric_set_reward: list[Optional[float]] = [None] * batch_size
        diff_text: list[Optional[str]] = [None] * batch_size
        rubric_set_metrics: dict[str, float] = {}
        bar_medium = float(self_distillation_cfg.get("rubric_set_bar_medium", 6.0))
        bar_low = float(self_distillation_cfg.get("rubric_set_bar_low", 3.0))
        gate_relative = bool(self_distillation_cfg.get("rubric_set_gate_relative", True))
        gate_margin = float(self_distillation_cfg.get("rubric_set_gate_margin", 0.0))
        if use_rubric_set_hint:
            try:
                rubric_set_text, rubric_set_reward, diff_text, rubric_set_metrics = (
                    self._generate_rubric_set_and_diff(
                        batch, response_texts, seq_scores_np, hint_types, rubric_strs, self_distillation_cfg
                    )
                )
            except Exception as e:  # noqa: BLE001
                print(f"[rubric_set] WARN: rubric_set/diff generation failed: {e}; "
                      f"falling back to abstract rubric hint for all buckets")
                import traceback
                traceback.print_exc()

        def _build_reflection(i: int, hint_kind: str) -> str:
            """Placeholder reflection text for self_rollout / best_rollout hints.

            NOTE: This is a stub. The real reflection source (e.g. a critique model or
            environment feedback) is expected to be wired in later. It intentionally keeps
            a stable interface so swapping the content does not touch the loss path.
            """
            fb = feedback_list[i]
            if fb is not None and isinstance(fb, str) and fb.strip():
                return fb.strip()
            return f"[reflection placeholder for {hint_kind}]"

        def _resolve_joint_hint(i: int) -> tuple[Optional[str], Optional[float], Optional[str], str]:
            """Resolve (content, reward, reflection, resolved_kind) for the sample's hint type.

            Falls back to "self_rollout" when the chosen hint's content is unavailable, so
            every rollout still receives a hint (distillation loss covers all rollouts).
            """
            kind = hint_types[i]
            if kind == "none":
                return None, None, None, "none"
            # ---- rubric_set branch: diff (medium) / demonstration (low) ----
            if kind == "diff":
                # Use diff only if it was generated (medium gate r_set >= bar_medium passed).
                if diff_text[i] is not None and str(diff_text[i]).strip():
                    return diff_text[i], rubric_set_reward[i], None, "diff"
                kind = "rubric"  # gate failed / unavailable -> fall back to abstract rubric
            if kind == "demonstration":
                # Low gate: relative mode -> use the rubric_set example only if it scores higher
                # than THIS rollout's own reward r_i (r_set > r_i + margin); absolute mode ->
                # r_set >= bar_low. Otherwise fall back to the abstract rubric.
                if rubric_set_text[i] is not None and rubric_set_reward[i] is not None:
                    if gate_relative:
                        gate_ok = rubric_set_reward[i] > float(seq_scores_np[i]) + gate_margin
                    else:
                        gate_ok = rubric_set_reward[i] >= bar_low
                    if gate_ok:
                        return rubric_set_text[i], rubric_set_reward[i], None, "demonstration"
                kind = "rubric"  # gate failed / unavailable -> fall back to abstract rubric
            if kind == "rubric":
                if rubric_strs[i] is not None:
                    return rubric_strs[i], None, None, "rubric"
                kind = "self_rollout"  # fallback
            if kind == "best_rollout":
                if group_best_str[i] is not None:
                    return group_best_str[i], group_best_reward[i], _build_reflection(i, "best_rollout"), "best_rollout"
                kind = "self_rollout"  # fallback
            # self_rollout (default / fallback): the rollout's own response
            return response_texts[i], float(seq_scores_np[i]), _build_reflection(i, "self_rollout"), "self_rollout"

        def _inject_reference(prompt_text: str, reference_section: str) -> str:
            last_output_marker = "### Output"
            last_idx = prompt_text.rfind(last_output_marker)
            if last_idx != -1:
                return prompt_text[:last_idx] + reference_section
            return prompt_text + "\n\n" + reference_section

        def _build_joint_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            content, reward, reflection, kind = _resolve_joint_hint(i)
            prompt_text = prompt_texts[i]
            # ---- rubric_set branch hint blocks (diff / demonstration) ----
            # Both are injected exactly like the abstract rubric hint (before the final
            # ranking / output instruction), so only the wrapper template differs.
            if content is not None and kind in ("rubric", "diff", "demonstration"):
                if kind == "rubric":
                    hint_block = self_distillation_cfg.rubric_template.format(rubric=content)
                elif kind == "diff":
                    hint_block = self_distillation_cfg.diff_hint_template.format(diff=content)
                else:  # demonstration
                    reward_val = f"{reward:.2f}" if reward is not None else "N/A"
                    hint_block = self_distillation_cfg.demonstration_hint_template.format(
                        rubric_set=content, reward=reward_val
                    )
                if "### Output" in prompt_text:
                    
                    reference_section = (
                        f"### Reference\n"
                        f"{hint_block}\n\n"
                        f"### Output"
                    )
                    prompt_text = _inject_reference(prompt_text, reference_section)
                else:
                    # Rubric-ranker format (2.json / test.json): inject the hint right before
                    
                    prompt_text = inject_hint_before_instruction(prompt_text, hint_block)
            elif content is not None:
                reward_val = f"{reward:.2f}" if reward is not None else "N/A"
                reflection_section = (
                    self_distillation_cfg.reflection_template.format(reflection=reflection)
                    if reflection is not None else ""
                )
                reference_section = (
                    f"### Reference\n"
                    f"{content}\n"
                    f"The document subset above scored {reward_val} out of 10 and can be used as a reference.\n"
                    f"{reflection_section}\n"
                    f"### Output"
                )
                prompt_text = _inject_reference(prompt_text, reference_section)
            return system_messages + [
                {"role": "user", "content": prompt_text},
            ]

        def _build_teacher_message(i: int) -> list[dict]:
            if joint_mode:
                return _build_joint_teacher_message(i)
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            has_rubric = use_rubric and rubric_strs[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # build solution section (rubric replaces solution when enabled)
            solution_section = ""
            if has_rubric:
                # Use rubric as the demonstration instead of successful rollout / ground truth
                solution_section = self_distillation_cfg.rubric_template.format(
                    rubric=rubric_strs[i]
                )
            elif has_solution:
                reward_val = f"{solution_rewards[i]:.2f}" if solution_rewards[i] is not None else ""
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i],
                    reward=reward_val,
                )

            # build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_section = self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_list[i]
                )

            
            prompt_text = prompt_texts[i]
            if has_solution and not has_rubric:
                reward_val = f"{solution_rewards[i]:.2f}" if solution_rewards[i] is not None else "N/A"
                reference_section = (
                    f"### Reference\n"
                    f"{solution_strs[i]}\n"
                    f"The document subset above scored {reward_val} out of 10. It is a useful reference but may not be globally optimal.\n\n"
                    f"### Output"
                )
                
                last_output_marker = "### Output"
                last_idx = prompt_text.rfind(last_output_marker)
                if last_idx != -1:
                    prompt_text = prompt_text[:last_idx] + reference_section
                else:
                    # Fallback: append reference section at the end
                    prompt_text = prompt_text + "\n\n" + reference_section

            # combine solution and feedback sections
            # if use_feedback or has_solution or has_rubric:
            #     reprompt_text = self_distillation_cfg.reprompt_template.format(
            #         prompt=prompt_text,
            #         solution="" if (has_solution and not has_rubric) else solution_section,
            #         feedback=feedback_section,
            #     )
            # else:
            reprompt_text = prompt_text

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]

        # Log one sample reprompt for debugging
        # if messages:
        #     import logging
        #     logger = logging.getLogger(__name__)
        #     sample_msg = messages[0][-1]["content"] if messages[0] else ""
        #     print(f"[Self-Distillation] Sample reprompt (idx=0, reward={solution_rewards[0]}):\n{sample_msg}")

        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", False) if self.config.data.apply_chat_template_kwargs else False
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Per-sample incorrect_format flag from the reward function (result["incorrect_format"]).
        # Carried into the actor as a tensor so the sample_route branch can, when
        # pg_only_on_incorrect_format is enabled, route ONLY malformed wrong samples to the
        # PG loss (and exclude them from distillation). Defaults to all-zero when unavailable.
        incorrect_format_raw = (reward_extra_infos_dict or {}).get("incorrect_format", None)
        if incorrect_format_raw is not None and len(incorrect_format_raw) == batch_size:
            incorrect_format_mask = torch.tensor(
                [1.0 if bool(v) else 0.0 for v in incorrect_format_raw],
                dtype=torch.float32,
                device=device,
            )
        else:
            incorrect_format_mask = torch.zeros(batch_size, dtype=torch.float32, device=device)

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask semantics by mode:
        # - normal SDPO: mask=1 for samples with teacher demonstration, mask=0 excluded from distillation
        # - distill_only: mask=1 for all samples (forced full distillation)
        # - sample_route: mask=1 for wrong samples (use KL distillation), mask=0 for correct samples (use PG)
        #   Wrong sample = reward < threshold; correct sample = reward >= threshold
        #   mask=-1 for noise samples (reward std too low, excluded from all losses)
        num_noise_samples = 0
        num_kept_high_stable = 0
        # - joint_grpo_distill: mask=1 for every rollout that receives a hint (distillation
        #   covers all such rollouts); mask=0 only for buckets whose hint type is "none".
        #   GRPO PG is applied to ALL rollouts separately in the actor (independent of mask).
        #   When all rollouts fail: all_fail_fallback="gt" -> mask=1 (use GT demonstration), "grpo" -> mask=0 (use GRPO)
        #   When use_rubric=True: wrong samples with rubric available always use distillation (mask=1)
        joint_excluded_incorrect_format_fraction = 0.0
        if joint_mode:
            # Every rollout gets a hint unless its bucket type resolves to "none".
            joint_mask_vals = [0.0 if hint_types[i] == "none" else 1.0 for i in range(batch_size)]
            
            if self_distillation_cfg.get("joint_exclude_incorrect_format", False):
                n_excluded = 0
                for i in range(batch_size):
                    if joint_mask_vals[i] > 0 and incorrect_format_mask[i] > 0:
                        joint_mask_vals[i] = 0.0
                        n_excluded += 1
                joint_excluded_incorrect_format_fraction = n_excluded / max(batch_size, 1)
            self_distillation_mask = torch.tensor(
                joint_mask_vals,
                dtype=torch.float32,
                device=device,
            )
        elif sample_route:
            seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
            all_fail_fallback = self_distillation_cfg.get("all_fail_fallback", "gt")
            uids_list = batch.non_tensor_batch["uid"]
            use_mean_threshold = self_distillation_cfg.get("use_mean_as_threshold", False)

            
            noise_std_threshold = self_distillation_cfg.get("noise_std_threshold", 0.0)
            if use_mean_threshold:
                uid_to_indices_route: dict[Any, list[int]] = defaultdict(list)
                for i in range(batch_size):
                    uid_to_indices_route[uids_list[i]].append(i)
                uid_mean_scores = {
                    uid: sum(seq_scores[j] for j in indices) / len(indices)
                    for uid, indices in uid_to_indices_route.items()
                }
                
                uid_max_scores = {
                    uid: max(seq_scores[j] for j in indices)
                    for uid, indices in uid_to_indices_route.items()
                }
                
                uid_std_scores = {
                    uid: float(np.std([seq_scores[j] for j in indices]))
                    for uid, indices in uid_to_indices_route.items()
                }
                min_distill_threshold = self_distillation_cfg.success_reward_threshold
            else:
                
                if noise_std_threshold > 0:
                    uid_to_indices_route: dict[Any, list[int]] = defaultdict(list)
                    for i in range(batch_size):
                        uid_to_indices_route[uids_list[i]].append(i)
                    uid_std_scores = {
                        uid: float(np.std([seq_scores[j] for j in indices]))
                        for uid, indices in uid_to_indices_route.items()
                    }
                else:
                    uid_std_scores = {}

            mask_values = []
            for i in range(batch_size):
                uid = uids_list[i]
                
                # if noise_std_threshold > 0 and uid_std_scores.get(uid, float('inf')) <= noise_std_threshold:
                #     mask_values.append(-1.0)
                #     num_noise_samples += 1
                #     continue
                
                group_std_v = uid_std_scores.get(uid, float('inf'))
                group_mean_v = uid_mean_scores.get(uid, 0)  #
                if noise_std_threshold > 0 and group_std_v <= noise_std_threshold:
                    if group_mean_v >= self_distillation_cfg.success_reward_threshold:
                        
                        mask_values.append(0.0)
                        num_kept_high_stable += 1
                        continue
                    else:
                        
                        mask_values.append(-1.0)
                        num_noise_samples += 1
                        continue
                if use_mean_threshold:
                    
                    if uid_max_scores[uid] < min_distill_threshold:
                        mask_values.append(0.0)
                        continue
                    is_correct = seq_scores[i] >= uid_mean_scores[uid]
                else:
                    is_correct = seq_scores[i] >= self_distillation_cfg.success_reward_threshold
                if is_correct:
                    # Correct sample -> always use PG (mask=0)
                    mask_values.append(0.0)
                else:
                    # Wrong sample -> check demonstration availability
                    if use_rubric and rubric_strs[i] is not None:
                        # Rubric available -> always use distillation with rubric
                        mask_values.append(1.0)
                    elif gt_fallback_flags[i]:
                        # All rollouts in this group failed, using GT fallback
                        if all_fail_fallback == "gt":
                            mask_values.append(1.0)  # Use GT demonstration, walk SDPO
                        else:
                            mask_values.append(0.0)  # Use GRPO instead
                    else:
                        # Group has successful rollouts, use them as demonstration
                        mask_values.append(1.0)
            self_distillation_mask = torch.tensor(
                mask_values,
                dtype=torch.float32,
                device=device
            )
        elif distill_only:
            self_distillation_mask = torch.ones(batch_size, dtype=torch.float32, device=device)
        else:
            self_distillation_mask = torch.tensor(
                [solution_strs[i] is not None or feedback_used[i] or (use_rubric and rubric_strs[i] is not None)
                 for i in range(batch_size)],
                dtype=torch.float32,
                device=device
            )

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)

        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid.get(uid, [])) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": (self_distillation_mask > 0).float().mean().item(),
            "self_distillation/distill_only": float(distill_only),
            "self_distillation/sample_route": float(sample_route),
            "self_distillation/gt_fallback_fraction": num_gt_fallback / batch_size,
            "self_distillation/dynamic_weighting": float(self_distillation_cfg.get("dynamic_weighting", False)),
            "self_distillation/all_fail_fallback_mode": 0.0 if self_distillation_cfg.get("all_fail_fallback", "gt") == "gt" else 1.0,
            "self_distillation/use_rubric": float(use_rubric),
            "self_distillation/rubric_available_fraction": num_rubric_available / batch_size if use_rubric else 0.0,
            "sample_route/noise_sample_fraction": num_noise_samples / batch_size if sample_route else 0.0,
            "sample_route/kept_high_fraction": num_kept_high_stable / batch_size if sample_route else 0.0,
            "sample_route/incorrect_format_fraction": incorrect_format_mask.mean().item(),
            # Fraction of samples that are BOTH wrong (mask>0) AND incorrect_format==1.
            # When pg_only_on_incorrect_format is enabled these are the wrong samples routed to PG (not distillation).
            "sample_route/wrong_incorrect_format_fraction": (
                ((self_distillation_mask > 0).float() * incorrect_format_mask).mean().item()
            ),
            "self_distillation/joint_grpo_distill": float(joint_mode),
            
            "joint/excluded_incorrect_format_fraction": joint_excluded_incorrect_format_fraction,
        }
        if joint_mode and hint_types is not None:
            # Assigned bucket fractions (before r_set gating / fallback).
            for _kind in ["rubric", "self_rollout", "best_rollout", "none", "diff", "demonstration"]:
                metrics[f"joint/hint_{_kind}_fraction"] = (
                    sum(1 for t in hint_types if t == _kind) / batch_size
                )
        if use_rubric_set_hint:
            # Resolved-kind fractions (after r_set gating / fallback) so we can see how many
            # medium/low buckets actually kept diff/demonstration vs. fell back to rubric.
            resolved_counts: dict[str, int] = defaultdict(int)
            for i in range(batch_size):
                _, _, _, rk = _resolve_joint_hint(i)
                resolved_counts[rk] += 1
            for _kind in ["rubric", "self_rollout", "best_rollout", "none", "diff", "demonstration"]:
                metrics[f"rubric_set/resolved_{_kind}_fraction"] = resolved_counts.get(_kind, 0) / batch_size
            metrics.update(rubric_set_metrics)

        # ------------------------------------------------------------------
        
        
        
        
        # ------------------------------------------------------------------
        distill_data_dir = self.config.trainer.get("distill_data_dir", None)
        if joint_mode and distill_data_dir:
            try:
                self._dump_distill_details(
                    batch=batch,
                    batch_size=batch_size,
                    seq_scores_np=seq_scores_np,
                    hint_types=hint_types,
                    rubric_strs=rubric_strs,
                    rubric_set_text=rubric_set_text,
                    rubric_set_reward=rubric_set_reward,
                    diff_text=diff_text,
                    response_texts=response_texts,
                    messages=messages,
                    resolve_fn=_resolve_joint_hint,
                    dump_path=distill_data_dir,
                )
            except Exception as e:  # noqa: BLE001
                import logging
                logging.getLogger(__name__).warning(
                    f"[tutoring_log] failed to write diagnostics; ignoring: {e}"
                )

        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
            "incorrect_format_mask": incorrect_format_mask,
        }), metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # Keep rubric_set / rubric_set_reward on the training batch (not needed for generation):
        # with rubric_set_source=data the self-distillation branch reads them directly from the
        # data instead of re-selecting a subset with the policy. Without this they'd be popped
        # into gen_batch and lost from the batch that reaches _maybe_build_self_distillation_batch.
        reward_model_keys = set(
            {"data_source", "reward_model", "extra_info", "uid", "raw_prompt", "index",
             "rubric_set", "rubric_set_reward"}
        ) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_indices_collected = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # Collect sample indices if available
            raw_indices = test_batch.non_tensor_batch.get("index", None)
            if raw_indices is not None:
                sample_indices_collected.extend(raw_indices.tolist() if hasattr(raw_indices, "tolist") else list(raw_indices))
            else:
                sample_indices_collected.extend([None] * len(input_texts))

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Keys to skip in reward dump (same as _log_rollout_data)
            _METRIC_ONLY_KEYS = {"len(pred)", "comprehensive", "conflict", "redundancy", "richness", "relevance", "truthfulness", "quality", "authority", "timeliness"}
            _SKIP_KEYS = {"score", "reward_invalid"}
            # eval_system is written to rollout log (not metric-only, not skipped)

            val_reward_info = {"score": sample_scores}
            for k, v in reward_extra_infos_dict.items():
                if k in _SKIP_KEYS or k in _METRIC_ONLY_KEYS or k == "reward":
                    continue
                if isinstance(v, list) and len(v) == len(sample_inputs):
                    val_reward_info[k] = v

            # Extract sample_index from collected indices
            val_sample_indices = sample_indices_collected if any(x is not None for x in sample_indices_collected) else None
            # sample_uids is already collected above

            val_rollout_n = self.config.actor_rollout_ref.rollout.val_kwargs.n

            self._dump_generations_fine(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                reward_info=val_reward_info,
                dump_path=val_data_dir,
                validate=True,
                rollout_n=val_rollout_n,
                uids=sample_uids if sample_uids else None,
                sample_indices=val_sample_indices,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        # Exclude reward-invalid samples from validation metrics.
        # Build valid indices (where reward_invalid is 0 or absent).
        val_reward_invalid_flags = reward_extra_infos_dict.get("reward_invalid", None)
        if val_reward_invalid_flags is not None and any(val_reward_invalid_flags):
            n_val_total = len(sample_scores)
            valid_indices = [i for i, f in enumerate(val_reward_invalid_flags) if not f]
            n_val_invalid = n_val_total - len(valid_indices)
            print(
                f"[ValRewardInvalid] Excluding {n_val_invalid}/{n_val_total} validation samples "
                f"with invalid reward from metrics."
            )
            # Filter all parallel lists
            data_source_lst_flat = np.concatenate(data_source_lst, axis=0)
            data_source_lst_filtered = data_source_lst_flat[valid_indices]
            sample_uids_filtered = [sample_uids[i] for i in valid_indices]
            filtered_infos_dict: dict[str, list] = {}
            for k, v in reward_extra_infos_dict.items():
                if k == "reward_invalid":
                    continue
                if isinstance(v, list) and len(v) == n_val_total:
                    filtered_infos_dict[k] = [v[i] for i in valid_indices]
            # sample_turns is a list of arrays, cannot easily filter here; pass empty
            sample_turns_filtered = []
        else:
            data_source_lst_filtered = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])
            sample_uids_filtered = sample_uids
            filtered_infos_dict = {k: v for k, v in reward_extra_infos_dict.items() if k != "reward_invalid"}
            sample_turns_filtered = sample_turns
            n_val_invalid = 0

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": [data_source_lst_filtered] if isinstance(data_source_lst_filtered, np.ndarray) else data_source_lst,
                "sample_uids": sample_uids_filtered,
                "sample_turns": sample_turns_filtered,
                "reward_extra_infos_dict": filtered_infos_dict,
            }
        metric_dict = self._val_metrics_update(data_source_lst_filtered, sample_uids_filtered, filtered_infos_dict, sample_turns_filtered)
        if n_val_invalid > 0:
            metric_dict["val-aux/reward_invalid_fraction"] = n_val_invalid / (n_val_invalid + len(valid_indices)) if (n_val_invalid + len(valid_indices)) > 0 else 0.0
        return metric_dict

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if f"@{n_max}" not in metric_name:
                        continue
                    if not any(metric_name.startswith(pfx) for pfx in ["mean"]):
                        continue
                    if metric_name.endswith("/std"):
                        continue
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    elif var_name == "overall_score_jp" and f"@{n_max}" in metric_name:
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Skip data steps without training (for resuming from earlier checkpoint but advancing data)
        skip_steps = self.config.trainer.get("skip_steps", 0)
        if skip_steps > 0:
            print(f"[SkipSteps] Skipping {skip_steps} data batches from dataloader without training...")
            skipped = 0
            for batch_dict in self.train_dataloader:
                skipped += 1
                if skipped >= skip_steps:
                    break
            print(f"[SkipSteps] Successfully skipped {skipped} batches. Training will start from data position step {self.global_steps + skipped - 1}.")

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # Exclude reward-invalid samples (e.g. RM timeout/error) from training.
                        # These samples should not contribute to GRPO group advantage or any loss.
                        reward_invalid_flags = reward_extra_infos_dict.get("reward_invalid", None)
                        if reward_invalid_flags is not None:
                            n_invalid = 0
                            for i, flag in enumerate(reward_invalid_flags):
                                if flag:
                                    n_invalid += 1
                                    # Zero out response_mask so loss functions ignore this sample
                                    batch.batch["response_mask"][i] = 0
                                    # Zero out token_level_scores so it doesn't affect reward stats
                                    batch.batch["token_level_scores"][i] = 0
                                    # Assign a unique uid so GRPO grouping excludes this sample
                                    # (single-sample group → advantage = 0)
                                    batch.non_tensor_batch["uid"][i] = f"__invalid_{i}_{uuid.uuid4()}"
                            batch_size = len(reward_invalid_flags)
                            metrics["reward_extra/invalid_fraction"] = n_invalid / batch_size if batch_size > 0 else 0.0
                            if n_invalid > 0:
                                print(
                                    f"[RewardInvalid] Excluded {n_invalid}/{batch_size} samples "
                                    f"with invalid reward (timeout/error) from training."
                                )

                        # Assign a stable per-rollout unique id BEFORE building the distill batch
                        # so the (later, actor-side) per-rollout / per-token OPSD advantage dumps can
                        # be joined back to {step}_distill.jsonl on `rollout_uid`.
                        self._maybe_add_rollout_uid(batch)

                        # Use the sanitized token_level_scores (reward_invalid samples already zeroed out)
                        # instead of the raw reward_tensor, so invalid samples are not mistakenly
                        # treated as "successful" in self-distillation grouping.
                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, batch.batch["token_level_scores"], reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                            and batch.batch["response_mask"].any()  # Skip if all response_masks are zero (e.g. all reward invalid)
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # Apply robust winsorization to token_level_rewards per uid group (before advantage computation)
                        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
                        if self_distillation_cfg and self_distillation_cfg.get("reward_winsorize", False):
                            batch = self._winsorize_rewards_by_uid(
                                batch,
                                reward_key="token_level_rewards",
                                trim_n=self_distillation_cfg.get("reward_winsorize_trim_n", 1),
                                max_deviation=self_distillation_cfg.get("reward_winsorize_max_deviation", 3.0),
                                variance_threshold=self_distillation_cfg.get("reward_winsorize_variance_threshold", 0.0),
                            )

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # Decide whether to dump per-rollout / per-token OPSD advantage this step.
                        # The tensors only exist inside the actor update (advantage-stacking branch),
                        # so we pass the request via meta_info and merge the rank shards afterwards.
                        opsd_dump_dir, opsd_dump_token_n = self._maybe_arm_opsd_adv_dump(batch)
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        if opsd_dump_dir is not None:
                            self._merge_opsd_adv_dump_shards(opsd_dump_dir, self.global_steps)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        rollout_extra_metrics = self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        metrics.update(rollout_extra_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
