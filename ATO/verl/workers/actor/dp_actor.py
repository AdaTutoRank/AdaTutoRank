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
Single Process Actor
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_dynamic_entropy_weights,
    compute_sdpo_backend_parity_metrics,
    compute_self_distillation_advantage,
    compute_self_distillation_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "sdpo":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        # ---- OPSD advantage/logprob dump (query & token granularity) ----
        # Requested by the driver via meta_info; only collected in the advantage-stacking branch
        # where per-token A_OPSD / combined_adv exist. Records are buffered here and written as a
        # per-rank shard at the end of the update; the driver merges the shards afterwards.
        self._opsd_dump_enabled = bool(data.meta_info.get("opsd_adv_dump", False))
        self._opsd_dump_dir = data.meta_info.get("opsd_adv_dump_dir", None)
        self._opsd_dump_step = int(data.meta_info.get("opsd_adv_dump_step", 0))
        self._opsd_dump_token_n = int(data.meta_info.get("opsd_adv_dump_token_n", 0))
        self._opsd_adv_records = []
        self._opsd_token_records = []
        self._opsd_token_rollouts_seen = 0

        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"
            # incorrect_format_mask is optional (only present when built by the trainer's
            # self-distillation batch). Select it if available so the sample_route branch can
            # route malformed wrong samples to PG when pg_only_on_incorrect_format is enabled.
            if "incorrect_format_mask" in data.batch.keys():
                self_distillation_required_keys.add("incorrect_format_mask")

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        # Include uid for dynamic_weighting per_prompt normalization
        dynamic_weighting_enabled = (
            self_distillation_enabled
            and self_distillation_cfg.get("sample_route", False)
            and self_distillation_cfg.get("dynamic_weighting", False)
        )
        if dynamic_weighting_enabled and "uid" not in non_tensor_select_keys and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        # Carry join keys through to the micro-batches for the OPSD advantage/logprob dump.
        if self._opsd_dump_enabled:
            for k in ("rollout_uid", "uid", "index"):
                if k in data.non_tensor_batch.keys() and k not in non_tensor_select_keys:
                    non_tensor_select_keys.append(k)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        did_update = False
        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                metrics["backend_parity/mini_batch_local_samples"] = [float(len(mini_batch))]
                metrics["backend_parity/mini_batch_dp_global_samples"] = [
                    float(
                        len(mini_batch)
                        * (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
                    )
                ]
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    if self_distillation_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    if teacher_regularization == "trust-region" and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    # Determine distillation mode: fixed indices vs dynamic top-K vs full logits
                    fixed_indices_list = getattr(self_distillation_cfg, "distillation_fixed_indices", None)
                    use_fixed_indices = fixed_indices_list is not None and len(fixed_indices_list) > 0
                    if use_fixed_indices:
                        # Fixed indices mode: use pre-defined token IDs for all positions
                        return_all_logps = False
                        distill_topk = None
                        response_length = model_inputs["responses"].size(-1)
                        batch_size_micro = model_inputs["responses"].size(0)
                        # Build fixed topk_indices: (bs, response_len, k)
                        fixed_indices_tensor = torch.tensor(
                            fixed_indices_list, dtype=torch.long, device=model_inputs["responses"].device
                        )
                        # Expand to (bs, response_len, k)
                        student_fixed_topk_indices = fixed_indices_tensor.unsqueeze(0).unsqueeze(0).expand(
                            batch_size_micro, response_length, -1
                        )
                        outputs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            topk_indices=student_fixed_topk_indices,
                        )
                        student_topk_indices = student_fixed_topk_indices
                    else:
                        return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                        distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                        outputs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_all_logps=return_all_logps,
                            distill_topk=distill_topk,
                        )
                        student_topk_indices = outputs.get("topk_indices") if distill_topk else None
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if (distill_topk or use_fixed_indices) else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        sample_route = self_distillation_cfg.get("sample_route", False)
                        # NOTE: Do NOT skip teacher forward even if self_distillation_mask is all-zero
                        # in this micro-batch. In FSDP, all ranks must synchronously call forward on
                        # the same module. Different ranks may have different mask patterns per micro-batch,
                        # so skipping would cause NCCL AllGather deadlock / nan.
                        skip_teacher_forward = False

                        teacher_log_prob = None
                        teacher_all_logps = None
                        teacher_topk_logps = None
                        teacher_entropy = None

                        if not skip_teacher_forward:
                            teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                            }
                            teacher_model = self.teacher_module or self.actor_module
                            if teacher_regularization == "trust-region" and (
                                self.teacher_module is None or self.teacher_module is self.actor_module
                            ):
                                raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                            with torch.no_grad():
                                # Enable entropy calculation for teacher when dynamic_weighting is enabled
                                teacher_calculate_entropy = self_distillation_cfg.get("dynamic_weighting", False) and self_distillation_cfg.get("sample_route", False)
                                # For fixed indices mode, pass the same fixed indices to teacher
                                teacher_topk_indices_arg = student_fixed_topk_indices if use_fixed_indices else student_topk_indices
                                teacher_outputs = self._forward_micro_batch(
                                    teacher_inputs,
                                    temperature=temperature,
                                    calculate_entropy=teacher_calculate_entropy,
                                    return_all_logps=return_all_logps,
                                    distill_topk=distill_topk,
                                    topk_indices=teacher_topk_indices_arg,
                                    module=teacher_model,
                                )
                            teacher_log_prob = teacher_outputs["log_probs"]
                            teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                            teacher_topk_logps = teacher_outputs.get("topk_logps") if (distill_topk or use_fixed_indices) else None
                            teacher_entropy = teacher_outputs.get("entropys") if teacher_calculate_entropy else None

                        # Check if sample_route mode: positive samples use PG loss, negative samples use KL distillation
                        sample_route = self_distillation_cfg.get("sample_route", False)
                        joint_grpo_distill = self_distillation_cfg.get("joint_grpo_distill", False)
                        if joint_grpo_distill:
                            # Joint objective. GRPO policy-gradient signal covers ALL rollouts
                            # (identical to the GRPO baseline); OPSD self-distillation is combined
                            # on top, weighted by lambda_opsd, over all rollouts that received a hint
                            # (self_distillation_mask > 0). Two combine modes (opsd_combine_mode):
                            
                            
                            
                            lambda_opsd = float(self_distillation_cfg.get("lambda_opsd", 1.0))
                            opsd_combine_mode = self_distillation_cfg.get("opsd_combine_mode", "loss")
                            
                            
                            
                            opsd_combine_form = self_distillation_cfg.get("opsd_combine_form", "additive")
                            opsd_convex_lambda = float(self_distillation_cfg.get("opsd_convex_lambda", 0.7))
                            distill_alpha = float(self_distillation_cfg.get("alpha", 0.0))
                            distill_mask = (
                                (self_distillation_mask > 0).float()
                                if self_distillation_mask is not None else None
                            )
                            has_distill_samples = distill_mask is not None and distill_mask.sum().item() > 0

                            policy_loss_fn = get_policy_loss_fn("vanilla")

                            if opsd_combine_mode == "advantage":
                                
                                
                                
                                if has_distill_samples:
                                    # A_ATD is defined against the frozen rollout policy
                                    # pi_theta_old, not the policy after minibatch updates.
                                    distill_adv = compute_self_distillation_advantage(
                                        student_log_probs=old_log_prob,
                                        teacher_log_probs=teacher_log_prob,
                                        response_mask=response_mask,
                                        alpha=distill_alpha,
                                        self_distillation_mask=distill_mask,
                                    )
                                else:
                                    distill_adv = torch.zeros_like(advantages)

                                if opsd_combine_form == "convex":
                                    # Paper Eq. 10:
                                    # A_ATO = lam*A_GRPO + (1-lam)*m_i*A_ATD.
                                    # distill_adv already contains m_i, so invalid-format
                                    # samples retain only lam*A_GRPO (not full-scale GRPO).
                                    combined_adv = (
                                        opsd_convex_lambda * advantages
                                        + (1.0 - opsd_convex_lambda) * distill_adv
                                    )
                                else:
                                    combined_adv = advantages + lambda_opsd * distill_adv
                                pg_loss, pg_metrics = policy_loss_fn(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=combined_adv,
                                    response_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=rollout_is_weights,
                                )
                                pg_metrics = dict(pg_metrics)

                                
                                
                                rl_abs = agg_loss(
                                    loss_mat=advantages.abs(), loss_mask=response_mask,
                                    loss_agg_mode="token-mean",
                                    batch_num_tokens=response_mask.sum().clamp(min=1.0),
                                ).detach()
                                
                                distill_loss_mask = response_mask * (
                                    distill_mask.unsqueeze(1) if distill_mask is not None else 1.0
                                )
                                distill_abs = agg_loss(
                                    loss_mat=distill_adv.abs(), loss_mask=distill_loss_mask,
                                    loss_agg_mode="token-mean",
                                    batch_num_tokens=distill_loss_mask.sum().clamp(min=1.0),
                                ).detach()
                                rl_abs_v = rl_abs.item()
                                distill_abs_v = distill_abs.item()
                                
                                
                                if opsd_combine_form == "convex":
                                    rl_weight = opsd_convex_lambda
                                    opsd_weight = 1.0 - opsd_convex_lambda
                                else:
                                    rl_weight = 1.0
                                    opsd_weight = lambda_opsd
                                scaled_rl_abs_v = rl_weight * rl_abs_v
                                scaled_distill_abs_v = opsd_weight * distill_abs_v
                                pg_metrics["joint/combine_mode"] = 1.0  # 1=advantage, 0=loss
                                pg_metrics["joint/combine_form_convex"] = (
                                    1.0 if opsd_combine_form == "convex" else 0.0
                                )
                                pg_metrics["joint/adv_rl_abs_mean"] = rl_abs_v
                                pg_metrics["joint/adv_distill_abs_mean"] = distill_abs_v
                                pg_metrics["joint/adv_distill_scaled_abs_mean"] = scaled_distill_abs_v
                                pg_metrics["joint/adv_distill_to_rl_ratio"] = (
                                    scaled_distill_abs_v / (scaled_rl_abs_v + 1e-8)
                                )

                                # OPSD advantage/logprob dump (query & token granularity).
                                
                                if self._opsd_dump_enabled and epoch_idx == 0:
                                    self._collect_opsd_adv_dump(
                                        model_inputs=model_inputs,
                                        response_mask=response_mask,
                                        advantages=advantages,
                                        distill_adv=distill_adv,
                                        combined_adv=combined_adv,
                                        student_log_prob=log_prob,
                                        teacher_log_prob=teacher_log_prob,
                                        distill_mask=distill_mask,
                                    )
                            else:
                                
                                pg_loss_rl, pg_metrics = policy_loss_fn(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    response_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=rollout_is_weights,
                                )
                                pg_metrics = dict(pg_metrics)

                                if has_distill_samples:
                                    distill_loss, _ = compute_self_distillation_loss(
                                        student_log_probs=log_prob,
                                        teacher_log_probs=teacher_log_prob,
                                        response_mask=response_mask,
                                        self_distillation_config=self_distillation_cfg,
                                        old_log_probs=old_log_prob,
                                        student_all_log_probs=student_all_logps,
                                        teacher_all_log_probs=teacher_all_logps,
                                        student_topk_log_probs=student_topk_logps,
                                        teacher_topk_log_probs=teacher_topk_logps,
                                        self_distillation_mask=distill_mask,
                                        loss_agg_mode=loss_agg_mode,
                                        rollout_is_weights=rollout_is_weights,
                                    )
                                else:
                                    # Keep a graph-connected zero so FSDP post-backward hooks fire on all ranks.
                                    distill_loss = log_prob.sum() * 0.0

                                pg_loss = pg_loss_rl + lambda_opsd * distill_loss

                                
                                
                                rl_loss_v = pg_loss_rl.detach().item()
                                distill_loss_v = distill_loss.detach().item()
                                scaled_distill_loss_v = lambda_opsd * distill_loss_v
                                pg_metrics["joint/combine_mode"] = 0.0  # 1=advantage, 0=loss
                                pg_metrics["joint/rl_loss"] = rl_loss_v
                                pg_metrics["joint/opsd_loss"] = distill_loss_v
                                pg_metrics["joint/opsd_loss_scaled"] = scaled_distill_loss_v
                                pg_metrics["joint/opsd_to_rl_abs_ratio"] = (
                                    abs(scaled_distill_loss_v) / (abs(rl_loss_v) + 1e-8)
                                )

                            pg_metrics["joint/lambda_opsd"] = lambda_opsd
                            pg_metrics["joint/opsd_convex_lambda"] = opsd_convex_lambda
                            pg_metrics["joint/distill_alpha"] = distill_alpha
                            pg_metrics["joint/distill_sample_fraction"] = (
                                distill_mask.mean().item() if distill_mask is not None else 0.0
                            )
                            pg_metrics["self_distillation/empty_target_batch"] = float(not has_distill_samples)
                        elif sample_route and self_distillation_mask is not None:
                            # self_distillation_mask: 1 = wrong (use KL distillation), 0 = correct (use PG advantage), -1 = noise (skip all losses)
                            noise_mask = (self_distillation_mask < 0).float()  # (bsz,) noise samples
                            pg_mask = (self_distillation_mask == 0).float()    # (bsz,) correct samples
                            distill_mask = (self_distillation_mask > 0).float()  # (bsz,) wrong samples

                            # wrong_sample_add_pg and pg_only_on_incorrect_format are two MUTUALLY
                            # EXCLUSIVE ways to add PG loss to wrong samples (validated in config):
                            #   - wrong_sample_add_pg: ALL wrong samples get PG loss on top of distillation.
                            #   - pg_only_on_incorrect_format: ONLY malformed wrong samples (incorrect_format==1)
                            #     get PG loss and are EXCLUDED from distillation; other wrong samples stay
                            #     distillation-only.
                            # When neither is enabled, only correct samples get PG (pure SRPO).
                            wrong_sample_add_pg = self_distillation_cfg.get("wrong_sample_add_pg", False)
                            pg_only_on_incorrect_format = self_distillation_cfg.get("pg_only_on_incorrect_format", False)
                            incorrect_format_mask = model_inputs.get("incorrect_format_mask", None)
                            wrong_pg_fraction = 0.0
                            if wrong_sample_add_pg:
                                pg_apply_mask = (self_distillation_mask >= 0).float()  # correct + wrong (exclude noise=-1)
                            elif pg_only_on_incorrect_format and incorrect_format_mask is not None:
                                icf = (incorrect_format_mask > 0).float()          # (bsz,)
                                wrong_pg_mask = distill_mask * icf                  # wrong & malformed -> PG
                                pg_apply_mask = pg_mask + wrong_pg_mask             # correct + (wrong & malformed)
                                distill_mask = distill_mask * (1.0 - icf)           # drop malformed from distillation
                                wrong_pg_fraction = wrong_pg_mask.mean().item()
                            else:
                                pg_apply_mask = pg_mask
                            distill_weight = float(self_distillation_cfg.get("distill_weight", 1.0))

                            has_pg_samples = pg_apply_mask.sum().item() > 0
                            has_distill_samples = distill_mask.sum().item() > 0

                            # Use a dummy loss connected to the model's computation graph.
                            # A plain torch.tensor(0.0) has no grad_fn and won't trigger FSDP's
                            # post-backward hooks, causing NCCL AllGather timeout on other ranks.
                            total_loss = log_prob.sum() * 0.0
                            use_dynamic_weighting = self_distillation_cfg.get("dynamic_weighting", False) and teacher_entropy is not None
                            pg_metrics = {
                                "sample_route/pg_sample_fraction": pg_mask.mean().item(),
                                "sample_route/distill_sample_fraction": distill_mask.mean().item(),
                                "sample_route/noise_sample_fraction": noise_mask.mean().item(),
                                "sample_route/wrong_pg_fraction": wrong_pg_fraction,
                                "self_distillation/empty_target_batch": float(not has_pg_samples and not has_distill_samples),
                                "sample_route/pg_loss": 0.0,
                                "sample_route/distill_loss": 0.0,
                            }

                            # PG loss (correct samples, or all non-noise samples when wrong_sample_add_pg)
                            if has_pg_samples:
                                pg_response_mask = response_mask * pg_apply_mask.unsqueeze(1)
                                # Guard against all-zero mask (e.g. all PG samples were reward_invalid)
                                if pg_response_mask.sum() > 0:
                                    policy_loss_fn = get_policy_loss_fn("vanilla")
                                    pg_loss_pos, _ = policy_loss_fn(
                                        old_log_prob=old_log_prob,
                                        log_prob=log_prob,
                                        advantages=advantages,
                                        response_mask=pg_response_mask,
                                        loss_agg_mode=loss_agg_mode,
                                        config=self.config,
                                        rollout_is_weights=rollout_is_weights,
                                    )
                                    total_loss = total_loss + pg_loss_pos
                                    pg_metrics["sample_route/pg_loss"] = pg_loss_pos.detach().item()

                            # KL distillation loss for negative samples (mask out positive samples)
                            if has_distill_samples:
                                # Compute dynamic entropy-based weights if enabled (SRPO)
                                dw_weights = None
                                if use_dynamic_weighting:
                                    distill_loss_mask = response_mask * distill_mask.unsqueeze(1)
                                    uids_for_dw = micro_batch.non_tensor_batch.get("uid", None) if self_distillation_cfg.get("entropy_norm_mode", "per_sample") == "per_prompt" else None
                                    dw_weights, _ = compute_dynamic_entropy_weights(
                                        teacher_entropy=teacher_entropy,
                                        loss_mask=distill_loss_mask,
                                        entropy_norm_mode=self_distillation_cfg.get("entropy_norm_mode", "per_sample"),
                                        uids=uids_for_dw,
                                    )

                                distill_loss, _ = compute_self_distillation_loss(
                                    student_log_probs=log_prob,
                                    teacher_log_probs=teacher_log_prob,
                                    response_mask=response_mask,
                                    self_distillation_config=self_distillation_cfg,
                                    old_log_probs=old_log_prob,
                                    student_all_log_probs=student_all_logps,
                                    teacher_all_log_probs=teacher_all_logps,
                                    student_topk_log_probs=student_topk_logps,
                                    teacher_topk_log_probs=teacher_topk_logps,
                                    self_distillation_mask=distill_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    rollout_is_weights=rollout_is_weights,
                                    dynamic_weights=dw_weights,
                                )
                                total_loss = total_loss + distill_weight * distill_loss
                                pg_metrics["sample_route/distill_loss"] = distill_loss.detach().item()

                            pg_loss = total_loss
                        else:
                            # Original SDPO: pure distillation loss for all samples
                            pg_loss, pg_metrics = compute_self_distillation_loss(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                response_mask=response_mask,
                                self_distillation_config=self_distillation_cfg,
                                old_log_probs=old_log_prob,
                                student_all_log_probs=student_all_logps,
                                teacher_all_log_probs=teacher_all_logps,
                                student_topk_log_probs=student_topk_logps,
                                teacher_topk_log_probs=teacher_topk_logps,
                                self_distillation_mask=self_distillation_mask,
                                loss_agg_mode=loss_agg_mode,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0

                        micro_batch_metrics.update(pg_metrics)
                        parity_metrics = compute_sdpo_backend_parity_metrics(
                            student_log_probs=log_prob,
                            old_log_probs=old_log_prob,
                            response_mask=response_mask,
                            teacher_log_probs=teacher_log_prob,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            total_loss=pg_loss,
                            pg_loss=pg_metrics.get("sample_route/pg_loss"),
                            distill_loss=pg_metrics.get("sample_route/distill_loss"),
                        )
                        micro_batch_metrics.update(parity_metrics)
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None and response_mask.any():
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()

        if self._opsd_dump_enabled:
            self._flush_opsd_adv_dump()

        # Reduce list-valued metrics to scalars before returning.
        # This is necessary because DataProto.concat merges metrics from multiple
        # DP ranks via list_of_dict_to_dict_of_list, which would create
        # list-of-lists if values are already lists. np.mean on list-of-lists
        # with different lengths causes ValueError (inhomogeneous shape).
        import numpy as np
        for k, v in list(metrics.items()):
            if isinstance(v, list) and len(v) > 0:
                key_lower = k.lower()
                if "max" in key_lower:
                    metrics[k] = float(np.max(v))
                elif "min" in key_lower:
                    metrics[k] = float(np.min(v))
                else:
                    metrics[k] = float(np.mean(v))

        return metrics

    @staticmethod
    def _masked_mean(mat_row: torch.Tensor, mask_row: torch.Tensor) -> float:
        denom = mask_row.sum().clamp(min=1.0)
        return float((mat_row * mask_row).sum().item() / denom.item())

    def _collect_opsd_adv_dump(
        self,
        *,
        model_inputs,
        response_mask,
        advantages,
        distill_adv,
        combined_adv,
        student_log_prob,
        teacher_log_prob,
        distill_mask,
    ):
        """Buffer per-rollout (query granularity) and per-token (token granularity) OPSD stats
        for the current micro-batch. All inputs are existing (bsz, response_len) tensors from the
        advantage-stacking branch, so this adds only cheap CPU copies + python bookkeeping."""
        bsz = response_mask.shape[0]
        rollout_uid = model_inputs.get("rollout_uid")
        uid = model_inputs.get("uid")
        sample_index = model_inputs.get("index")
        responses = model_inputs["responses"]
        tok = getattr(self, "tokenizer", None)

        # Move to CPU float once for the whole micro-batch.
        mask_c = response_mask.detach().float().cpu()
        adv_c = advantages.detach().float().cpu()
        dadv_c = distill_adv.detach().float().cpu()
        cadv_c = combined_adv.detach().float().cpu()
        slp_c = student_log_prob.detach().float().cpu()
        tlp_c = teacher_log_prob.detach().float().cpu() if teacher_log_prob is not None else None
        dmask_c = distill_mask.detach().float().cpu() if distill_mask is not None else None
        resp_c = responses.detach().cpu()

        def _rid(j):
            return str(rollout_uid[j]) if rollout_uid is not None else None

        def _uid(j):
            return str(uid[j]) if uid is not None else None

        def _sidx(j):
            if sample_index is None:
                return None
            v = sample_index[j]
            return v.item() if hasattr(v, "item") else v

        for j in range(bsz):
            m = mask_c[j]
            valid = m.bool()
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            got_hint = bool(dmask_c[j].item() > 0) if dmask_c is not None else False
            self._opsd_adv_records.append(
                {
                    "step": self._opsd_dump_step,
                    "rollout_uid": _rid(j),
                    "uid": _uid(j),
                    "sample_index": _sidx(j),
                    "got_hint": got_hint,
                    "num_response_tokens": n_valid,
                    "a_rl": self._masked_mean(adv_c[j], m),
                    "a_opsd_mean": self._masked_mean(dadv_c[j], m),
                    "a_opsd_abs_mean": self._masked_mean(dadv_c[j].abs(), m),
                    "combined_adv_mean": self._masked_mean(cadv_c[j], m),
                    "teacher_logp_mean": (self._masked_mean(tlp_c[j], m) if tlp_c is not None else None),
                    "student_logp_mean": self._masked_mean(slp_c[j], m),
                }
            )

            # Token-level dump for the first N rollouts encountered on this rank.
            if self._opsd_token_rollouts_seen >= self._opsd_dump_token_n:
                continue
            self._opsd_token_rollouts_seen += 1

            valid_pos = valid.nonzero(as_tuple=True)[0].tolist()
            token_ids = [int(resp_c[j, p].item()) for p in valid_pos]
            if tok is not None:
                try:
                    token_strs = tok.convert_ids_to_tokens(token_ids)
                except Exception:  # noqa: BLE001
                    token_strs = [str(t) for t in token_ids]
            else:
                token_strs = [str(t) for t in token_ids]

            for local_idx, p in enumerate(valid_pos):
                self._opsd_token_records.append(
                    {
                        "step": self._opsd_dump_step,
                        "rollout_uid": _rid(j),
                        "uid": _uid(j),
                        "sample_index": _sidx(j),
                        "token_idx": local_idx,
                        "token_id": token_ids[local_idx],
                        "token_str": token_strs[local_idx],
                        "student_logp": float(slp_c[j, p].item()),
                        "teacher_logp": (float(tlp_c[j, p].item()) if tlp_c is not None else None),
                        "a_opsd": float(dadv_c[j, p].item()),
                        "combined_adv": float(cadv_c[j, p].item()),
                    }
                )

    def _flush_opsd_adv_dump(self):
        """Write this rank's buffered OPSD dump records to per-rank shard files. The driver merges
        (and de-duplicates across sequence-parallel ranks) after the actor update returns."""
        if not self._opsd_dump_dir:
            return
        if not self._opsd_adv_records and not self._opsd_token_records:
            return
        import json

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        os.makedirs(self._opsd_dump_dir, exist_ok=True)
        step = self._opsd_dump_step

        def _write(records, kind):
            if not records:
                return
            fn = os.path.join(self._opsd_dump_dir, f"{step}_{kind}.part{rank}.jsonl")
            with open(fn, "w") as f:
                for e in records:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")

        _write(self._opsd_adv_records, "distill_adv")
        _write(self._opsd_token_records, "distill_token")
