# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode == "sdpo".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "none".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    use_mean_as_threshold: bool = False  # When True, use per-uid mean reward as threshold instead of fixed success_reward_threshold
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.0
    # When True, the Megatron actor's `_update_teacher` will load the teacher
    # module to GPU before the EMA step and offload it back to CPU afterwards.
    # When False (default), it assumes the worker layer has already put the
    # teacher on GPU for the duration of `update_actor` (cheaper: one
    # load/offload pair per training step instead of two).
    teacher_ema_offload: bool = False
    distillation_topk: Optional[int] = None
    distillation_fixed_indices: Optional[List[int]] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "  Here is a strong result that can be used as a reference:\n\n"
        "  {successful_previous_attempt}\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    distill_only: bool = False  # When True, all samples participate in distillation regardless of reward
    sample_route: bool = False  # When True, positive samples use PG advantage loss, negative samples use KL distillation with ground truth
    dynamic_weighting: bool = False  # When True (requires sample_route=True), apply entropy-based dynamic weighting to SDPO distillation loss
    entropy_norm_mode: str = "per_sample"  # Normalization scope for dynamic weights: "per_prompt" or "per_sample"
    all_fail_fallback: str = "gt"  # Strategy when all rollouts in a group fail: "gt" (use ground truth as demonstration) or "grpo" (use GRPO advantage)
    wrong_sample_add_pg: bool = False  # When True (requires sample_route=True), wrong samples ALSO receive the negative-advantage GRPO PG loss on top of the distillation loss (SRPO+GRPO hybrid). This restores the "suppress malformed/verbose outputs" signal that pure SRPO lacks, while distillation still provides the directional target.
    pg_only_on_incorrect_format: bool = True  # When True (requires sample_route=True; MUTUALLY EXCLUSIVE with wrong_sample_add_pg), among wrong samples ONLY those with incorrect_format==1 receive the GRPO PG loss, and those PG samples are EXCLUDED from the distillation loss. The remaining well-formatted low-score wrong samples keep distillation-only. Enabling both this and wrong_sample_add_pg raises a config error.
    distill_weight: float = 1.0  # Scaling factor applied to the sample_route distillation loss when combined with PG. Only affects the total loss magnitude, not the logged raw sample_route/distill_loss.
    use_rubric_as_demonstration: bool = False  # When True, use rubric from extra_info as teacher demonstration instead of successful rollout / ground truth
    rubric_template: str = (
        "\n"
        "  The optimal subset should cover these query-specific points:\n\n"
        "  {rubric}\n"
    )  # Template for formatting rubric section. Uses {rubric} placeholder.
    # Optional: path to a yaml file with key `rubric_hint_template`. If set (non-empty),
    # its content will OVERRIDE `rubric_template` at runtime. Use this to avoid feeding
    # a multi-line template (with '#'/'{}') through Hydra CLI overrides.
    rubric_template_path: Optional[str] = None
    noise_std_threshold: float = 0.0  # When > 0 and sample_route=True, groups with reward std <= this value are treated as noise and excluded from all losses
    reward_winsorize: bool = False  # When True, apply robust winsorization to per-uid reward scores before advantage computation
    reward_winsorize_trim_n: int = 1  # Number of extremes to trim in standard Winsorize layer
    reward_winsorize_max_deviation: float = 3.0  # Maximum allowed deviation from median in the MAD-based clip layer
    reward_winsorize_variance_threshold: float = 0.0  # Only apply winsorization when group variance exceeds this value (0 = always apply)

    # ---------------------------------------------------------------------
    # Joint GRPO + OPSD self-distillation (L = L_RL + lambda_opsd * L_OPSD)
    # ---------------------------------------------------------------------
    # When True, the GRPO policy-gradient loss is applied to ALL rollouts (unchanged
    # from the GRPO baseline) and the self-distillation (OPSD) loss is ADDED on top,
    # weighted by ``lambda_opsd``. This is different from ``sample_route`` (which routes
    # each sample to EITHER PG OR distillation). The distillation loss covers all
    # rollouts that receive a hint (see the hint-bucket config below).
    joint_grpo_distill: bool = False
    # lambda weight for the OPSD distillation term in joint mode: total = pg_loss + lambda_opsd * distill_loss
    lambda_opsd: float = 1.0
    
    #   "loss"      -> L = L_RL + lambda_opsd * L_OPSD
    
    
    
    
    
    opsd_combine_mode: str = "advantage"
    
    
    
    
    opsd_combine_form: str = "convex"
    
    opsd_convex_lambda: float = 0.7
    
    
    
    
    joint_exclude_incorrect_format: bool = True
    # ---------------------------------------------------------------------
    # Adaptive tutoring hints: assign one of 3 hint types per rollout by reward bucket.
    # Buckets are defined by ``hint_thresholds`` (ascending cut points on the 0-10 reward
    # scale). With N thresholds there are N+1 buckets; ``hint_bucket_types`` gives the hint
    # type for each bucket (length must equal len(hint_thresholds)+1).
    # Valid hint types:
    #   "rubric"       -> h*  : reference rubric from extra_info (gold hint)
    #   "self_rollout" -> h~  : the rollout's own response (+ reflection placeholder)
    #   "best_rollout" -> h~  : the highest-reward rollout in the group (+ reflection placeholder)
    #   "none"         -> no hint for this bucket (still gets GRPO PG, excluded from distillation)
    # NOTE: hint *content* is currently a lightweight placeholder for the reflection part;
    # the source is expected to be refined later. When a chosen hint's content is
    # unavailable (e.g. rubric missing), it falls back to "self_rollout" so every rollout
    # still receives a hint (loss covers all rollouts).
    # ---------------------------------------------------------------------
    hint_thresholds: List[float] = field(default_factory=lambda: [3.0, 6.0])
    hint_bucket_types: List[str] = field(default_factory=lambda: ["demonstration", "diff", "rubric"])
    # Placeholder reflection template used by "self_rollout"/"best_rollout" hints. Uses the
    # {reflection} placeholder. Reflection text source is a stub for now (see ray_trainer).
    reflection_template: str = (
        "\n"
        "  Here is a reflection on the result above:\n\n"
        "  {reflection}\n"
    )

    # ---------------------------------------------------------------------
    
    # ---------------------------------------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    use_rubric_set_hint: bool = False
    
    
    
    
    
    
    
    
    rubric_set_source: str = "policy"
    
    
    rubric_set_max_response_length: int = 0
    
    
    
    
    
    rubric_set_select_template: str = ""
    
    rubric_set_select_template_path: Optional[str] = None
    
    
    
    diff_prompt_template: str = ""
    
    diff_prompt_template_path: Optional[str] = None
    
    
    
    diff_max_response_length: int = 256
    
    
    diff_hint_template: str = (
        "\n"
        "  Use the following corrective direction to improve the subset:\n\n"
        "  {diff}\n"
    )
    
    
    demonstration_hint_template: str = (
        "\n"
        "  Here is a reference subset scored {reward} out of 10:\n\n"
        "  {rubric_set}\n"
    )
    
    
    
    rubric_set_bar_medium: float = 6.0
    rubric_set_bar_low: float = 3.0
    
    
    
    
    
    rubric_set_gate_relative: bool = True
    
    rubric_set_gate_margin: float = 0.0
    
    rubric_set_remove_thinking: bool = True

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")

        # wrong_sample_add_pg and pg_only_on_incorrect_format are two mutually exclusive
        # PG-augmentation modes for wrong samples; at most one may be enabled at a time.
        if self.wrong_sample_add_pg and self.pg_only_on_incorrect_format:
            raise ValueError(
                "self_distillation.wrong_sample_add_pg and "
                "self_distillation.pg_only_on_incorrect_format are mutually exclusive; "
                "enable at most one of them."
            )

        teacher_regularization = self.teacher_regularization
        if isinstance(teacher_regularization, str):
            teacher_regularization = teacher_regularization.lower().replace("_", "-")
        if teacher_regularization == "disabled":
            teacher_regularization = "none"

        # BaseConfig fields are frozen after initialization; use object.__setattr__ for canonicalization.
        object.__setattr__(self, "teacher_regularization", teacher_regularization)

        valid_teacher_regularization = ["ema", "trust-region", "none"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.distillation_fixed_indices is not None:
            if self.distillation_topk is not None:
                raise ValueError(
                    "self_distillation.distillation_fixed_indices and distillation_topk are mutually exclusive. "
                    "Set distillation_topk=null when using fixed indices."
                )
            if len(self.distillation_fixed_indices) == 0:
                raise ValueError("self_distillation.distillation_fixed_indices must be a non-empty list of token IDs.")
            if not self.full_logit_distillation:
                raise ValueError(
                    "self_distillation.distillation_fixed_indices requires full_logit_distillation=True."
                )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        valid_entropy_norm_modes = ["per_prompt", "per_sample"]
        if self.entropy_norm_mode not in valid_entropy_norm_modes:
            raise ValueError(
                f"self_distillation.entropy_norm_mode must be one of {valid_entropy_norm_modes}, got {self.entropy_norm_mode}"
            )
        valid_all_fail_fallback = ["gt", "grpo"]
        if self.all_fail_fallback not in valid_all_fail_fallback:
            raise ValueError(
                f"self_distillation.all_fail_fallback must be one of {valid_all_fail_fallback}, got {self.all_fail_fallback}"
            )

        # Joint mode validation
        if self.joint_grpo_distill:
            if self.sample_route:
                raise ValueError(
                    "self_distillation.joint_grpo_distill and sample_route are mutually exclusive; "
                    "joint mode applies GRPO PG to ALL rollouts and adds the distillation loss on top, "
                    "whereas sample_route routes each sample to EITHER PG OR distillation."
                )
            if self.lambda_opsd < 0:
                raise ValueError(f"self_distillation.lambda_opsd must be >= 0, got {self.lambda_opsd}")
            valid_opsd_combine_modes = ["loss", "advantage"]
            if self.opsd_combine_mode not in valid_opsd_combine_modes:
                raise ValueError(
                    f"self_distillation.opsd_combine_mode must be one of {valid_opsd_combine_modes}, "
                    f"got '{self.opsd_combine_mode}'"
                )
            valid_opsd_combine_forms = ["additive", "convex"]
            if self.opsd_combine_form not in valid_opsd_combine_forms:
                raise ValueError(
                    f"self_distillation.opsd_combine_form must be one of {valid_opsd_combine_forms}, "
                    f"got '{self.opsd_combine_form}'"
                )
            if self.opsd_combine_form == "convex" and not 0.0 <= self.opsd_convex_lambda <= 1.0:
                raise ValueError(
                    f"self_distillation.opsd_convex_lambda must be in [0,1], got {self.opsd_convex_lambda}"
                )

        # rubric_set adaptive-hint branch validation
        if self.use_rubric_set_hint:
            if not self.joint_grpo_distill:
                raise ValueError(
                    "self_distillation.use_rubric_set_hint requires joint_grpo_distill=True."
                )
            if self.rubric_set_bar_low > self.rubric_set_bar_medium:
                raise ValueError(
                    "self_distillation.rubric_set_bar_low must be <= rubric_set_bar_medium, "
                    f"got low={self.rubric_set_bar_low}, medium={self.rubric_set_bar_medium}"
                )
            if self.rubric_set_source != "policy":
                raise ValueError(
                    "AdaTutoRank-ATO requires self_distillation.rubric_set_source='policy', "
                    f"got {self.rubric_set_source!r}"
                )

        valid_hint_types = ["rubric", "self_rollout", "best_rollout", "none", "diff", "demonstration"]
        if self.hint_bucket_types is not None:
            for t in self.hint_bucket_types:
                if t not in valid_hint_types:
                    raise ValueError(
                        f"self_distillation.hint_bucket_types entries must be one of {valid_hint_types}, got '{t}'"
                    )
            expected_len = (len(self.hint_thresholds) if self.hint_thresholds is not None else 0) + 1
            if len(self.hint_bucket_types) != expected_len:
                raise ValueError(
                    "self_distillation.hint_bucket_types length must equal len(hint_thresholds)+1 "
                    f"(got {len(self.hint_bucket_types)} types for {len(self.hint_thresholds) if self.hint_thresholds else 0} thresholds)"
                )
        if self.hint_thresholds is not None and len(self.hint_thresholds) > 1:
            if any(self.hint_thresholds[i] > self.hint_thresholds[i + 1] for i in range(len(self.hint_thresholds) - 1)):
                raise ValueError(
                    f"self_distillation.hint_thresholds must be sorted ascending, got {self.hint_thresholds}"
                )


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'sdpo'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
