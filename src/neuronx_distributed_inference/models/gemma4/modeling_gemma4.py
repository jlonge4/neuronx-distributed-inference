# coding=utf-8
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
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
PyTorch Gemma 4 model for NXD inference.

Key architectural differences from Gemma 3:
  - Dual head dims: local/SWA layers use head_dim=256, global layers use global_head_dim=512
  - Asymmetric GQA: num_key_value_heads (local) vs num_global_key_value_heads (global)
  - Global layers use partial RoPE: only partial_rotary_factor=0.25 of dims are rotated
  - No query_pre_attn_scalar (removed in Gemma 4)
  - vocab_size: 262144 (was 262208 in Gemma 3)
  - sliding_window: 1024 for 31B (was model-specific in Gemma 3)
  - Layer pattern still 5 local : 1 global (every 6th layer is global, 1-indexed)

TODO: final_logit_softcapping=30.0 is not yet applied (Gemma 2/4 feature).
TODO: attention_k_eq_v=True means K and V share weights — verify state dict behaviour
      with the loaded HF checkpoint.
"""

import copy
from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding, apply_rotary_pos_emb


# ── RMSNorm ───────────────────────────────────────────────────────────────────

class NeuronGemma4RMSNorm(nn.Module):
    """Gemma-style RMSNorm: weight initialised to zeros, scale applied as (1 + weight)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Gemma convention: (1 + weight) not weight, see https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


def _get_rmsnorm_cls():
    """Return HF's implementation on CPU (NeuronRMSNorm doesn't run on CPU)."""
    if cpu_mode():
        try:
            from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
            return Gemma4RMSNorm
        except ImportError:
            # transformers may not have Gemma 4 yet; fall through to our impl
            pass
    return NeuronGemma4RMSNorm


# ── Partial Rotary Embedding ──────────────────────────────────────────────────

class PartialRotaryEmbedding(nn.Module):
    """
    Applies RoPE to the first `partial_rotary_factor` fraction of head dimensions.

    Used by Gemma 4 global (full) attention layers which rotate only 25% of dims
    and leave the remaining 75% unchanged.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        partial_rotary_factor: float = 0.25,
    ):
        super().__init__()
        self.full_dim = dim
        self.rotary_dim = int(dim * partial_rotary_factor)
        self._rope = RotaryEmbedding(
            dim=self.rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )

    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, full_dim] — used only for device/dtype
        # Returns cos/sin of shape [bs, seq_len, rotary_dim]
        return self._rope(x[..., : self.rotary_dim], position_ids)


def _apply_partial_rotary_pos_emb(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to the first `rotary_dim` dims of Q and K; pass the rest through unchanged.

    Q, K  : [bs, num_heads, seq_len, head_dim]
    cos/sin: [bs, seq_len, rotary_dim]
    """
    Q_rot, Q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
    K_rot, K_pass = K[..., :rotary_dim], K[..., rotary_dim:]
    Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos, sin)
    Q = torch.cat([Q_rot, Q_pass], dim=-1)
    K = torch.cat([K_rot, K_pass], dim=-1)
    return Q, K


# ── Per-layer config helper ───────────────────────────────────────────────────

def _is_global_layer(layer_idx: int, layer_types=None) -> bool:
    """
    Return True if this layer uses full (global) attention.

    Respects an explicit `layer_types` list from the HF config if present,
    otherwise falls back to the standard Gemma pattern: every 6th layer
    (1-indexed) is global.
    """
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx] == "full_attention"
    return (layer_idx + 1) % 6 == 0


def get_updated_configs(config: InferenceConfig) -> List[InferenceConfig]:
    """
    Build a per-layer InferenceConfig list for Gemma 4.

    Global layers get:
      - sliding_window = None
      - head_dim       = global_head_dim  (e.g. 512)
      - num_key_value_heads = num_global_key_value_heads (if set)
    Local/SWA layers keep the defaults from config.
    """
    layer_types = getattr(config, "layer_types", None)
    global_head_dim = getattr(config, "global_head_dim", config.head_dim * 2)
    num_global_kv_heads = getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads

    updated = []
    for i in range(config.num_hidden_layers):
        cfg = copy.deepcopy(config)
        if _is_global_layer(i, layer_types):
            cfg.sliding_window = None
            cfg.head_dim = global_head_dim
            cfg.num_key_value_heads = num_global_kv_heads
        updated.append(cfg)
    return updated


# ── NeuronConfig / InferenceConfig ────────────────────────────────────────────

class Gemma4NeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronGemma4Attention


class Gemma4InferenceConfig(InferenceConfig):
    def __init__(self, neuron_config: NeuronConfig, fused_spec_config=None, load_config=None):
        # Attributes to pull from the HF config (or text_config for multimodal checkpoints)
        self.attributes = [
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "sliding_window",
        ]

        self.neuron_config = neuron_config
        self.fused_spec_config = fused_spec_config

        if load_config is not None:
            load_config(self)
        else:
            self.load_config()

        # Gemma 4 checkpoints nest the text config under `text_config`
        text_config = getattr(self, "text_config", None)
        if text_config is not None:
            for attr in self.attributes:
                setattr(self, attr, getattr(text_config, attr))

            # Pull Gemma 4-specific fields from text_config if available
            for extra in (
                "global_head_dim",
                "num_global_key_value_heads",
                "layer_types",
                "rope_parameters",
                "final_logit_softcapping",
                "attention_k_eq_v",
            ):
                val = getattr(text_config, extra, None)
                if val is not None:
                    setattr(self, extra, val)

        # ── Defaults (overridden by anything loaded from the checkpoint above) ──
        if not hasattr(self, "max_position_embeddings"):
            setattr(self, "max_position_embeddings", 262144)
        if not hasattr(self, "vocab_size"):
            setattr(self, "vocab_size", 262144)           # 262144, not 262208 like Gemma 3
        if not hasattr(self, "pad_token_id"):
            setattr(self, "pad_token_id", 0)
        if not hasattr(self, "rms_norm_eps"):
            setattr(self, "rms_norm_eps", 1e-6)
        if not hasattr(self, "hidden_act"):
            setattr(self, "hidden_act", "gelu_pytorch_tanh")

        # RoPE: derive local/global theta from rope_parameters if present
        rope_params = getattr(self, "rope_parameters", None)
        if rope_params is not None:
            sliding = rope_params.get("sliding_attention", {})
            full = rope_params.get("full_attention", {})
            self.local_rope_theta = sliding.get("rope_theta", 10000.0)
            self.global_rope_theta = full.get("rope_theta", 1_000_000.0)
            self.partial_rotary_factor = full.get("partial_rotary_factor", 0.25)
        else:
            if not hasattr(self, "local_rope_theta"):
                self.local_rope_theta = 10000.0
            if not hasattr(self, "global_rope_theta"):
                self.global_rope_theta = 1_000_000.0
            if not hasattr(self, "partial_rotary_factor"):
                self.partial_rotary_factor = 0.25

        # global_head_dim defaults to 2× local head_dim when not in checkpoint
        if not hasattr(self, "global_head_dim") or self.global_head_dim is None:
            self.global_head_dim = self.head_dim * 2

        # num_global_key_value_heads falls back to num_key_value_heads
        if not hasattr(self, "num_global_key_value_heads") or self.num_global_key_value_heads is None:
            self.num_global_key_value_heads = self.num_key_value_heads

        self.add_derived_config()
        self.validate_config()

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return self.attributes

    @classmethod
    def get_neuron_config_cls(cls) -> Type[Gemma4NeuronConfig]:
        return Gemma4NeuronConfig


# ── Attention ─────────────────────────────────────────────────────────────────

class NeuronGemma4Attention(NeuronAttentionBase):
    """
    Gemma 4 attention module.

    Local (SWA) layers:
      - head_dim=256, full RoPE at theta=10000, sliding_window KV cache

    Global (full) layers:
      - head_dim=512, partial RoPE (25% of dims) at theta=1M, full KV cache
      - Potentially fewer KV heads (num_global_key_value_heads=4 in the 31B model)
    """

    def __init__(self, config: Gemma4InferenceConfig):
        # head_dim and num_key_value_heads have already been set per-layer
        # by get_updated_configs() before this __init__ is called.
        head_dim = config.head_dim
        is_global = config.sliding_window is None

        if is_global:
            rotary_emb = PartialRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.global_rope_theta,
                partial_rotary_factor=config.partial_rotary_factor,
            )
        else:
            rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.local_rope_theta,
            )

        self._is_global = is_global
        self._rotary_dim = int(head_dim * config.partial_rotary_factor) if is_global else None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,          # we attach norms manually below
            use_scaled_rope=None,
            sliding_window=config.sliding_window,
            post_transpose_layernorm=True,
        )

        rmsnorm_cls = _get_rmsnorm_cls()
        self.q_layernorm = rmsnorm_cls(hidden_size=head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = rmsnorm_cls(hidden_size=head_dim, eps=config.rms_norm_eps)

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Override to apply partial RoPE on global attention layers."""
        if not self._is_global:
            return super().apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        # Global layer: partial RoPE — rotate only the first rotary_dim dims
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)
        Q, K = _apply_partial_rotary_pos_emb(Q, K, cos_cache, sin_cache, self._rotary_dim)
        return Q, K, cos_cache, sin_cache


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class NeuronGemma4DecoderLayer(nn.Module):
    """
    Gemma 4 decoder layer.

    Same 4-norm structure as Gemma 3:
      input_layernorm → attention → post_attention_layernorm → residual
      pre_feedforward_layernorm → MLP → post_feedforward_layernorm → residual
    """

    def __init__(self, config: Gemma4InferenceConfig, layer_idx: int):
        super().__init__()
        self.is_sliding_window_attention = config.sliding_window is not None
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        rmsnorm_cls = _get_rmsnorm_cls()
        self.self_attn = NeuronGemma4Attention(config)
        self.mlp = NeuronLlamaMLP(config)   # gelu_pytorch_tanh, same gate/up/down structure
        self.input_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = rmsnorm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        mask = local_mask
        if not self.is_sliding_window_attention or local_mask is None:
            mask = attention_mask

        # Gemma scaled word embeddings: multiply by sqrt(hidden_size) at layer 0
        if self.layer_idx == 0:
            hidden_states = hidden_states * (self.hidden_size ** 0.5)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


# ── Text Model ────────────────────────────────────────────────────────────────

class NeuronGemma4TextModel(NeuronBaseModel):

    def setup_attr_for_model(self, config: Gemma4InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: Gemma4InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )

        updated_configs = get_updated_configs(config)
        self.layers = nn.ModuleList(
            [NeuronGemma4DecoderLayer(cfg, idx) for idx, cfg in enumerate(updated_configs)]
        )
        self.norm = _get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


# ── CausalLM ──────────────────────────────────────────────────────────────────

class NeuronGemma4ForCausalLM(NeuronBaseForCausalLM):
    """
    Neuron inference wrapper for Gemma 4 text generation (e.g. gemma-4-31B-IT).

    The published checkpoint is Gemma4ForConditionalGeneration (multimodal).
    All text weights live under model.language_model.* so we override
    _STATE_DICT_MODEL_PREFIX to strip that full path automatically.
    """

    _model_cls = NeuronGemma4TextModel
    # HF checkpoint keys: model.language_model.layers.*.  Strip to get layers.*
    _STATE_DICT_MODEL_PREFIX = "model.language_model."

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        try:
            from transformers import Gemma4ForConditionalGeneration
            return Gemma4ForConditionalGeneration.from_pretrained(model_path, **kwargs)
        except ImportError:
            raise ImportError(
                "Gemma 4 requires transformers >= 4.52. "
                "Please upgrade: pip install --upgrade transformers"
            )

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_config = config.neuron_config

        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree

        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

            # HF uses q_norm / k_norm; NXD attention_base expects q_layernorm / k_layernorm
            q_norm_key = f"layers.{i}.self_attn.q_norm.weight"
            k_norm_key = f"layers.{i}.self_attn.k_norm.weight"

            if q_norm_key in state_dict:
                state_dict[f"layers.{i}.self_attn.q_layernorm.weight"] = (
                    state_dict.pop(q_norm_key).detach().clone()
                )
            if k_norm_key in state_dict:
                state_dict[f"layers.{i}.self_attn.k_layernorm.weight"] = (
                    state_dict.pop(k_norm_key).detach().clone()
                )


        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return Gemma4InferenceConfig
