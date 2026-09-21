"""Huggingface Gemma-4 with Compressed Context Memory (CCM port, 2026-09-20).

A direct port of the official CCM semantics (ccm_llama.py, snu-mllab/
Context-Memory) onto the transformers 5.x Gemma4 text backbone
(modeling_gemma4.py@5.17, Gemma4ForCausalLM text-only path).

Semantic invariants kept (identical to the qwen3 port, ccm_qwen3.py):
  - conditional LoRA: the q/k/v/o projections are LinearMask
    (nn.Linear subclass); the peft_custom LoRA forward applies the
    delta only where comp_mask == 1 (COMP/SUM positions);
  - input-side compression tokens: COMP/SUM rows go through the frozen
    (resized) token embedding; no auto-encoding of the compressed text;
  - SUM overwrite happens AFTER q/k-norm and RoPE and BEFORE the
    attention matmul, on the post-RoPE [B, H, L, D] states;
  - merge_recur visibility folded into the additive 4D mask;
  - position policy "skip": text positions advance contiguously, each
    COMP/SUM carries position 1..m cycling.

Gemma4 deltas vs the qwen3/llama hosts (all native Gemma4 kept intact):
  - 5:1 hybrid attention: the causal mask is built per layer type
    ("full_attention" vs "sliding_attention", 512-token window).  The
    CCM visibility mask is folded into BOTH masks.  DailyDialog
    sequences (~200-300 tokens) stay inside the window, so sliding
    layers see the full history.
  - KV sharing (layers >= num_hidden_layers - num_kv_shared_layers have
    no k/v projections and reuse shared_kv_states per layer type): the
    SUM merge / Gamma recurrence / memory callback run ONLY on
    non-shared layers (the ones that actually compute K/V).  Shared
    layers inherit the merged states through shared_kv_states, so the
    merge is applied exactly once per layer type - applying Gamma again
    on shared layers would double-count.
  - PLE (per-layer embeddings): frozen.  Both embedding tables
    (embed_tokens and embed_tokens_per_layer) are resized for the
    COMP/SUM rows; the PLE rows stay at zero, the main-embedding rows
    are trainable (official CCM trains the comp embeddings).
  - heads*head_dim != hidden: sliding layers 8x256=2048, global layers
    8x512=4096 vs hidden 2560 - the q_proj up-projects and o_proj
    down-projects; the Llama divisibility assert is dropped (qwen3
    precedent).
  - final_logit_softcapping is applied to the logits (native Gemma4);
    the eval log-likelihood path must consume the same softcapped
    logits to stay on-protocol.

This port trains and evaluates in full-sequence mode only: use_cache
and generation are not implemented (v1); passing use_cache=True raises.
"""

import math
from collections import UserDict
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextMLP,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..data.mask import get_comp_mask

# Oracle ceiling test hook (R9 lineage, kept per-host so this file never
# imports ccm_llama - transformers 5.x compatibility).
_ORACLE_LOGITS = None


def set_oracle_logits(logits):
    """[B, t_max] float tensor or None; per-dialogue merge-weight logits."""
    global _ORACLE_LOGITS
    _ORACLE_LOGITS = logits


def clear_oracle_logits():
    global _ORACLE_LOGITS
    _ORACLE_LOGITS = None


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype,
                      device: torch.device,
                      past_key_values_length: int = 0) -> torch.Tensor:
    """Additive 4D causal mask (same construction as ccm_llama.py)."""
    bsz, tgt_len = input_ids_shape
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((tgt_len, tgt_len), min_dtype, dtype=dtype,
                      device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length,
                                      dtype=dtype, device=device), mask],
                         dim=-1)
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype,
                 tgt_len: Optional[int] = None) -> torch.Tensor:
    """2D padding mask -> additive 4D (same construction as
    ccm_llama.py)."""
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                     torch.finfo(dtype).min)


class LinearMask(nn.Linear):
    """Linear function with compression mask as an argument.
    The mask is used for conditional LoRA at
    src/peft_custom/lora.py - Linear() - forward().

    (Own copy, identical semantics to ccm_llama.LinearMask; peft_custom
    matches targets via isinstance(nn.Linear), so the module path is
    irrelevant.)
    """

    def forward(self, input: Tensor, comp_mask=None) -> Tensor:
        return F.linear(input, self.weight, self.bias)


def update_position_ids(comp_mask, comp_token, attention_mask=None,
                        type_="skip"):
    """Position ids for inputs with compression/sum tokens (verbatim
    port of ccm_llama.update_position_ids)."""
    if attention_mask is None:
        attention_mask = torch.ones_like(comp_mask)

    valid_position = attention_mask.to(comp_mask.dtype) * (1.0 - comp_mask)
    n_comp_tok = len(comp_token)

    if type_ == "skip":
        position_ids = valid_position.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # 1, 2, ..., n_comp_tok
        # Assume comp_token and sum_token has the same length
        position_ids_comp = (comp_mask.long().cumsum(-1) - 1) % n_comp_tok + 1
        position_ids = position_ids + (comp_mask * position_ids_comp).long()
    else:
        raise AssertionError(f"Unknown positional embedding type {type_}")

    return position_ids


class Gemma4CCMTextAttention(nn.Module):
    """Gemma4 hybrid attention with the CCM merge/visibility semantics.

    Mirrors the native Gemma4TextAttention (5.17) with four CCM deltas:
    LinearMask projections (conditional LoRA), the post-RoPE SUM
    overwrite, the Gamma recurrence scan and the memory callback - all
    restricted to the NON-shared layers that actually compute K/V.
    Shared layers (is_kv_shared_layer) reuse shared_kv_states and skip
    the merge: applying it there would double-count Gamma on the same
    states.
    """

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = (config.layer_types[layer_idx]
                           if hasattr(config, "layer_types") else None)
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding \
            else None

        layer_config = config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        self.use_alternative_attention = (
            config.attention_k_eq_v and not self.is_sliding)
        self.num_key_value_groups = (
            config.num_attention_heads // layer_config.num_key_value_heads)
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout

        # Shared KV layers: the k/v projections and norms simply do not
        # exist (native Gemma4), so the conditional LoRA only covers
        # q/o on those layers - an architecture fact, not a bug.
        first_kv_shared_layer_idx = self.config.num_hidden_layers - \
            getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_layers) - 1
            - prev_layers[::-1].index(config.layer_types[layer_idx]))

        self.q_proj = LinearMask(
            config.hidden_size, config.num_attention_heads * self.head_dim,
            bias=config.attention_bias)
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim,
                                    eps=config.rms_norm_eps)

        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim,
                                        eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim,
                                        eps=config.rms_norm_eps,
                                        with_scale=False)
            self.k_proj = LinearMask(
                config.hidden_size,
                layer_config.num_key_value_heads * self.head_dim,
                bias=config.attention_bias)
            self.v_proj = (
                LinearMask(
                    config.hidden_size,
                    layer_config.num_key_value_heads * self.head_dim,
                    bias=config.attention_bias)
                if not self.use_alternative_attention else None)

        self.o_proj = LinearMask(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=config.attention_bias)

        # RPBE: Gamma residual slot (attached by rpbe.hosts.ccm.ccm_patch).
        self.gamma = None
        # RPBE: memory extraction callback (post-merge K/V at SUM rows).
        self.mem_callback = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        shared_kv_states: dict,
        attention_mask: Optional[torch.Tensor] = None,
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        sum_row_pos: Optional[torch.Tensor] = None,
        sum_row_valid: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states,
                                   comp_mask=comp_mask).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin,
                                            unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer:
            # Native Gemma4: reuse the K/V of the last non-sharing layer
            # of this type (already merged there, if applicable).
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states,
                                     comp_mask=comp_mask).view(hidden_shape)
            value_states = (
                self.v_proj(hidden_states, comp_mask=comp_mask).view(
                    hidden_shape)
                if self.v_proj is not None else key_states)

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin,
                                              unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

            # CCM: overwrite the SUM rows with the post-RoPE weighted
            # mean of the same-slot preceding COMP rows (uniform 1/j
            # officially; oracle-ceiling weights when hooked).
            if sum_attn_mask is not None:
                sum_attn_mask = sum_attn_mask.to(key_states.dtype)
                key_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1),
                                            key_states)
                value_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1),
                                              value_states)
                no_sum_mask = (1 - sum_mask).to(key_states.dtype) \
                    .unsqueeze(1).unsqueeze(-1)

                # RPBE: Gamma recurrence scan (t_max >= 2), same math
                # as the Llama/qwen3 hosts.
                if self.gamma is not None and sum_row_pos is not None \
                        and int(sum_row_pos.shape[1]) >= 2:
                    n_heads = key_states.shape[1]
                    head_dim = key_states.shape[3]
                    n_slots = int(sum_row_pos.shape[2])
                    t_max = int(sum_row_pos.shape[1])
                    n_rows = t_max * n_slots
                    idx = sum_row_pos.reshape(bsz := input_shape[0],
                                              n_rows).unsqueeze(1)
                    idx = idx.expand(bsz, n_heads, n_rows)
                    idx = idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k_sum = torch.gather(key_states, 2, idx)
                    v_sum = torch.gather(value_states, 2, idx)
                    k_base = torch.gather(key_comp_avg, 2, idx).reshape(
                        bsz, n_heads, t_max, n_slots, head_dim)
                    v_base = torch.gather(value_comp_avg, 2, idx).reshape(
                        bsz, n_heads, t_max, n_slots, head_dim)
                    t = torch.arange(1, t_max + 1,
                                     device=key_states.device).float()
                    t = t.view(1, 1, t_max, 1, 1)
                    prev_shift = torch.zeros_like(k_base)
                    prev_shift[:, :, 1:, :, :] = k_base[:, :, :-1, :, :]
                    k_cur = t * k_base - (t - 1) * prev_shift
                    prev_shift = torch.zeros_like(v_base)
                    prev_shift[:, :, 1:, :, :] = v_base[:, :, :-1, :, :]
                    v_cur = t * v_base - (t - 1) * prev_shift

                key_states = no_sum_mask * key_states + key_comp_avg
                value_states = no_sum_mask * value_states + value_comp_avg

                # RPBE: per-turn residual recurrence over the SUM rows.
                if self.gamma is not None and sum_row_pos is not None \
                        and int(sum_row_pos.shape[1]) >= 2:
                    n_heads = key_states.shape[1]
                    head_dim = key_states.shape[3]
                    n_slots = int(sum_row_pos.shape[2])
                    t_max = int(sum_row_pos.shape[1])
                    res_prev_k = torch.zeros(
                        bsz, n_heads, n_slots, head_dim,
                        dtype=key_states.dtype, device=key_states.device)
                    res_prev_v = torch.zeros_like(res_prev_k)
                    res_all_k = torch.zeros_like(k_base)
                    res_all_v = torch.zeros_like(v_base)
                    valid = sum_row_valid.to(key_states.dtype).unsqueeze(1)
                    valid = valid.unsqueeze(-1)  # [B, 1, T, n_slots, 1]
                    for t_i in range(2, t_max + 1):
                        tt = torch.full((bsz, 1), t_i,
                                        dtype=torch.float32,
                                        device=key_states.device)
                        res_t_k = self.gamma(
                            k_base[:, :, t_i - 2] + res_prev_k,
                            k_cur[:, :, t_i - 1], tt) \
                            * valid[:, :, t_i - 1]
                        res_t_v = self.gamma(
                            v_base[:, :, t_i - 2] + res_prev_v,
                            v_cur[:, :, t_i - 1], tt) \
                            * valid[:, :, t_i - 1]
                        res_prev_k = res_t_k
                        res_prev_v = res_t_v
                        res_all_k[:, :, t_i - 1] = res_t_k
                        res_all_v[:, :, t_i - 1] = res_t_v
                    for b in range(bsz):
                        key_states[b] = key_states[b].index_add(
                            1, sum_row_pos[b].reshape(-1),
                            res_all_k[b].reshape(n_heads, -1, head_dim))
                        value_states[b] = value_states[b].index_add(
                            1, sum_row_pos[b].reshape(-1),
                            res_all_v[b].reshape(n_heads, -1, head_dim))

                # RPBE: memory extraction (post-merge, pre-attention).
                if self.mem_callback is not None:
                    self.mem_callback(key_states, value_states, sum_mask,
                                      sum_row_pos)

            if self.store_full_length_kv:
                shared_kv_states[self.layer_type] = key_states, value_states

        # GQA attention: repeat KV heads, additive 4D mask, fp32 softmax.
        # NOTE (gemma4 delta): native Gemma4 sets self.scaling = 1.0, so
        # the attention scores are NOT divided by sqrt(head_dim) (the
        # sqrt scaling only applies when scaling is None in the native
        # eager path).  Multiplying by 1.0 keeps bit parity (G1 gate).
        key_states_r = repeat_kv(key_states, self.num_key_value_groups)
        value_states_r = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states_r.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(
                query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_r)
        attn_output = attn_output.transpose(1, 2)
        # Gemma4 delta: heads * head_dim != hidden_size; o_proj
        # down-projects to the residual width.
        attn_output = attn_output.reshape(
            input_shape[0], input_shape[1],
            self.config.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output, comp_mask=comp_mask)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class Gemma4CCMTextDecoderLayer(nn.Module):
    """Gemma4 decoder layer with the CCM attention; the PLE pipeline and
    the pre/post-norm residual stack stay native."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4CCMTextAttention(config=config,
                                                layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size,
                                             eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = nn.Buffer(torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(
                self.hidden_size, self.hidden_size_per_layer_input,
                bias=False)
            self.per_layer_projection = nn.Linear(
                self.hidden_size_per_layer_input, self.hidden_size,
                bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        sum_row_pos: Optional[torch.Tensor] = None,
        sum_row_valid: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            shared_kv_states=shared_kv_states,
            attention_mask=attention_mask,
            comp_mask=comp_mask,
            sum_mask=sum_mask,
            sum_attn_mask=sum_attn_mask,
            sum_row_pos=sum_row_pos,
            sum_row_valid=sum_row_valid,
            output_attentions=output_attentions,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states, attn_weights


class Gemma4CCMTextModel(Gemma4PreTrainedModel):
    """Gemma4 text decoder with the CCM mask/position pipeline.

    The mask pipeline (get_comp_sum_mask) is ported verbatim from the
    official LlamaModelCCM; the forward keeps the native Gemma4
    pre/post-norm stack, the PLE pipeline and the shared_kv_states
    mechanism.  The causal mask is materialized per layer type
    (full_attention / sliding_attention) as additive 4D tensors and the
    CCM visibility mask is folded into both.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx,
            embed_scale=config.hidden_size ** 0.5)
        self.layers = nn.ModuleList(
            [Gemma4CCMTextDecoderLayer(config, layer_idx=layer_idx)
             for layer_idx in range(config.num_hidden_layers)])
        self.norm = Gemma4RMSNorm(config.hidden_size,
                                  eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.unique_layer_types = set(self.config.layer_types)

        # PLE (native, frozen in training; resized alongside the main
        # embedding for the COMP/SUM rows via resize_ple_embeddings).
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers
                * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input ** 0.5)
            self.per_layer_input_scale = 2.0 ** -0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers
                * config.hidden_size_per_layer_input,
                bias=False)
            self.per_layer_model_projection_scale = config.hidden_size ** -0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input,
                eps=config.rms_norm_eps)

        # Native Gemma4: drop the k/v proj and norms of the shared
        # layers when loading a full checkpoint.
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if layer.self_attn.is_kv_shared_layer:
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}"
                     for name in ("k_proj", "v_proj", "k_norm", "v_norm")])

        self.post_init()

        # CCM: compression token bookkeeping.
        self.comp_token = None
        self.sum_token = None
        self.comp_relative_embedding = getattr(
            config, "comp_relative_embedding", "skip")

        # RPBE: flipped by attach_gamma().
        self._gamma_attached = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def resize_ple_embeddings(self, new_num_tokens):
        """Grow the PLE table to cover the COMP/SUM rows (zero-init,
        frozen).  Called by Gemma4ForCausalLM_CCM.resize_token_embeddings
        so both tables move together."""
        ple = self.embed_tokens_per_layer
        old_num = ple.num_embeddings
        if new_num_tokens <= old_num:
            return
        old_w = ple.weight.data
        new_w = torch.zeros(new_num_tokens, old_w.shape[1],
                            dtype=old_w.dtype, device=old_w.device)
        new_w[:old_num] = old_w
        ple.num_embeddings = new_num_tokens
        ple.weight = nn.Parameter(new_w, requires_grad=False)

    def get_comp_sum_mask(self, input_ids):
        """comp/sum masks and the merge/visibility plan (official port,
        see ccm_qwen3.py for the shared lineage)."""
        comp_mask = None
        if self.comp_token is not None:
            comp_mask = get_comp_mask(input_ids, self.comp_token).to(
                input_ids.device)

        sum_attn_mask = None
        sum_mask = None
        sum_row_pos = None
        sum_row_valid = None
        if self.sum_token is not None:
            sum_mask = get_comp_mask(input_ids, self.sum_token).to(
                input_ids.device)
            if sum_mask.sum() == 0:
                sum_mask = None

            if sum_mask is not None:
                assert comp_mask.sum() > 0, \
                    "sum_attn_mask is not None, but comp_mask is None"
                batch_size, seq_len = input_ids.shape
                sum_attn_mask = torch.zeros(
                    (batch_size, seq_len, seq_len),
                    device=input_ids.device).float()

                for k in range(len(self.comp_token)):
                    comp_loc_k = input_ids == self.comp_token[k]
                    sum_loc_k = input_ids == self.sum_token[k]
                    attn_k = sum_loc_k.unsqueeze(2) & comp_loc_k.unsqueeze(1)
                    sum_attn_mask += attn_k.float()

                sum_attn_mask = torch.tril(sum_attn_mask)

                if _ORACLE_LOGITS is not None:
                    # Oracle ceiling hook (same as the Llama host).
                    _lg = _ORACLE_LOGITS
                    for _b in range(batch_size):
                        for _k in range(len(self.sum_token)):
                            _spos = (input_ids[_b]
                                     == self.sum_token[_k]).nonzero(
                                as_tuple=False).flatten()
                            _cpos = (input_ids[_b]
                                     == self.comp_token[_k]).nonzero(
                                as_tuple=False).flatten()
                            for _j, _p in enumerate(_spos):
                                _nh = int((_cpos < _p).sum().item())
                                if _nh <= 0:
                                    continue
                                _w = torch.softmax(
                                    _lg[_b, :_nh].float(), dim=0)
                                sum_attn_mask[_b, _p, _cpos[:_nh]] = \
                                    _w.to(sum_attn_mask.dtype)
                else:
                    sum_attn_mask = sum_attn_mask / torch.clamp(
                        sum_attn_mask.sum(-1, keepdim=True),
                        min=0.1)

                # RPBE: per-slot SUM positions in turn order.
                if self._gamma_attached:
                    n_slots = len(self.sum_token)
                    per_slot = []
                    t_max = 0
                    for k in range(n_slots):
                        loc = (input_ids == self.sum_token[k]).nonzero()
                        by_batch = {}
                        for b, p in loc.tolist():
                            by_batch.setdefault(int(b), []).append(int(p))
                        per_slot.append(by_batch)
                        t_max = max(t_max, max(
                            (len(v) for v in by_batch.values()), default=0))
                    sum_row_pos = torch.zeros(batch_size, t_max, n_slots,
                                              dtype=torch.long,
                                              device=input_ids.device)
                    sum_row_valid = torch.zeros(batch_size, t_max, n_slots,
                                                dtype=torch.bool,
                                                device=input_ids.device)
                    for k, by_batch in enumerate(per_slot):
                        for b, ps in by_batch.items():
                            for t_i, p in enumerate(ps):
                                sum_row_pos[b, t_i, k] = p
                                sum_row_valid[b, t_i, k] = True

                # Conditional LoRA + positional embedding mask.
                comp_mask = comp_mask + sum_mask

        return comp_mask, sum_mask, sum_attn_mask, sum_row_pos, sum_row_valid

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_comp: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_id_offset: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # CCM: compression masks for conditional inference and merge.
        comp_mask, sum_mask, sum_attn_mask, sum_row_pos, sum_row_valid = \
            self.get_comp_sum_mask(input_ids)

        output_attentions = (output_attentions if output_attentions
                             is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None
                                else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else False  # v1
        if use_cache:
            raise NotImplementedError(
                "ccm_gemma4 v1 is full-sequence only (use_cache=False)")
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        # CCM: skip position ids (text contiguous, COMP/SUM cycle 1..m).
        if position_ids is None:
            if self.comp_relative_embedding == "base" or comp_mask is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = update_position_ids(
                    comp_mask,
                    self.comp_token,
                    attention_mask,
                    type_=self.comp_relative_embedding)
            if pos_id_offset is not None:
                position_ids += pos_id_offset
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # CCM: materialize the per-layer-type causal masks (additive 4D)
        # and fold the merge_recur visibility into both.
        min_val = torch.finfo(inputs_embeds.dtype).min
        full_mask = _make_causal_mask((batch_size, seq_length),
                                      inputs_embeds.dtype,
                                      device=inputs_embeds.device)
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=seq_length).to(inputs_embeds.device)
            full_mask = expanded_attn_mask + full_mask

        sliding_mask = full_mask.clone()
        if "sliding_attention" in self.unique_layer_types:
            window = int(self.config.sliding_window)
            q_idx = torch.arange(seq_length, device=inputs_embeds.device)
            dist = q_idx[None, :] - q_idx[:, None]  # [q, k] = q - k
            window_out = dist >= window
            sliding_mask[:, :, window_out] = min_val

        causal_mask_mapping = {
            "full_attention": full_mask,
            "sliding_attention": sliding_mask,
        }
        if attention_mask_comp is not None:
            attention_mask_comp_float = torch.full_like(full_mask, min_val)
            attention_mask_comp_float = attention_mask_comp_float \
                .masked_fill(attention_mask_comp.bool(), 0.0)
            causal_mask_mapping["full_attention"] = \
                full_mask + attention_mask_comp_float
            causal_mask_mapping["sliding_attention"] = \
                sliding_mask + attention_mask_comp_float

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # PLE: token-identity + context projection (native pipeline).
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.get_per_layer_inputs(
                input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs)

        # Embed positions per layer type (p-RoPE on global layers).
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type)

        # shared_kv_states: the KV-sharing dict (native mechanism; the
        # merged states flow into it through store_full_length_kv).
        shared_kv_states = UserDict()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            per_layer_input = (
                per_layer_inputs[:, :, idx, :]
                if per_layer_inputs is not None else None)

            layer_outputs = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[
                    self.config.layer_types[idx]],
                attention_mask=causal_mask_mapping[
                    self.config.layer_types[idx]],
                comp_mask=comp_mask,
                sum_mask=sum_mask,
                sum_attn_mask=sum_attn_mask,
                sum_row_pos=sum_row_pos,
                sum_row_valid=sum_row_valid,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states,
                                     all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # -- PLE helpers (verbatim native semantics) --

    def get_per_layer_inputs(self, input_ids, inputs_embeds):
        """Token-identity component of PLE (native)."""
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("PLE not enabled in this config")
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (inputs_embeds[:, :, None, :]
                     == self.embed_tokens.weight[None, None, :, :]
                     * self.config.hidden_size ** 0.5)
                    .all(dim=3).nonzero()[:, 2])
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError:
                    raise RuntimeError(
                        "inputs_embeds do not match the embedding weights; "
                        "provide exact inputs_embeds")
        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input)

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs):
        """Context-aware component of PLE (native)."""
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("PLE not enabled in this config")
        # fp32 GEMM (CUBLAS fix): the 2560->10752 bf16 GEMM trips
        # CUBLAS_STATUS_NOT_SUPPORTED on the training shapes.  The
        # autocast Linear fastpath keeps casting to the module weight
        # dtype (bf16) regardless of the input upcast, so the GEMM is
        # wrapped in a locally DISABLED autocast: fp32 x fp32 then runs
        # on the fp32 GEMM path.  Single layer, cheap, safe.
        _proj = self.per_layer_model_projection
        with torch.autocast(device_type="cuda",
                            dtype=torch.bfloat16, enabled=False):
            per_layer_projection = F.linear(
                inputs_embeds.float(), _proj.weight.float(),
                _proj.bias.float() if _proj.bias is not None else None) \
                * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.to(inputs_embeds.dtype)
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input)
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        return (per_layer_projection + per_layer_inputs) \
            * self.per_layer_input_scale


class Gemma4ForCausalLM_CCM(Gemma4PreTrainedModel):
    """Gemma4 causal LM with CCM compression tokens (text-only path of
    Gemma4ForConditionalGeneration).

    The lm_head stays tied to the input embeddings; the PLE table is
    grown alongside (zero rows, frozen) in resize_token_embeddings.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Gemma4TextConfig):
        super().__init__(config)
        self.model = Gemma4CCMTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        self.post_init()

        # CCM: compression token bookkeeping.
        self.comp_token = None
        self.sum_token = None
        self.comp_relative_embedding = getattr(
            config, "comp_relative_embedding", "skip")

    def update_comp_token(self, comp_token, sum_token):
        self.comp_token = comp_token
        self.model.comp_token = comp_token
        self.sum_token = sum_token
        self.model.sum_token = sum_token

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: int = None,
                                pad_to_multiple_of: int = None,
                                mean_resizing: bool = False):
        """Resize the main embedding + tied lm_head (mean_resizing=False
        keeps the 4.x random-init semantics of the llama line) and grow
        the PLE table to match (zero rows, frozen)."""
        res = super().resize_token_embeddings(
            new_num_tokens, pad_to_multiple_of, mean_resizing)
        if new_num_tokens is not None \
                and self.model.hidden_size_per_layer_input:
            self.model.resize_ple_embeddings(new_num_tokens)
        return res

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        attention_mask_comp=None,
        inputs_embeds=None,
        pos_id_offset=None,
        first_time=False,
        **kwargs,
    ):
        """Interface stub for peft_custom compatibility (generation
        itself raises in the forward: v1 is full-sequence only)."""
        first_time_with_compression = first_time \
            and pos_id_offset is not None

        comp_mask = None
        if self.comp_token is not None:
            comp_mask = get_comp_mask(input_ids, self.comp_token).to(
                input_ids.device)
            comp_mask_all = comp_mask
            if self.sum_token is not None:
                sum_mask = get_comp_mask(input_ids, self.sum_token).to(
                    input_ids.device)
                comp_mask_all = comp_mask + sum_mask

        input_len = input_ids.shape[1]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            if self.comp_relative_embedding == "base":
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = update_position_ids(
                    comp_mask_all,
                    self.comp_token,
                    attention_mask[:, -input_len:],
                    type_=self.comp_relative_embedding)
            if pos_id_offset is not None:
                position_ids += pos_id_offset

        if past_key_values and not first_time_with_compression:
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "attention_mask_comp": attention_mask_comp,
            "pos_id_offset": pos_id_offset,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx) for past_state
                in layer_past),)
        return reordered_past

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_comp: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_id_offset: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (output_attentions if output_attentions
                             is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None
                                else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_comp=attention_mask_comp,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pos_id_offset=pos_id_offset,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
