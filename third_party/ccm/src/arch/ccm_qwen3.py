""" Huggingface Qwen3 with Compressed Context Memory (CCM port, 2026-09-17).

A direct port of the official CCM semantics (third_party/ccm/src/arch/
ccm_llama.py, snu-mllab/Context-Memory@a89dd08) onto the frozen
transformers 4.56.2 Qwen3 backbone (modeling_qwen3.py@cd74917).

Semantic invariants kept (CCM_Qwen3_Porting_Audit.md):
  - conditional LoRA: q/k/v/o projections are LinearMask; the
    comp_mask gate is only 1 at COMP/SUM positions;
  - input-side compression tokens: the COMP/SUM rows are embedded
    through the (frozen + resized) token embedding, no auto-encoding
    of the compressed text;
  - SUM overwrite happens AFTER q/k-norm and RoPE and BEFORE the
    attention matmul (no cache is used in this port's v1);
  - merge_recur visibility: get_comp_attn_mask_recur on the SUM tokens
    folded into the additive 4D mask (attention_mask_comp);
  - position policy "skip": text positions advance contiguously, each
    COMP/SUM carries position 1..m cycling.

Qwen3 deltas vs the Llama host (all native Qwen3 kept intact):
  - GQA: k/v projections output num_key_value_heads * head_dim and the
    SUM merge / Gamma scan operate on the KV head count;
  - q_norm / k_norm run after the projections (native order) and the
    merge consumes the post-norm, post-RoPE states;
  - model-level Qwen3RotaryEmbedding shared across layers;
  - tied lm_head (resize_token_embeddings covers both sides).

This port trains and evaluates in full-sequence mode only: use_cache
and generation are not implemented (v1); passing use_cache=True raises.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..data.mask import get_comp_mask
from .ccm_llama import (
    LinearMask,
    set_oracle_logits as _set_oracle_logits,
    clear_oracle_logits as _clear_oracle_logits,
    _ORACLE_LOGITS,
    update_position_ids,
)

# Re-export the oracle-ceiling hooks so callers keep one import site.
set_oracle_logits = _set_oracle_logits
clear_oracle_logits = _clear_oracle_logits


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
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                     torch.finfo(dtype).min)


class Qwen3CCMAttention(nn.Module):
    """Qwen3 GQA attention with the CCM merge/visibility semantics.

    Mirrors the official Qwen3Attention (4.56.2) with four CCM deltas:
    LinearMask projections (conditional LoRA), the post-RoPE SUM
    overwrite, the Gamma recurrence scan, and the memory callback.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int = -1):
        self.layer_idx = layer_idx
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # NOTE (qwen3 delta): unlike Llama, Qwen3 does not satisfy
        # hidden == heads * head_dim (2560 vs 32*128=4096 for the 4B):
        # the q projection up-projects and o_proj down-projects.  The
        # Llama divisibility assert is intentionally dropped here.
        # CCM: LinearMask accepts comp_mask for the conditional LoRA.
        self.q_proj = LinearMask(self.hidden_size,
                                 self.num_heads * self.head_dim,
                                 bias=config.attention_bias)
        self.k_proj = LinearMask(self.hidden_size,
                                 self.num_key_value_heads * self.head_dim,
                                 bias=config.attention_bias)
        self.v_proj = LinearMask(self.hidden_size,
                                 self.num_key_value_heads * self.head_dim,
                                 bias=config.attention_bias)
        self.o_proj = LinearMask(self.num_heads * self.head_dim,
                                 self.hidden_size,
                                 bias=config.attention_bias)
        # Native Qwen3 q/k norms (kept; merge consumes post-norm states).
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # RPBE: Gamma residual slot (attached by
        # rpbe.hosts.ccm.ccm_patch.attach_gamma; None = official).
        self.gamma = None
        # RPBE: memory extraction callback (post-merge K/V at SUM rows).
        self.mem_callback = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        sum_row_pos: Optional[torch.Tensor] = None,
        sum_row_valid: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if past_key_value is not None:
            raise NotImplementedError(
                "ccm_qwen3 v1 is full-sequence only; use_cache/past is "
                "not implemented")

        # Conditional LoRA path (comp_mask gate at COMP/SUM positions).
        query_states = self.q_proj(
            hidden_states, comp_mask=comp_mask).view(
                bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(
            hidden_states, comp_mask=comp_mask).view(
                bsz, q_len, self.num_key_value_heads,
                self.head_dim).transpose(1, 2)
        value_states = self.v_proj(
            hidden_states, comp_mask=comp_mask).view(
                bsz, q_len, self.num_key_value_heads,
                self.head_dim).transpose(1, 2)

        # Native q/k norm (per-head RMSNorm on the head dim).
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # RoPE with the CCM skip positions (already indexed cos/sin).
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # CCM: overwrite the SUM rows with the post-RoPE weighted mean of
        # the same-slot preceding COMP rows (uniform 1/j officially).
        if sum_attn_mask is not None:
            sum_attn_mask = sum_attn_mask.to(key_states.dtype)
            key_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1),
                                        key_states)
            value_comp_avg = torch.matmul(sum_attn_mask.unsqueeze(1),
                                          value_states)
            no_sum_mask = (1 - sum_mask).to(key_states.dtype).unsqueeze(1) \
                .unsqueeze(-1)

            # RPBE: Gamma recurrence scan (t_max >= 2), same math as the
            # Llama host (see ccm_llama.py for the full derivation).
            if self.gamma is not None and sum_row_pos is not None \
                    and int(sum_row_pos.shape[1]) >= 2:
                n_heads = key_states.shape[1]
                head_dim = key_states.shape[3]
                n_slots = int(sum_row_pos.shape[2])
                t_max = int(sum_row_pos.shape[1])
                n_rows = t_max * n_slots
                idx = sum_row_pos.reshape(bsz, n_rows).unsqueeze(1)
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
                res_prev_k = torch.zeros(bsz, n_heads, n_slots, head_dim,
                                         dtype=key_states.dtype,
                                         device=key_states.device)
                res_prev_v = torch.zeros_like(res_prev_k)
                res_all_k = torch.zeros_like(k_base)
                res_all_v = torch.zeros_like(v_base)
                valid = sum_row_valid.to(key_states.dtype).unsqueeze(1)
                valid = valid.unsqueeze(-1)  # [B, 1, T, n_slots, 1]
                for t_i in range(2, t_max + 1):
                    tt = torch.full((bsz, 1), t_i, dtype=torch.float32,
                                    device=key_states.device)
                    res_t_k = self.gamma(
                        k_base[:, :, t_i - 2] + res_prev_k,
                        k_cur[:, :, t_i - 1], tt) * valid[:, :, t_i - 1]
                    res_t_v = self.gamma(
                        v_base[:, :, t_i - 2] + res_prev_v,
                        v_cur[:, :, t_i - 1], tt) * valid[:, :, t_i - 1]
                    res_prev_k = res_t_k
                    res_prev_v = res_t_v
                    res_all_k[:, :, t_i - 1] = res_t_k
                    res_all_v[:, :, t_i - 1] = res_t_v
                # OUT-OF-PLACE index_add (freeze-host fix): the old
                # per-batch slice assignment `key_states[b] = ...` is an
                # in-place write; with the host frozen the K/V tensors
                # carry no requires_grad and the in-place op silently
                # DROPS the Gamma residual from the autograd graph
                # (backward dies with "element 0 does not require
                # grad").  A single out-of-place index_add over the
                # batch-flattened index keeps the graph.
                _off = torch.arange(bsz, device=key_states.device) \
                    * key_states.shape[2]
                _idx_k = (sum_row_pos + _off[:, None, None]).reshape(-1)
                key_states = key_states.index_add(
                    1, _idx_k, res_all_k.reshape(-1, n_heads, head_dim))
                value_states = value_states.index_add(
                    1, _idx_k, res_all_v.reshape(-1, n_heads, head_dim))

            # RPBE: memory extraction (post-merge, pre-attention).
            if self.mem_callback is not None:
                self.mem_callback(key_states, value_states, sum_mask,
                                  sum_row_pos)

        past_key_value = None

        # GQA attention: repeat KV heads, additive 4D mask, fp32 softmax.
        key_states_r = repeat_kv(key_states, self.num_key_value_groups)
        value_states_r = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(
            query_states, key_states_r.transpose(2, 3)) / math.sqrt(
                self.head_dim)

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
        # Qwen3 delta: heads * head_dim (4096) != hidden_size (2560);
        # o_proj down-projects to the residual width.
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output, comp_mask=comp_mask)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen3CCMDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3Config, layer_idx: int = -1):
        self.layer_idx = layer_idx
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3CCMAttention(config=config,
                                           layer_idx=self.layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size,
                                            eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        comp_mask: Optional[torch.Tensor] = None,
        sum_mask: Optional[torch.Tensor] = None,
        sum_attn_mask: Optional[torch.Tensor] = None,
        sum_row_pos: Optional[torch.Tensor] = None,
        sum_row_valid: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        max_id: Optional[int] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor,
                                                 torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = \
            self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                comp_mask=comp_mask,
                sum_mask=sum_mask,
                sum_attn_mask=sum_attn_mask,
                sum_row_pos=sum_row_pos,
                sum_row_valid=sum_row_valid,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen3CCMModel(Qwen3PreTrainedModel):
    """Qwen3 decoder with the CCM mask/position pipeline.

    The mask pipeline (get_comp_sum_mask) and the skip position policy
    are ported verbatim from the official LlamaModelCCM; the forward
    keeps the native Qwen3 pre-norm residual stack.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3CCMDecoderLayer(config, layer_idx=i)
             for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

        ## CCM: compression token bookkeeping
        self.comp_token = None
        self.sum_token = None
        self.comp_relative_embedding = getattr(
            config, "comp_relative_embedding", "skip")

        ## RPBE: flipped by attach_gamma()
        self._gamma_attached = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds,
                                        past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype,
                tgt_len=input_shape[-1]).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask)
        return combined_attention_mask

    def get_comp_sum_mask(self, input_ids):
        """comp/sum masks and the merge/visibility plan (official port).

        Returns (comp_mask, sum_mask, sum_attn_mask, sum_row_pos,
        sum_row_valid); sum_row_* are built only when _gamma_attached
        (RPBE), exactly as on the Llama host.
        """
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
                            _spos = (input_ids[_b] == self.sum_token[_k]
                                     ).nonzero(as_tuple=False).flatten()
                            _cpos = (input_ids[_b] == self.comp_token[_k]
                                     ).nonzero(as_tuple=False).flatten()
                            for _j, _p in enumerate(_spos):
                                _nh = int((_cpos < _p).sum().item())
                                if _nh <= 0:
                                    continue
                                _w = torch.softmax(
                                    _lg[_b, :_nh].float(), dim=0)
                                sum_attn_mask[_b, _p,
                                              _cpos[:_nh]] = _w.to(
                                    sum_attn_mask.dtype)
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pos_id_offset: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        ## CCM: compression masks for conditional inference and merge
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
                "ccm_qwen3 v1 is full-sequence only (use_cache=False)")
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

        past_key_values_length = 0
        seq_length_with_past = seq_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )

        ## CCM: skip position ids (text contiguous, COMP/SUM cycle 1..m)
        if position_ids is None:
            if self.comp_relative_embedding == "base" or comp_mask is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, past_key_values_length:]
            else:
                position_ids = update_position_ids(
                    comp_mask,
                    self.comp_token,
                    attention_mask[:, -seq_length:],
                    type_=self.comp_relative_embedding)
            if pos_id_offset is not None:
                position_ids += pos_id_offset
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        ## CCM: fold the merge_recur visibility mask into the 4D mask
        if attention_mask_comp is not None:
            min_val = torch.tensor(torch.finfo(attention_mask.dtype).min)
            attention_mask_comp_float = torch.full_like(attention_mask,
                                                        min_val)
            attention_mask_comp_float = attention_mask_comp_float \
                .masked_fill(attention_mask_comp.bool(), 0.0)
            attention_mask = attention_mask + attention_mask_comp_float

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = None

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                comp_mask=comp_mask,
                sum_mask=sum_mask,
                sum_attn_mask=sum_attn_mask,
                sum_row_pos=sum_row_pos,
                sum_row_valid=sum_row_valid,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_id=None,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
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


class Qwen3ForCausalLM_CCM(Qwen3PreTrainedModel):
    """Qwen3 causal LM with CCM compression tokens.

    The lm_head stays tied to the input embeddings (native Qwen3);
    resize_token_embeddings grows both sides with the COMP/SUM rows.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}  # dict form required by transformers 5.x

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3CCMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)
        self.post_init()

        ## CCM: compression token bookkeeping
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
        """Interface stub for peft_custom compatibility.

        (generation itself raises in the model forward: v1 is
        full-sequence only).  Kept verbatim from the Llama host so the
        vendored PeftModel wrapper constructs cleanly."""
        first_time_with_compression = first_time and pos_id_offset is not None

        comp_mask = sum_mask = None
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
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pos_id_offset=pos_id_offset,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

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
