"""Gamma attach hook for the vendored CCM llama (plan v2 L2).

``attach_gamma`` puts one :class:`GammaResidual` on every layer's
``LlamaAttention`` and flips the model's ``_gamma_attached`` flag; the
vendored forward then builds the previous-turn masks and applies the
recurrence scan in the merge block (see the RPBE modification markers in
``third_party/ccm/src/arch/ccm_llama.py``).

The three arms must be able to share an identical data-sampling RNG:
``paired_seed_hash`` derives the per-arm seed from the base seed and the
byte content of every trainable parameter (LoRA + Gamma + ...) plus the
COMP/SUM embedding rows, so equal configuration <=> equal sampling stream
(plan L2 closure 5).
"""

import hashlib
import struct
from typing import List

from .gamma_residual import GammaResidual

N_TOK_LOCK = 2  # every run explicitly pins n_tok = 2 (plan L2)


def _base_model(model):
    """Descend the .model chain (PEFT wrapper -> CausalLM -> LlamaModelCCM)
    to the object that owns the transformer layers."""
    base = model
    while not hasattr(base, "layers") and hasattr(base, "model"):
        base = base.model
    return base


def attach_gamma(model, *, head_dim=None, hidden=64, time_dim=16,
                 init_scale=0.02) -> List[GammaResidual]:
    """Attach one GammaResidual per transformer layer.

    The residual is shared across the K and V merges and across the two
    COMP/SUM slots.  Call after ``update_comp_token`` (n_tok is locked
    to 2) and after PEFT wrapping if the run uses LoRA.
    """
    node = model
    while node is not None:
        if "Stream" in type(node).__name__:
            raise NotImplementedError(
                "Gamma is not implemented for the stream variant yet; use "
                "the non-stream model for training and evaluation (plan L2)")
        node = getattr(node, "model", None)
    base = _base_model(model)
    comp = getattr(base, "comp_token", None)
    sums = getattr(base, "sum_token", None)
    if comp is None or sums is None:
        raise ValueError("set comp/sum tokens (update_comp_token) before "
                         "attaching Gamma")
    if len(comp) != N_TOK_LOCK or len(sums) != N_TOK_LOCK:
        raise ValueError(
            "Gamma requires n_tok == {} (got comp={}, sum={})"
            .format(N_TOK_LOCK, len(comp), len(sums)))
    modules = []
    # Follow the model's current device (attach may happen after .cuda()).
    device = next(base.parameters()).device
    for layer in base.layers:
        attn = layer.self_attn
        if getattr(attn, "gamma", None) is not None:
            raise ValueError("Gamma already attached to this model")
        if head_dim is None:
            head_dim = attn.head_dim
        attn.gamma = GammaResidual(head_dim, time_dim=time_dim, hidden=hidden,
                                   init_scale=init_scale).to(device)
        modules.append(attn.gamma)
    base._gamma_attached = True
    return modules


def wrap_lora(model, *, r: int = 8, target_modules=None):
    """LoRA wrap with the official conditional-LoRA path first.

    The vendored attention calls ``q_proj(x, comp_mask=...)``; the
    official peft_custom LoRA layer accepts that keyword, while modern
    peft only forwards it on newer versions (>= ~0.20).  Prefer the
    official path when importable (cloud peft 0.4), fall back to modern
    peft otherwise (local tests on peft 0.20).
    """
    from peft import LoraConfig
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(r=int(r), lora_alpha=2 * int(r), lora_dropout=0.0,
                        bias="none", task_type="CAUSAL_LM",
                        target_modules=list(target_modules))
    try:
        from src import peft_custom  # vendored, comp_mask-aware
        return peft_custom.get_peft_model(model, config)
    except ImportError:
        from peft import get_peft_model
        return get_peft_model(model, config)


def paired_seed_hash(base_seed: int, model, *, extra_components=()) -> str:
    """Per-arm seed derivation (plan L2 closure 5).

    Hashes the base seed together with every trainable parameter byte
    (LoRA + Gamma + anything else trainable) and the COMP/SUM embedding
    rows of both the input embedding and the lm_head, so the three arms
    share an identical data-sampling RNG only when their trainable
    components are configured identically.  ``extra_components`` may add
    (name, bytes) pairs for any other frozen measurement-affecting state.
    """
    h = hashlib.sha256()
    h.update(b"ccm-paired-seed-v1")
    h.update(struct.pack(">q", int(base_seed)))
    seen = set()
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            h.update(name.encode())
            h.update(p.detach().float().cpu().reshape(-1).contiguous()
                     .numpy().tobytes())
    # COMP/SUM embedding rows participate even when frozen (closure 5).
    base = _base_model(model)
    emb = base.embed_tokens.weight.detach().float().cpu()
    head = getattr(model, "lm_head", None)
    head_w = head.weight.detach().float().cpu() if head is not None else None
    for token in (list(getattr(model, "comp_token", None) or [])
                  + list(getattr(model, "sum_token", None) or [])):
        h.update(b"embed_row:%d" % int(token))
        h.update(emb[int(token)].reshape(-1).contiguous().numpy().tobytes())
        if head_w is not None:
            h.update(b"head_row:%d" % int(token))
            h.update(head_w[int(token)].reshape(-1).contiguous()
                     .numpy().tobytes())
    for name, data in extra_components:
        h.update(name.encode())
        h.update(bytes(data))
    return h.hexdigest()
