# Vendored from snu-mllab/Context-Memory

- Source: https://github.com/snu-mllab/Context-Memory
- Commit: a89dd08e2c9587ec9c6c3ad339bb154c33e6b41a ("add comment")
- License: MIT (see LICENSE)
- Vendored for the CCM-merge × RPBE cross-domain experiment
  (docs/CCM_RPBE_plan_v2.md).  Modifications are kept minimal and
  documented:

## L0 (protocol + transformers compatibility)

- `src/arch/ccm_llama.py`: `_make_causal_mask`/`_expand_mask` inlined from
  transformers 4.29; `LlamaMLP(config)` constructor; RoPE constructor is
  version-gated (transformers >= 5 takes the whole config, 4.44 the
  `(dim, max_position_embeddings)` form).
- `src/arch/generation_utils.py`: the `GreedySearch*` output classes were
  removed in transformers 5.x; a local fallback defines them (4.44 keeps
  the upstream import).
- `src/arch/ccm_llama_stream.py`: same transformers-compat treatment.
- `src/arguments.py`: dataclass default_factory; Union-container
  overrides for `lr_scheduler_kwargs`/`fsdp_config`.
- `src/model.py`: T5 import isolation.
- `src/data/dialogue/data.py`: `DIALOG_MIRROR` local mirror (the official
  yanran.li URL is dead); `clean_split=True` restores the official
  train/validation/test split.
- `src/data/collator.py`, `src/data/dialogue/collator.py`: dict return.
- `src/callbacks.py`: wandb `Settings`.

## L2 (Gamma residual, plan v2)

- `src/arch/ccm_llama.py`:
  - `LlamaAttention` gains a `gamma` slot (None = official behavior).
  - The merge block computes the previous-turn mean and the current COMP
    K/V from the ORIGINAL unmerged states, then applies the recurrence
    scan `M_t = mean + R_theta(M_{t-1}, h_t, t)` per turn (eager loop;
    all three arms run with torch.compile disabled).  With the gate
    `s == 0` the added residual is exactly 0.0, so the official
    arithmetic merge is reproduced bit-identically (Test A).
  - `get_comp_sum_mask` additionally returns (when `_gamma_attached`,
    flipped by `rpbe.hosts.ccm.attach_gamma`): the previous-turn COMP
    mean mask, the previous same-slot SUM selector, and the per-row turn
    counts.  Official-reproduction mode is untouched (flag off).
- The learnable piece lives in `src/rpbe/hosts/ccm/` (gamma_residual +
  ccm_patch), NOT in this vendored copy.
- NOT yet ported: `src/arch/ccm_llama_stream.py` (stream eval variant);
  `attach_gamma` refuses stream models.
