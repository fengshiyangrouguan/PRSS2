"""Frozen item encoder E_0 (task book v4.1 §1.1 / §2.1 / §2.8).

    X_v[j] = P_rc . E_0(serialize(item_j))  +  t_type(type(j))  +  r_depth(depth(j))

Everything here is FROZEN: the CodeBERT weights, the chunking, the pooling, the
projection and the type/depth buffers. Nothing receives gradient, and the whole
path runs under `no_grad` and returns detached tensors.

Frozen encoder spec (§1.1):
    model_id   microsoft/codebert-base
    revision   3b0952feddeffad0063f274080e3c23d75e7eb39
    hidden     768  -> output 256
    chunking   non-overlapping windows of at most 256 tokens INCLUDING special
               tokens; masked mean per chunk; chunks combined weighted by their
               valid token count
    projection fixed Rademacher P_rc in R^{768x256}, entries +/-1/sqrt(256), seed 0
    pooling    masked mean
    gradient   torch.no_grad() + detach()

Serialization REUSES the official formatters (§2.8). No second text template:
    Trace        -> OmegaEngine._format_raw_traces([trace])
    InjectedCode -> OmegaEngine._format_context_stack([code])
    query        -> OmegaEngine._format_raw_traces(current_traces)

The encoder emits X_v and q_emb ONLY. `Q_v` is NOT produced here: it depends on
Gamma and must be recomputed every step (§2.1).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import torch

from meta_n.core.meta_layer import InjectedCode, Trace
from meta_n.rpbe import config as C

TYPE_TRACE = 0
TYPE_CODE = 1
N_TYPES = 2


class EncoderError(RuntimeError):
    pass


def _rademacher(shape: Tuple[int, int], seed: int, scale_dim: int,
                dtype=torch.float32):
    g = torch.Generator().manual_seed(int(seed))
    bits = torch.randint(0, 2, shape, generator=g).to(dtype) * 2 - 1
    return bits / math.sqrt(float(scale_dim))


class FrozenItemEncoder:
    """`U_v -> X_v, q_emb`. Deterministic, gradient-free, offline once loaded."""

    def __init__(self, model_path: str,
                 revision: str = C.ENCODER_REVISION,
                 device: Optional[str] = None,
                 max_tokens: int = C.ENCODER_MAX_TOKENS,
                 d_e: int = C.D_E,
                 proj_seed: int = C.P_RC_SEED,
                 type_depth_seed: int = C.GAMMA_INIT_SEED,
                 init_scale: float = C.GAMMA_INIT_SCALE,
                 lazy: bool = False):
        self.model_path = str(model_path)
        self.revision = str(revision)
        self.max_tokens = int(max_tokens)
        self.d_e = int(d_e)
        self.device = device
        self.hidden = C.ENCODER_HIDDEN
        # Fixed projection + fixed type/depth buffers: plain attributes, never
        # Parameters, so they can never enter an optimizer.
        self.P_rc = _rademacher((self.hidden, self.d_e), proj_seed, self.d_e)
        g = torch.Generator().manual_seed(int(type_depth_seed))
        self.t_type = torch.stack(
            [torch.randn(self.d_e, generator=g) * init_scale
             for _ in range(N_TYPES)])
        self.r_depth = torch.stack(
            [torch.randn(self.d_e, generator=g) * init_scale
             for _ in range(C.MAX_DEPTH + 1)])
        self._tok = None
        self._model = None
        if not lazy:
            self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(
            self.model_path, revision=self.revision)
        self._model = AutoModel.from_pretrained(
            self.model_path, revision=self.revision)
        self._model.eval()
        if self.device:
            self._model.to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)

    @property
    def _dev(self):
        return next(self._model.parameters()).device

    # -- serialization (§2.8) ---------------------------------------------
    @staticmethod
    def _engine():
        from meta_n.core.omega import OmegaEngine
        return object.__new__(OmegaEngine)

    @classmethod
    def serialize_item(cls, item: Any) -> Tuple[str, int, int]:
        """-> (text, type_id, depth). Official formatters only."""
        eng = cls._engine()
        if isinstance(item, Trace):
            from meta_n.core.omega import OmegaEngine
            return (OmegaEngine._format_raw_traces(eng, [item]),
                    TYPE_TRACE, int(getattr(item, "depth", 0) or 0))
        if isinstance(item, InjectedCode):
            from meta_n.core.omega import OmegaEngine
            return (OmegaEngine._format_context_stack(eng, [item]),
                    TYPE_CODE, int(getattr(item, "source_depth", 0) or 0))
        raise EncoderError(
            "unsupported item type {}; U_v items must be Trace or InjectedCode"
            .format(type(item).__name__))

    @staticmethod
    def serialize_query(current_traces: Sequence[Trace]) -> str:
        """The query's LOCAL observation: the current cut's traces only.

        Never the child/grandchild, never C_v (§2.8).
        """
        from meta_n.core.omega import OmegaEngine
        eng = object.__new__(OmegaEngine)
        return OmegaEngine._format_raw_traces(eng, list(current_traces))

    # -- chunked encoding (§1.1) ------------------------------------------
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """`E_0(text)` -> [d_e], detached."""
        if self._model is None:
            raise EncoderError("encoder not loaded; pass lazy=False")
        if not text:
            return torch.zeros(self.d_e)
        ids = self._tok(text, add_special_tokens=True,
                        return_tensors="pt")["input_ids"][0]
        # Non-overlapping windows of at most max_tokens INCLUDING special
        # tokens: keep 2 slots for the two specials the tokenizer adds.
        body = max(1, self.max_tokens - 2)
        pooled = []
        weights = []
        for start in range(0, int(ids.shape[0]), body):
            chunk = ids[start:start + body]
            if chunk.numel() == 0:
                continue
            enc = self._tok.decode(chunk, skip_special_tokens=True)
            batch = self._tok(enc, add_special_tokens=True,
                              return_tensors="pt")
            batch = {k: v.to(self._dev) for k, v in batch.items()}
            out = self._model(**batch).last_hidden_state[0]      # [T, 768]
            mask = batch["attention_mask"][0].unsqueeze(-1).to(out.dtype)
            n_valid = float(mask.sum())
            pooled.append((out * mask).sum(0) / max(n_valid, 1.0))
            weights.append(n_valid)
        if not pooled:
            return torch.zeros(self.d_e)
        P = torch.stack(pooled, 0)                               # [C, 768]
        w = torch.tensor(weights, dtype=P.dtype,
                         device=P.device).clamp(min=1.0)
        mean = (P * w.unsqueeze(-1)).sum(0) / w.sum()             # [768]
        # P_rc is [hidden, d_e]: the projection is mean^T P_rc -> [d_e].
        vec = mean @ self.P_rc.to(mean.device, mean.dtype)
        return vec.detach().cpu()

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        """`q_emb` -> [d_e], detached."""
        return self.encode_text(text)

    @torch.no_grad()
    def encode(self, items: Sequence[Any]) -> torch.Tensor:
        """`U_v -> X_v` -> [n_v, d_e], detached, with type/depth added."""
        rows: List[torch.Tensor] = []
        for item in items:
            text, type_id, depth = self.serialize_item(item)
            base = self.encode_text(text)
            d = max(0, min(int(depth), C.MAX_DEPTH))
            rows.append(base + self.t_type[type_id] + self.r_depth[d])
        if not rows:
            return torch.zeros(0, self.d_e)
        return torch.stack(rows, 0).detach()


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

_STUB = "stub"


def self_test() -> int:
    """Runs fully offline against a stub embedder; the real CodeBERT path is
    exercised by `--with-model` (still zero API cost)."""
    import os
    import sys
    print("encoder.py acceptance")
    print("=" * 68)

    have_model = "--with-model" in sys.argv

    # --- geometry / freeze properties, no model needed --------------------
    enc = FrozenItemEncoder(_STUB, lazy=True)
    assert enc.P_rc.shape == (768, C.D_E)
    mags = enc.P_rc.abs().reshape(-1)
    assert torch.allclose(mags, torch.full_like(mags, 1.0 / math.sqrt(C.D_E)))
    assert not enc.P_rc.requires_grad
    assert enc.t_type.shape == (N_TYPES, C.D_E)
    assert enc.r_depth.shape == (C.MAX_DEPTH + 1, C.D_E)
    print("OK  frozen buffers     P_rc [768,256] entries +/-1/sqrt(256); "
          "t_type [2,256]; r_depth [11,256]; none require grad")

    # deterministic projection and buffers
    enc2 = FrozenItemEncoder(_STUB, lazy=True)
    assert torch.equal(enc.P_rc, enc2.P_rc)
    assert torch.equal(enc.t_type, enc2.t_type) and torch.equal(enc.r_depth,
                                                               enc2.r_depth)
    print("OK  deterministic      projection + type/depth buffers reproduce "
          "exactly across instances")

    # --- serialization reuses the OFFICIAL formatters ---------------------
    tr = Trace(task_id="t", script="print(1)", stdout="o", success=True)
    code = InjectedCode(pre_process="def p(): pass", rationale="r",
                        code_library={"f": "def f(): return 1"},
                        source_depth=3)
    txt_t, ty_t, dp_t = FrozenItemEncoder.serialize_item(tr)
    txt_c, ty_c, dp_c = FrozenItemEncoder.serialize_item(code)
    assert ty_t == TYPE_TRACE and ty_c == TYPE_CODE
    assert dp_c == 3 and txt_c.strip() and txt_t.strip()
    q = FrozenItemEncoder.serialize_query([tr])
    assert isinstance(q, str) and q.strip()
    import inspect
    from meta_n.core.omega import OmegaEngine
    src = inspect.getsource(FrozenItemEncoder.serialize_item)
    assert "_format_raw_traces" in src and "_format_context_stack" in src
    assert hasattr(OmegaEngine, "_format_context_stack")
    print("OK  official serialize Trace-> _format_raw_traces, InjectedCode-> "
          "_format_context_stack, query-> _format_raw_traces")

    # rejecting a foreign item type keeps U_v well defined
    try:
        FrozenItemEncoder.serialize_item(object())
        raise AssertionError("accepted a foreign item type")
    except EncoderError:
        pass
    print("OK  item types         non-Trace/InjectedCode rejected")

    # --- the real model, still zero API cost ------------------------------
    if have_model:
        import os as _os
        path = _os.environ.get("CODEBERT_PATH", "")
        if not path or not _os.path.isdir(path):
            print("!!  --with-model given but CODEBERT_PATH is unset/missing")
            return 1
        real = FrozenItemEncoder(path)
        X = real.encode([tr, code, tr])
        assert X.shape == (3, C.D_E), X.shape
        assert not X.requires_grad
        qv = real.encode_query("some query text")
        assert qv.shape == (C.D_E,) and not qv.requires_grad
        # same text -> same vector; different text -> different vector
        assert torch.allclose(real.encode_text(txt_t), real.encode_text(txt_t))
        assert not torch.allclose(real.encode_text(txt_t), real.encode_text(txt_c))
        # an item's row differs from its bare text encoding (type/depth added)
        assert not torch.allclose(X[0], real.encode_text(txt_t))
        print("OK  real CodeBERT     X_v [3,256] detached; encode_query [256]; "
              "identical text repeats, distinct text differs")

    paid = 0
    led = os.environ.get("META_N_REQUEST_LEDGER", "").strip()
    if led and os.path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))
    if not have_model:
        print("    (model-dependent checks skipped; pass --with-model)")

    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
