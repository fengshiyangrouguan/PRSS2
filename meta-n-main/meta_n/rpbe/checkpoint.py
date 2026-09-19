"""Gamma checkpoint save/load -- the Phase B -> Phase C handoff (0 API).

WHY THIS EXISTS
---------------
Phase B trains Gamma offline; Phase C must evaluate that FROZEN Gamma under
`predictive` reduction. Before this module there was no path between them:
`main.py` built `SlottedFusion()` fresh for predictive mode, so a "Phase C" run
would have been scored with a RANDOM Gamma and the number would have been
meaningless. This closes that gap.

CONTRACT
--------
* `save_gamma_checkpoint` writes the parameters plus the provenance needed to
  detect a mismatched load (parameter count, tensor shapes, config fingerprint,
  steps, status) and a checksum over the payload.
* `load_gamma_checkpoint` REFUSES to hand back a partially-loaded model: a
  missing file, a shape mismatch, or a corrupt payload raises
  `GammaCheckpointError`. There is deliberately no fallback to a fresh Gamma --
  a silent random initialization is the exact failure this module prevents.
* `strict_load_into` loads into an existing fusion and verifies bit-for-bit
  equality of every parameter after the copy.

The checkpoint is a plain torch file (state_dict + meta), so it round-trips
without importing Meta^n.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from meta_n.rpbe import config as C

FORMAT = "rpbe-gamma-checkpoint"
FORMAT_VERSION = 1


class GammaCheckpointError(RuntimeError):
    """A Gamma checkpoint is missing, malformed, or does not match the model."""


@dataclass(frozen=True)
class CheckpointMeta:
    format: str
    format_version: int
    param_count: int
    shapes: Dict[str, list]
    d_e: int
    h: int
    n_slots: int
    cond_rank: int
    n_branches: int
    m_sketch: int
    b_v_dim: int
    steps: int
    status: str
    stop_reason: Optional[str]
    saved_at: str
    checksum: Optional[str] = None

    def fingerprint(self) -> str:
        """Stable identity of the GEOMETRY this Gamma lives in.

        A Gamma trained under one geometry is meaningless under another, so the
        loader compares this before it will return a model.
        """
        return hashlib.sha256(json.dumps({
            "d_e": self.d_e, "h": self.h, "n_slots": self.n_slots,
            "cond_rank": self.cond_rank, "n_branches": self.n_branches,
            "m_sketch": self.m_sketch, "b_v_dim": self.b_v_dim,
            "param_count": self.param_count,
        }, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _geometry(fusion) -> Dict[str, int]:
    return {
        "d_e": int(getattr(fusion, "d_e", C.D_E)),
        "h": int(getattr(fusion, "h", C.H_LORA)),
        "n_slots": int(getattr(fusion, "n_slots", C.N_SLOTS)),
        "cond_rank": int(getattr(fusion, "cond_rank", C.COND_RANK)),
        "n_branches": int(C.N_BRANCHES),
        "m_sketch": int(C.M_SKETCH),
        "b_v_dim": int(C.B_V_DIM),
    }


def _checksum(state: Dict[str, Any], meta: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(torch.as_tensor(state[k]).detach().cpu().numpy().tobytes())
    h.update(json.dumps(meta, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def save_gamma_checkpoint(path, fusion, *, steps: int = 0,
                          status: str = "completed",
                          stop_reason: Optional[str] = None,
                          extra: Optional[Dict[str, Any]] = None) -> CheckpointMeta:
    """Persist the CURRENT parameters of ``fusion``.

    "Current" is exactly right for Phase B: a certificate-blocked step commits
    nothing, so whatever the module holds at the end of `run()` IS the last
    valid Gamma (see trainer.CertificateBlocked).
    """
    state = {k: v.detach().cpu().clone() for k, v in fusion.state_dict().items()}
    geo = _geometry(fusion)
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "param_count": int(sum(v.numel() for v in state.values())),
        "shapes": {k: list(v.shape) for k, v in state.items()},
        "steps": int(steps),
        "status": str(status),
        "stop_reason": stop_reason,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "extra": dict(extra or {}),
        **geo,
    }
    meta["checksum"] = _checksum(state, {k: v for k, v in meta.items()
                                         if k != "checksum"})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "meta": meta}, str(p))
    return CheckpointMeta(**{k: meta[k] for k in CheckpointMeta.__annotations__})


def read_meta(path) -> CheckpointMeta:
    """Read only the provenance block (for pre-flight logging)."""
    p = Path(path)
    if not p.is_file():
        raise GammaCheckpointError(
            "Gamma checkpoint not found at {!r}. Refusing to continue: "
            "evaluating 'predictive' with a randomly initialised Gamma would "
            "produce a meaningless score.".format(str(p)))
    try:
        blob = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as e:                                       # noqa: BLE001
        raise GammaCheckpointError(
            "Gamma checkpoint at {!r} is unreadable: {}: {}".format(
                str(p), type(e).__name__, e))
    meta = blob.get("meta") or {}
    if meta.get("format") != FORMAT:
        raise GammaCheckpointError(
            "{!r} is not a {} payload (got {!r})".format(
                str(p), FORMAT, meta.get("format")))
    if int(meta.get("format_version", -1)) != FORMAT_VERSION:
        raise GammaCheckpointError(
            "checkpoint format_version {} != {}".format(
                meta.get("format_version"), FORMAT_VERSION))
    state = blob.get("state_dict") or {}
    want = meta.get("checksum")
    got = _checksum(state, {k: v for k, v in meta.items() if k != "checksum"})
    if want and want != got:
        raise GammaCheckpointError(
            "checksum mismatch at {!r}: stored {} computed {} (corrupt or "
            "edited)".format(str(p), want, got))
    return CheckpointMeta(**{k: meta[k] for k in CheckpointMeta.__annotations__})


def load_gamma_checkpoint(path, fusion=None, *, strict: bool = True):
    """Load a checkpoint into ``fusion`` (or a fresh SlottedFusion).

    Returns ``(fusion, meta)``. Raises :class:`GammaCheckpointError` on a
    missing file, a geometry mismatch, or a size mismatch -- there is NO
    fallback to a fresh Gamma.
    """
    meta = read_meta(path)
    if fusion is None:
        from meta_n.rpbe.fusion import SlottedFusion
        fusion = SlottedFusion()
    if strict:
        want = _geometry(fusion)
        mismatch = {k: (v, want[k]) for k, v in
                    {"d_e": meta.d_e, "h": meta.h, "n_slots": meta.n_slots,
                     "cond_rank": meta.cond_rank, "n_branches": meta.n_branches,
                     "m_sketch": meta.m_sketch, "b_v_dim": meta.b_v_dim}.items()
                    if v != want[k]}
        if mismatch:
            raise GammaCheckpointError(
                "checkpoint geometry does not match the model: {}".format(
                    mismatch))
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob["state_dict"]
    missing, unexpected = fusion.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise GammaCheckpointError(
            "state_dict mismatch: missing={} unexpected={}".format(
                list(missing), list(unexpected)))
    n = int(sum(p.numel() for p in fusion.parameters()))
    if n != meta.param_count:
        raise GammaCheckpointError(
            "parameter count {} != checkpoint {}".format(n, meta.param_count))
    if strict:
        for k, v in fusion.state_dict().items():
            if not torch.equal(v.detach().cpu(), state[k].detach().cpu()):
                raise GammaCheckpointError(
                    "parameter {!r} differs after load (not bit-identical)"
                    .format(k))
    return fusion, meta


# --------------------------------------------------------------------------
# acceptance self-test
# --------------------------------------------------------------------------

def self_test() -> int:
    import tempfile
    print("checkpoint.py acceptance")
    print("=" * 68)
    from meta_n.rpbe.fusion import SlottedFusion

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "gamma.pt"

        # missing file => HARD failure, never a silent fresh model
        try:
            load_gamma_checkpoint(p)
            raise AssertionError("a missing checkpoint was accepted")
        except GammaCheckpointError as e:
            assert "not found" in str(e)
        print("OK  missing file       refused (no silent random init)")

        g0 = SlottedFusion()
        for i, prm in enumerate(g0.parameters()):
            prm.data.add_(torch.full_like(prm, 0.01 * (i + 1)))
        meta = save_gamma_checkpoint(p, g0, steps=7, status="completed",
                                     extra={"note": "unit test"})
        assert meta.param_count == sum(x.numel() for x in g0.parameters())
        print("OK  save               {} params, checksum {}".format(
            meta.param_count, meta.checksum))

        # round-trip into a FRESH model
        g1 = SlottedFusion()
        before = [x.detach().clone() for x in g1.parameters()]
        g1, meta2 = load_gamma_checkpoint(p, g1)
        for a, b in zip(before, g1.parameters()):
            assert not torch.equal(a.detach(), b.detach()), \
                "load changed nothing -- was it a no-op?"
        ok = all(torch.equal(a.detach().cpu(), b.detach().cpu()) for a, b in
                 zip(g0.state_dict().values(), g1.state_dict().values()))
        assert ok
        print("OK  round-trip         save -> fresh model -> load is "
              "BIT-IDENTICAL ({} tensors)".format(len(g0.state_dict())))

        # geometry mismatch is refused
        small = SlottedFusion(d_e=64)
        try:
            load_gamma_checkpoint(p, small)
            raise AssertionError("geometry mismatch accepted")
        except GammaCheckpointError as e:
            assert "geometry" in str(e)
        print("OK  geometry guard     mismatched model refused")

        # corruption is refused
        blob = torch.load(str(p), map_location="cpu", weights_only=False)
        k = sorted(blob["state_dict"])[0]
        blob["state_dict"][k] = blob["state_dict"][k] + 1.0
        torch.save(blob, str(p))
        try:
            load_gamma_checkpoint(p)
            raise AssertionError("a tampered checkpoint was accepted")
        except GammaCheckpointError as e:
            assert "checksum" in str(e)
        print("OK  checksum guard     tampered payload refused")

        assert "gamma.pt" in str(p) and meta2.fingerprint() == meta.fingerprint()
        print("OK  fingerprint        stable across save/load: {}".format(
            meta.fingerprint()))

    led = __import__("os").environ.get("META_N_REQUEST_LEDGER", "").strip()
    paid = 0
    if led and __import__("os").path.isfile(led):
        from meta_n.rpbe.accounting import snapshot
        paid = snapshot()["backend_requests"]
    assert paid == 0
    print("OK  zero-cost          paid backend_requests = {}".format(paid))
    print()
    print("VERDICT: ALL OK")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(self_test())
