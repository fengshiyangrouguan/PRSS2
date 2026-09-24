#!/usr/bin/env python3
"""CCM-merge x RPBE training entry (plan v2 L5).

Three arms, identical task samples / seed / cadence:

  ccm_merge       official protocol: LoRA on merge_recur, one pass, one
                  optimizer step per ``grad_accum`` microbatch.
  gamma_task_only  Gamma attached, LoRA, the SAME two-pass window replay
                  as ours with lambda = 0 (matched control).
  ours            Gamma + RPBE: pass 1 (no grad) collects cut rows
                  incrementally into a KFMomentWindow that closes at
                  >= kf_min_cuts effective cuts; pass 2 replays the whole
                  window's microbatch dicts and trains task CE plus the
                  exact surrogate (numerically zero, gradient = window
                  J).  One optimizer step per closed window.

Native actuation (--rpbe-native-compression, review 2026-09-22): the
RPBE recursion node is the native CCM merge step (M_{t-1}, u_t) -> M_t
with depth D=L and Z_t = M_t; the actuated params are the conditional
LoRA + COMP/SUM comp-embedding rows (the backbone is frozen, no Gamma
module).  ``ours`` + flag = Native-Ours: the treewise QP runs in ADAMW
PROPOSAL space — d_0 = joint proposal (g_task + lambda*g_pred through
the shadow AdamW state), d* = Proj_F(d_0) with half-spaces
q_i^T d >= -kappa*||q_i||*||d_task||, and theta <- theta + d* is
written back directly.  ``gamma_task_only`` + flag = Native-task-only:
the ordinary AdamW step on the same data stream.  The gamma line keeps
the V11 gradient-space treewise implementation unchanged.

RNG protocol (plan L5): each microbatch's pass-1 forward runs under a
saved RNG state that is restored right after (builder counters are NOT
restored — cut ids stay monotonic), so the data-sampling stream matches
the official single-pass arm exactly.  The model carries zero dropout
(LLaMA default), so pass-2 replays the identical forward; this is
asserted at startup (L6 test 3).  Pass 2 restores the window-start state
(RNG + builder counter) so occurrence ids align with the replay plan.

Lambda follows the r_eff calibration rule (plan L5); --calibrate-lambda
measures r_eff = ||g_KF|| / ||g_task|| on the first closed window and
exits with the derived lambda (gradient space on the gamma line,
proposal space on the native line).
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
CCM = HERE.parent / "third_party" / "ccm"
for p in (str(SRC), str(CCM)):
    if p not in sys.path:
        sys.path.insert(0, p)

from rpbe.hosts.ccm.adapter import CCMHostAdapter
from rpbe.hosts.ccm.ccm_patch import (N_TOK_LOCK, attach_gamma,
                                      paired_seed_hash, wrap_lora)
from rpbe.hosts.ccm.gamma_residual import GammaResidual
from rpbe.llm.dialogue_records import (DialogueCutBuilder, DialogueMeta,
                                       Llmmaps, MEM_TAU)
from rpbe.llm.utterance_embed import UtteranceEmbed
from rpbe.llm.mem_lift import JMemLift
from rpbe.loss import KFMomentWindow
from rpbe.training.checkpoint import _restore_rng, _rng_state


class BranchEnsembleWindow:
    """Review round 8: four independent 32-dim sketch branches.

    Wraps one :class:`KFMomentWindow` per measurement branch.  Each
    branch sees the same rows with ``p_override`` sliced to its own
    32-dim sketch; branches close independently and the ensemble score
    is ``J_ens = (1/N) sum_r J_r``.  The branches are NEVER concatenated
    into one 128-dim covariance — that would restore the small-sample
    ill-conditioning the ensemble exists to avoid.  The replay gradient
    is the branch-mean of the per-cut adjoints (dJ_ens/dz = (1/N) sum_r
    dJ_r/dz; a failed branch contributes zero).
    """

    def __init__(self, branches):
        self.ws = list(branches)
        self.nb = len(self.ws)

    def add(self, rows):
        for br, w in enumerate(self.ws):
            sub = [dataclasses.replace(r, p_override=r.p_override[br])
                   for r in rows]
            w.add(sub)
        return {}, {}, []

    def window_ready(self) -> bool:
        return all(w.window_ready() for w in self.ws)

    def _threshold(self, tau: str) -> float:
        return self.ws[0]._threshold(tau)

    def close_replay(self):
        closed: dict = {}
        plans = []
        diags = []
        j_branches = []
        for br, w in enumerate(self.ws):
            c, p, d = w.close_replay()
            for tau, j in c.items():
                closed[tau] = closed.get(tau, 0.0) + float(j) / self.nb
            plans.append(p)
            diags.append(d)
            j_branches.append({tau: float(jj) for tau, jj in c.items()})
        plan: dict = {}
        diag: dict = {}
        for tau in closed:
            oid_all = set()
            for p in plans:
                oid_all |= set(p.get(tau, {}).get("by_oid", {}).keys())
            by_oid = {}
            for oid in oid_all:
                gs = [p.get(tau, {}).get("by_oid", {}).get(oid)
                      for p in plans]
                gs = [g for g in gs if g is not None]
                if gs:
                    by_oid[oid] = (sum(gs) / self.nb).float()
            plan[tau] = {"by_batch": [[]], "by_oid": by_oid}
            d0 = None
            for p_d in diags:
                if tau in p_d:
                    d0 = dict(p_d[tau])
                    break
            if d0 is None:
                diag[tau] = {"failed": None, "below_threshold": True}
            else:
                d0["J_branches"] = [jb.get(tau, float("nan"))
                                    for jb in j_branches]
                d0["J_ens"] = closed.get(tau, 0.0)
                if d0.get("J_shuffled") is not None:
                    d0["J_real_minus_shuffled"] = (
                        float(closed.get(tau, 0.0))
                        - float(d0["J_shuffled"]))
                diag[tau] = d0
        return closed, plan, diag

N_TOK = N_TOK_LOCK  # comp slots; 2 comp + 2 sum = 4 added tokens


def parse_args():
    p = argparse.ArgumentParser("CCM x RPBE training (plan v2 L5)")
    p.add_argument("--arm", required=True,
                   choices=["ccm_merge", "gamma_task_only", "ours",
                            "ccm_merge_official"],
                   help="ccm_merge_official: official fixed-accumulation "
                        "cadence reproduction arm (frozen cadence is "
                        "window-matched for the three main arms; the "
                        "official arm is reported separately)")
    p.add_argument("--host", default="llama",
                   choices=["llama", "qwen3", "gemma4"],
                   help="backbone host: llama = vendored LlamaModelCCM "
                        "(R8-R10 line); qwen3 = Qwen3-4B CCM port "
                        "(feature_QWEN)")
    p.add_argument("--frozen", default="",
                   help="override the frozen spec path (llama v2 "
                        "comp-trainable experiment)")
    p.add_argument("--micro-batch", type=int, default=1,
                   help="dialogues per collated batch (Qwen3-line "
                        "throughput option, 2026-09-17).  The per-dialogue "
                        "equal-weight gradient normalization is PRESERVED "
                        "bit-for-bit (row-wise mean CE summed over the "
                        "window's dialogue count), so batch>1 changes only "
                        "numerical reorder inside the kernels, not the "
                        "objective.  RPBE arms currently require 1 "
                        "(fail-fast).  The STRICT official merge training "
                        "lives in scripts/train_ccm_merge.py (user ruling "
                        "2026-09-17), not here.")
    p.add_argument("--model-name-or-path", required=True)
    p.add_argument("--dialog-mirror", required=True,
                   help="DIALOG_MIRROR: ijcnlp_dailydialog layout dir")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--schedule-total-steps", type=int, default=None,
                   help="LR schedule length (warmup+cosine); defaults to max_steps. The e50 pilot keeps 1000 to preserve the original schedule while stopping at 500.")
    p.add_argument("--grad-accum", type=int, default=128,
                   help="microbatch per update in --merge-cadence official "
                        "(official-reproduction arm only)")
    p.add_argument("--merge-cadence", default="window-matched",
                   choices=["window-matched", "official"],
                   help="window-matched: the ccm_merge arm fires an update "
                        "on the same adaptive boundary as the RPBE arms "
                        "(>= min-effective-cuts dialogues with k>=3), so "
                        "task exposure and scheduler cadence are identical "
                        "across the three arms (frozen_method.json "
                        "cadence; review P0-2).  official: fixed "
                        "grad-accum microbatch cadence for the "
                        "ccm_merge_official reproduction reference.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="max_grad_norm (official Trainer default 1.0; "
                        "frozen_method.json training.grad_clip is "
                        "authoritative and an explicit override is "
                        "refused)")
    p.add_argument("--relative-embedding", default="skip",
                   choices=["skip", "base"])
    # RPBE
    p.add_argument("--kf-lambda", type=float, default=1e-3)
    p.add_argument("--kf-min-cuts", type=int, default=128,
                   help="effective-cuts gate (window closes at >= this)")
    p.add_argument("--ridge-eps", type=float, default=1e-3)
    p.add_argument("--sketch-dim", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--rpbe-seed", type=int, default=0)
    p.add_argument("--gamma-hidden", type=int, default=64)
    p.add_argument("--official-host", action="store_true",
                   help="review round 7: build the REAL official host — "
                        "SeparatedEmbedding + trainable COMP/SUM embeddings "
                        "+ the released Step-2 compression adapter on top of "
                        "the Step-1 foundation merge (no resize+freeze).")
    p.add_argument("--official-adapter", default="",
                   help="path to the released Step-2 compression adapter "
                        "(llama-7b-no-online-merge_recur-ntok2)")
    p.add_argument("--foundation", default="",
                   help="path to an official Step-1 default-LoRA adapter "
                        "dir (e.g. llama-7b-no); its weights are MERGED "
                        "into the base model before training — the "
                        "official two-stage protocol.  Empty = train "
                        "from theta_0 on the raw LLaMA (single-stage).")
    p.add_argument("--calibrate-lambda", action="store_true",
                   help="measure r_eff on the first closed window and exit")
    # Tree-wise feasibility projection (TGN final-spec alignment,
    # 2026-09-15: row-normalized half spaces + 1e-6 full certificate +
    # CERT_FAIL => skip the representation step; ported from
    # tgb_link_loop._cstr_group_close_treewise, the b523cf3 lineage)
    p.add_argument("--rpbe-constrain-mode", default="aggregate",
                   choices=["aggregate", "treewise"],
                   help="aggregate = classic summed RPBE surrogate "
                        "(additive update, no projection); "
                        "treewise = per-tree half-space feasibility "
                        "projection on the Gamma scope (final spec)")
    p.add_argument("--rpbe-kappa", type=float, default=0.05,
                   help="treewise guardrail: g_i.d >= -kappa||g_i||||t|| "
                        "(LARGER is looser; kappa>=1 never binds)")
    p.add_argument("--proj-iters", type=int, default=400,
                   help="FISTA iterations for the treewise dual QP")
    p.add_argument("--rpbe-lr", type=float, default=None,
                   help="independent AdamW lr for the Gamma params "
                        "(None = unified args.lr for all params; "
                        "the scheduler scales both groups by the same "
                        "warmup/cosine factor, so the ratio rpbe_lr/lr "
                        "is preserved through training)")
    p.add_argument("--rpbe-gamma-only", action="store_true",
                   help="R10 structure (review 2026-09-16): the RPBE "
                        "surrogate updates GAMMA ONLY — task CE keeps "
                        "training LoRA/COMP embeddings/Gamma, but the "
                        "auxiliary gradient on non-Gamma params is "
                        "discarded.  The global lambda calibration no "
                        "longer dilutes r_eff across LoRA/COMP.")
    p.add_argument("--rpbe-native-compression", action="store_true",
                   help="COMP-RPBE native actuation (review 2026-09-22): "
                        "the RPBE recursion node is the native CCM merge "
                        "step (M_{t-1}, u_t) -> M_t with depth D=L and "
                        "Z_t = M_t; the actuated parameters are the "
                        "native compression params (conditional LoRA + "
                        "COMP/SUM comp-embedding rows) instead of a "
                        "post-merge Gamma residual.  Skips attach_gamma, "
                        "freezes the backbone, and runs the "
                        "proposal-space treewise projection (ours).  "
                        "Requires the native actuation spec "
                        "(frozen_method_qwen3_native.json); mutually "
                        "exclusive with --freeze-host / "
                        "--rpbe-gamma-only / --rpbe-lr.")
    p.add_argument("--supervisor-mode", default="current",
                   choices=["current", "root_only"],
                   help="predictive supervisor variant (sweep, review "
                        "2026-09-23): current = 2Obs (local u_{t+1}->"
                        "u_{t+2} + short-suffix->y); root_only = every "
                        "cut supervises the final target y with the "
                        "remaining suffix (u_{t+1}..u_L, c) as context "
                        "(single row per cut, w=1.0)")
    p.add_argument("--s4-supervisor", action="store_true",
                   help="FORMAL S4 supervisor (review 2026-09-23, "
                        "frozen candidate): keep the original 2Obs "
                        "topology, replace the CountSketch/Ky-Fan "
                        "scoring of every observation with the "
                        "real-token NLL gap "
                        "Delta_{t,h} = l_comp - sg(l_full), where "
                        "l_comp = -log p(Y | M_t, C) and l_full = "
                        "-log p_ref(Y | U_t, C) share the SAME future "
                        "conditioning C (only the compression of the "
                        "history differs; the full reference is a "
                        "1-turn merge = lossless).  RAW gap — no hinge, "
                        "no epsilon (margin gating is a later ablation)."
                        "  Native actuation only.")
    p.add_argument("--s4-deficit-gate", action="store_true",
                   help="DB-DG-S4 (review 2026-09-24): only "
                        "observations with Delta_{t,h} > 0 (compression "
                        "actually loses vs the full reference) feed the "
                        "ACTIVE predictive center; the QP rows stay "
                        "UNGATED (learned interfaces keep protection).")
    p.add_argument("--s4-depth-mults", default="",
                   help="DB-DG-S4 depth reweight (review 2026-09-24): "
                        "comma-separated multipliers for t<=2, t=3..5, "
                        "t>=6 applied to the ACTIVE predictive center "
                        "only (never the QP rows), window-mean "
                        "normalized over ALL observations, e.g. "
                        "'0.5,1.0,1.5'.  Empty = no depth weighting.")
    p.add_argument("--aliased-measurement", action="store_true",
                   help="probe switch (review 2026-09-24): sketch EVERY "
                        "logical layer in J_mem (the old aliased "
                        "behavior) instead of the unique physical KV "
                        "providers — used only for the paired probe "
                        "comparing q-norm/task-cosine/Pr between the "
                        "two measurements.")
    p.add_argument("--history-gamma", action="store_true",
                   help="Stage-B history-branch correction (review "
                        "2026-09-24): keep the native compressor "
                        "FROZEN (LoRA + COMP rows), attach a zero-init "
                        "Gamma whose residual enters WITH the history "
                        "weight (t-1)/t — turn-3 structurally "
                        "identical to the Stage-A checkpoint; S4/QP/"
                        "cuts unchanged.  Requires "
                        "--rpbe-native-compression.")
    p.add_argument("--no-qp", action="store_true",
                   help="No-QP arm (review 2026-09-24 deep-floor "
                        "probe): skip the treewise feasibility "
                        "projection entirely and write d* = d_0 back "
                        "(same task + lambda S4 center, no feasibility "
                        "correction).  Pairs with --s4-supervisor; the "
                        "dirs are still collected for diagnostics.")
    p.add_argument("--early-cut-drop-ratio", type=float, default=0.0,
                   help="cut-depth balancing (review 2026-09-24): "
                        "deterministically drop this fraction of t<=2 "
                        "rows from the TREEWISE QP constraint set "
                        "(every 1/ratio-th early row skipped; the "
                        "aggregate S4 center d_0 is UNCHANGED).  "
                        "Endpoint-level sampling does not imply "
                        "interface-level balance in recursive "
                        "computation (audit: t1-2 rows carry 51.5% of "
                        "the gradient mass while t6-13 carry 20.7%).")
    p.add_argument("--gradient-audit", action="store_true",
                   help="gradient-mass audit (review 2026-09-24): "
                        "record every S4 observation's (endpoint depth "
                        "L, horizon h in {1=local, 2=root}, q_{L,h}) "
                        "during the first closed window, then dump the "
                        "(L,h) decomposition of N / sum||q|| / mass "
                        "share / q^T d_task into gradient_audit.json "
                        "and exit (no optimizer step).")
    p.add_argument("--s4-ref-cache", action="store_true",
                   help="S4 speed fix (review 2026-09-23): batch-"
                        "precompute the full-reference NLL table for "
                        "the WHOLE window before the S4 loop (no_grad, "
                        "chunked multi-row forwards) and look it up "
                        "per observation.  The full reference is a "
                        "stop-grad scalar, so q_{t,h} = grad l_comp is "
                        "UNCHANGED — this is a moving-reference "
                        "batching, not an approximation (the l_full "
                        "values are live).")
    p.add_argument("--s4-ref-chunk", type=int, default=32,
                   help="chunk rows per batched reference forward "
                        "(--s4-ref-cache)")
    p.add_argument("--no-warmup", action="store_true",
                   help="explicit warmup skip (fork/resume semantics, "
                        "review 2026-09-23): the first window steps at "
                        "the peak lr and the cosine descends from "
                        "there.  Needed on RESUME of a fork arm — "
                        "--resume-from does not set --init-from, so "
                        "without this flag the scheduler rebuild would "
                        "silently reintroduce the 30-step warmup.")
    # monitoring
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--resume-from", default="",
                   help="resume from an adapter-only checkpoint.pt")
    p.add_argument("--init-from", default="",
                   help="two-stage flow (Qwen3 line): initialize the "
                        "trainable weights (LoRA + COMP rows) from a "
                        "merge-stage checkpoint and train from step 0 "
                        "(no optimizer/data-stream state).  Gamma stays "
                        "zero-init by design; non-Gamma missing keys "
                        "still fail fast.")
    p.add_argument("--freeze-host", action="store_true",
                   help="final training protocol (review ruling): after "
                        "--init-from, freeze the ENTIRE host "
                        "(conditional LoRA + COMP/SUM + backbone) and "
                        "train Gamma ONLY.  The two arms (task-only vs "
                        "ours) then differ solely in the Gamma training "
                        "objective — the clean causal attribution the "
                        "paper needs.")
    p.add_argument("--max-pending-mbs", type=int, default=2048,
                   help="degenerate-window guard: pending cap before abort")
    p.add_argument("--max-windows", type=int, default=0,
                   help="verification runs: stop after this many closed "
                        "windows (0 = run to max_steps).  max_steps stays "
                        "frozen at 1000, so short verification runs use "
                        "this instead of changing the step budget.")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


FROZEN_PATH = Path(__file__).resolve().parents[1] / "configs" / "ccm" \
    / "frozen_method.json"


def _frozen_path(args):
    """Per-host frozen spec.  --frozen overrides everything (used by the
    llama v2 comp-trainable experiment).  The native-compression flag
    selects the native actuation spec automatically (review 2026-09-22)
    unless an explicit --frozen is given."""
    if getattr(args, "frozen", ""):
        return Path(args.frozen)
    if getattr(args, "rpbe_native_compression", False):
        if getattr(args, "history_gamma", False):
            _name = "frozen_method_gemma4_stageB.json"
        else:
            _name = ("frozen_method_qwen3_native.json"
                     if getattr(args, "host", "llama") == "qwen3"
                     else "frozen_method_gemma4_native.json")
        return Path(__file__).resolve().parents[1] / "configs" / "ccm" \
            / _name
    name = ("frozen_method_qwen3.json"
            if getattr(args, "host", "llama") == "qwen3"
            else "frozen_method_gemma4_joint.json"
            if getattr(args, "host", "llama") == "gemma4"
            else "frozen_method.json")
    return Path(__file__).resolve().parents[1] / "configs" / "ccm" / name


def enforce_frozen(args):
    """The frozen method spec is authoritative (review P0-3 tail).

    The trainer entry point MUST read configs/ccm/frozen_method.json and
    refuse an inconsistent CLI override — a silent drift (e.g. a leftover
    grad_clip=5.0 default while the reported protocol is the official
    1.0) would train a different method than the one reviewed and
    frozen.  Every bound key is checked here; values the CLI does not
    carry are read back for logging only.

    Review ruling #2 extends the binds to the full method vector
    (lambda_kf, z_dim, gamma_hidden, merge_cadence, max_steps, LoRA rank)
    and pins the arm/cadence pairing: the ccm_merge_official reproduction
    arm is the ONLY arm allowed the fixed official cadence, the three main
    arms must use the machine value "window-matched", and lambda_kf is
    either null (calibration-only run permitted, anything else refused)
    or a committed number that overrides the CLI.
    """
    with open(_frozen_path(args), encoding="utf-8") as f:
        fz = json.load(f)
    binds = [
        ("--ridge-eps", "ridge_eps", fz["rpbe"]["ridge_eps"]),
        ("--sketch-dim", "sketch_dim", fz["rpbe"]["sketch_dim_m"]),
        ("--rpbe-seed", "rpbe_seed", fz["rpbe"]["rpbe_map_seed"]),
        ("--kf-min-cuts", "kf_min_cuts",
         fz["window"]["min_effective_cuts"]),
        ("--grad-clip", "grad_clip", fz["training"]["grad_clip"]),
        ("--z-dim", "z_dim", fz["rpbe"]["z_v_dim"]),
        ("--gamma-hidden", "gamma_hidden", fz["gamma"]["hidden"]),
        ("--lora-r", "lora_r", fz["adapter"]["lora_r"]),
        ("--max-steps", "max_steps", fz["training"]["steps"]),
        ("--schedule-total-steps", "schedule_total_steps",
         fz["training"].get("schedule_total_steps", fz["training"]["steps"])),
        ("--checkpoint-every", "checkpoint_every",
         fz["training"]["checkpoint_every"]),
    ]
    for flag, name, frozen_val in binds:
        cli_val = getattr(args, name)
        if cli_val != frozen_val:
            raise SystemExit(
                "[frozen] {}={} conflicts with frozen_method.json (={}); "
                "the frozen spec is authoritative — edit the spec file "
                "to override".format(flag, cli_val, frozen_val))
    # Arm/cadence pairing: only the official reproduction arm may run
    # the fixed accumulation cadence; the three main arms share the
    # adaptive "window-matched" boundary (review P0-2 / ruling #2).
    if args.arm == "ccm_merge_official":
        if args.merge_cadence != "official":
            raise SystemExit(
                "[frozen] ccm_merge_official must run with "
                "--merge-cadence official (frozen_method.json "
                "training.merge_cadence = window-matched applies to the "
                "three main arms only)")
    elif args.merge_cadence != "window-matched":
        raise SystemExit(
            "[frozen] arm {} must run with the frozen window-matched "
            "cadence; --merge-cadence official is reserved for "
            "ccm_merge_official".format(args.arm))
    # Review ruling (2026-09-21): the freeze-host protocol is MANDATORY
    # for the main RPBE arms — a forgotten --freeze-host silently reverts
    # to LoRA+COMP+Gamma joint training (the exact confound the
    # freeze-host design removes).  Same for --official-host where the
    # spec pins the official build (legacy resize host retired).
    for _key, _flag in (("freeze_host", "freeze_host"),
                        ("official_host", "official_host")):
        _section = fz.get(_key)
        if not isinstance(_section, dict) or not _section.get("enabled"):
            continue
        if args.arm in ("ours", "gamma_task_only") \
                and not getattr(args, _flag, False):
            raise SystemExit(
                "[frozen] {}.enabled=true in the frozen spec: the {} "
                "arm MUST pass --{}".format(_key, args.arm,
                                            _flag.replace("_", "-")))
    # Actuation guard (review 2026-09-22, COMP-RPBE native design): the
    # spec's actuation.mode pins WHICH parameter block the RPBE
    # recursion actuates.  mode=native (conditional LoRA + COMP/SUM rows,
    # proposal-space treewise) and the gamma flag must agree with the
    # spec in BOTH directions — a native spec without the flag would
    # silently train the old post-merge Gamma, and the flag with a gamma
    # spec would silently attach no Gamma while the spec still binds
    # gamma.hidden.
    act_mode = fz.get("actuation", {}).get("mode", "gamma")
    native_flag = bool(getattr(args, "rpbe_native_compression", False))
    if act_mode == "native":
        _why = []
        if args.arm not in ("ours", "gamma_task_only"):
            _why.append("arm {} is not an RPBE arm".format(args.arm))
        if not native_flag:
            _why.append("missing --rpbe-native-compression")
        if getattr(args, "freeze_host", False):
            _why.append("--freeze-host is the Gamma-only protocol")
        if getattr(args, "rpbe_gamma_only", False):
            _why.append("--rpbe-gamma-only has no scope under native "
                        "actuation")
        if getattr(args, "rpbe_lr", None) is not None:
            _why.append("--rpbe-lr would silently route ALL native "
                        "params into the rpbe_lr group")
        if getattr(args, "host", "llama") not in ("qwen3", "gemma4"):
            _why.append("native actuation v1 is Qwen3/Gemma4-only")
        if args.arm == "ours" \
                and getattr(args, "rpbe_constrain_mode", "") != "treewise" \
                and not getattr(args, "s4_supervisor", False):
            _why.append("native ours requires --rpbe-constrain-mode "
                        "treewise (the aggregate branch would add the "
                        "raw RPBE gradient onto the LoRA directly); "
                        "S4 aggregate is allowed as the two-phase "
                        "post-conflict form (review 2026-09-23)")
        if _why:
            raise SystemExit(
                "[frozen] actuation.mode=native in the frozen spec: "
                + "; ".join(_why))
    elif native_flag:
        raise SystemExit(
            "[frozen] --rpbe-native-compression given but the spec's "
            "actuation.mode is '{}' — native flag + gamma spec (or a "
            "stale spec) would silently train the wrong parameter "
            "block".format(act_mode))
    if getattr(args, "s4_supervisor", False) and (
            args.arm != "ours"
            or not getattr(args, "rpbe_native_compression", False)):
        raise SystemExit(
            "[s4] --s4-supervisor is the native-ours formal supervisor "
            "(--arm ours --rpbe-native-compression required)")
    # Lambda authority: the calibration-only run writes the derived
    # lambda into the frozen spec; afterwards the number overrides any
    # CLI value and re-calibration is refused.
    # Calibration measures r_eff on theta_0 — resuming first would run
    # several optimizer steps, so the calibration branch would be skipped
    # and the run would silently continue as normal training (audit #3).
    if args.calibrate_lambda and args.resume_from:
        raise SystemExit(
            "[frozen] lambda calibration must start from fresh theta_0; "
            "--calibrate-lambda and --resume-from are mutually exclusive")
    lam = fz["rpbe"]["lambda_calibration"].get("lambda_kf")
    if args.arm == "ours":
        if lam is None:
            if not args.calibrate_lambda:
                raise SystemExit(
                    "[frozen] rpbe.lambda_calibration.lambda_kf is null — "
                    "the ours arm may only run the calibration-only pass "
                    "(--calibrate-lambda).  Commit its derived_lambda "
                    "into configs/ccm/frozen_method.json before the "
                    "formal runs.")
        else:
            if args.calibrate_lambda:
                raise SystemExit(
                    "[frozen] lambda_kf={} is committed in "
                    "frozen_method.json; re-running --calibrate-lambda "
                    "would retrain the calibration.  Delete the field "
                    "first if a recalibration is really intended.".format(
                        lam))
            if args.kf_lambda != lam:
                print("[frozen] CLI --kf-lambda={} overridden by the "
                      "committed lambda_kf={} "
                      "(frozen_method.json)".format(args.kf_lambda, lam),
                      flush=True)
                args.kf_lambda = lam
    elif args.calibrate_lambda:
        raise SystemExit(
            "[frozen] --calibrate-lambda measures r_eff of the surrogate "
            "(arm 'ours'); arm '{}' does not train the auxiliary "
            "term".format(args.arm))
    return fz


def build_tokenizer(args):
    if args.host == "qwen3":
        from transformers import AutoTokenizer
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # Qwen3 has no pad token: use <|endoftext|> (in the vocab) so the
        # pad id never collides with EOS (<|im_end|>), keeping the EOS
        # exclusion semantics clean.
        if tok.pad_token_id is None:
            tok.pad_token = "<|endoftext|>"
        tok.padding_side = "left"
        # CRITICAL alignment (2026-09-17): the Qwen3 tokenizer vocab is
        # 151669 while the model config.vocab_size is 151936 (reserved
        # rows).  Comp tokens must get ids >= 151936 so the
        # SeparatedEmbedding routes them to the TRAINABLE comp rows;
        # otherwise they fall on frozen pretrained main-table rows and
        # comp_embeddings never receives a gradient (verified: grad 0).
        cfg_vocab = Qwen3Config.from_pretrained(
            args.model_name_or_path).vocab_size
        if len(tok) < cfg_vocab:
            tok.add_tokens(
                ["<|extra_{}|>".format(i)
                 for i in range(cfg_vocab - len(tok))])
        added = [f"<COMP{k}>" for k in range(N_TOK)] \
            + [f"<SUM{k}>" for k in range(N_TOK)]
        tok.add_special_tokens({"additional_special_tokens": added})
        ids = [tok.convert_tokens_to_ids(f"<COMP{k}>") for k in range(N_TOK)]  + [tok.convert_tokens_to_ids(f"<SUM{k}>") for k in range(N_TOK)]
        assert ids[0] >= cfg_vocab, \
            "comp ids must exceed config.vocab_size for SeparatedEmbedding"
        tok.comp_token_id = ids[:N_TOK]
        tok.sum_token_id = ids[N_TOK:]
        tok._qwen3_host = True  # eval collator dispatch marker
        return tok
    if args.host == "gemma4":
        from transformers import AutoTokenizer
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig)
        tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # Gemma4 ships without a pad token; config default pad id is 0
        # (a real reserved row, distinct from eos=1/bos=2), which keeps
        # the EOS exclusion semantics clean (qwen3 lesson).
        if tok.pad_token_id is None:
            tok.pad_token_id = 0
            tok.pad_token = "<pad>"
        tok.padding_side = "left"
        # CRITICAL alignment (qwen3 lesson, resize route): comp ids must
        # land >= config.vocab_size so resize_token_embeddings grows the
        # main table (trainable comp rows) and the PLE table (zero rows)
        # past the frozen pretrained rows.
        cfg_vocab = Gemma4TextConfig.from_pretrained(
            args.model_name_or_path).vocab_size
        if len(tok) < cfg_vocab:
            tok.add_tokens(
                ["<|extra_{}|>".format(i)
                 for i in range(cfg_vocab - len(tok))])
        added = [f"<COMP{k}>" for k in range(N_TOK)] \
            + [f"<SUM{k}>" for k in range(N_TOK)]
        tok.add_special_tokens({"additional_special_tokens": added})
        ids = [tok.convert_tokens_to_ids(f"<COMP{k}>") for k in range(N_TOK)]  + [tok.convert_tokens_to_ids(f"<SUM{k}>") for k in range(N_TOK)]
        assert ids[0] >= cfg_vocab, \
            "comp ids must exceed config.vocab_size (resize route)"
        tok.comp_token_id = ids[:N_TOK]
        tok.sum_token_id = ids[N_TOK:]
        tok._gemma4_host = True  # eval collator dispatch marker
        return tok
    from transformers import LlamaTokenizer
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    tok.bos_token_id = tok.bos_token_id or 1
    tok.eos_token_id = tok.eos_token_id or 2
    tok.padding_side = "left"
    added = [f"<COMP{k}>" for k in range(N_TOK)] \
        + [f"<SUM{k}>" for k in range(N_TOK)]
    tok.add_special_tokens({"additional_special_tokens": added})
    ids = [tok.convert_tokens_to_ids(f"<COMP{k}>") for k in range(N_TOK)]  + [tok.convert_tokens_to_ids(f"<SUM{k}>") for k in range(N_TOK)]
    tok.comp_token_id = ids[:N_TOK]
    tok.sum_token_id = ids[N_TOK:]
    return tok


def build_model(args, device):
    if args.host == "qwen3":
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from src.arch.ccm_qwen3 import Qwen3ForCausalLM_CCM
        from src.utils import SeparatedEmbedding
        if args.official_host or args.foundation:
            raise SystemExit(
                "[qwen3] --official-host/--foundation are Llama-line "
                "artifacts; the Qwen3 host trains single-stage from "
                "theta_0 (frozen backbone + conditional LoRA)")
        config = Qwen3Config.from_pretrained(args.model_name_or_path)
        config.comp_relative_embedding = args.relative_embedding
        # Two-stage official layout (train_ccm_merge semantics): the
        # COMP/SUM rows live in a SeparatedEmbedding; the lm_head keeps
        # the base vocab (no resize).
        model = Qwen3ForCausalLM_CCM.from_pretrained(
            args.model_name_or_path, config=config,
            torch_dtype=torch.bfloat16 if device.type == "cuda"
            else torch.float32)
        model.model.embed_tokens = SeparatedEmbedding(
            model.model.embed_tokens, 2 * N_TOK)
        model.update_comp_token(
            [config.vocab_size + k for k in range(N_TOK)],
            [config.vocab_size + N_TOK + k for k in range(N_TOK)])
        return model.to(device)
    if args.host == "gemma4":
        import torch as _t
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig)
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration)
        from src.arch.ccm_gemma4 import Gemma4ForCausalLM_CCM
        if args.official_host:
            raise SystemExit(
                "[gemma4] --official-host is a Llama-line artifact; "
                "the gemma4 host builds from the released "
                "Gemma-4-E4B checkpoint (+ optional --foundation merge)")
        text_cfg = Gemma4TextConfig.from_pretrained(
            args.model_name_or_path)
        text_cfg.comp_relative_embedding = args.relative_embedding
        model = Gemma4ForCausalLM_CCM(text_cfg)
        # Load the released multimodal checkpoint and keep the text
        # decoder only: model.language_model.* -> model.*, lm_head.
        dtype = (torch.bfloat16 if device.type == "cuda"
                 else torch.float32)
        full = Gemma4ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, torch_dtype=dtype)
        prefix = "model.language_model."
        text_sd = {}
        for k, v in full.state_dict().items():
            if k.startswith(prefix):
                text_sd["model." + k[len(prefix):]] = v
            elif k == "lm_head.weight":
                text_sd[k] = v
        del full
        _t.cuda.empty_cache() if device.type == "cuda" else None
        missing, unexpected = model.load_state_dict(text_sd, strict=False)
        print("[gemma4] load text-only: {} missing / {} unexpected".format(
            len(missing), len(unexpected)), flush=True)
        if unexpected:
            raise RuntimeError(
                "unexpected keys in text-only load: {}".format(
                    unexpected[:8]))
        # .to(bf16) matches the native from_pretrained dtype conversion:
        # persistent=False buffers (embed_scale etc.) are NOT in the
        # state_dict, so they stay fp32 otherwise and shift every embed
        # output by the bf16 rounding of the scale (G1 gate catch).
        model.to(device, torch.bfloat16)
        if args.foundation:
            # Stage-1 default-LoRA merge (official two-stage protocol):
            # the Step-1 artifact is an adapter-only checkpoint from
            # train_ccm_step1.py ("trainable_state_dict" of plain peft
            # LoRA keys).  Merge W' = W + B@A * (alpha/r) into the base
            # BEFORE attaching the conditional LoRA (alpha=16, r=8 from
            # the Step-1 Table-14 config).
            ck = torch.load(args.foundation, map_location=device,
                            weights_only=False)
            tsd = ck.get("trainable_state_dict", ck)
            n_merged = 0
            with torch.no_grad():
                for k in list(tsd):
                    if ".lora_A." not in k:
                        continue
                    b_key = k.replace(".lora_A.", ".lora_B.")
                    base_key = re.sub(
                        r"^base_model\.model\.model\.layers\.(\d+)\."
                        r"self_attn\.(\w+_proj)\.lora_A\.default\.weight$",
                        r"model.layers.\1.self_attn.\2.weight", k)
                    if base_key == k:
                        continue
                    A = tsd[k].float().to(device)
                    B = tsd[b_key].float().to(device)
                    delta = (B @ A) * (16.0 / 8.0)
                    with torch.no_grad():
                        tgt = dict(model.named_parameters())[base_key]
                        tgt.data += delta.to(tgt.dtype)
                    n_merged += 1
            print("[gemma4] foundation merged: {} LoRA modules from {}"
                  .format(n_merged, args.foundation), flush=True)
        # Merge-chain embedding layout (review fix 2026-09-23): the
        # gemma Stage-2 checkpoints carry the SeparatedEmbedding comp-
        # row keys (train_ccm_merge semantics) — the RPBE double-table
        # resize has no comp_embeddings key and strict-load of a merge
        # ckpt fails.  Align build_model with build_model_merge.
        from src.utils import SeparatedEmbedding
        model.model.embed_tokens = SeparatedEmbedding(
            model.model.embed_tokens, 2 * N_TOK)
        # The PLE table is looked up by the SAME input_ids — grow it to
        # match (zero rows, frozen), same as the merge chain.
        model.model.resize_ple_embeddings(
            text_cfg.vocab_size + 2 * N_TOK)
        model.update_comp_token(
            [text_cfg.vocab_size + k for k in range(N_TOK)],
            [text_cfg.vocab_size + N_TOK + k for k in range(N_TOK)])
        return model
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = args.relative_embedding
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.resize_token_embeddings(32000 + 2 * N_TOK)
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    return model.to(device)


def build_official_host(args, device):
    """Review round 7: the REAL official host.  The Step-1 default LoRA
    is merged into the base weights; the 4 COMP/SUM tokens live in a
    SeparatedEmbedding whose comp_embeddings are TRAINABLE (the official
    protocol trains them); the released Step-2 compression adapter is
    loaded through the official conditional-LoRA (peft_custom) wrapper.
    Everything is frozen except comp_embeddings + the adapter LoRA
    params."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from src.arch.ccm_llama import LlamaForCausalLM_CCM
    from src.model import load_lora_weight, peft_custom
    from src.utils import SeparatedEmbedding
    from peft import LoraConfig
    config = LlamaConfig.from_pretrained(args.model_name_or_path)
    config.comp_relative_embedding = args.relative_embedding
    # fp32 throughout (review round 7): the official separate-embed path
    # keeps comp_embeddings in fp32, and mixing fp16 weights with fp32
    # embed outputs breaks the peft conditional-LoRA forward.
    model = LlamaForCausalLM_CCM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=torch.float32)
    model = model.to(device)
    load_lora_weight(args.foundation, model, merge=True)
    model.update_comp_token([32000 + k for k in range(N_TOK)],
                            [32000 + N_TOK + k for k in range(N_TOK)])
    model.model.embed_tokens = SeparatedEmbedding(model.model.embed_tokens,
                                                  2 * N_TOK)
    # fp32 already (model is fp32)
    adapter_dir = args.official_adapter
    lora_cfg = LoraConfig().from_pretrained(adapter_dir)
    model = peft_custom.get_peft_model(model, lora_cfg)
    load_lora_weight(adapter_dir, model, merge=False)
    for _p in model.parameters():
        _p.requires_grad_(False)
    # peft wraps one level deeper: PeftModel -> base Llama -> LlamaModel
    model.base_model.model.model.embed_tokens.comp_embeddings.weight \
        .requires_grad_(True)
    for _n, _p in model.named_parameters():
        if "lora_" in _n:
            _p.requires_grad_(True)
    model._official_host = True
    print("[official-host] SeparatedEmbedding + trainable comp embeddings "
          "+ Step-2 adapter loaded", flush=True)
    return model


def wrap_lora(model, r):
    from rpbe.hosts.ccm.ccm_patch import wrap_lora as _wrap
    return _wrap(model, r=int(r))


def build_dataset(args, tokenizer):
    from src.arguments import CompressionArguments
    os.environ["DIALOG_MIRROR"] = args.dialog_mirror
    comp_args = CompressionArguments(attn_type="merge_recur",
                                     num_comp_tokens=N_TOK,
                                     add_comp_token=True,
                                     relative_embedding=args.relative_embedding)
    if args.host == "qwen3":
        from src.data.dialogue.qwen3_data import (
            Qwen3DialogueDataset, Qwen3DialogueCollator)
        dialog = Qwen3DialogueDataset(tokenizer, mirror=args.dialog_mirror)
        collator = Qwen3DialogueCollator(
            dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
            comp_token=tokenizer.comp_token_id,
            sum_token=tokenizer.sum_token_id,
            pad_token=tokenizer.pad_token_id,
            label_pad_token_id=-100)
        return dialog, collator
    if args.host == "gemma4":
        from src.data.dialogue.gemma4_data import (
            Gemma4DialogueDataset, Gemma4DialogueCollator)
        dialog = Gemma4DialogueDataset(tokenizer, mirror=args.dialog_mirror)
        collator = Gemma4DialogueCollator(
            dataset=dialog, tokenizer=tokenizer, comp_args=comp_args,
            comp_token=tokenizer.comp_token_id,
            sum_token=tokenizer.sum_token_id,
            pad_token=tokenizer.pad_token_id,
            label_pad_token_id=-100)
        return dialog, collator
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.data import DialogueDataset
    from src.data.dialogue.collator import DataCollatorForDialogue_LLAMA
    dialog = DialogueDataset(tokenizer, comp_token=tokenizer.comp_token_id,
                             online=True, add_comp_token=True,
                             clean_split=True)
    collator = DataCollatorForDialogue_LLAMA(
        dialog=dialog, tokenizer=tokenizer, comp_args=comp_args,
        comp_token=tokenizer.comp_token_id, sum_token=tokenizer.sum_token_id,
        padding="left", pad_token=tokenizer.pad_token_id,
        label_pad_token_id=-100)
    return dialog, collator


# ---------------------------------------------------------------------
# Review round 8 (turn-14 legal-cut scarcity): depth-stratified candidate
# pools.  Depth L = compressed-history turn count; a dialogue contributes
# to pool L iff its ORIGINAL length >= L + 2 (L history turns + 1
# immediate context + 1 target).  N_TURNS_OF_L[L] = L + 2 is the fixed
# prefix length measured in TURNS, so the cut is the memory after exactly
# L compressions and the target is always the (L+2)-th turn — strictly
# legal by construction (same dialogue, contiguous, complete utterances,
# no EOS/padding/truncation crossing, never the trailing suffix, and the
# memory state is real).
# Turn-14/L=13 requires the original dialogue to be long enough, and its
# target is FIXED at the 15th turn with a 13-turn compressed history.
#
# NAMING (audit fix 2026-09-21): this quantity is a TURN COUNT (L + 2).
# It is NOT the CCM "k" of parse_meta / data_flow.jsonl, which counts
# context turns as len(blocks) + 1 (= L + 1).  The two conventions differ
# by exactly 1, so calling both "k" is precisely the off-by-one depth
# misreading an audit is meant to catch (audit #1 fell into that class).
# Every turn-count use below is spelled n_turns_*; every block-count use
# stays meta["k"].  Do not reintroduce a bare "k" for either.
# ---------------------------------------------------------------------
DEPTH_LEVELS = (1, 2, 4, 8, 13)          # L = compressed-history turns
N_TURNS_OF_L = {L: L + 2 for L in DEPTH_LEVELS}  # TURN COUNT per depth
DEPTH_CAP = 3.0 / len(DEPTH_LEVELS)      # max oversampling vs uniform


def build_depth_pools(train_items):
    """L -> list of ORIGINAL dialogue indices eligible at that depth.

    One original dialogue appears in every pool its length permits; the
    per-window dedup (one dialogue per window) is enforced at sampling
    time, not here.
    """
    pools = {L: [] for L in DEPTH_LEVELS}
    for i, item in enumerate(train_items):
        n = len(item["dialog"])
        for L in DEPTH_LEVELS:
            if n >= N_TURNS_OF_L[L]:
                pools[L].append(i)
    return pools


def depth_sampling_probs(pools, alpha=None):
    """Depth sampling weights.

    Default (alpha=None): the R10 legacy rule q_L ~ 1/sqrt(n_L) with the
    3x-uniform oversampling cap (Llama-line frozen spec; unchanged).

    Qwen3-line (alpha from frozen_method_qwen3.json sampling.q_alpha,
    user ruling 2026-09-17): q_L ∝ n_L^alpha (alpha=1 = natural
    frequency).  Combined with the per-dialogue replay cap
    (sampling.max_replays), q_L ∝ n_L makes every depth pool exhaust its
    n_L x K sample budget at the SAME time — the depth mix stays stable
    across the whole budget and every dialogue gets ~K exposures (the
    official 12-epoch semantics).  The R10 inverse-sqrt upweighting is
    retired on the Qwen3 line: it gave the L=13 pool ~60% of the stream
    and replayed its 573 dialogues 52-120x each (turn_15 overfitting,
    diagnosed 2026-09-17)."""
    n = np.array([max(len(pools[L]), 1) for L in DEPTH_LEVELS],
                 dtype=np.float64)
    if alpha is None:
        w = 1.0 / np.sqrt(n)
        q = w / w.sum()
        for _ in range(20):
            over = q > DEPTH_CAP
            if not over.any():
                break
        excess = float((q[over] - DEPTH_CAP).sum())
        q[over] = DEPTH_CAP
        q[~over] += excess * q[~over] / q[~over].sum()
        return q

    # q_L ∝ n_L^alpha (Qwen3-line natural-frequency family)
    w = n ** float(alpha)
    return (w / w.sum()).astype(np.float64)


def parse_meta(batch, comp_ids, sum_ids, sample_id_global, orig_ids=None,
               raw_dialogs=None):
    """Deterministic per-sample metadata from the padded collator batch.

    Returns a list (one per batch row) of dicts: k, L, blocks (C0/S0
    positions per turn), utterance_spans, prompt_end, raw_dialog.  Every
    turn block is [C0, C1, S0, S1] right after its utterance; the final
    context turn carries no block.  ``sample_id_global`` is the
    stream-global sample index (unique across epochs) used as the tree
    identity.  ``raw_dialogs`` (review ruling 2026-09-21) carries the
    collator's own tokenized turns [u_1..u_L, c, y] per row — the ONLY
    source for chi/phi; spans/ids stay model-position metadata.
    """
    ids = batch["input_ids"]
    labels = batch["labels"]
    B, L = ids.shape
    metas = []
    for b in range(B):
        row = ids[b]
        c0 = (row == comp_ids[0]).nonzero(as_tuple=False).flatten().tolist()
        blocks = []
        ok = True
        for pos in c0:
            if pos + 3 >= L or row[pos + 1] != comp_ids[1] \
                    or row[pos + 2] not in sum_ids \
                    or row[pos + 3] not in sum_ids:
                ok = False
                break
            blocks.append((pos, pos + 2))  # (C0 pos, S0 pos)
        n_completion = int((labels[b] != -100).sum())
        prompt_end = L - n_completion
        k = len(blocks) + 1  # context turns = blocks + final blockless turn
        depth_L = len(blocks)  # compressed-history turns (review ruling:
                               # the canonical cut depth L = k - 1)
        utterance_spans = []
        prev_end = -1
        for (c0_pos, _s0) in blocks:
            utterance_spans.append((prev_end + 1, c0_pos))
            prev_end = c0_pos + 3  # S1 position
        utterance_spans.append((prev_end + 1, prompt_end))
        raw = None
        if raw_dialogs is not None and b < len(raw_dialogs):
            raw = [list(u) for u in raw_dialogs[b]]
            if ok:
                assert len(raw) == depth_L + 2, (
                    "raw_dialog turns {} != L + 2 = {}".format(
                        len(raw), depth_L + 2))
        metas.append({"sample_id": int(sample_id_global) + b, "row": b,
                      "k": k, "L": depth_L,
                      "blocks": blocks,
                      "utterance_spans": utterance_spans,
                      "prompt_end": prompt_end, "ok": ok,
                      "raw_dialog": raw,
                      # Review round 8: stable ORIGINAL dialogue id for
                      # the tree identity (fall back to the stream cursor
                      # when the caller does not supply it).
                      "orig_id": int(orig_ids[b]) if orig_ids is not None
                      else -1})
    return metas


def run_forward(model, batch, device, grad_enabled):
    # labels are NOT passed here: the task CE is computed once by
    # task_ce (passing them would compute the same loss a second time).
    # attention_mask_comp IS passed: the official merge_recur collator
    # derives it from the SUM tokens and the vendored LlamaModel folds
    # it into the attention mask (comp/sum positions blocked from the
    # causal stream), so omitting it silently trains a DIFFERENT
    # attention pattern than the official protocol (caught by the L6.5
    # gate-2 parity run).  fp16 autocast matches the official Trainer:
    # the vendored conditional-LoRA layer computes its lora branch in
    # fp32, so the forward MUST run under autocast (mixed fp16/fp32
    # without autocast raises on the fp16 base path).
    ctx = torch.enable_grad() if grad_enabled else torch.no_grad()
    amc = batch.get("attention_mask_comp")
    # LOCAL FIX (r7 official host): the official host is built fp32
    # throughout (build_official_host), so autocast is disabled there --
    # under autocast the merge_recur residual scatter (ccm_llama.py
    # index_add onto key_states) mixes fp32 key_states with fp16
    # res_all buffers and raises.  Autocast remains for the legacy fp16
    # resize host (its fp16 base path requires the mixed-mode forward).
    _official = bool(getattr(model, "_official_host", False))
    # Autocast dtype follows the backbone weight dtype (fp16 Llama /
    # bf16 Qwen3); fp32 hosts (official) skip autocast entirely.
    _dt = next(model.parameters()).dtype
    _acast_dt = _dt if _dt in (torch.float16, torch.bfloat16) \
        else torch.float16
    with ctx:
        with torch.autocast(device_type="cuda", dtype=_acast_dt,
                            enabled=(device.type == "cuda" and not _official)):
            return model(input_ids=batch["input_ids"].to(device),
                         attention_mask=batch["attention_mask"].to(device),
                         attention_mask_comp=amc.to(device)
                         if amc is not None else None)


def task_ce_shifted(out, labels, device):
    """Official CCM task CE: SHIFTED sum + valid count.

    The vendored model's internal loss (ccm_llama.py ~line 922) shifts
    logits/labels by one ("tokens < n predict n") before the CE; the
    official CompSeq2SeqTrainer backprops that per-microbatch loss
    directly.  This helper returns (shifted_sum, n_valid) so callers can
    build the per-microbatch MEAN (= sum / valid, the official
    reduction) and normalize by the window size where the L6.5 review
    requires it.  The token-normalized CE log line uses sum / total
    valid tokens.
    """
    logits = out.logits
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    n_valid = int((shift_labels != -100).sum())
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1), ignore_index=-100, reduction="sum")
    return loss, n_valid


def task_ce_rows(out, labels, device):
    """Per-dialogue-row shifted CE (Qwen3-line micro_batch>1).

    Returns (row_sum [B], row_n_valid [B]): the row-wise reduction keeps
    the equal-weight-per-dialogue objective intact when several
    dialogues share one collated batch (row_mean.sum()/N_dialogues ==
    the batch=1 objective bit-for-bit)."""
    logits = out.logits
    labs = labels.to(device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labs[..., 1:].contiguous()
    B = shift_logits.shape[0]
    row_n = (shift_labels != -100).sum(-1)  # [B]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), ignore_index=-100, reduction="none")
    row_sum = loss.view(B, -1).sum(-1)  # [B]
    return row_sum, row_n


def collect_replay_z(meta, adapter, device, v=None):
    """Pass-2 extraction ONLY: z_t at block_idx v = t - 1 (gradient-
    connected).  No builder, no chi, no p — those finished their job at
    the pass-1 window close (L6.5 review structural fix).  Review ruling
    (2026-09-21): block_idx = t - 1 is the ONLY cut indexing rule; v=None
    means the terminal cut t = L (block_idx L - 1)."""
    if v is None:
        v = int(meta["L"]) - 1
    s0_pos = meta["blocks"][v][1]
    sum_positions = torch.tensor([[s0_pos, s0_pos + 1]],
                                 dtype=torch.long, device=device)
    return adapter.extract_z(sum_positions)[0]


def collect_rows(meta, adapter, builder, utter_embed, phi_embed,
                 embed_tokens, batch, device,
                 supervisor_mode="current"):
    """Review ruling (2026-09-21): ONE CUT PER ACTUAL COMPRESSION LAYER.

      t = 1..L,  block_idx = t - 1   (the ONLY cut indexing rule)

    with raw_dialog = [u_1..u_L, c, y] (len L + 2, tokenized turns):

      t < L:  obs1  C = u_{t+1},  Y = u_{t+2}          (w = 0.5)
              obs2  C = (u_{t+1}, u_{t+2}, one-update), Y = y  (w = 0.5)
      t = L:  single legal observation  C = c,  Y = y  (w = 1.0)
              — the terminal cut does not fabricate a second future.

    supervisor_mode="root_only" (supervisor sweep, review 2026-09-23):
    EVERY t supervises the final root target y with the remaining
    suffix as the context — (M_t, C_t^{suffix}) -> y with
    C_t = (u_{t+1}..u_L, c), a single row per cut at w = 1.0.  No
    surrogate local target u_{t+2}.

    Chi/phi are built from the RAW utterance tokens (the collator's
    sample() already had them; the old code carved spans out of the
    collated input_ids, dragging BOS / chat headers / separators into
    the "utterance" sketches).  The model forward is used ONLY to lift
    z_t from block_idx t - 1.  Chi/phi stay frozen input-embedding
    sketches (no extra forward, no grad flow)."""
    if not meta["ok"] or meta["L"] < 1:
        return []
    L = int(meta["L"])
    raw = meta["raw_dialog"]
    assert len(raw) == L + 2, (
        "raw_dialog length {} != L + 2 = {}".format(len(raw), L + 2))
    # extract_z's gather expands sum_positions along the BATCH dim, so
    # a multi-row [n_cuts, 2] input would index batch 1..n_cuts against
    # a batch-1 cache.  Extract one cut at a time (each call is a cheap
    # per-layer gather, not a forward).
    zs = [adapter.extract_z(torch.tensor(
        [[meta["blocks"][t - 1][1], meta["blocks"][t - 1][1] + 1]],
        dtype=torch.long, device=device))[0]
        for t in range(1, L + 1)]  # [L, z_dim]
    dm = DialogueMeta(sample_id=int(meta["sample_id"]), k=int(meta["k"]),
                      sum_positions=[(p + 2, p + 3) for (p, _s)
                                     in meta["blocks"]],
                      utterance_spans=list(meta["utterance_spans"]),
                      orig_id=int(meta.get("orig_id", -1)),
                      L=L, raw_dialog=list(raw))

    def _tok(idx):
        return torch.tensor(list(raw[idx]), dtype=torch.long,
                            device=device).unsqueeze(0)

    rows = []
    for t in range(1, L + 1):
        if supervisor_mode == "root_only":
            # (M_t, suffix) -> y for EVERY t: the suffix is the
            # remaining future dialogue (u_{t+1}..u_L, c); the terminal
            # cut's suffix is just c (same as the current design).
            suffix_ids = torch.tensor(
                [tok for idx in range(t, L + 1) for tok in raw[idx]],
                dtype=torch.long, device=device).unsqueeze(0)
            y_ids = _tok(L + 1)
            chi1 = utter_embed(embed_tokens, suffix_ids, tag=0)
            phi1 = phi_embed(embed_tokens, y_ids, tag=0)
            chi2, phi2 = chi1, phi1  # unused (single row)
            rows.extend(builder.build(dm, zs[t - 1], chi1[0], chi2[0],
                                      phi1[0], phi2[0], v=t - 1,
                                      single=True))
            continue
        if t < L:
            u_nxt = _tok(t)          # u_{t+1}  (raw index t)
            u_nx2 = _tok(t + 1)      # u_{t+2}
            y_ids = _tok(L + 1)      # y (target)
            chi1 = utter_embed(embed_tokens, u_nxt, tag=0)
            phi1 = phi_embed(embed_tokens, u_nx2, tag=0)
            chi2 = utter_embed.combine(embed_tokens, u_nxt, u_nx2, tag=1)
            phi2 = phi_embed(embed_tokens, y_ids, tag=1)
        else:
            c_ids = _tok(L)          # c (context turn)
            y_ids = _tok(L + 1)      # y (target)
            chi1 = utter_embed(embed_tokens, c_ids, tag=0)
            phi1 = phi_embed(embed_tokens, y_ids, tag=0)
            chi2, phi2 = chi1, phi1  # unused on the single-row path
        rows.extend(builder.build(dm, zs[t - 1], chi1[0], chi2[0],
                                  phi1[0], phi2[0], v=t - 1))
    return rows


def s4_obs_list(meta, v):
    """(h, comp_dlg, full_dlg, w) per observation of cut v (S4 formal
    supervisor, review 2026-09-23).

    Both dialogues share the SAME future conditioning (C, Y); only the
    history's information source differs: comp compresses u_1..u_t into
    M_t (t-turn merge), full gives u_1..u_t as ONE lossless context
    turn (a 1-turn merge is the identity).  Weights follow the original
    2Obs topology (0.5/0.5 below the terminal, single w=1 at t=L).
    """
    _t = int(v) + 1
    _raw = meta["raw_dialog"]
    _L = int(meta["L"])
    _hist = [list(_raw[_x]) for _x in range(_t)]
    _hist_full = [tok for _x in range(_t) for tok in _raw[_x]]
    if _t < _L:
        _ctx2 = [tok for _x in range(_t, _t + 2) for tok in _raw[_x]]
        return [
            (1, _hist + [list(_raw[_t])] + [list(_raw[_t + 1])],
             [_hist_full] + [list(_raw[_t])] + [list(_raw[_t + 1])],
             0.5),
            (2, _hist + [_ctx2] + [list(_raw[_L + 1])],
             [_hist_full] + [_ctx2] + [list(_raw[_L + 1])],
             0.5),
        ]
    return [(1, _hist + [list(_raw[_L])] + [list(_raw[_L + 1])],
             [_hist_full] + [list(_raw[_L])] + [list(_raw[_L + 1])],
             1.0)]


def batch_surrogate(z_rows_by_oid, batch_terms, lam, device):
    """Numerically zero surrogate; gradient = exact window J (plan L5).

    K = 1 here (one optimizer step per closed window), so the auxiliary
    is simply -lambda * sum_v (<sg(g_v), z_v> - sg(<g_v, z_v>)).

    ``batch_terms`` is ``[(occurrence_id, g), ...]`` for ONE replay batch
    only — the caller resolves which of the window's gradients belong to
    this batch through occurrence ids (review P0-1: never slice the
    window plan by batch position; the plan covers only the
    cut-producing batches while the pending list mixes in k < 4
    microbatches)."""
    terms = []
    for oid, g in batch_terms:
        z = z_rows_by_oid.get(oid)
        if z is None:
            continue
        gd = g.detach()
        terms.append((gd * z).sum() - (gd * z.detach()).sum())
    if not terms:
        return torch.zeros((), device=device), 0
    return -lam * sum(terms), len(terms)


def _fista_nonneg(Q, c, iters):
    """min_{mu>=0} 1/2 mu^T Q mu - c^T mu (Q PSD) by accelerated projected
    GD.  Tiny (N x N, N~128-300) dual of the tree-wise feasibility QP.
    (TGN final-spec port, 2026-09-15: eigvalsh exact step 1/lambda_max —
    the TGN loop uses 30 power iterations to estimate the same bound; at
    CCM's N the exact value is both cheaper and at least as tight.)"""
    L = float(torch.linalg.eigvalsh(Q).max().clamp(min=1e-12))
    mu = torch.zeros_like(c)
    y = mu.clone()
    tk = 1.0
    for _ in range(iters):
        grad = Q @ y - c
        mu_new = torch.clamp(y - grad / L, min=0.0)
        tk_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * tk * tk))
        y = mu_new + ((tk - 1.0) / tk_new) * (mu_new - mu)
        mu, tk = mu_new, tk_new
    return mu


def _json_safe(obj):
    """NaN/Inf -> None for jsonl writes (NaN is not valid JSON and
    breaks strict parsers; None serializes as a legal null)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def adamw_proposal(optimizer, params, grads, step_count,
                   lr_override=None):
    """SHADOW AdamW proposal WITHOUT stepping (review 2026-09-22,
    proposal-space treewise projection).

    Feeds ``grads`` through the optimizer's CURRENT first/second
    moments (exp_avg / exp_avg_sq, read but NOT written) and returns the
    AdamW update vector

        d = -lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta)

    with m_hat/v_hat bias-corrected by ``step_count`` (the step the
    optimizer WOULD take).  beta1/beta2/eps/lr are read from the
    param_groups; theta is detached.  This is the proposal vector the
    QP projects in the native actuation line: because AdamW carries
    momentum, second moments and coordinate-wise preconditioning,

        g -> Proj(g) -> AdamW   !=   g -> AdamW(g) -> Proj(d)

    and the frozen method definition (paper logic) is the
    proposal-space projection — project the AdamW proposal, then write
    the projected vector back as the parameter update.

    lr_override (review 2026-09-22 warmup fix): lr is a COMMON scalar
    factor of the proposal — the QP direction space and the lambda
    calibration ratio are lr-invariant, so those callers pass
    lr_override=1.0 (direction space); the write-back multiplies by the
    real current lr (0 during warmup step 0 keeps the warmup
    semantics).  Callers that need the exact optimizer step (test 5)
    leave it None to read the real lr."""
    out = []
    for p, g in zip(params, grads):
        _pg = optimizer.param_groups[0]
        for _grp in optimizer.param_groups:
            if any(pp is p for pp in _grp["params"]):
                _pg = _grp
                break
        grp = optimizer.param_groups[0]
        betas = tuple(grp["betas"])
        eps = float(grp.get("eps", 1e-8))
        lr = (float(_pg["lr"]) if lr_override is None
              else float(lr_override))
        wd = float(_pg.get("weight_decay", 0.0))
        st = optimizer.state.get(p, {})
        m = st.get("exp_avg", torch.zeros_like(p))
        v = st.get("exp_avg_sq", torch.zeros_like(p))
        b1, b2 = betas
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        bc1 = 1.0 - b1 ** max(int(step_count), 1)
        bc2 = 1.0 - b2 ** max(int(step_count), 1)
        m_hat = m_new / bc1
        v_hat = v_new / bc2
        step_dir = m_hat / (v_hat.sqrt() + eps)
        if wd:
            step_dir = step_dir + wd * p.detach()
        out.append(-lr * step_dir.to(p.dtype))
    return out


def proposal_norm(optimizer, params, step_count, lr_override=None):
    """L2 norm of the shadow AdamW proposal for the params' CURRENT
    .grad (calibration: r_eff measured in proposal space on the native
    line; the gamma line keeps the plain gradient-norm calibration).
    lr_override=1.0 for the lambda calibration: lr is a common factor,
    so the r_eff RATIO is lr-invariant (warmup may set the real lr to
    0 at theta_0, which would make both norms trivially zero)."""
    grads = [p.grad.detach().float() if p.grad is not None
             else torch.zeros_like(p, dtype=torch.float32)
             for p in params]
    props = adamw_proposal(optimizer, params, grads, step_count,
                           lr_override=lr_override)
    return float(
        torch.cat([d.reshape(-1) for d in props]).double().norm())


def treewise_feasibility_projection(g_task_repr, repr_params, G, kappa,
                                    iters=400, cert_tol=1e-6, min_norm=1e-9,
                                    b_norm=None, device=None):
    """Tree-wise RPBE Feasibility Projection — TGN final-spec alignment
    (2026-09-15, ported from tgb_link_loop._cstr_group_close_treewise,
    the b523cf3 lineage; the Cimmino refinement there is kappa=0-only and
    is NOT needed on the kappa>0 path CCM runs).

    One global center direction t (whose MEANING is given by the
    caller's vector space); every per-oid RPBE direction g_j stays
    SEPARATE and contributes one half-space
    g_j.d >= -kappa*||g_j||*||t||.  Solve

        d* = argmin_d 1/2||d - t||^2   s.t.  H d >= b,  b_j = -kappa*||t||

    with the ROW-NORMALIZED H (rows g_j/||g_j||; final-spec
    normalization — the same constraint set, a far better-conditioned
    dual), through the N x N dual (mu >= 0): d* = t + H^T mu.  After
    solving, EVERY row is certified against the 1e-6 tolerance (the
    final-spec full certificate).  On success writes ONLY the
    representation params: p.grad = g_task - sum_j mu_j g_j.  On
    certificate FAILURE returns ok=False WITHOUT writing anything — the
    caller must SKIP the representation step (reviewer requirement: no
    constrained update is executed on an uncertified solution).

    VECTOR SPACE of t / G is the CALLER's choice (review 2026-09-22):
    the gamma line passes the GRADIENT-space joint proposal
    (t = -g_task_repr); the native line passes the AdamW
    PROPOSAL-space center (t = d_0 from adamw_proposal, with
    b_j = -kappa*||g_j||*||d_task||).  The QP math is identical — the
    frozen method definition is the proposal-space projection, because
    g -> Proj(g) -> AdamW != g -> AdamW(g) -> Proj(d).

    Method fact (review 2026-09-22): lambda does NOT affect the pure
    half-space feasibility region (q_i -> lambda q_i cancels on both
    sides of the inequality).  Lambda is meaningful only through the
    JOINT proposal center d_0 = d_joint(lambda), which is why the
    method is joint-proposal + feasibility (scheme B), not pure
    feasibility (scheme A).

    CCM scale note: the window keeps N ~ 1-3 hundred directions, far
    below the TGN 2000+ regime, so the ACTIVE SET is the full row set
    (the TGN 3-round one-row-at-a-time expansion is a solver-capacity
    device for large N and would only add rounds here).  The projection
    runs on SCALED (fp16 GradScaler) gradients — the QP is invariant to
    a uniform positive rescaling of t and G, so the math is unaffected.

    CPU-memory streaming (2026-09-23, S4 2Obs at 5.9M repr dims): the
    S4-2Obs window holds ~600 rows x 5.9M fp32 (~14GB); stacking the
    full G, H and K matrices peaked at ~80GB of CPU RAM and the system
    OOM-killer took the training process.  The row normalization is
    per-row, the Gram K = H H^T and the matvecs are exact block sums —
    every stage streams in 64-row blocks, so the peak is the G list
    itself (~14GB) + one block (~1.5GB).  The math is UNCHANGED
    (exact streaming Gram; review 2026-09-22 pre-authorized).  G may be
    a tensor (gamma line) or a list of rows (native line — no stack).
    """
    sizes = [p.numel() for p in repr_params]
    t = torch.cat([-x.flatten().float() for x in g_task_repr])
    nt = float(t.norm())
    if isinstance(G, torch.Tensor):
        _G_rows = list(G)
    else:
        _G_rows = list(G)
    n_rows = len(_G_rows) if _G_rows else 0
    diag = {"n_dirs": n_rows,
            "task_norm": nt, "cos_mean": None, "cos_med": None,
            "cos_p5": None, "cos_min": None, "frac_below": None,
            "frac_below_grid": None, "feasible_d0": None,
            "active": 0, "n_viol": None, "max_viol": None,
            "corr_ratio": 0.0, "d_norm_ratio": None,
            "fista_iter": 0, "note": None, "cert_fail": False,
            # CCM-specific legacy fields (kept for cross-line digests)
            "proj_n_valid": 0, "proj_n_active_init": 0,
            "proj_n_mu_pos": 0, "proj_cos_min": 0.0,
            "proj_min_slack_after": None,
            "proj_max_viol_before": 0.0}

    def _write_task_only():
        # Gamma .grad <- the aggregate task gradient alone (d = t), the
        # degenerate / feasible / probe close semantics.
        for gt, p in zip(g_task_repr, repr_params):
            p.grad = gt.to(p.device)

    if not _G_rows:
        diag["note"] = "no_dirs"
        return True, diag
    if nt <= 1e-12:
        diag["note"] = "zero_task"
        _write_task_only()
        return True, diag
    # b_norm (native proposal-space line, review 2026-09-22): the
    # half-space bound is -kappa*||q_j||*||d_task|| — the TASK proposal
    # norm, NOT the joint-center norm ||t|| = ||d_0||.  The gamma line
    # passes b_norm=None and keeps b_j = -kappa*||t|| (V11 semantics
    # unchanged).
    nb = float(b_norm) if b_norm is not None else nt
    diag["b_norm"] = nb
    k_eff = kappa * (nb / max(nt, 1e-30))
    _CH = 64
    # GPU offload (2026-09-23): the two-level block Gram is the per-
    # window CPU bottleneck (~4-6 min).  With device set, each block is
    # uploaded (64 x 5.9M x 4B ~ 1.5GB per transfer, ~10s total over
    # PCIe for all 100 pairs) and the matmuls run on the GPU in
    # milliseconds; the n x n results come back to the CPU (FISTA stays
    # CPU, 600x600 is small).  None keeps the pure-CPU path (tests,
    # gamma line).
    _dev = device
    _t_q = t.to(_dev) if _dev is not None else t

    def _blocks():
        for _c0 in range(0, n_rows, _CH):
            yield _G_rows[_c0:_c0 + _CH], _c0

    # ---- pass 1: norms + valid mask (streaming) ----------------------
    ng_parts = []
    for _blk, _ in _blocks():
        _Gc = torch.stack(_blk).float()
        ng_parts.append(_Gc.norm(dim=1))
        del _Gc
    ng = torch.cat(ng_parts)
    valid = ng > min_norm
    if not bool(valid.any()):
        diag["note"] = "no_valid_dirs"
        _write_task_only()
        return True, diag
    v_idx = torch.nonzero(valid).reshape(-1)
    n_valid = int(v_idx.numel())
    _pos_of = {int(_r): int(_k) for _k, _r in enumerate(v_idx.tolist())}

    # ---- cos statistics (streaming H @ t) ----------------------------
    cos_parts = []
    for _blk, _ in _blocks():
        _Gc = torch.stack(_blk).float()
        _ngc = _Gc.norm(dim=1).clamp(min=min_norm)
        _Hc = _Gc / _ngc[:, None]
        cos_parts.append((_Hc.to(_dev) if _dev is not None else _Hc)
                         @ _t_q)
        del _Gc, _Hc
    cos = torch.cat([_p.cpu() for _p in cos_parts]) / nt
    cos_valid = cos[valid]
    cos_np = cos_valid.detach().cpu().numpy()
    diag["proj_n_valid"] = n_valid
    diag["cos_mean"] = float(np.mean(cos_np))
    diag["cos_med"] = float(np.median(cos_np))
    diag["cos_p5"] = float(np.percentile(cos_np, 5.0))
    diag["cos_min"] = float(np.min(cos_np))
    diag["proj_cos_min"] = float(np.min(cos_np))
    diag["frac_below"] = float(np.mean(cos_np < -k_eff))
    diag["frac_below_grid"] = {
        ("%.2f" % kk): float(np.mean(cos_np < -kk))
        for kk in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)}
    diag["feasible_d0"] = bool(np.all(cos_np >= -k_eff))
    diag["proj_max_viol_before"] = float(
        np.clip(-cos_np - k_eff, 0.0, None).max())
    viol0 = np.flatnonzero(cos_np < -k_eff)
    diag["proj_n_active_init"] = int(len(viol0))
    if len(viol0) == 0:
        # d0 = t is already feasible: task-only close (final-spec path)
        diag["note"] = "feasible_d0"
        diag["max_viol"] = 0.0
        _write_task_only()
        return True, diag
    # ---- full-set QP: K = H H^T, c = b - H t (streaming exact sums) ---
    # K[i,j] = h_i . h_j pairs exist ACROSS blocks, so the Gram needs a
    # two-level block loop (each row pair computed exactly once); the
    # single-level block-diagonal sum silently dropped every cross-block
    # entry (caught by the multi-block equivalence test, 2026-09-23).
    _blk_list = list(range(0, n_rows, _CH))

    def _blk_H(_c0):
        _blk = _G_rows[_c0:_c0 + _CH]
        _Gc = torch.stack(_blk).float()
        _ngc = _Gc.norm(dim=1).clamp(min=min_norm)
        _m = _ngc > min_norm
        if not bool(_m.any()):
            return None, None
        _Hv = _Gc[_m] / _ngc[_m, None]
        if _dev is not None:
            _Hv = _Hv.to(_dev)
        _pos = [_pos_of[_c0 + _k] for _k in range(len(_m))
                if bool(_m[_k])]
        return _Hv, _pos

    K = torch.zeros(n_valid, n_valid)
    c_vec = -kappa * nb * torch.ones(n_valid)
    for _c0 in _blk_list:
        _H0, _p0 = _blk_H(_c0)
        if _H0 is None:
            continue
        c_vec[_p0] -= (_H0 @ _t_q).cpu()
        for _c1 in _blk_list:
            _H1, _p1 = _blk_H(_c1)
            if _H1 is None:
                continue
            _k_sub = (_H0 @ _H1.t()).cpu()
            K[torch.tensor(_p0).unsqueeze(1),
              torch.tensor(_p1).unsqueeze(0)] += _k_sub
        del _H0
    mu = _fista_nonneg(K, c_vec, iters)
    # ---- corr = H^T mu (streaming) -----------------------------------
    corr = torch.zeros(t.numel())
    for _blk, _c0 in _blocks():
        _Gc = torch.stack(_blk).float()
        _ngc = _Gc.norm(dim=1).clamp(min=min_norm)
        _m = _ngc > min_norm
        if not bool(_m.any()):
            del _Gc
            continue
        _Hv = _Gc[_m] / _ngc[_m, None]
        _pos = [_pos_of[_c0 + _k] for _k in range(len(_m))
                if bool(_m[_k])]
        if _dev is not None:
            corr += (_Hv.to(_dev).t()
                     @ mu[_pos].to(_dev)).cpu()
        else:
            corr += _Hv.t() @ mu[_pos]
        del _Gc, _Hv
    d = t + corr
    # ---- full certificate (streaming H @ d) --------------------------
    viol_parts = []
    for _blk, _ in _blocks():
        _Gc = torch.stack(_blk).float()
        _ngc = _Gc.norm(dim=1).clamp(min=min_norm)
        _Hc = _Gc / _ngc[:, None]
        if _dev is not None:
            viol_parts.append((_Hc.to(_dev) @ d.to(_dev)).cpu())
        else:
            viol_parts.append(_Hc @ d)
        del _Gc, _Hc
    viol = ((-kappa * nb - torch.cat(viol_parts)[valid])
            / (nt + 1e-30))
    max_viol = float(viol.max()) if viol.numel() else 0.0
    diag["active"] = int((mu > 1e-8).sum())
    diag["proj_n_mu_pos"] = int((mu > 1e-8).sum())
    diag["fista_iter"] = iters
    diag["n_viol"] = int(max_viol > cert_tol)
    diag["max_viol"] = max_viol
    diag["corr_ratio"] = float((d - t).norm() / max(nt, 1e-12))
    diag["d_norm_ratio"] = float(d.norm()) / max(nt, 1e-12)
    diag["proj_min_slack_after"] = float(viol.min())
    if max_viol > cert_tol:
        # Hard certificate FAILED: write NOTHING (final-spec CERT_FAIL —
        # the caller skips the representation step for this window).
        diag["note"] = "cert_fail"
        diag["cert_fail"] = True
        return False, diag
    with torch.no_grad():
        for p, cp, gt in zip(repr_params, torch.split(corr, sizes),
                             g_task_repr):
            # .to(p.device): the QP runs on CPU (V11 OOM fix — the
            # G matrix GPU peak was the last 4GB that blew the card).
            p.grad = (gt - cp.view_as(p)).to(p.device)
    return True, diag


def repr_grad_norm(params):
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += float((p.grad.detach().double() ** 2).sum())
            n += 1
    return math.sqrt(total) if n else 0.0


def bad_grad_names(params, k=3):
    """Names of the first k parameters whose grad is non-finite
    (CCM_AUX_DIAG diagnostic: which param carries the NaN/Inf)."""
    bad = []
    for n, p in params:
        if p.grad is not None and not bool(
                torch.isfinite(p.grad.detach()).all()):
            bad.append(n)
            if len(bad) >= k:
                break
    return bad


def params_digest(params):
    """sha256 over the trainable weights (CPU copy).  The lambda
    calibration asserts the digest is unchanged across its measurements
    so the r_eff values really live on theta_0 (review P0-4)."""
    h = hashlib.sha256()
    for p in params:
        h.update(np.ascontiguousarray(
            p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def trainable_state_dict(model):
    """LoRA + Gamma + any other trainable params ONLY (the frozen 7B
    backbone is NOT stored; a full state_dict costs ~13GB per file)."""
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad}


def new_token_rows(model):
    """The COMP/SUM rows appended by resize_token_embeddings are frozen
    (never in trainable_state_dict) BUT their concrete random values
    affect inference through the comp-token embedding lookup.  Save
    them explicitly so the eval-side build can restore the exact rows
    (review ruling 2026-09-05: eval must reconstruct the SAME model).

    Legacy resize path only: the --official-host path keeps COMP/SUM in
    a SeparatedEmbedding whose comp_embeddings are TRAINABLE and are
    already part of trainable_state_dict."""
    if getattr(model, "_official_host", False):
        return None
    emb = model.get_input_embeddings()
    if not hasattr(emb, "weight"):
        # SeparatedEmbedding (train_ccm_merge path): the comp rows are
        # TRAINABLE and already part of trainable_state_dict — nothing
        # extra to save.
        return None
    lm = model.lm_head
    n = 2 * N_TOK
    return {
        "input_embed_rows": emb.weight[-n:].detach().cpu(),
        "lm_head_rows": lm.weight[-n:].detach().cpu(),
    }


def save_trainable(path, model, **extra):
    payload = {"model": trainable_state_dict(model)}
    _ntr = new_token_rows(model)
    if _ntr is not None:
        payload["new_token_rows"] = _ntr
    payload.update(extra)
    torch.save(payload, path)


def _remap_optimizer_state(ckpt_opt, optimizer, params):
    """R9 (2026-09-16): adapt a checkpoint optimizer state to the current
    param-group layout (single-group <-> --rpbe-lr two-group switch).

    Same code + same build => the checkpoint's flat param order equals
    the current process's, so per-param state entries map by flat index;
    the CURRENT group definitions (with their lr choices) are kept, so a
    resumed run applies the new rpbe_lr instead of the checkpoint's."""
    old_groups = ckpt_opt["param_groups"]
    n_new = sum(len(g["params"]) for g in optimizer.param_groups)
    if len(old_groups) != 1 or len(old_groups[0]["params"]) != len(params):
        return ckpt_opt   # unknown layout: let load_state_dict surface it
    # current groups' flattened order -> global param index
    idx_of = {id(p): i for i, p in enumerate(params)}
    flat = [idx_of[id(p)] for g in optimizer.param_groups
            for p in g["params"]]
    if sorted(flat) != list(range(len(params))):
        raise RuntimeError("param groups do not partition the params")
    pos_of = {old_i: p for p, old_i in enumerate(flat)}
    state = {pos_of[int(k)]: v for k, v in ckpt_opt["state"].items()}
    groups = []
    off = 0
    for g in optimizer.param_groups:
        ng = dict(g)
        ng["params"] = list(range(off, off + len(g["params"])))
        off += len(g["params"])
        groups.append(ng)
    return {"state": state, "param_groups": groups}


def load_trainable(path, model, optimizer, device,
                   load_optimizer: bool = True):
    payload = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"],
                                                strict=False)
    if unexpected:
        raise RuntimeError("unexpected keys in checkpoint: {}"
                           .format(sorted(unexpected)[:5]))
    # Fail-fast (review 2026-09-16): a missing key that is a CURRENT
    # trainable param would silently keep its init value (e.g. an old
    # checkpoint without Gamma leaves a zero-init Gamma).
    trainable_names = {n for n, p in model.named_parameters()
                       if p.requires_grad}
    bad_missing = sorted(set(missing) & trainable_names)
    if bad_missing:
        raise RuntimeError(
            "checkpoint is missing trainable params (stale ckpt?): {}"
            .format(bad_missing[:5]))
    if load_optimizer and "optimizer" in payload:
        opt_ckpt = payload["optimizer"]
        if len(opt_ckpt["param_groups"]) != len(optimizer.param_groups):
            params = [p for p in model.parameters() if p.requires_grad]
            opt_ckpt = _remap_optimizer_state(opt_ckpt, optimizer, params)
        optimizer.load_state_dict(opt_ckpt)
    return payload


def main():
    args = parse_args()
    fz = enforce_frozen(args)  # review P0-3: frozen spec is authoritative
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    tokenizer = build_tokenizer(args)
    if args.official_host:
        model = build_official_host(args, device)
    else:
        model = build_model(args, device)
    if args.foundation and not args.official_host \
            and args.host != "gemma4":
        # Official two-stage protocol: merge the Step-1 default LoRA
        # (llama-7b-no) into the base weights BEFORE attaching our
        # conditional LoRA / Gamma.  The official merge path adds
        # (B @ A) * scaling to the frozen params directly and needs no
        # LoRA structure on the model.  (gemma4 merges its Step-1
        # adapter INSIDE build_model's gemma4 branch — its Step-1 ckpt
        # is a plain torch payload, not an HF adapter directory.)
        from src.model import load_lora_weight
        load_lora_weight(args.foundation, model, merge=True)
        print("[foundation] merged {}".format(args.foundation), flush=True)
    if not args.official_host:
        model = wrap_lora(model, args.lora_r)
        # The PEFT wrap can replace the CausalLM wrapper; re-assert the
        # comp/sum token registration on the wrapped object.
        model.update_comp_token(
            [tokenizer.comp_token_id[k] for k in range(N_TOK)],
            [tokenizer.sum_token_id[k] for k in range(N_TOK)])
        if args.host in ("qwen3", "gemma4"):
            # Two-stage flow (user ruling 2026-09-17): the COMP rows
            # stay TRAINABLE in stage 2 — the merge checkpoint is the
            # init, task gradients keep refining them (RPBE still
            # touches Gamma only under --rpbe-gamma-only).
            model.base_model.model.model.embed_tokens \
                .comp_embeddings.weight.requires_grad_(True)
    use_rpbe = args.arm in ("ours", "gamma_task_only")
    if use_rpbe and args.micro_batch > 1:
        raise SystemExit(
            "[micro-batch] RPBE arms require --micro-batch 1 (the "
            "two-pass replay and window machinery are batch=1 "
            "protocols); batch>1 is currently a task-only option")
    if use_rpbe and (not args.rpbe_native_compression
                     or getattr(args, "history_gamma", False)):
        attach_gamma(model, hidden=args.gamma_hidden)
        if getattr(args, "history_gamma", False):
            print("[history-gamma] Stage-B: Gamma attached on top of "
                  "the native compressor", flush=True)
    if args.rpbe_native_compression and use_rpbe:
        # Native actuation (review 2026-09-22): the RPBE recursion node
        # is the native CCM merge step (M_{t-1}, u_t) -> M_t; the
        # actuated parameters are the native compression params —
        # conditional LoRA + COMP/SUM comp-embedding rows.  Everything
        # else (the backbone) is frozen, so task and RPBE gradients only
        # move the compression params.  No Gamma module is attached, so
        # the SUM-block K/V states lifted by extract_z ARE the pure
        # CCM mean M_t (Z_t = M_t by construction).
        n_frozen = 0
        if getattr(args, "history_gamma", False):
            # Stage-B (review 2026-09-24): freeze the native compressor
            # (LoRA + COMP rows + backbone) and train the history-branch
            # Gamma ONLY.  turn-3 is structurally identical to the
            # Stage-A checkpoint ((t-1)/t = 0 at t=1).
            for _n, _p in model.named_parameters():
                if "gamma" in _n:
                    _p.requires_grad_(True)
                elif _p.requires_grad:
                    _p.requires_grad_(False)
                    n_frozen += 1
        else:
            for _n, _p in model.named_parameters():
                if "lora_" in _n or "comp_embeddings" in _n:
                    continue
                if _p.requires_grad:
                    _p.requires_grad_(False)
                    n_frozen += 1
        # NOTE: peft's get_peft_model usually already froze the base
        # (mark_only_lora_as_trainable), so n_frozen is often 0 — the
        # loop is an idempotent guard, the RESULTING state is what
        # matters (trainable = LoRA + comp rows).
        _base = model
        while not hasattr(_base, "layers") and hasattr(_base, "model"):
            _base = _base.model
        if not getattr(args, "history_gamma", False):
            assert not getattr(_base, "_gamma_attached", False), \
                "native actuation must not carry an attached Gamma"
            for _layer in _base.layers:
                assert getattr(_layer.self_attn, "gamma", None) is None, \
                    "native actuation: found a layer-attached Gamma"
        # The LoRA stays train() (dropout pinned to 0.0 in wrap_lora),
        # so pass-1/pass-2 replay remains bit-identical (the RNG
        # protocol precondition).
        if getattr(args, "history_gamma", False):
            # Stage-B (review 2026-09-24): frozen host eval() + Gamma
            # train().
            model.eval()
            _base2 = model
            while not hasattr(_base2, "layers") \
                    and hasattr(_base2, "model"):
                _base2 = _base2.model
            for _layer in _base2.layers:
                if _layer.self_attn.gamma is not None:
                    _layer.self_attn.gamma.train()
        else:
            model.train()
        print("[native-compression] frozen {} backbone params; "
              "trainable = {}".format(
                  n_frozen,
                  "history-branch Gamma (Stage-B)"
                  if getattr(args, "history_gamma", False)
                  else "conditional LoRA + COMP/SUM rows (pure "
                       "CCM-merge forward, Z_t = M_t)"),
              flush=True)
    if args.init_from:
        # Two-stage init: the merge checkpoint carries LoRA + COMP rows
        # (trainable in stage 2) but NO Gamma — Gamma keeps its zero
        # init by design (stage 2 starts from the exact merge model).
        _payload = torch.load(args.init_from, map_location=device,
                              weights_only=False)
        _missing, _unexpected = model.load_state_dict(
            _payload["model"], strict=False)
        if _unexpected:
            raise RuntimeError("init-from unexpected keys: {}"
                               .format(sorted(_unexpected)[:5]))
        _trainable = {n for n, p in model.named_parameters()
                      if p.requires_grad}
        _bad = sorted(set(_missing) & _trainable)
        # Gamma keys are the ONLY allowed missing trainable keys.
        _bad = [k for k in _bad if "gamma" not in k]
        if _bad:
            raise RuntimeError(
                "init-from checkpoint is missing trainable params: {}"
                .format(_bad[:5]))
        print("[init-from] loaded {} (Gamma zero-init kept)".format(
            args.init_from), flush=True)
    if args.freeze_host:
        # Final protocol (review ruling): the host is the CCM-merge
        # checkpoint, PERMANENTLY frozen.  Only Gamma trains, so
        # Ours - Task-only can only come from predictive preservation.
        n_frozen = 0
        for n, p in model.named_parameters():
            if "gamma" not in n and p.requires_grad:
                p.requires_grad_(False)
                n_frozen += 1
        # Review ruling (2026-09-21): the frozen host must run its
        # forward EXACTLY as at inference.  The official conditional
        # LoRA carries lora_dropout=0.05, which stays ACTIVE while the
        # model is in train() mode — frozen params would still perturb
        # the host forward.  eval() the whole host, then re-enable
        # train() ONLY on the Gamma modules (gradients keep flowing
        # through the frozen host to Gamma either way).
        model.eval()
        for _m in model.modules():
            if isinstance(_m, GammaResidual):
                _m.train()
        print("[freeze-host] frozen {} non-Gamma trainable params "
              "(train Gamma only); host eval() + Gamma train()"
              .format(n_frozen), flush=True)
    cfg = model.model.config
    # Projection scope: the RPBE-actuated representation parameters.
    # gamma line: the Gamma residual of every layer (post-merge
    # correction).  native line (review 2026-09-22): the native
    # compression params — conditional LoRA + COMP/SUM rows — which
    # equals the full trainable set after the backbone freeze.
    repr_params = []
    if use_rpbe:
        if args.rpbe_native_compression:
            repr_params = [p for p in model.parameters()
                           if p.requires_grad]
        else:
            _base = model
            while not hasattr(_base, "layers") and hasattr(_base, "model"):
                _base = _base.model
            for _layer in _base.layers:
                _g = getattr(_layer.self_attn, "gamma", None)
                if _g is not None:
                    repr_params.extend(list(_g.parameters()))
    repr_set = {id(p) for p in repr_params}

    # RNG protocol precondition: zero dropout everywhere (pass-1/pass-2
    # replay must be identical; LLaMA ships with dropout 0).
    assert getattr(cfg, "attention_dropout", 0.0) == 0.0 \
        and getattr(cfg, "hidden_dropout", 0.0) == 0.0, \
        "two-pass replay requires zero model dropout"
    if args.lora_r:
        pass  # lora_dropout pinned to 0.0 in wrap_lora

    adapter = maps = builder = utter_embed = window = None
    if use_rpbe:
        # GQA hosts (qwen3) lift the memory from the KV heads (8) with
        # the config's explicit head_dim (128); the Llama host uses all
        # attention heads (32, head_dim = hidden / heads).  gemma4 is
        # heterogeneous: sliding layers 2 KV heads x 256, full layers
        # 2 x 512 — the flat dim is the per-layer sum (review fix
        # 2026-09-23, JMemLift per_layer_dims).
        per_layer_dims = None
        if args.host == "qwen3":
            n_heads = cfg.num_key_value_heads
            head_dim = getattr(cfg, "head_dim",
                               cfg.hidden_size // cfg.num_attention_heads)
        elif args.host == "gemma4":
            n_heads = cfg.num_key_value_heads
            head_dim = 0
            per_layer_dims = [
                int(cfg.per_layer_config[i].num_key_value_heads)
                * int(cfg.per_layer_config[i].head_dim)
                for i in range(cfg.num_hidden_layers)]
            # Unique-provider measurement (review 2026-09-24): the
            # shared KV layers alias ONE physical provider K/V — sketch
            # only the providers so each physical memory state counts
            # once in J_mem (aliased sketching distorted the predictive
            # gradient norm AND direction).
            _n_shared = int(getattr(cfg, "num_kv_shared_layers", 0))
            unique_layer_ids = (
                None if getattr(args, "aliased_measurement", False)
                else list(range(cfg.num_hidden_layers - _n_shared)))
        else:
            n_heads = cfg.num_attention_heads
            head_dim = cfg.hidden_size // cfg.num_attention_heads
        adapter = CCMHostAdapter(model, n_layers=cfg.num_hidden_layers,
                                 n_heads=n_heads,
                                 head_dim=head_dim,
                                 z_dim=args.z_dim, seed=args.rpbe_seed,
                                 per_layer_dims=per_layer_dims,
                                 unique_layer_ids=(
                                     unique_layer_ids
                                     if args.host == "gemma4"
                                     else None))
        # Review round 8: FOUR independent 32-dim sketch branches (the
        # single 64-dim sketch had too much collision variance for the
        # rare turn-14 rows).  J_ens = mean_r J_r at window close; the
        # branches share d_chi = 64 and the depth bucket encoding.
        maps = Llmmaps(d_chi=64, d_phi=32, m=32,
                       n_branches=Llmmaps.N_BRANCHES,
                       seed=args.rpbe_seed).to(device)
        builder = DialogueCutBuilder(maps, z_dim=args.z_dim,
                                     seed=args.rpbe_seed)
        utter_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=64,
                                     seed=args.rpbe_seed,
                                     combine_dim=1).to(device)
        phi_embed = UtteranceEmbed(hidden_dim=cfg.hidden_size, d_chi=32,
                                   seed=args.rpbe_seed + 100).to(device)
        branches = [KFMomentWindow({MEM_TAU: args.z_dim}, min_ratio=2.0,
                                   min_abs=args.kf_min_cuts,
                                   eps=args.ridge_eps, fixed_maps=maps,
                                   strict=False, autoclose=False,
                                   # Review round 8: paired OAS shrinkage;
                                   # the fixed ridge keeps only its
                                   # numerical-safety role.
                                   oas=True)
                    for _ in range(Llmmaps.N_BRANCHES)]
        window = BranchEnsembleWindow(branches)

    params = [p for p in model.parameters() if p.requires_grad]
    # Official CCM protocol (L6.5 review P0-2): AdamW with weight_decay=0,
    # cosine decay to zero, 3% warmup, and the official Trainer's default
    # max_grad_norm=1.0.  Same scheduler on all three arms (per-step).
    # --rpbe-lr (R9 add, 2026-09-16): an independent base lr for the
    # Gamma group; the single scheduler multiplies BOTH groups by the
    # same warmup/cosine factor, so the group ratio is preserved.
    if args.rpbe_lr is not None and repr_params:
        repr_ids = {id(p) for p in repr_params}
        optimizer = torch.optim.AdamW(
            [{"params": [p for p in params if id(p) in repr_ids],
              "lr": args.rpbe_lr},
             {"params": [p for p in params if id(p) not in repr_ids],
              "lr": args.lr}],
            lr=args.lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, int(args.schedule_total_steps
                            if args.schedule_total_steps is not None
                            else args.max_steps))
    # Fork semantics (review 2026-09-23): --init-from continues from an
    # ALREADY-TRAINED theta (e.g. merge s300/s500), so the warmup that
    # exists for random initializations is skipped — the first window
    # steps at the peak lr and the cosine descends from there.  (The old
    # behavior restarted the scheduler from zero, wasting ~30 of the 50
    # windows in the warmup trough.)  --no-warmup forces the same on a
    # RESUME of a fork arm (resume rebuilds the scheduler; without the
    # flag it would reintroduce the 30-step warmup).
    warmup_steps = 0 if (args.init_from or args.no_warmup) \
        else max(1, int(0.03 * total_steps))

    def _lr_lambda(s):
        if warmup_steps > 0 and s < warmup_steps:
            return float(s) / float(warmup_steps)
        progress = float(s - warmup_steps) / float(
            max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda"
                 and not getattr(model, "_official_host", False)))

    dialog, collator = build_dataset(args, tokenizer)
    comp_ids = tokenizer.comp_token_id
    sum_ids = tokenizer.sum_token_id
    pad_id = tokenizer.pad_token_id
    embed_tokens = model.get_input_embeddings()

    train_items = dialog.train_dataset
    n_items = len(train_items)
    sample_cursor = 0
    # Review round 8: depth-stratified candidate pools + sampling probs.
    # Persisted for the statistical report (per-level candidate counts,
    # unique dialogues, the q_L used, and the achieved per-window mix).
    pools = build_depth_pools(train_items)
    # Qwen3-line sampling (frozen_method_qwen3.json sampling.*): natural
    # frequency q_L ∝ n_L^q_alpha + per-dialogue replay cap.  The Llama
    # line has neither field -> legacy R10 rule (alpha=None) and no cap.
    _sampling = fz.get("sampling", {})
    q_alpha = _sampling.get("q_alpha")
    max_replays = int(_sampling.get("max_replays", 0))  # 0 = unlimited
    depth_probs = depth_sampling_probs(pools, alpha=q_alpha)
    replay_count = {}  # dialogue id -> samples taken so far
    if max_replays:
        print("[sampling] q_alpha={} max_replays={} (Qwen3 official-"
              "semantics line)".format(q_alpha, max_replays), flush=True)
    pool_all = list(range(n_items))
    seen_dialogs = set()  # one ORIGINAL dialogue per window (cleared at
                          # every window close)
    depth_win = {L: 0 for L in DEPTH_LEVELS}  # per-window counters
    save_json(out / "depth_pools.json", {
        "levels": {str(L): {"n_pool": len(pools[L]),
                            "q": float(depth_probs[i])}
                   for i, L in enumerate(DEPTH_LEVELS)},
        "n_items": n_items,
        "n_turns_of_L": {str(L): N_TURNS_OF_L[L] for L in DEPTH_LEVELS},
        "n_turns_of_L_units": "TURNS (prefix length in dialogue turns). "
                              "NOT meta['k']: meta.k = len(blocks) + 1 "
                              "= L + 1 counts context turns and is the "
                              "convention data_flow.jsonl records. The two "
                              "differ by exactly 1 by construction.",
        "rule": ("q_L ~ 1/sqrt(n_L), 3x-uniform oversampling cap" if
                 q_alpha is None else
                 "q_L ~ n_L^{} (natural frequency) with per-dialogue "
                 "replay cap {}".format(q_alpha, max_replays))
        + "; one original dialogue per window; fixed prefix = "
          "n_turns_of_L[L] = L + 2 turns",
        "note_L1": "depth L=1 dialogues (3 turns) emit the TERMINAL cut "
                   "t=1=L with (C, Y) = (c, y), w=1 — one RPBE row "
                   "(review ruling 2026-09-21: Gamma and RPBE now cover "
                   "every compression layer t=1..L, so L=1 is inside the "
                   "method).",
    })

    def next_batch():
        # Review round 8 (depth-stratified legal cuts): draw depth L with
        # q_L (legacy 1/sqrt(n_L) capped, or the Qwen3-line natural
        # frequency), then one dialogue from pool L that has NOT appeared
        # in this window (one dialogue per window), and collate it at the
        # FIXED prefix N_TURNS_OF_L[L] = L + 2 turns (turn count, not the
        # meta.k = L + 1 block convention).  The same RNG stream drives both arms
        # (seed_all), so task-only and ours see the identical sampling
        # stream.
        # Qwen3-line replay cap: a dialogue leaves its pool after
        # max_replays samples (official 12-epoch semantics); exhausted
        # depth layers drop out of the draw (the q weights renormalize),
        # and when every layer is exhausted the counters reset for the
        # next replay cycle.
        # micro_batch>1 (Qwen3 line): draw args.micro_batch dialogues the
        # same way and collate them together; the window tail may return
        # a short batch (fewer unseen dialogues remain).
        nonlocal sample_cursor
        items = []
        orig_ids = []
        Ls = []
        for _ in range(args.micro_batch):
            avail = [L for L in DEPTH_LEVELS
                     if any(replay_count.get(i, 0) < max_replays
                            for i in pools[L])] if max_replays \
                else list(DEPTH_LEVELS)
            if not avail:
                replay_count.clear()
                avail = list(DEPTH_LEVELS)
            q_avail = np.array([depth_probs[DEPTH_LEVELS.index(L)]
                                for L in avail], dtype=np.float64)
            q_avail = q_avail / q_avail.sum()
            L = avail[int(np.random.choice(len(avail), p=q_avail))]
            cand = [i for i in pools[L]
                    if (replay_count.get(i, 0) < max_replays
                        if max_replays else True)
                    and i not in seen_dialogs]
            if not cand and max_replays:
                cand = [i for i in pools[L]
                        if replay_count.get(i, 0) < max_replays]
            if not cand:
                # Pool exhausted within this window: fall back to any
                # unseen dialogue (dedup preserved; depth mix degrades).
                cand = [i for i in pool_all if i not in seen_dialogs]
            if not cand:
                break  # window tail: return the short batch
            orig_id = int(random.choice(cand))
            replay_count[orig_id] = replay_count.get(orig_id, 0) + 1
            seen_dialogs.add(orig_id)
            item = dict(train_items[orig_id])
            item["dialog"] = list(item["dialog"])[:N_TURNS_OF_L[L]]
            item["fixed_depth"] = True
            items.append(item)
            orig_ids.append(orig_id)
            Ls.append(L)
        if not items:
            raise RuntimeError(
                "degenerate depth window: every dialogue already seen "
                "(window grew past the whole pool)")
        batch = collator(items)
        # Review ruling (2026-09-21): the RAW tokenized turns ride along
        # with the batch — chi/phi are built from these, never from
        # spans carved out of the collated ids.
        raw_dialogs = [list(item["dialog"]) for item in items]
        sample_cursor += len(items)
        return batch, sample_cursor - len(items), orig_ids, Ls, \
            raw_dialogs

    threshold = window._threshold(MEM_TAU) if window else None
    save_json(out / "config.json", {
        "arm": args.arm, "seed": args.seed, "cli": vars(args),
        "paired_seed_hash": paired_seed_hash(args.seed, model)
        if use_rpbe else "n/a",
        "actuation": "native" if args.rpbe_native_compression else "gamma",
        "theta0_digest": params_digest(params),
        "n_repr_params": len(repr_params),
        "threshold": threshold,
    })
    print("arm={} seed={} threshold={} n_train={}".format(
        args.arm, args.seed, threshold, n_items), flush=True)

    step = 0
    last_logged_step = 0  # dedup: log only when step changes (L7 smoke)
    last_saved_step = -1
    # Review ruling: ``step`` counts global window attempts (HF Trainer
    # global_step, advances even on AMP skips; the frozen 1000-step budget
    # is GLOBAL steps, not applied-update count).  optimizer_steps_executed
    # counts optimizer.step() invocations (an lr=0 warmup-first execution
    # counts too — "executed", not "applied"); amp_skipped_steps counts
    # GradScaler overflow skips; scheduler_steps counts scheduler.step()
    # calls and must equal optimizer_steps_executed (Trainer 4.44.2: LR
    # advances only on a real step).
    optimizer_steps_executed = 0
    amp_skipped_steps = 0
    scheduler_steps = 0
    # treewise final-spec: windows whose QP certificate failed at 1e-6
    # close WITHOUT a representation step (TGN CERT_FAIL semantics).
    cert_skip_steps = 0
    total_task_sum = 0.0
    total_tokens = 0
    total_microbatches = 0
    total_kf = 0.0
    kf_closed = 0
    aux_terms = 0
    below_threshold = 0
    lambda_kf = args.kf_lambda
    t_start = time.time()
    t_win_start = t_start
    # CCM_PROFILE=1: per-window breakdown of pass1/collect/pass2/bwd
    # wall time (diagnostic for the L6.5 step-time gate).
    profile = os.environ.get("CCM_PROFILE") == "1"
    PROF = {}

    def _pf(key, t0):
        PROF[key] = PROF.get(key, 0.0) + time.perf_counter() - t0

    pending = []
    cut_records = []
    pass1_rngs = []          # per-microbatch pass-1 RNG (P0 fix, R10)
    window_start_state = None
    # Review P0-2: the ccm_merge arm counts effective cuts (dialogues
    # with k >= 3, review round 8: depth L = 1 contributes cuts too) in
    # the same stream and fires its update on the same boundary as the
    # RPBE windows, so all three arms see the same task samples at the
    # same scheduler step.
    merge_eff_cuts = 0
    # L6.5 gate 1: every arm hashes its (sample_id, k) data stream so the
    # three arms can be compared bit-for-bit after a run; the raw stream
    # is also written for prefix comparison across unequal window sizes.
    data_flow_hash = hashlib.sha256()
    data_flow_len = 0
    # Review (cadence collision): per-window boundary records
    # w{ordinal}:{n_mb}:{n_cut_mb} hashed into boundary_hash so the three
    # arms' window boundaries are comparable window by window, not only
    # through the final data-flow hash.  boundary_records keeps the raw
    # window strings so a resumed run can rebuild boundary_hash from
    # scratch (sha256 state is not serializable); data_flow.jsonl plays
    # the same role for data_flow_hash.
    boundary_hash = hashlib.sha256()
    boundary_records = []
    data_flow_path = out / "data_flow.jsonl"
    # Review round 8 statistical report: per-window depth-stratified
    # coverage (per-level cut counts, unique dialogues, duplication rate,
    # row-level ESS, turn-14 coverage).  Written on every window close;
    # the dedup set and per-window counters reset with the window.
    depth_diag_path = out / "depth_diag.jsonl"

    def close_depth_window(n_mb, step_val, n_cuts, n_rows=None):
        uniq = len(seen_dialogs)
        with depth_diag_path.open("a") as f:
            f.write(json.dumps({
                "step": int(step_val),
                "n_mb": int(n_mb),
                "per_L": {str(L): int(depth_win[L]) for L in DEPTH_LEVELS},
                "unique_dialogs": int(uniq),
                "dup_rate": float(1.0 - uniq / max(int(n_mb), 1)),
                "n_cuts": int(n_cuts),
                # Review ruling (2026-09-21): rows per cut are 2 for
                # t < L and 1 for the terminal cut (L=1 emits its
                # terminal row) — pass the exact count from cut_records
                # when available (RPBE arms), else the 2*n_cuts bound.
                "ess_rows": float(n_rows if n_rows is not None
                                  else 2 * n_cuts),
                "L13_cover": float(depth_win[13] / max(int(n_mb), 1)),
            }) + "\n")
        seen_dialogs.clear()
        for L in DEPTH_LEVELS:
            depth_win[L] = 0
    if args.resume_from:
        payload = load_trainable(args.resume_from, model, optimizer, device)
        step = int(payload.get("step", 0))
        lambda_kf = float(payload.get("lambda_kf", args.kf_lambda))
        # R10 recalibration (review 2026-09-16): the committed frozen
        # lambda is authoritative — a resume payload carrying the OLD
        # lambda must not silently override a freshly calibrated one.
        if abs(lambda_kf - float(args.kf_lambda)) > 1e-12:
            print("[frozen] resume payload lambda={} overridden by "
                  "committed lambda={}".format(lambda_kf, args.kf_lambda),
                  flush=True)
            lambda_kf = float(args.kf_lambda)
        if "sample_cursor" in payload:
            sample_cursor = int(payload["sample_cursor"])
        if "rng" in payload:
            _restore_rng(payload["rng"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        if "scheduler" in payload:
            # Audit #3: without the scheduler state a resumed 1000-step
            # run restarts the LR phase (warmup + cosine) from zero.
            # R9: a single-group checkpoint scheduler has one base_lr;
            # the rpbe-lr two-group run needs one per group (the group
            # ratio is carried by the optimizer's lr, so the scheduler's
            # base_lrs are rebuilt from the CURRENT group lrs).
            sch_ckpt = payload["scheduler"]
            if len(sch_ckpt.get("base_lrs", [])) != len(scheduler.base_lrs):
                sch_ckpt = dict(sch_ckpt)
                sch_ckpt["base_lrs"] = [float(g["lr"])
                                        for g in optimizer.param_groups]
            scheduler.load_state_dict(sch_ckpt)
        if "builder_oid" in payload and builder is not None:
            builder.next_oid = int(payload["builder_oid"])
        # Counters (review ruling: full resume of the aggregate counters
        # so a resumed run's final summary covers the WHOLE run, not just
        # the post-resume part).
        if "optimizer_steps_executed" in payload:
            optimizer_steps_executed = int(payload["optimizer_steps_executed"])
        if "amp_skipped_steps" in payload:
            amp_skipped_steps = int(payload["amp_skipped_steps"])
        if "cert_skip_steps" in payload:
            cert_skip_steps = int(payload["cert_skip_steps"])
        if "scheduler_steps" in payload:
            scheduler_steps = int(payload["scheduler_steps"])
        if "total_task_sum" in payload:
            total_task_sum = float(payload["total_task_sum"])
        if "total_tokens" in payload:
            total_tokens = int(payload["total_tokens"])
        if "total_microbatches" in payload:
            total_microbatches = int(payload["total_microbatches"])
        if "total_kf" in payload:
            total_kf = float(payload["total_kf"])
        if "kf_closed" in payload:
            kf_closed = int(payload["kf_closed"])
        if "aux_terms" in payload:
            aux_terms = int(payload["aux_terms"])
        if "boundary_records" in payload:
            boundary_records = list(payload["boundary_records"])
        # Rebuild the stream hashes from the on-disk records.  The jsonl
        # is kept (append mode) across resume; a fresh run truncates it.
        for rec in boundary_records:
            boundary_hash.update(str(rec).encode())
        if data_flow_path.exists():
            for line in data_flow_path.open():
                row = json.loads(line)
                data_flow_hash.update(struct.pack(
                    ">qq", int(row["did"]) % n_items, int(row["k"])))
                data_flow_len += 1
            if data_flow_len != int(payload.get("data_flow_len",
                                                data_flow_len)):
                raise RuntimeError(
                    "resume: data_flow.jsonl rows ({}) != checkpoint "
                    "data_flow_len ({})".format(
                        data_flow_len, payload.get("data_flow_len")))
        elif payload.get("data_flow_len", 0):
            raise RuntimeError(
                "resume: data_flow.jsonl missing but checkpoint recorded "
                "{} rows".format(payload.get("data_flow_len")))
        print("RESUME step={} lambda={} cursor={} executed={} skips={} "
              "sched={} ce_sum={:.2f} tokens={} aux={} boundary={}"
              .format(step, lambda_kf, sample_cursor,
                      optimizer_steps_executed, amp_skipped_steps,
                      scheduler_steps, total_task_sum, total_tokens,
                      aux_terms, len(boundary_records)),
              flush=True)
    else:
        data_flow_path.unlink(missing_ok=True)

    def pass1_one(batch, meta_list):
        """Incremental pass 1: forward (no grad) + rows into the window.
        Returns [(meta, occurrence_id)] per cut in this batch (the pass-2
        replay contract; builder counters stay monotonic)."""
        batch_cuts = []
        with torch.no_grad():
            _t = time.perf_counter()
            run_forward(model, batch, device, grad_enabled=False)
            _pf("pass1_fwd", _t)
            for meta in meta_list:
                _t = time.perf_counter()
                rows = collect_rows(meta, adapter, builder, utter_embed,
                                     phi_embed,
                                    embed_tokens, batch, device,
                                    supervisor_mode=args.supervisor_mode)
                _pf("collect_rows", _t)
                if rows:
                    _t = time.perf_counter()
                    window.add(rows)
                    _pf("window_add", _t)
                    # R10 chain-wise: every cut of the dialogue is
                    # replayed — record (meta, oid, v) ONCE per cut
                    # (2 horizon rows per cut, EXCEPT the deepest cut
                    # whose context-turn obs1 is skipped -> 1 row).
                    _seen_oids = set()
                    for r in rows:
                        if r.occurrence_id in _seen_oids:
                            continue
                        _seen_oids.add(r.occurrence_id)
                        batch_cuts.append(
                            (meta, r.occurrence_id,
                             int(r.context["cut_turn"])))
            adapter.clear()
        return batch_cuts

    def pass2_one(batch, cut_meta, g_by_oid, lam):
        """Pass-2 replay: task CE (official MEAN per microbatch, divided
        by the window size below) + exact surrogate.  ``cut_meta`` is the
        pass-1 [(meta, oid)] record — NO builder/chi/p here (L6.5
        review: pass 2 only re-extracts the gradient-connected z).

        Gradients are mapped through OCCURRENCE IDs (review P0-1): this
        batch's cuts carry their pass-1 occurrence ids; a cut is replayed
        iff its id is in the window plan's ``by_oid``.  Batch positions
        are never used to slice the plan (k < 4 microbatches interleave
        and the plan covers only cut-producing batches)."""
        _t = time.perf_counter()
        out = run_forward(model, batch, device, grad_enabled=True)
        _pf("pass2_fwd", _t)
        task_sum, n_valid = task_ce_shifted(out, batch["labels"], device)
        task_mean = task_sum / max(n_valid, 1)
        aux = torch.zeros((), device=device)
        n_terms = 0
        if g_by_oid:
            z_by_oid = {}
            batch_terms = []
            _t = time.perf_counter()
            for meta, oid, v in cut_meta:
                g = g_by_oid.get(oid)
                if g is not None:
                    z_by_oid[oid] = collect_replay_z(meta, adapter,
                                                     device, v=v)
                    batch_terms.append((oid, g))
            adapter.clear()
            _pf("collect_z", _t)
            _t = time.perf_counter()
            aux, n_terms = batch_surrogate(z_by_oid, batch_terms, lam,
                                           device)
            _pf("surrogate", _t)
            if os.environ.get("CCM_AUX_DIAG") == "1":
                # Diagnostic (acceptance zero-update-diff investigation):
                # is the surrogate gradient actually connected to the
                # replay forward graph on the GPU/fp16 path?
                z_req = sum(1 for z in z_by_oid.values()
                            if z.requires_grad)
                print("AUXDIAG mb={} n_terms={} lam={} z_req={}/{} "
                      "aux.requires_grad={} aux_abs={:.3e}".format(
                          len(batch_terms), n_terms, lam, z_req,
                          len(z_by_oid), aux.requires_grad,
                          abs(float(aux))), flush=True)
        return task_mean, task_sum, n_valid, aux, n_terms

    def grad_step():
        # Review ruling (2026-09-02, L6.5 CONDITIONAL PASS pending item):
        # HF Trainer 4.44.2 advances the scheduler ONLY when the optimizer
        # step actually ran; an AMP overflow skip (GradScaler backs the
        # scale off and discards the update) must NOT change the LR.
        # The skip is detected by the scale backoff (the same signal
        # Accelerator uses), not by params_digest — a digest compare
        # cannot tell "optimizer ran with lr=0" from "optimizer skipped".
        nonlocal optimizer_steps_executed, amp_skipped_steps
        nonlocal scheduler_steps
        scale_before = float(scaler.get_scale())
        if os.environ.get("CCM_AUX_DIAG") == "1":
            digest_before = params_digest(params)
        scaler.unscale_(optimizer)
        if os.environ.get("CCM_AUX_DIAG") == "1":
            # Diagnostic (NaN investigation): after unscale_, is the raw
            # gradient finite?  If the BAD grads vanish here, the NaN was
            # fp16 scaled-overflow (GradScaler cold start); if they
            # persist, the backward graph itself produces non-finite grads.
            named = [(n, p) for n, p in model.named_parameters()
                     if p.requires_grad and p.grad is not None]
            bad = bad_grad_names(named, k=5)
            norm = repr_grad_norm(params) if not bad else float("nan")
            print("AUXDIAG unscale norm={:.6f} bad={}".format(
                norm, bad), flush=True)
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        amp_skipped = (scaler.is_enabled()
                       and float(scaler.get_scale()) < scale_before)
        optimizer_steps_executed += 1
        if not amp_skipped:
            scheduler.step()
            scheduler_steps += 1
        else:
            amp_skipped_steps += 1
        # Review ruling: the independent scheduler counter must stay glued
        # to the real-execution counter (Trainer 4.44.2): every optimizer
        # execution advances either the scheduler or the skip counter.
        assert optimizer_steps_executed == scheduler_steps + amp_skipped_steps, (
            "step counters diverged: executed={} sched={} skips={}".format(
                optimizer_steps_executed, scheduler_steps,
                amp_skipped_steps))
        if os.environ.get("CCM_AUX_DIAG") == "1":
            digest_changed = params_digest(params) != digest_before
            print("AUXDIAG grad_step digest_changed={} amp_skipped={} "
                  "scale_now={:.0f}".format(
                      digest_changed, amp_skipped, scaler.get_scale()),
                  flush=True)
        return not amp_skipped

    while step < args.max_steps:
        batch, sample_id, orig_id, L, raw_dialogs = next_batch()
        for _l in L:
            depth_win[_l] += 1
        # Data-stream hash for every arm (parse_meta is pure, no RNG).
        # Review round 8: the stream records the STABLE dialogue id and
        # its depth level — the per-step cursor is no longer the identity.
        metas = parse_meta(batch, comp_ids, sum_ids, sample_id,
                           orig_ids=orig_id, raw_dialogs=raw_dialogs)
        for m in metas:
            data_flow_hash.update(struct.pack(
                ">qq", int(m["orig_id"]) if m["orig_id"] >= 0
                else int(m["sample_id"]) % n_items, int(m["k"])))
            data_flow_len += 1
        with data_flow_path.open("a") as f:
            for m in metas:
                # L = compressed-history turns (blocks); k = L + 1 keeps
                # the legacy context-turn convention.
                f.write(json.dumps({
                    "did": int(m["orig_id"]) if m["orig_id"] >= 0
                    else int(m["sample_id"]) % n_items,
                    "L": int(m["L"]),
                    "k": int(m["k"])}) + "\n")
        if window_start_state is None:
            # Builder counters are NOT rewound here: pass 2 never touches
            # the builder, so occurrence ids stay monotonic across the
            # whole run (L6.5 review P0-1).
            window_start_state = {"rng": _rng_state()}
        pending.append((batch, sample_id))
        if use_rpbe:
            # Incremental pass 1: RNG restored after each microbatch so
            # the data-sampling stream matches the single-pass arm; the
            # cut rows and their occurrence ids accumulate monotonically.
            # Microbatches with NO possible cut (L < 1) skip the pass-1
            # forward entirely: no row can come from them (L6.5 perf:
            # DailyDialog's effective-cut rate is ~50%, halving the
            # pass-1 cost).  Review ruling (2026-09-21): L = 1 now DOES
            # emit the terminal (c, y) cut, so the gate is L >= 1.
            # P0 fix (R10, review 2026-09-16): the official Step-2 LoRA
            # carries lora_dropout=0.05 and the model is in TRAIN mode —
            # dropout consumes CUDA RNG inside the forward.  Pass 1
            # restores the SAME state after every microbatch (all masks
            # identical); pass 2 replays the forward stream linearly, so
            # WITHOUT per-microbatch restoration z_pass2 != z_pass1 and
            # the exact-replay surrogate is broken.  Keep each
            # microbatch's pre-forward RNG and restore it before its
            # pass-2 forward: mask_i^pass2 == mask_i^pass1.
            state = {"rng": _rng_state()}
            pass1_rngs.append(state["rng"])
            if any(m["ok"] and m["L"] >= 1 for m in metas):
                batch_cuts = pass1_one(batch, metas)
            else:
                batch_cuts = []
            cut_records.append(batch_cuts)
            _restore_rng(state["rng"])
            if window.window_ready():
                _t = time.perf_counter()
                closed, plan, diag = window.close_replay()
                _pf("close_replay", _t)
                # Review round 5 (CCA saturation audit): persist the
                # dialogue-grouped shuffle diagnostic per window.
                _d = diag.get(MEM_TAU, {}) if isinstance(diag, dict) else {}
                with (out / "window_diag.jsonl").open("a") as _f:
                    _f.write(json.dumps(_json_safe({
                        "step": int(step),
                        "M_unique": _d.get("M_unique"),
                        "M_rows": _d.get("M_rows"),
                        "M_unique_trees": _d.get("M_unique_trees"),
                        "J_real_minus_shuffled":
                            _d.get("J_real_minus_shuffled"),
                        "J_shuffled_mean": _d.get("J_shuffled"),
                        "J_shuffled_n": _d.get("J_shuffled_n"),
                        # R9 (review 2026-09-16): within-depth null — the
                        # depth bucket b_L lives in p, so the GLOBAL
                        # shuffle breaks depth pairing and can stay large
                        # on pure depth identity.  within-L keeps depth
                        # intact; only future CONTENT is deranged.
                        "J_shuffled_within_L":
                            _d.get("J_shuffled_within_L"),
                        "J_real_minus_shuffled_within_L":
                            _d.get("J_real_minus_shuffled_within_L"),
                        # Review round 8: per-branch scores + ensemble.
                        "J_ens": _d.get("J_ens"),
                        "J_branches": _d.get("J_branches"),
                        # OAS shrinkage intensities (branch 0's window
                        # statistics, detached by construction).
                        "alpha_z": _d.get("alpha_z"),
                        "alpha_p": _d.get("alpha_p"),
                    })) + "\n")
                # P0-1 fix: capture the REAL post-pass-1 data-stream RNG
                # position; pass 2 replays from the window start and the
                # stream resumes from the captured position afterwards.
                resume_rng = _rng_state()
                _restore_rng(window_start_state["rng"])
                # Review P0-1: consume the window gradients through the
                # occurrence-id index.  The legacy per-batch slice was
                # aligned to the CUT-PRODUCING batches only, so indexing
                # it with the pending position silently dropped ~99% of
                # the cuts once k < 4 microbatches interleaved
                # (aux_terms was 11 over 7 windows instead of 896).
                g_by_oid = plan.get(MEM_TAU, {}).get("by_oid", {})
                # Audit #3: plan completeness — every cut pass 1 recorded
                # must carry a window gradient.  Without this assert a
                # numeric window failure (strict=False returns an empty
                # by_oid) would silently decay ours toward
                # gamma_task_only for the REST of the run — the 11/896
                # class of bug, now fail-fast on every window (checked
                # for the calibration window too).
                n_cut_win = sum(1 for cr in cut_records for _ in cr)
                if len(g_by_oid) != n_cut_win:
                    raise RuntimeError(
                        "RPBE replay incomplete: plan={} cuts={} diag={}"
                        .format(len(g_by_oid), n_cut_win,
                                diag.get(MEM_TAU)))
                if args.calibrate_lambda and kf_closed == 0 \
                        and not args.resume_from:
                    # Review P0-4 (lambda calibration timing): the
                    # calibration must run on theta_0 — BEFORE this
                    # window's optimizer/scheduler step — replaying the
                    # theta_0 adjoint plan against theta_0 z's.  The
                    # previous order ran the real pass 2 and grad_step
                    # first, then measured on theta_1: the surrogate
                    # became J_z(theta_1)^T dJ/dz|_{theta_0} (mixed
                    # point) and the task gradient was measured at
                    # theta_1 too, so the derived lambda was not the
                    # frozen spec's r_eff calibration value.  Asserts:
                    # trainable params unchanged and no optimizer /
                    # scheduler step has fired.
                    digest0 = params_digest(params)
                    assert step == 0, \
                        "lambda calibration must fire before the first " \
                        "optimizer step (step={})".format(step)
                    # r_eff on this window: separate norm measurements.
                    # g_task uses the real training scale (per-microbatch
                    # MEAN divided by len(pending)); g_kf uses the real
                    # pass-2 scale (aux NOT divided — review P0-3).
                    # Native line (review 2026-09-22): r_eff is measured
                    # in PROPOSAL space via the shadow AdamW proposal —
                    # lambda scales the predictive proposal inside the
                    # joint center d_0, so the ratio must be the
                    # proposal-norm ratio.  At theta_0 the optimizer
                    # state is empty (first bias-correction step).
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    for b, sid in pending:
                        fwd_out = run_forward(model, b, device,
                                              grad_enabled=True)
                        task_sum_m, n_valid_m = task_ce_shifted(
                            fwd_out, b["labels"], device)
                        (task_sum_m / max(n_valid_m, 1)
                         / float(len(pending))).backward()
                    if os.environ.get("CCM_CALIB_DIAG") == "1":
                        _gd = [p for p in repr_params
                               if p.grad is not None]
                        print("CALIBDIAG task: n_grad={}/{} lr={} "
                              "g_abs0={}".format(
                                  len(_gd), len(repr_params),
                                  optimizer.param_groups[0]["lr"],
                                  float(_gd[0].grad.abs().sum())
                                  if _gd else None), flush=True)
                    if args.rpbe_native_compression:
                        g_task = proposal_norm(optimizer, repr_params, 1,
                                               lr_override=1.0)
                    else:
                        g_task = repr_grad_norm(repr_params)
                    g_task_all = repr_grad_norm(params)
                    optimizer.zero_grad(set_to_none=True)
                    _restore_rng(window_start_state["rng"])
                    for i, (b, sid) in enumerate(pending):
                        fwd_out = run_forward(model, b, device,
                                              grad_enabled=True)
                        # Same occurrence-id mapping as pass 2 (P0-1):
                        # only this batch's cuts, resolved via by_oid.
                        z_by_oid = {}
                        batch_terms = []
                        for meta, oid, v in cut_records[i]:
                            g = g_by_oid.get(oid)
                            if g is not None:
                                z_by_oid[oid] = collect_replay_z(
                                    meta, adapter, device, v=v)
                                batch_terms.append((oid, g))
                        adapter.clear()
                        aux, n_aux = batch_surrogate(
                            z_by_oid, batch_terms, 1.0, device)
                        # A microbatch without terms (k < 4) contributes
                        # nothing; its zero aux has no grad_fn, so skip.
                        if n_aux:
                            aux.backward()
                    _restore_rng(resume_rng)
                    if args.rpbe_native_compression:
                        g_kf = proposal_norm(optimizer, repr_params, 1,
                                             lr_override=1.0)
                    else:
                        g_kf = repr_grad_norm(repr_params)
                    g_kf_all = repr_grad_norm(params)
                    if params_digest(params) != digest0:
                        raise RuntimeError(
                            "lambda calibration changed the trainable "
                            "params — theta_0 violated")
                    r_eff = g_kf / max(g_task, 1e-30)
                    r_eff_all = g_kf_all / max(g_task_all, 1e-30)
                    derived = 0.1 / max(r_eff, 1e-30)
                    save_json(out / "calibration.json", {
                        "g_task_repr": g_task, "g_kf_gamma": g_kf,
                        "r_eff_gamma": r_eff,
                        "g_task_all": g_task_all, "g_kf_all": g_kf_all,
                        "r_eff_all": r_eff_all,
                        "derived_lambda": derived,
                        "optimizer_steps": 0, "scheduler_steps": 0,
                        "scope": "native" if args.rpbe_native_compression
                        else "gamma",
                        "space": "proposal" if args.rpbe_native_compression
                        else "gradient",
                        "rule": (
                            "native-scope calibration (review "
                            "2026-09-22): lambda = 0.1 / r_eff measured "
                            "in PROPOSAL space on the native compression "
                            "params at theta_0"
                            if args.rpbe_native_compression else
                            "R10 gamma-scope calibration (review "
                            "2026-09-16): lambda = 0.1 / r_eff measured "
                            "on the GAMMA group only at theta_0, "
                            "chain-wise window")})
                    print(json.dumps({"g_task_repr": g_task,
                                      "g_kf_gamma": g_kf,
                                      "r_eff_gamma": r_eff,
                                      "r_eff_all": r_eff_all,
                                      "derived_lambda": derived,
                                      "scope": "native"
                                      if args.rpbe_native_compression
                                      else "gamma",
                                      "space": "proposal"
                                      if args.rpbe_native_compression
                                      else "gradient",
                                      "theta0_verified": True},
                                     indent=2), flush=True)
                    return
                # Audit #3: pass 2 must replay every planned cut exactly
                # once (n_cut_win and the plan-completeness check ran
                # above, before the calibration branch).
                aux_before = aux_terms
                optimizer.zero_grad(set_to_none=True)
                task_sum = 0.0
                n_tokens = 0
                treewise = (args.rpbe_constrain_mode == "treewise"
                            and args.arm == "ours")
                if treewise:
                    # Tree-wise feasibility projection (TGN final-spec
                    # port, 2026-09-15): pass 1 = task CE alone (snapshot
                    # grads); pass 2 = per-oid surrogate backwards collect
                    # per-tree Gamma influence dirs (never summed);
                    # non-Gamma params accumulate the ordinary summed aux
                    # gradient.  The QP projects the Gamma task gradient
                    # into the intersection of the per-tree half-spaces
                    # and certifies EVERY row at 1e-6; a failed
                    # certificate skips the representation step below.
                    if args.rpbe_native_compression:
                        # Native proposal-space line (review 2026-09-22):
                        # the QP needs BOTH the pure task proposal
                        # (constraint norm ||d_task||) and the joint
                        # proposal (center d_0).  OOM fix (2026-09-22,
                        # second): with the LoRA trainable, EVERY
                        # layer's hidden carries grad, so one tree's
                        # graph holds the full 36-layer saved
                        # activations — retaining the whole window's
                        # graphs (128 trees) blew the 40GB card twice.
                        # Per-batch: task backward (retain for THIS
                        # batch's aux) -> accumulate the PURE task
                        # increment (grad delta) -> aux backward FREES
                        # the graph.  pure_acc holds the exact pure
                        # task gradient in scaled space.
                        pure_acc = {}
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])  # P0: mask==pass1
                            task_mean, task_raw, n_valid, _aux, _n = \
                                pass2_one(b, cut_records[i], g_by_oid,
                                          (0.0 if args.s4_supervisor
                                           else lambda_kf))
                            _before = {
                                id(p): p.grad.detach().clone()
                                for p in params if p.grad is not None}
                            if task_mean.requires_grad:
                                # S4 supervisor (review 2026-09-23):
                                # NO retain — the S4 loop runs its OWN
                                # per-observation forwards, so the
                                # sketch surrogate graph is not needed.
                                scaler.scale(
                                    task_mean
                                    / float(len(pending))).backward(
                                        retain_graph=(
                                            not args.s4_supervisor))
                            for p in params:
                                if p.grad is None:
                                    continue
                                _prev = _before.get(id(p))
                                _inc = p.grad.detach() - (
                                    _prev if _prev is not None
                                    else torch.zeros_like(p.grad))
                                if id(p) in pure_acc:
                                    pure_acc[id(p)].add_(_inc)
                                else:
                                    pure_acc[id(p)] = _inc.clone()
                            if _n and _aux.requires_grad \
                                    and not args.s4_supervisor:
                                scaler.scale(_aux).backward()  # frees graph
                            task_sum += float(task_raw.detach())
                            n_tokens += n_valid
                        task_grads_pure = pure_acc
                        if not args.s4_supervisor:
                            task_grads = {
                                id(p): p.grad.detach().clone()
                                for p in params if p.grad is not None}
                    else:
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])   # P0: mask == pass1
                            # V11 (review): the QP center is the JOINT
                            # proposal q = t + a_lambda, not the bare
                            # task proposal t.  Running pass2_one with
                            # the real lambda accumulates the aggregate
                            # RPBE gradient ONTO the Gamma task
                            # gradient, so the snapshot below
                            # (g_task_repr) is exactly the joint
                            # proposal direction; the projection then
                            # trims only the components that hurt a
                            # local interface.  Non-Gamma params keep
                            # pure task (rpbe_gamma_only structure,
                            # unchanged).
                            task_mean, task_raw, n_valid, _aux, _n = \
                                pass2_one(
                                    b, cut_records[i], g_by_oid,
                                    0.0 if args.arm == "gamma_task_only"
                                    else lambda_kf)
                            if task_mean.requires_grad:
                                # freeze-host fix: k<3 batches carry no
                                # COMP/SUM rows, so Gamma never touches
                                # the output and task_mean has no graph.
                                # Skip their (zero) backward instead of
                                # dying.
                                scaler.scale(
                                    task_mean
                                    / float(len(pending))).backward(
                                        retain_graph=True)
                            if _n and args.arm == "ours" \
                                    and _aux.requires_grad:
                                # V11: accumulate the aggregate RPBE
                                # gradient onto Gamma so the snapshot
                                # below is the joint proposal
                                # q = t + a_lambda (mirrors the aggregate
                                # branch's aux backward).  freeze-host
                                # fix: _n>0 but a graph-less _aux (all
                                # oids filtered out) has nothing to add.
                                scaler.scale(_aux).backward()
                            task_sum += float(task_raw.detach())
                            n_tokens += n_valid
                        task_grads = {
                            id(p): p.grad.detach().clone()
                            for p in params if p.grad is not None}
                    optimizer.zero_grad(set_to_none=True)
                    aux_other = {id(p): torch.zeros_like(p)
                                 for p in params
                                 if id(p) not in repr_set}
                    native_g_joint = None   # true-value g_joint for the
                    native_amp_skip = False  # native proposal write-back
                    dirs = []
                    s4_pred_acc = {}
                    audit_rows = [] if args.gradient_audit else None
                    _early_drop_step = (int(round(
                        1.0 / args.early_cut_drop_ratio))
                        if args.early_cut_drop_ratio > 0 else 0)
                    _early_cnt = 0
                    # DB-DG-S4 state (review 2026-09-24)
                    _dbdg = bool(args.s4_deficit_gate) \
                        or bool(args.s4_depth_mults)
                    _dbdg_mults = None
                    if args.s4_depth_mults:
                        _pm = [float(x) for x in
                               args.s4_depth_mults.split(",")]
                        assert len(_pm) == 3, args.s4_depth_mults
                        _dbdg_mults = _pm
                    _dbdg_abar = 1.0
                    _dbdg_diag = None
                    if _dbdg:
                        _wsum = 0.0
                        _wasum = 0.0
                        for _i2 in range(len(pending)):
                            for meta, oid, v in cut_records[_i2]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                for _h, _cd, _fd, _w in s4_obs_list(
                                        meta, int(v)):
                                    _tt = int(v) + 1
                                    _aa = (_dbdg_mults[0] if _tt <= 2
                                           else _dbdg_mults[1]
                                           if _tt <= 5
                                           else _dbdg_mults[2]) \
                                        if _dbdg_mults else 1.0
                                    _wsum += _w
                                    _wasum += _w * _aa
                        _dbdg_abar = (_wasum / _wsum
                                      if _wsum > 0 else 1.0)
                        _dbdg_diag = {
                            "mass_raw": [0.0, 0.0, 0.0],
                            "mass_active": [0.0, 0.0, 0.0],
                            "n_gate": [0, 0, 0],
                            "n_tot": [0, 0, 0]}
                    if args.rpbe_native_compression \
                            and args.s4_supervisor:
                        # S4 formal supervisor (review 2026-09-23
                        # frozen candidate): the treewise directions are
                        # the per-observation real-token NLL gap
                        # gradients q_{t,h} = grad_theta w_h * Delta,
                        # Delta = l_comp - sg(l_full).  Each observation
                        # runs its OWN comp-dialogue forward (grad) and
                        # full-dialogue forward (no_grad, detach) with
                        # the SAME future conditioning — only the
                        # history compression differs.  The per-obs
                        # backward also accumulates the lambda-weighted
                        # aggregate predictive gradient (the joint
                        # proposal center d_0).
                        # Speed fix (2026-09-23): the observations of
                        # ONE tree batch into a single multi-row
                        # forward (<=2 rows); the per-row CE backwards
                        # share that graph (retain until the last row —
                        # the same rolling-release pattern as the
                        # sketch dirs loop).  Method-identical to the
                        # per-obs forwards, ~2x fewer forwards.
                        # --s4-ref-cache: the full-reference NLLs for
                        # the WHOLE window are precomputed in chunked
                        # no_grad forwards before the loop and looked
                        # up per observation (q_{t,h} is unchanged —
                        # l_full is a stop-grad scalar).
                        _ref_table = {}
                        if args.s4_ref_cache:
                            _ref_items = []
                            for _i2 in range(len(pending)):
                                for meta, oid, v in cut_records[_i2]:
                                    if g_by_oid.get(oid) is None:
                                        continue
                                    for _h, _cd, _fd, _w in s4_obs_list(
                                            meta, int(v)):
                                        _ref_items.append(
                                            ((oid, _h), _fd))
                            with torch.no_grad():
                                for _c0 in range(
                                        0, len(_ref_items),
                                        args.s4_ref_chunk):
                                    _chunk = _ref_items[
                                        _c0:_c0 + args.s4_ref_chunk]
                                    _b3 = collator(
                                        [{"dialog": _d,
                                          "fixed_depth": True}
                                         for _, _d in _chunk])
                                    _b3 = {kk: vv.to(device)
                                           for kk, vv in _b3.items()}
                                    _f3 = run_forward(
                                        model, _b3, device,
                                        grad_enabled=False)
                                    _sh3 = _f3.logits[..., :-1, :] \
                                        .contiguous()
                                    _sl3 = _b3["labels"][..., 1:] \
                                        .contiguous()
                                    for _j3, (_key, _d) in \
                                            enumerate(_chunk):
                                        _m3 = _sl3[_j3] != -100
                                        _ref_table[_key] = float(
                                            F.cross_entropy(
                                                _sh3[_j3][_m3],
                                                _sl3[_j3][_m3]))
                        for i, (b, sid) in enumerate(pending):
                            for meta, oid, v in cut_records[i]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                _obs = s4_obs_list(meta, int(v))
                                _b2 = collator(
                                    [{"dialog": _d, "fixed_depth": True}
                                     for _, _d, _, _ in _obs])
                                _b2 = {kk: vv.to(device)
                                       for kk, vv in _b2.items()}
                                _f2 = run_forward(model, _b2, device,
                                                  grad_enabled=True)
                                _sh2 = _f2.logits[..., :-1, :] \
                                    .contiguous()
                                _sl2 = _b2["labels"][..., 1:] \
                                    .contiguous()
                                for _j, (_h, _cdlg, _fdlg, _w) in \
                                        enumerate(_obs):
                                    _m2 = _sl2[_j] != -100
                                    _ce_c = F.cross_entropy(
                                        _sh2[_j][_m2], _sl2[_j][_m2])
                                    if _ref_table:
                                        _ce_f = _ref_table[(oid, _h)]
                                        _ce_f = torch.tensor(
                                            _ce_f, dtype=_ce_c.dtype,
                                            device=_ce_c.device)
                                    else:
                                        with torch.no_grad():
                                            _b3 = collator(
                                                [{"dialog": _fdlg,
                                                  "fixed_depth": True}])
                                            _b3 = {kk: vv.to(device)
                                                   for kk, vv
                                                   in _b3.items()}
                                            _f3 = run_forward(
                                                model, _b3, device,
                                                grad_enabled=False)
                                            _sh3 = _f3.logits[
                                                ..., :-1, :] \
                                                .contiguous()
                                            _sl3 = _b3["labels"][
                                                ..., 1:].contiguous()
                                            _m3 = _sl3 != -100
                                            _ce_f = F.cross_entropy(
                                                _sh3[_m3], _sl3[_m3])
                                    # raw gap, stop-grad full reference
                                    _loss_h = scaler.scale(
                                        lambda_kf * _w
                                        * (_ce_c - _ce_f.detach()))
                                    optimizer.zero_grad(
                                        set_to_none=True)
                                    _loss_h.backward(
                                        retain_graph=(
                                            _j < len(_obs) - 1))
                                    _skip_row = False
                                    if _early_drop_step \
                                            and int(v) + 1 <= 2:
                                        _early_cnt += 1
                                        if _early_cnt % _early_drop_step \
                                                == 0:
                                            _skip_row = True
                                    # DB-DG-S4 (review 2026-09-24):
                                    # deficit gate + depth reweight
                                    # apply to the ACTIVE center only.
                                    _center_scale = 1.0
                                    _dbdg_g = 2
                                    if _dbdg:
                                        _tt = int(v) + 1
                                        _dbdg_g = (0 if _tt <= 2
                                                   else 1 if _tt <= 5
                                                   else 2)
                                        _delta = float(
                                            _ce_c.detach()) - float(
                                                _ce_f.detach()
                                                if torch.is_tensor(
                                                    _ce_f) else _ce_f)
                                        _gate = 1.0 if _delta > 0 \
                                            else 0.0
                                        _aa = _dbdg_mults[_dbdg_g] \
                                            if _dbdg_mults else 1.0
                                        _center_scale = (_aa
                                                         / _dbdg_abar) \
                                            * _gate
                                    if not _skip_row:
                                        dirs.append(torch.cat(
                                            [p.grad.reshape(-1).float()
                                             if p.grad is not None else
                                             torch.zeros(p.numel(),
                                                         dtype=torch.float32,
                                                         device=p.device)
                                             for p in repr_params]).cpu())
                                        if audit_rows is not None:
                                            audit_rows.append(
                                                (int(meta["L"]),
                                                 int(_h), int(v),
                                                 dirs[-1]))
                                    for p in params:
                                        if p.grad is None:
                                            continue
                                        if _dbdg:
                                            _gn = float(
                                                p.grad.detach()
                                                .double().norm())
                                            _dbdg_diag["mass_raw"][
                                                _dbdg_g] += _gn
                                            _dbdg_diag["n_tot"][
                                                _dbdg_g] += 1
                                            if _center_scale > 0:
                                                _dbdg_diag["n_gate"][
                                                    _dbdg_g] += 1
                                                _dbdg_diag[
                                                    "mass_active"][
                                                    _dbdg_g] += (
                                                        _gn
                                                        * _center_scale)
                                        if _center_scale != 1.0:
                                            if id(p) in s4_pred_acc:
                                                s4_pred_acc[id(p)].add_(
                                                    p.grad.detach()
                                                    * _center_scale)
                                            else:
                                                s4_pred_acc[id(p)] = (
                                                    p.grad.detach()
                                                    .clone()
                                                    * _center_scale)
                                        else:
                                            if id(p) in s4_pred_acc:
                                                s4_pred_acc[id(p)].add_(
                                                    p.grad.detach())
                                            else:
                                                s4_pred_acc[id(p)] = (
                                                    p.grad.detach()
                                                    .clone())
                                adapter.clear()
                        # joint (scaled space) = pure task + lambda-w-
                        # eighted aggregate S4 gradient.
                        task_grads = {}
                        for p in params:
                            _v = task_grads_pure.get(id(p))
                            _v = (_v.clone() if _v is not None
                                  else torch.zeros_like(p))
                            _e = s4_pred_acc.get(id(p))
                            if _e is not None:
                                _v = _v + _e
                            task_grads[id(p)] = _v
                    else:
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])  # P0: mask==pass1
                            fwd_out = run_forward(model, b, device,
                                                  grad_enabled=True)
                            # OOM fix (review 2026-09-22): retain_graph on
                            # EVERY cut kept this batch's full graph alive
                            # for the whole window — with the LoRA trainable
                            # (native), all 36 layers' activations carry
                            # grad, so 128 trees of retained graphs blew the
                            # 40GB card.  Retain only until the batch's LAST
                            # cut; its backward frees the graph.  Gradient
                            # VALUES are unchanged (retain only controls
                            # graph lifetime).
                            _cuts = [(meta, oid, v) for meta, oid, v
                                     in cut_records[i]
                                     if g_by_oid.get(oid) is not None]
                            for _j, (meta, oid, v) in enumerate(_cuts):
                                g = g_by_oid.get(oid)
                                optimizer.zero_grad(set_to_none=True)
                                z = collect_replay_z(meta, adapter, device,
                                                     v=v)
                                gd = g.detach()
                                aux_i = -lambda_kf * ((gd * z).sum()
                                                      - (gd * z.detach()).sum())
                                # retain_graph: the SAME fwd_out graph is
                                # replayed for every cut of this batch
                                # (treewise bug found by the probe audit:
                                # the second cut's backward died on a freed
                                # graph — treewise mode had never run).
                                aux_i.backward(
                                    retain_graph=(_j < len(_cuts) - 1))
                                # CPU-side accumulation (OOM fix):
                                # the dirs matrix is ~n_dirs x n_gamma fp32
                                # (~4GB for 1098 dirs); keeping the list on
                                # GPU plus the torch.stack copy doubled the
                                # peak and blew the 40GB card.  Structural
                                # zeros (native): the last layer's q/o LoRA
                                # has NO path in z's graph -> grad None,
                                # pad with zeros (review 2026-09-22).
                                dirs.append(torch.cat(
                                    [p.grad.reshape(-1).float()
                                     if p.grad is not None else
                                     torch.zeros(p.numel(),
                                                 dtype=torch.float32,
                                                 device=p.device)
                                     for p in repr_params]).cpu())
                                for p in params:
                                    if id(p) not in repr_set \
                                            and p.grad is not None:
                                        aux_other[id(p)].add_(
                                            p.grad.detach())
                            adapter.clear()
                    optimizer.zero_grad(set_to_none=True)
                    g_task_repr = [
                        task_grads.get(id(p), torch.zeros_like(p))
                        for p in repr_params]
                    if os.environ.get("CCM_TREEWISE_PROBE") == "1" \
                            and dirs:
                        # Audit (review): cos(g_task, g_v) distribution
                        # on the CURRENT window — answers whether the
                        # treewise constraint would actually bind.
                        # Constraint semantics: t = -g_task, row g_j,
                        # g_j.d >= -kappa||g_j||||t|| i.e.
                        # cos(g_j, g_task) <= +kappa is the safe side;
                        # BOTH tails are reported for reviewer judgement.
                        import numpy as _np
                        _t_flat = torch.cat(
                            [x.reshape(-1).float()
                             for x in g_task_repr])
                        _nt = float(_t_flat.norm())
                        _G = torch.stack(dirs)
                        _nr = _G.norm(dim=1)
                        _cos = ((_G @ _t_flat)
                                / (_nr * _nt).clamp(min=1e-12)).tolist()
                        _cos = [float(c) for c in _cos]
                        _a = _np.array(_cos)
                        _probe = {
                            "n_dirs": int(len(_cos)),
                            "cos_mean": float(_a.mean()),
                            "cos_med": float(_np.median(_a)),
                            "cos_min": float(_a.min()),
                            "cos_max": float(_a.max()),
                            "frac_below_neg_kappa": float(
                                (_a < -args.rpbe_kappa).mean()),
                            "frac_above_pos_kappa": float(
                                (_a > args.rpbe_kappa).mean()),
                            "kappa": float(args.rpbe_kappa),
                            "cos_list": _cos,
                        }
                        with (out / "treewise_probe.json").open("w") \
                                as _f:
                            json.dump(_probe, _f, indent=2)
                        print("[treewise-probe] n={} mean={:.4f} "
                              "med={:.4f} frac<-k={:.3f} frac>+k={:.3f} "
                              "min={:.4f} max={:.4f}".format(
                                  _probe["n_dirs"], _probe["cos_mean"],
                                  _probe["cos_med"],
                                  _probe["frac_below_neg_kappa"],
                                  _probe["frac_above_pos_kappa"],
                                  _probe["cos_min"], _probe["cos_max"]),
                              flush=True)
                        raise SystemExit(
                            "CCM_TREEWISE_PROBE: window probed, "
                            "no optimizer step executed")
                    if args.rpbe_native_compression:
                        # Shared pre-projection block (P0 fix +
                        # interface probe share): NO live p.grad exists
                        # here — the pure/joint snapshots taken BEFORE
                        # the dirs loop are the only gradient source.
                        # They were recorded in SCALED space: divide by
                        # the current scale to recover the true values
                        # (no scaler.unscale_ needed).
                        scale_f = float(scaler.get_scale())
                        g_joint_true = [
                            (task_grads.get(id(p), torch.zeros_like(p))
                             .detach() / scale_f)
                            for p in repr_params]
                        g_task_true = [
                            (task_grads_pure.get(
                                id(p), torch.zeros_like(p)).detach()
                             / scale_f)
                            for p in repr_params]
                        _inf = any(
                            not bool(torch.isfinite(g).all())
                            for g in g_joint_true)
                    if args.rpbe_native_compression \
                            and os.environ.get("CCM_INTERFACE_PROBE") == "1":
                        # Interface-space probe (review 2026-09-23):
                        # compare the predictive adjoint lifted BACK to
                        # memory space, r_pred = J_mem^T a_i, with the
                        # CE gradient AT the memory, r_task =
                        # grad_{M_i} L_task — BEFORE either is pulled
                        # through the LoRA Jacobian.  cos ~ 0 here =>
                        # the phi/J predictive notion itself is
                        # task-orthogonal (supervision geometry);
                        # cos clearly negative here but ~0 in parameter
                        # space => actuation/Jacobian support.
                        _lift = adapter.j_mem
                        cos_if = []
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])
                            optimizer.zero_grad(set_to_none=True)
                            fwd_out = run_forward(model, b, device,
                                                  grad_enabled=True)
                            for _entry in adapter._cache:
                                if _entry is not None:
                                    _entry[0].retain_grad()
                                    _entry[1].retain_grad()
                            _tsm, _nvm = task_ce_shifted(
                                fwd_out, b["labels"], device)
                            (_tsm / max(_nvm, 1)
                             / float(len(pending))).backward()
                            for meta, oid, v in cut_records[i]:
                                _a = g_by_oid.get(oid)
                                if _a is None:
                                    continue
                                _pos = torch.tensor(
                                    [[meta["blocks"][v][1],
                                      meta["blocks"][v][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp, _vp = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp.append(torch.gather(_k.grad, 2,
                                                            _idx))
                                    _vp.append(torch.gather(_v.grad, 2,
                                                            _idx))
                                _r_task = JMemLift.pack_sum_mem(
                                    _kp, _vp)[0]        # [full_dim]
                                _r_pred = _lift.transpose(
                                    _a.unsqueeze(0))[0]  # [full_dim]
                                # dtype align: K/V grads are bf16, the
                                # adjoint is float32.
                                _rt = _r_task.float()
                                _rp = _r_pred.float()
                                _den = (_rt.norm()
                                        * _rp.norm()).clamp(
                                            min=1e-12)
                                cos_if.append(
                                    float((_rt @ _rp) / _den))
                            adapter.clear()
                        cos_if = [float(c) for c in cos_if]
                        _ci = np.array(cos_if)
                        # Block-wise support overlap (review 2026-09-23):
                        # split the flat repr dims by (layer, proj) and
                        # measure per-block energy shares and the
                        # block-level cos(q_i^block, g_task^block).
                        _gt_flat = torch.cat(
                            [g.reshape(-1).float().cpu()
                             for g in g_task_true])
                        _G = torch.stack(dirs)
                        _blocks = []
                        _off = 0
                        for _n, _p in model.named_parameters():
                            if not _p.requires_grad:
                                continue
                            _num = _p.numel()
                            _blocks.append((_n, _off, _off + _num))
                            _off += _num
                        _bnorm_ratio_q = {}
                        _bnorm_ratio_t = {}
                        _bcos = {}
                        _beta_b = {}
                        _gnorms = _G.norm(dim=1).clamp(min=1e-12)
                        for _n, _s, _e in _blocks:
                            _qb = _G[:, _s:_e]
                            _gb = _gt_flat[_s:_e]
                            _bnorm_ratio_q[_n] = float(
                                (_qb.norm(dim=1) / _gnorms).mean())
                            _bnorm_ratio_t[_n] = float(
                                _gb.norm() / _gt_flat.norm().clamp(
                                    min=1e-12))
                            _bcos[_n] = float(
                                ((_qb @ _gb) / (
                                    _qb.norm(dim=1).clamp(min=1e-12)
                                    * _gb.norm().clamp(min=1e-12)))
                                .mean())
                            # eta_b (review 2026-09-23): the block's
                            # TRUE contribution to the global alignment
                            # q^T d / (||q|| ||d||) — a large local cos
                            # on a tiny-energy block contributes ~0.
                            _beta_b[_n] = float(
                                ((_qb @ _gb) / (
                                    _gnorms
                                    * _gt_flat.norm().clamp(
                                        min=1e-12))).mean())
                        _probe = {
                            "interface_cos_mean": float(_ci.mean()),
                            "interface_cos_med": float(np.median(_ci)),
                            "interface_cos_p5": float(
                                np.percentile(_ci, 5.0)),
                            "interface_cos_min": float(_ci.min()),
                            "interface_cos_max": float(_ci.max()),
                            "n_cuts": int(len(cos_if)),
                            "block_norm_ratio_q": _bnorm_ratio_q,
                            "block_norm_ratio_t": _bnorm_ratio_t,
                            "block_cos": _bcos,
                            "block_eta": _beta_b,
                        }
                        with (out / "interface_probe.json").open("w") \
                                as _f:
                            json.dump(_probe, _f, indent=1)
                        _top_q = sorted(
                            _bnorm_ratio_q.items(),
                            key=lambda kv: -kv[1])[:4]
                        _top_t = sorted(
                            _bnorm_ratio_t.items(),
                            key=lambda kv: -kv[1])[:4]
                        print("[interface-probe] n_cuts={} cos: "
                              "mean={:.4f} med={:.4f} p5={:.4f} "
                              "min={:.4f} max={:.4f}".format(
                                  len(cos_if), _probe["interface_cos_mean"],
                                  _probe["interface_cos_med"],
                                  _probe["interface_cos_p5"],
                                  _probe["interface_cos_min"],
                                  _probe["interface_cos_max"]),
                              flush=True)
                        print("[interface-probe] top energy q: {} | "
                              "top energy t: {}".format(
                                  _top_q, _top_t), flush=True)
                        optimizer.zero_grad(set_to_none=True)
                    if args.rpbe_native_compression \
                            and os.environ.get("CCM_S3_PROBE") == "1":
                        # S3 probe (review 2026-09-23): full-context
                        # teacher vs compressed prediction at EVERY cut.
                        # The recursive interface stays Z_t = M_t; what
                        # changes is the future predictive test:
                        #   pi_full_j = p(·| u_{1:L}, c, y_{<j})   (frozen
                        #       teacher, the standard CCM forward at y)
                        #   pi_comp_j = p(·| M_t, u_{t+1:L}, c, y_{<j})
                        #       — a REORGANIZED dialogue: u_1..u_t
                        #       compressed into M_t, the remaining
                        #       suffix as the context turn, y as target
                        #   D_t = mean_j KL(pi_full_j || pi_comp_j)
                        #   a^S3 = grad_{M_t} D_t  (memory space),
                        #   q^S3 = grad_theta D_t (parameter space)
                        # Four numbers vs the CE task direction.
                        _gt_flat = torch.cat(
                            [g.reshape(-1).float().cpu()
                             for g in g_task_true])
                        r_task_cache = {}
                        teacher_cache = {}
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])
                            optimizer.zero_grad(set_to_none=True)
                            fwd_out = run_forward(model, b, device,
                                                  grad_enabled=True)
                            _lg = fwd_out.logits
                            _lb = b["labels"]
                            _sh = _lg[..., :-1, :].contiguous()
                            _sl = _lb[..., 1:].contiguous()
                            _mask = _sl != -100
                            _t_logits = _sh[_mask].detach()
                            for _entry in adapter._cache:
                                if _entry is not None:
                                    _entry[0].retain_grad()
                                    _entry[1].retain_grad()
                            _tsm, _nvm = task_ce_shifted(
                                fwd_out, b["labels"], device)
                            (_tsm / max(_nvm, 1)
                             / float(len(pending))).backward()
                            for meta, oid, v in cut_records[i]:
                                _pos = torch.tensor(
                                    [[meta["blocks"][v][1],
                                      meta["blocks"][v][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp, _vp = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp.append(torch.gather(_k.grad, 2,
                                                            _idx))
                                    _vp.append(torch.gather(_v.grad, 2,
                                                            _idx))
                                r_task_cache[oid] = JMemLift.pack_sum_mem(
                                    _kp, _vp)[0].float()
                            teacher_cache[i] = (_t_logits,
                                               int(_mask.sum()))
                            adapter.clear()
                        cos_mem_s3 = []
                        cos_par_s3 = []
                        q_agg = None
                        for i, (b, sid) in enumerate(pending):
                            for meta, oid, v in cut_records[i]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                _t = int(v) + 1
                                _raw = meta["raw_dialog"]
                                _L = int(meta["L"])
                                _suffix = [tok for _x in
                                           range(_t, _L + 1)
                                           for tok in _raw[_x]]
                                _new_dialog = ([list(_raw[_x])
                                                for _x in range(_t)]
                                               + [_suffix]
                                               + [list(_raw[_L + 1])])
                                _item = {"dialog": _new_dialog,
                                         "fixed_depth": True}
                                _b2 = collator([_item])
                                _b2 = {kk: vv.to(device)
                                       for kk, vv in _b2.items()}
                                optimizer.zero_grad(set_to_none=True)
                                _f2 = run_forward(model, _b2, device,
                                                  grad_enabled=True)
                                for _entry in adapter._cache:
                                    if _entry is not None:
                                        _entry[0].retain_grad()
                                        _entry[1].retain_grad()
                                _lg2 = _f2.logits
                                _lb2 = _b2["labels"]
                                _sh2 = _lg2[..., :-1, :].contiguous()
                                _sl2 = _lb2[..., 1:].contiguous()
                                _mask2 = _sl2 != -100
                                _s_logits = _sh2[_mask2]
                                _t_logits, _n_tok = teacher_cache[i]
                                _n_s = int(_mask2.sum())
                                _n_min = min(_n_tok, _n_s)
                                _kl = F.kl_div(
                                    torch.log_softmax(
                                        _s_logits[:_n_min], dim=-1),
                                    torch.softmax(
                                        _t_logits[:_n_min], dim=-1),
                                    reduction="none").sum(-1)
                                _D = _kl.mean()
                                _D.backward()
                                _meta2 = parse_meta(_b2, comp_ids,
                                                    sum_ids, sid)[0]
                                _pos2 = torch.tensor(
                                    [[_meta2["blocks"][_t - 1][1],
                                      _meta2["blocks"][_t - 1][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp2, _vp2 = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos2.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp2.append(torch.gather(_k.grad, 2,
                                                             _idx))
                                    _vp2.append(torch.gather(_v.grad, 2,
                                                             _idx))
                                _a_s3 = JMemLift.pack_sum_mem(
                                    _kp2, _vp2)[0].float()
                                _r_task = r_task_cache[oid]
                                _den_m = (_a_s3.norm()
                                          * _r_task.norm()).clamp(
                                              min=1e-12)
                                cos_mem_s3.append(
                                    float((_a_s3 @ _r_task) / _den_m))
                                _q_flat = torch.cat(
                                    [p.grad.reshape(-1).float()
                                     if p.grad is not None else
                                     torch.zeros(p.numel(),
                                                 dtype=torch.float32,
                                                 device=p.device)
                                     for p in repr_params]).cpu()
                                _den_p = (_q_flat.norm()
                                          * _gt_flat.norm()).clamp(
                                              min=1e-12)
                                cos_par_s3.append(
                                    float((_q_flat @ _gt_flat)
                                          / _den_p))
                                q_agg = (_q_flat if q_agg is None
                                         else q_agg + _q_flat)
                            adapter.clear()
                        cos_mem_s3 = [float(c) for c in cos_mem_s3]
                        cos_par_s3 = [float(c) for c in cos_par_s3]
                        _cm = np.array(cos_mem_s3)
                        _cp = np.array(cos_par_s3)
                        _agg_cos_s3 = float(
                            ((q_agg @ _gt_flat)
                             / (q_agg.norm()
                                * _gt_flat.norm()).clamp(
                                    min=1e-12))) if q_agg is not None \
                            else 0.0
                        _s3 = {
                            "n_cuts": int(len(cos_mem_s3)),
                            "cos_mem_mean": float(_cm.mean()),
                            "cos_mem_med": float(np.median(_cm)),
                            "cos_mem_p5": float(np.percentile(_cm, 5.0)),
                            "cos_mem_min": float(_cm.min()),
                            "cos_mem_max": float(_cm.max()),
                            "cos_par_mean": float(_cp.mean()),
                            "cos_par_med": float(np.median(_cp)),
                            "cos_par_p5": float(np.percentile(_cp, 5.0)),
                            "cos_par_min": float(_cp.min()),
                            "frac_par_neg_kappa": float(
                                np.mean(_cp < -args.rpbe_kappa)),
                            "cos_agg": _agg_cos_s3,
                        }
                        with (out / "s3_probe.json").open("w") as _f:
                            json.dump(_s3, _f, indent=1)
                        print("[s3-probe] n={} mem_cos: mean={:.4f} "
                              "p5={:.4f} min={:.4f} | par_cos: "
                              "mean={:.4f} p5={:.4f} min={:.4f} "
                              "Pr<-k={:.4f} | AGG={:.4f}".format(
                                  _s3["n_cuts"],
                                  _s3["cos_mem_mean"], _s3["cos_mem_p5"],
                                  _s3["cos_mem_min"],
                                  _s3["cos_par_mean"], _s3["cos_par_p5"],
                                  _s3["cos_par_min"],
                                  _s3["frac_par_neg_kappa"],
                                  _s3["cos_agg"]), flush=True)
                        optimizer.zero_grad(set_to_none=True)
                    if args.rpbe_native_compression \
                            and os.environ.get("CCM_S4_PROBE") == "1":
                        # S4 probe (review 2026-09-23): ground-truth
                        # next-token predictive GAP.  Same geometry as
                        # the final task (real tokens, real logits,
                        # real CE):
                        #   l_full  = -log p(y_j | U_t, C_t, y_<j)
                        #             (the standard CCM forward at y)
                        #   l_comp  = -log p(y_j | M_t, C_t, y_<j)
                        #             (reorganized dialogue: u_1..u_t
                        #             compressed, suffix as context)
                        #   Delta_t = mean_j(l_comp - l_full)
                        #   a^S4 = grad_{M_t} l_comp  (the hinge [.]+
                        #   is a training-time choice; the probe reads
                        #   the raw Delta distribution and the
                        #   directions)
                        # Four numbers + the Delta distribution.
                        _gt_flat = torch.cat(
                            [g.reshape(-1).float().cpu()
                             for g in g_task_true])
                        r_task_cache = {}
                        teacher_ce_cache = {}
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])
                            optimizer.zero_grad(set_to_none=True)
                            fwd_out = run_forward(model, b, device,
                                                  grad_enabled=True)
                            for _entry in adapter._cache:
                                if _entry is not None:
                                    _entry[0].retain_grad()
                                    _entry[1].retain_grad()
                            _tsm, _nvm = task_ce_shifted(
                                fwd_out, b["labels"], device)
                            (_tsm / max(_nvm, 1)
                             / float(len(pending))).backward()
                            teacher_ce_cache[i] = float(
                                _tsm / max(_nvm, 1))
                            for meta, oid, v in cut_records[i]:
                                _pos = torch.tensor(
                                    [[meta["blocks"][v][1],
                                      meta["blocks"][v][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp, _vp = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp.append(torch.gather(_k.grad, 2,
                                                            _idx))
                                    _vp.append(torch.gather(_v.grad, 2,
                                                            _idx))
                                r_task_cache[oid] = JMemLift.pack_sum_mem(
                                    _kp, _vp)[0].float()
                            adapter.clear()
                        delta_list = []
                        cos_mem_s4 = []
                        cos_par_s4 = []
                        q_agg = None
                        for i, (b, sid) in enumerate(pending):
                            for meta, oid, v in cut_records[i]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                _t = int(v) + 1
                                _raw = meta["raw_dialog"]
                                _L = int(meta["L"])
                                _suffix = [tok for _x in
                                           range(_t, _L + 1)
                                           for tok in _raw[_x]]
                                _new_dialog = ([list(_raw[_x])
                                                for _x in range(_t)]
                                               + [_suffix]
                                               + [list(_raw[_L + 1])])
                                _item = {"dialog": _new_dialog,
                                         "fixed_depth": True}
                                _b2 = collator([_item])
                                _b2 = {kk: vv.to(device)
                                       for kk, vv in _b2.items()}
                                optimizer.zero_grad(set_to_none=True)
                                _f2 = run_forward(model, _b2, device,
                                                  grad_enabled=True)
                                for _entry in adapter._cache:
                                    if _entry is not None:
                                        _entry[0].retain_grad()
                                        _entry[1].retain_grad()
                                _lg2 = _f2.logits
                                _lb2 = _b2["labels"]
                                _sh2 = _lg2[..., :-1, :].contiguous()
                                _sl2 = _lb2[..., 1:].contiguous()
                                _mask2 = _sl2 != -100
                                _s_logits = _sh2[_mask2]
                                _s_labs = _sl2[_mask2]
                                _ce_s = F.cross_entropy(_s_logits,
                                                        _s_labs)
                                _delta = float(
                                    _ce_s.detach()) - teacher_ce_cache[i]
                                delta_list.append(_delta)
                                _ce_s.backward()
                                _meta2 = parse_meta(_b2, comp_ids,
                                                    sum_ids, sid)[0]
                                _pos2 = torch.tensor(
                                    [[_meta2["blocks"][_t - 1][1],
                                      _meta2["blocks"][_t - 1][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp2, _vp2 = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos2.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp2.append(torch.gather(_k.grad, 2,
                                                             _idx))
                                    _vp2.append(torch.gather(_v.grad, 2,
                                                             _idx))
                                _a_s4 = JMemLift.pack_sum_mem(
                                    _kp2, _vp2)[0].float()
                                _r_task = r_task_cache[oid]
                                _den_m = (_a_s4.norm()
                                          * _r_task.norm()).clamp(
                                              min=1e-12)
                                cos_mem_s4.append(
                                    float((_a_s4 @ _r_task) / _den_m))
                                _q_flat = torch.cat(
                                    [p.grad.reshape(-1).float()
                                     if p.grad is not None else
                                     torch.zeros(p.numel(),
                                                 dtype=torch.float32,
                                                 device=p.device)
                                     for p in repr_params]).cpu()
                                _den_p = (_q_flat.norm()
                                          * _gt_flat.norm()).clamp(
                                              min=1e-12)
                                cos_par_s4.append(
                                    float((_q_flat @ _gt_flat)
                                          / _den_p))
                                q_agg = (_q_flat if q_agg is None
                                         else q_agg + _q_flat)
                            adapter.clear()
                        delta_list = [float(d) for d in delta_list]
                        cos_mem_s4 = [float(c) for c in cos_mem_s4]
                        cos_par_s4 = [float(c) for c in cos_par_s4]
                        _dd = np.array(delta_list)
                        _cm4 = np.array(cos_mem_s4)
                        _cp4 = np.array(cos_par_s4)
                        _agg_cos_s4 = float(
                            ((q_agg @ _gt_flat)
                             / (q_agg.norm()
                                * _gt_flat.norm()).clamp(
                                    min=1e-12))) if q_agg is not None \
                            else 0.0
                        _s4 = {
                            "n_cuts": int(len(delta_list)),
                            "delta_mean": float(_dd.mean()),
                            "delta_med": float(np.median(_dd)),
                            "delta_p5": float(np.percentile(_dd, 5.0)),
                            "delta_min": float(_dd.min()),
                            "delta_max": float(_dd.max()),
                            "frac_delta_neg": float(
                                np.mean(_dd < 0.0)),
                            "frac_delta_gt_0p1": float(
                                np.mean(_dd > 0.1)),
                            "cos_mem_mean": float(_cm4.mean()),
                            "cos_mem_med": float(np.median(_cm4)),
                            "cos_mem_p5": float(np.percentile(_cm4, 5.0)),
                            "cos_mem_min": float(_cm4.min()),
                            "cos_par_mean": float(_cp4.mean()),
                            "cos_par_p5": float(np.percentile(_cp4, 5.0)),
                            "cos_par_min": float(_cp4.min()),
                            "frac_par_neg_kappa": float(
                                np.mean(_cp4 < -args.rpbe_kappa)),
                            "cos_agg": _agg_cos_s4,
                        }
                        with (out / "s4_probe.json").open("w") as _f:
                            json.dump(_s4, _f, indent=1)
                        print("[s4-probe] n={} delta: mean={:.4f} "
                              "med={:.4f} p5={:.4f} min={:.4f} "
                              "frac_neg={:.4f} frac>0.1={:.4f} | "
                              "mem_cos: mean={:.4f} p5={:.4f} | "
                              "par_cos: mean={:.4f} p5={:.4f} "
                              "Pr<-k={:.4f} | AGG={:.4f}".format(
                                  _s4["n_cuts"], _s4["delta_mean"],
                                  _s4["delta_med"], _s4["delta_p5"],
                                  _s4["delta_min"],
                                  _s4["frac_delta_neg"],
                                  _s4["frac_delta_gt_0p1"],
                                  _s4["cos_mem_mean"],
                                  _s4["cos_mem_p5"],
                                  _s4["cos_par_mean"],
                                  _s4["cos_par_p5"],
                                  _s4["frac_par_neg_kappa"],
                                  _s4["cos_agg"]), flush=True)
                        optimizer.zero_grad(set_to_none=True)
                    if args.rpbe_native_compression \
                            and os.environ.get("CCM_S42OBS_PROBE") == "1":
                        # S4-2Obs probe (review 2026-09-23): KEEP the
                        # original 2Obs recursive topology and swap only
                        # the scoring geometry — every observation
                        # becomes a real-token NLL gap:
                        #   l_comp = -log p(Y | M_t, C),   l_full =
                        #   -log p_ref(Y | U_t, C) with U_t given as an
                        #   UNCOMPRESSED single context turn (a 1-turn
                        #   merge is lossless).
                        #   h=1 local:  C=u_{t+1}, Y=u_{t+2}
                        #   h=2 upward: C=(u_{t+1},u_{t+2}), Y=y
                        #   t=L:       C=c, Y=y (single obs)
                        # Report cos_mem / cos_param / AGG / Pr<-k and
                        # the Delta distribution SEPARATELY per obs and
                        # for the 0.5/0.5 merge.
                        _gt_flat = torch.cat(
                            [g.reshape(-1).float().cpu()
                             for g in g_task_true])
                        r_task_cache = {}
                        for i, (b, sid) in enumerate(pending):
                            _restore_rng(pass1_rngs[i])
                            optimizer.zero_grad(set_to_none=True)
                            fwd_out = run_forward(model, b, device,
                                                  grad_enabled=True)
                            for _entry in adapter._cache:
                                if _entry is not None:
                                    _entry[0].retain_grad()
                                    _entry[1].retain_grad()
                            _tsm, _nvm = task_ce_shifted(
                                fwd_out, b["labels"], device)
                            (_tsm / max(_nvm, 1)
                             / float(len(pending))).backward()
                            for meta, oid, v in cut_records[i]:
                                _pos = torch.tensor(
                                    [[meta["blocks"][v][1],
                                      meta["blocks"][v][1] + 1]],
                                    dtype=torch.long, device=device)
                                _kp, _vp = [], []
                                for _entry in adapter._cache:
                                    _k, _v = _entry[0], _entry[1]
                                    _idx = _pos.unsqueeze(1).expand(
                                        -1, _k.shape[1], -1)
                                    _idx = _idx.unsqueeze(-1).expand(
                                        -1, -1, -1, _k.shape[-1])
                                    _kp.append(torch.gather(_k.grad, 2,
                                                            _idx))
                                    _vp.append(torch.gather(_v.grad, 2,
                                                            _idx))
                                r_task_cache[oid] = JMemLift.pack_sum_mem(
                                    _kp, _vp)[0].float()
                            adapter.clear()

                        def _fwd_ce(dialog_turns):
                            _item = {"dialog": dialog_turns,
                                     "fixed_depth": True}
                            _b2 = collator([_item])
                            _b2 = {kk: vv.to(device)
                                   for kk, vv in _b2.items()}
                            _f2 = run_forward(model, _b2, device,
                                              grad_enabled=True)
                            _lg2 = _f2.logits
                            _lb2 = _b2["labels"]
                            _sh2 = _lg2[..., :-1, :].contiguous()
                            _sl2 = _lb2[..., 1:].contiguous()
                            _mask2 = _sl2 != -100
                            return _f2, _sh2[_mask2], _sl2[_mask2], _b2

                        stats = {h: {"delta": [], "mem_cos": [],
                                     "par_cos": [], "q_agg": None}
                                 for h in (1, 2)}
                        for i, (b, sid) in enumerate(pending):
                            for meta, oid, v in cut_records[i]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                _t = int(v) + 1
                                _raw = meta["raw_dialog"]
                                _L = int(meta["L"])
                                _hist = [list(_raw[_x])
                                         for _x in range(_t)]
                                _hist_full = [tok for _x in range(_t)
                                              for tok in _raw[_x]]
                                _obs = []
                                if _t < _L:
                                    # h=1 local: u_{t+1} -> u_{t+2}
                                    _obs.append((1,
                                                 _hist + [list(_raw[_t])]
                                                 + [list(_raw[_t + 1])],
                                                 [_hist_full]
                                                 + [list(_raw[_t])]
                                                 + [list(_raw[_t + 1])]))
                                    # h=2 upward: (u_{t+1}, u_{t+2}) -> y
                                    _obs.append((2,
                                                 _hist
                                                 + [[tok for _x in
                                                     range(_t, _t + 2)
                                                     for tok in
                                                     _raw[_x]]]
                                                 + [list(_raw[_L + 1])],
                                                 [_hist_full]
                                                 + [[tok for _x in
                                                     range(_t, _t + 2)
                                                     for tok in
                                                     _raw[_x]]]
                                                 + [list(_raw[_L + 1])]))
                                else:
                                    # terminal: c -> y
                                    _obs.append((1,
                                                 _hist + [list(_raw[_L])]
                                                 + [list(_raw[_L + 1])],
                                                 [_hist_full]
                                                 + [list(_raw[_L])]
                                                 + [list(_raw[_L + 1])]))
                                for _h, _comp_dlg, _full_dlg in _obs:
                                    optimizer.zero_grad(set_to_none=True)
                                    _f2, _s_logits, _s_labs, _b2 = \
                                        _fwd_ce(_comp_dlg)
                                    for _entry in adapter._cache:
                                        if _entry is not None:
                                            _entry[0].retain_grad()
                                            _entry[1].retain_grad()
                                    _ce_s = F.cross_entropy(_s_logits,
                                                            _s_labs)
                                    # backward BEFORE the full-context
                                    # forward: the no_grad full forward
                                    # would overwrite the adapter cache
                                    # with graph-less tensors, leaving
                                    # .grad None at the gather below.
                                    _ce_s.backward()
                                    _meta2 = parse_meta(_b2, comp_ids,
                                                        sum_ids, sid)[0]
                                    _L2 = int(_meta2["L"])
                                    _pos2 = torch.tensor(
                                        [[_meta2["blocks"][_L2 - 1][1],
                                          _meta2["blocks"][_L2 - 1][1]
                                          + 1]],
                                        dtype=torch.long, device=device)
                                    _kp2, _vp2 = [], []
                                    for _entry in adapter._cache:
                                        _k, _v = _entry[0], _entry[1]
                                        _idx = _pos2.unsqueeze(1).expand(
                                            -1, _k.shape[1], -1)
                                        _idx = _idx.unsqueeze(-1).expand(
                                            -1, -1, -1, _k.shape[-1])
                                        _kp2.append(torch.gather(
                                            _k.grad, 2, _idx))
                                        _vp2.append(torch.gather(
                                            _v.grad, 2, _idx))
                                    _a_h = JMemLift.pack_sum_mem(
                                        _kp2, _vp2)[0].float()
                                    _q_flat = torch.cat(
                                        [p.grad.reshape(-1).float()
                                         if p.grad is not None else
                                         torch.zeros(p.numel(),
                                                     dtype=torch.float32,
                                                     device=p.device)
                                         for p in repr_params]).cpu()
                                    with torch.no_grad():
                                        _f3, _t_logits, _t_labs, _b3 = \
                                            _fwd_ce(_full_dlg)
                                        _ce_f = F.cross_entropy(
                                            _t_logits, _t_labs)
                                    _delta = float(
                                        _ce_s.detach()) - float(
                                            _ce_f.detach())
                                    stats[_h]["delta"].append(_delta)
                                    _r_task = r_task_cache[oid]
                                    _den_m = (_a_h.norm()
                                              * _r_task.norm()).clamp(
                                                  min=1e-12)
                                    stats[_h]["mem_cos"].append(
                                        float((_a_h @ _r_task) / _den_m))
                                    _den_p = (_q_flat.norm()
                                              * _gt_flat.norm()).clamp(
                                                  min=1e-12)
                                    stats[_h]["par_cos"].append(
                                        float((_q_flat @ _gt_flat)
                                              / _den_p))
                                    _q = stats[_h]["q_agg"]
                                    stats[_h]["q_agg"] = (
                                        _q_flat if _q is None
                                        else _q + _q_flat)
                                    adapter.clear()
                        s42 = {}
                        for _h in (1, 2):
                            _st = stats[_h]
                            _dd = np.array(_st["delta"])
                            _cm = np.array(_st["mem_cos"])
                            _cp = np.array(_st["par_cos"])
                            _qa = _st["q_agg"]
                            _agg = float(
                                ((_qa @ _gt_flat)
                                 / (_qa.norm()
                                    * _gt_flat.norm()).clamp(
                                        min=1e-12))) if _qa is not None \
                                else 0.0
                            s42["obs{}".format(_h)] = {
                                "n": int(len(_st["delta"])),
                                "delta_mean": float(_dd.mean()),
                                "delta_med": float(np.median(_dd)),
                                "delta_p5": float(
                                    np.percentile(_dd, 5.0)),
                                "frac_delta_neg": float(
                                    np.mean(_dd < 0.0)),
                                "frac_delta_gt_0p1": float(
                                    np.mean(_dd > 0.1)),
                                "mem_cos_mean": float(_cm.mean()),
                                "mem_cos_p5": float(
                                    np.percentile(_cm, 5.0)),
                                "par_cos_mean": float(_cp.mean()),
                                "par_cos_p5": float(
                                    np.percentile(_cp, 5.0)),
                                "frac_par_neg_kappa": float(
                                    np.mean(_cp < -args.rpbe_kappa)),
                                "cos_agg": _agg,
                            }
                        # 0.5/0.5 merge (terminal rows count once).
                        _ddm = np.array(stats[1]["delta"]
                                        + stats[2]["delta"])
                        _cmm = np.array(stats[1]["mem_cos"]
                                        + stats[2]["mem_cos"])
                        _cpm = np.array(stats[1]["par_cos"]
                                        + stats[2]["par_cos"])
                        _qam = None
                        for _h in (1, 2):
                            if stats[_h]["q_agg"] is not None:
                                _qam = (stats[_h]["q_agg"] if _qam is None
                                        else _qam + stats[_h]["q_agg"])
                        s42["merge"] = {
                            "n": int(len(_ddm)),
                            "delta_mean": float(_ddm.mean()),
                            "mem_cos_mean": float(_cmm.mean()),
                            "par_cos_mean": float(_cpm.mean()),
                            "frac_par_neg_kappa": float(
                                np.mean(_cpm < -args.rpbe_kappa)),
                            "cos_agg": float(
                                ((_qam @ _gt_flat)
                                 / (_qam.norm()
                                    * _gt_flat.norm()).clamp(
                                        min=1e-12)))
                            if _qam is not None else 0.0,
                        }
                        with (out / "s4_2obs_probe.json").open("w") \
                                as _f:
                            json.dump(s42, _f, indent=1)
                        print("[s4-2obs] " + json.dumps(
                            {k: {kk: (round(vv, 4) if isinstance(
                                vv, float) else vv)
                                 for kk, vv in v.items()}
                             for k, v in s42.items()},
                            indent=1), flush=True)
                        optimizer.zero_grad(set_to_none=True)
                    if args.rpbe_native_compression:
                        # Native proposal-space projection (review
                        # 2026-09-22).  P0 fix (2026-09-22 review): the
                        # dirs-collection loop ends with
                        # zero_grad(set_to_none=True), so NO live p.grad
                        # exists here — computed in the shared
                        # pre-projection block above.
                        if _inf:
                            # AMP overflow: no proposal / no QP / no
                            # update.  scaler backs off; the window
                            # closes like a normal skip.
                            proj_ok, proj_diag = True, {
                                "note": "amp_skip", "cert_fail": False,
                                "amp_skip": True, "n_dirs": 0,
                                "task_norm": 0.0, "b_norm": 0.0}
                            cert_fail = False
                            native_amp_skip = True
                        else:
                            native_amp_skip = False
                            # Global clip, the same threshold as
                            # grad_step, applied to EACH proposal's
                            # gradient source INDEPENDENTLY: g_task is
                            # the exact gradient the task-only arm's
                            # optimizer would consume, g_joint the one
                            # the ours arm consumes.
                            def _clip_gs(gs):
                                _tot = math.sqrt(sum(
                                    float((g.double() ** 2).sum())
                                    for g in gs))
                                if _tot > args.grad_clip:
                                    _c = args.grad_clip / _tot
                                    for g in gs:
                                        g.mul_(_c)
                            _clip_gs(g_joint_true)
                            _clip_gs(g_task_true)
                            native_g_joint = g_joint_true
                            _bc = optimizer_steps_executed + 1
                            # Direction space (warmup fix, review
                            # 2026-09-22): lr is a common scalar factor,
                            # the QP and ||d_task|| are lr-invariant —
                            # project the lr=1 direction, multiply the
                            # real current lr back at write-back (0 at
                            # warmup step 0 keeps the warmup semantics).
                            d0 = adamw_proposal(
                                optimizer, repr_params, g_joint_true,
                                _bc, lr_override=1.0)
                            d_task = adamw_proposal(
                                optimizer, repr_params, g_task_true,
                                _bc, lr_override=1.0)
                            d_task_norm = float(
                                torch.cat([x.reshape(-1) for x in d_task])
                                .double().norm())
                            if audit_rows is not None:
                                # gradient-mass audit (review
                                # 2026-09-24): decompose the effective
                                # predictive gradient mass by
                                # (endpoint depth L, horizon h) and
                                # dump, then exit before the QP.
                                _dt_flat = torch.cat(
                                    [x.reshape(-1).double()
                                     for x in d_task]).cpu()
                                _cells = {}
                                _tot = 0.0
                                for _L, _h, _t, _q in audit_rows:
                                    _qd = _q.double()
                                    _nrm = float(_qd.norm())
                                    # Cut-depth t bucket (review
                                    # 2026-09-24): deep trees
                                    # recursively CONTAIN the shallow
                                    # semantics — the endpoint L
                                    # overcounts long-horizon mass, so
                                    # decompose by the obs's own
                                    # recursion depth t (v is the
                                    # 0-based cut index; t = v + 1).
                                    _tb = ("t1-2" if _t + 1 <= 2
                                           else "t3-5" if _t + 1 <= 5
                                           else "t6-13")
                                    _k = "{}_{}".format(
                                        _tb, "local" if _h == 1
                                        else "root")
                                    _c = _cells.setdefault(
                                        _k, {"N": 0, "sum_norm": 0.0,
                                             "mass2": 0.0, "proj": 0.0})
                                    _c["N"] += 1
                                    _c["sum_norm"] += _nrm
                                    _c["mass2"] += _nrm * _nrm
                                    _c["proj"] += float(
                                        (_qd * _dt_flat).sum())
                                    _tot += _nrm
                                _audit = {"d_task_norm":
                                          float(_dt_flat.norm()),
                                          "total_mass": _tot,
                                          "cells": {}}
                                for _k in sorted(_cells):
                                    _c = _cells[_k]
                                    _cos = (_c["proj"]
                                            / (_c["sum_norm"]
                                               * float(_dt_flat.norm()))
                                            if _c["sum_norm"] > 0
                                            and _dt_flat.norm() > 0
                                            else None)
                                    _audit["cells"][_k] = {
                                        "N": _c["N"],
                                        "sum_norm":
                                            round(_c["sum_norm"], 3),
                                        "mass_share":
                                            round(_c["sum_norm"] / _tot,
                                                  4),
                                        "mass2": round(_c["mass2"], 3),
                                        "proj_d_task":
                                            round(_c["proj"], 3),
                                        "cos_to_d_task":
                                            round(_cos, 6)
                                            if _cos is not None
                                            else None,
                                    }
                                with (out / "gradient_audit.json") \
                                        .open("w") as _f:
                                    json.dump(_audit, _f, indent=1)
                                print("[gradient-audit] dumped {} -> {}"
                                      .format(len(audit_rows),
                                              out / "gradient_audit.json"),
                                      flush=True)
                                sys.exit(0)
                            # Streaming (2026-09-23): pass the dirs
                            # LIST (no stack — the QP streams in
                            # 64-row blocks); the sign flip negates
                            # each row.
                            _G = [-_r for _r in dirs] if dirs else None
                            if getattr(args, "no_qp", False):
                                # No-QP arm (review 2026-09-24 deep-
                                # floor probe): d* = d_0, write the
                                # proposal back directly (the QP's
                                # successful write-back is p.grad =
                                # -d*, so here p.grad = -d_0).
                                with torch.no_grad():
                                    for _p, _dv in zip(repr_params, d0):
                                        _p.grad = (-_dv).to(_p.device)
                                proj_diag = {"note": "no_qp",
                                             "n_dirs":
                                                 len(dirs) if dirs
                                                 else 0}
                                proj_ok = True
                            else:
                                proj_ok, proj_diag = \
                                    treewise_feasibility_projection(
                                        [x.detach().cpu() for x in d0],
                                        repr_params, _G,
                                        args.rpbe_kappa,
                                        iters=args.proj_iters,
                                        cert_tol=1e-4,
                                        b_norm=d_task_norm,
                                        device=device)
                            proj_diag["space"] = "proposal"
                            proj_diag["d_task_norm"] = d_task_norm
                            # Proposal-space conflict statistics (short
                            # test C, review 2026-09-22): the frozen
                            # method works in PROPOSAL space, so the
                            # meaningful conflict measure is
                            # c_i = q_i^T d_task / (||q_i|| ||d_task||)
                            # — "if I took the task-only AdamW update,
                            # how many predictive interfaces would
                            # violate the tolerance band?".  The
                            # corr_ratio from the QP diag is exactly
                            # ||d* - d_0|| / ||d_0|| (corr = d*_qp - t,
                            # ||t|| = ||d_0||).
                            if dirs:
                                _dt_flat = torch.cat(
                                    [x.reshape(-1).float().cpu()
                                     for x in d_task])
                                # Streaming (2026-09-23): per-row cos
                                # values in blocks + the aggregate as a
                                # running sum (no full stack).
                                _c_parts = []
                                _agg = None
                                for _c0 in range(0, len(dirs), 64):
                                    _Gb = torch.stack(
                                        dirs[_c0:_c0 + 64]).float()
                                    _nrb = _Gb.norm(dim=1)
                                    _c_parts.append(
                                        ((_Gb @ _dt_flat)
                                         / (_nrb
                                            * _dt_flat.norm()).clamp(
                                                min=1e-12)))
                                    _sb = _Gb.sum(0)
                                    _agg = (_sb if _agg is None
                                            else _agg + _sb)
                                    del _Gb, _sb
                                _c = torch.cat(_c_parts).numpy()
                                # Aggregate predictive direction vs the
                                # task proposal (supervisor sweep,
                                # review 2026-09-23): distinguishes
                                # "treewise normalization eats the
                                # signal" from "the supervisor itself
                                # produces a task-orthogonal direction".
                                # (dirs are scaled-space gradients; the
                                # cosine is scale-invariant.)
                                _agg_cos = float(
                                    ((_agg @ _dt_flat)
                                     / (_agg.norm()
                                        * _dt_flat.norm()).clamp(
                                            min=1e-12)))
                                proj_diag.update({
                                    "conflict_cos_mean":
                                        float(_c.mean()),
                                    "conflict_cos_med":
                                        float(np.median(_c)),
                                    "conflict_cos_p5":
                                        float(np.percentile(_c, 5.0)),
                                    "conflict_cos_min":
                                        float(_c.min()),
                                    "conflict_frac_neg_kappa":
                                        float(np.mean(
                                            _c < -args.rpbe_kappa)),
                                    "conflict_agg_cos":
                                        _agg_cos,
                                })
                            cert_fail = bool(
                                proj_diag.get("cert_fail"))
                            if not proj_ok:
                                for p in params:
                                    p.grad = None
                            else:
                                # The QP wrote p.grad = the projected
                                # proposal d* for every repr param
                                # (repr_set == all trainable params
                                # under native actuation).
                                pass
                    else:
                        proj_ok, proj_diag = treewise_feasibility_projection(
                            [x.detach().cpu() for x in g_task_repr],
                            repr_params,
                            (torch.stack(dirs)
                             if dirs else None),
                            args.rpbe_kappa, iters=args.proj_iters,
                            # V11 (self-ruled 2026-09-21): the TGN
                            # final-spec 1e-6 certificate is a fp32
                            # contract; CCM runs the QP on fp16
                            # GradScaler-SCALED gradients, so the
                            # achievable max_viol floors at ~1e-5
                            # (measured: 8.37e-06 / 1e-05 across the
                            # first windows).  Keeping 1e-6 made EVERY
                            # window cert_fail and skipped every repr
                            # update — the V11 joint-QP experiment
                            # would degenerate to task-only.  1e-4 is
                            # the fp16-achievable line; the
                            # skip-on-failure semantics is unchanged.
                            cert_tol=1e-4)
                        cert_fail = bool(proj_diag.get("cert_fail"))
                        if not proj_ok:
                            # CERT_FAIL: no constrained update is
                            # executed — zero the non-Gamma grads too
                            # and skip the representation step
                            # (final-spec reviewer requirement; TGN:
                            # "skipping repr step").
                            for p in params:
                                if id(p) in repr_set:
                                    p.grad = None
                                else:
                                    p.grad = torch.zeros_like(p)
                        else:
                            for p in params:
                                if id(p) in repr_set:
                                    continue  # projection already
                                              # wrote .grad
                                p.grad = task_grads.get(
                                    id(p), torch.zeros_like(p))
                                if not args.rpbe_gamma_only:
                                    # R10 structure: discard the
                                    # non-Gamma auxiliary gradient
                                    # (review 2026-09-16)
                                    p.grad = (p.grad + aux_other.get(
                                        id(p),
                                        torch.zeros_like(p)))
                    with (out / "window_diag.jsonl").open("a") as _f:
                        _f.write(json.dumps(_json_safe(
                            {"step": int(step), "event": "treewise_proj",
                             **proj_diag})) + "\n")
                        if _dbdg_diag is not None:
                            _f.write(json.dumps(_json_safe(
                                {"step": int(step), "event": "s4_dbdg",
                                 "abar": _dbdg_abar,
                                 "mults": _dbdg_mults,
                                 "mass_raw":
                                     [round(x, 2) for x in
                                      _dbdg_diag["mass_raw"]],
                                 "mass_active":
                                     [round(x, 2) for x in
                                      _dbdg_diag["mass_active"]],
                                 "pr_delta_pos": [
                                     round(
                                         _dbdg_diag["n_gate"][_g]
                                         / _dbdg_diag["n_tot"][_g], 4)
                                     if _dbdg_diag["n_tot"][_g]
                                     else None
                                     for _g in range(3)],
                                 })) + "\n")
                    if cert_fail:
                        print("[rpbe-cstr] window={} CERT_FAIL "
                              "max_viol={:.3e} skipping repr step".format(
                                  kf_closed + 1,
                                  proj_diag.get("max_viol", float("inf"))),
                              flush=True)
                    aux_terms += n_cut_win
                elif args.rpbe_native_compression \
                        and args.s4_supervisor:
                    # S4 AGGREGATE phase (review 2026-09-23 two-phase
                    # form): after the treewise phase has pruned the
                    # predictive gap, the per-interface conflict dies
                    # out (active -> 0-3) and the QP idles.  The
                    # aggregate phase keeps the lambda-weighted S4
                    # gradient WITHOUT dirs/QP: joint = task + lambda *
                    # sum w_h q_{t,h}, one ordinary AdamW step.  Cost
                    # drops to the comp-dialogue forwards only.
                    pure_acc = {}
                    for i, (b, sid) in enumerate(pending):
                        _restore_rng(pass1_rngs[i])  # P0: mask==pass1
                        task_mean, task_raw, n_valid, _aux, _n = \
                            pass2_one(b, cut_records[i], g_by_oid, 0.0)
                        _before = {
                            id(p): p.grad.detach().clone()
                            for p in params if p.grad is not None}
                        if task_mean.requires_grad:
                            scaler.scale(
                                task_mean
                                / float(len(pending))).backward()
                        for p in params:
                            if p.grad is None:
                                continue
                            _prev = _before.get(id(p))
                            _inc = p.grad.detach() - (
                                _prev if _prev is not None
                                else torch.zeros_like(p.grad))
                            if id(p) in pure_acc:
                                pure_acc[id(p)].add_(_inc)
                            else:
                                pure_acc[id(p)] = _inc.clone()
                        task_sum += float(task_raw.detach())
                        n_tokens += n_valid
                    # Reference table (chunked batch, same as treewise).
                    _ref_table = {}
                    if args.s4_ref_cache:
                        _ref_items = []
                        for _i2 in range(len(pending)):
                            for meta, oid, v in cut_records[_i2]:
                                if g_by_oid.get(oid) is None:
                                    continue
                                for _h, _cd, _fd, _w in s4_obs_list(
                                        meta, int(v)):
                                    _ref_items.append(((oid, _h), _fd))
                        with torch.no_grad():
                            for _c0 in range(0, len(_ref_items),
                                             args.s4_ref_chunk):
                                _chunk = _ref_items[
                                    _c0:_c0 + args.s4_ref_chunk]
                                _b3 = collator(
                                    [{"dialog": _d,
                                      "fixed_depth": True}
                                     for _, _d in _chunk])
                                _b3 = {kk: vv.to(device)
                                       for kk, vv in _b3.items()}
                                _f3 = run_forward(
                                    model, _b3, device,
                                    grad_enabled=False)
                                _sh3 = _f3.logits[..., :-1, :] \
                                    .contiguous()
                                _sl3 = _b3["labels"][..., 1:] \
                                    .contiguous()
                                for _j3, (_key, _d) in \
                                        enumerate(_chunk):
                                    _m3 = _sl3[_j3] != -100
                                    _ref_table[_key] = float(
                                        F.cross_entropy(
                                            _sh3[_j3][_m3],
                                            _sl3[_j3][_m3]))
                    # Per-tree comp forwards, gradients ACCUMULATED
                    # (no per-obs isolation, no dirs).
                    for i, (b, sid) in enumerate(pending):
                        for meta, oid, v in cut_records[i]:
                            if g_by_oid.get(oid) is None:
                                continue
                            _obs = s4_obs_list(meta, int(v))
                            _b2 = collator(
                                [{"dialog": _d, "fixed_depth": True}
                                 for _, _d, _, _ in _obs])
                            _b2 = {kk: vv.to(device)
                                   for kk, vv in _b2.items()}
                            _f2 = run_forward(model, _b2, device,
                                              grad_enabled=True)
                            _sh2 = _f2.logits[..., :-1, :] \
                                .contiguous()
                            _sl2 = _b2["labels"][..., 1:] \
                                .contiguous()
                            for _j, (_h, _cdlg, _fdlg, _w) in \
                                    enumerate(_obs):
                                _m2 = _sl2[_j] != -100
                                _ce_c = F.cross_entropy(
                                    _sh2[_j][_m2], _sl2[_j][_m2])
                                if _ref_table:
                                    _ce_f = torch.tensor(
                                        _ref_table[(oid, _h)],
                                        dtype=_ce_c.dtype,
                                        device=_ce_c.device)
                                else:
                                    with torch.no_grad():
                                        _b3 = collator(
                                            [{"dialog": _fdlg,
                                              "fixed_depth": True}])
                                        _b3 = {kk: vv.to(device)
                                               for kk, vv
                                               in _b3.items()}
                                        _f3 = run_forward(
                                            model, _b3, device,
                                            grad_enabled=False)
                                        _sh3 = _f3.logits[
                                            ..., :-1, :] \
                                            .contiguous()
                                        _sl3 = _b3["labels"][
                                            ..., 1:].contiguous()
                                        _m3 = _sl3 != -100
                                        _ce_f = F.cross_entropy(
                                            _sh3[_m3], _sl3[_m3])
                                _loss_h = scaler.scale(
                                    lambda_kf * _w
                                    * (_ce_c - _ce_f.detach()))
                                _loss_h.backward(
                                    retain_graph=(
                                        _j < len(_obs) - 1))
                        adapter.clear()
                    aux_terms += n_cut_win
                else:
                    for i, (b, sid) in enumerate(pending):
                        _restore_rng(pass1_rngs[i])   # P0: mask == pass1
                        task_mean, task_raw, n_valid, aux, n_terms = \
                            pass2_one(
                                b, cut_records[i], g_by_oid,
                                0.0 if args.arm == "gamma_task_only"
                                else lambda_kf)
                        # Per-microbatch normalization: the official
                        # Trainer scales each microbatch MEAN loss by
                        # 1/accumulation, so the gradient scale is
                        # window-length independent (L6.5 review P0-2).
                        # The KF surrogate is NOT rescaled: it is the
                        # exact window-J gradient.
                        if args.rpbe_gamma_only and n_terms:
                            # R10 structure: task and RPBE backwards are
                            # SPLIT; the auxiliary gradient is kept on
                            # Gamma only, non-Gamma params keep the pure
                            # task gradient (review 2026-09-16: a global
                            # r_eff=0.1 does not mean Gamma received 10%
                            # of the RPBE signal when LoRA/COMP absorb
                            # most of it).
                            if task_mean.requires_grad:
                                # freeze-host fix (see treewise branch).
                                scaler.scale(
                                    task_mean
                                    / float(len(pending))).backward(
                                        retain_graph=True)
                            task_snap = {id(p): p.grad.detach().clone()
                                         for p in params
                                         if p.grad is not None
                                         and id(p) not in repr_set}
                            if os.environ.get("CCM_GRAD_GROUP") == "1":
                                task_snap_all = {
                                    id(p): p.grad.detach().clone()
                                    for p in params if p.grad is not None}
                            if aux.requires_grad:
                                # freeze-host fix: same graph-less case as
                                # the else branch below — nothing to add.
                                scaler.scale(aux).backward()
                            if os.environ.get("CCM_GRAD_GROUP") == "1":
                                # per-group r_eff (review 2026-09-16):
                                # task vs RPBE gradient norms for
                                # Gamma / LoRA / COMP embeddings.  The
                                # aux loss already carries lambda, so
                                # aux_norm/task_norm IS the group r_eff.
                                groups = {"gamma": [], "lora": [],
                                          "comp": [], "other": []}
                                for n, p in model.named_parameters():
                                    if p.requires_grad and p.grad is not None:
                                        if id(p) in repr_set:
                                            groups["gamma"].append((n, p))
                                        elif "lora_" in n:
                                            groups["lora"].append((n, p))
                                        elif "comp" in n or "embedding" in n:
                                            groups["comp"].append((n, p))
                                        else:
                                            groups["other"].append((n, p))
                                _t0 = task_snap_all
                                for _gn, _gps in groups.items():
                                    if not _gps:
                                        continue
                                    _nt = sum(float((_t0[id(p)].double()
                                                     ** 2).sum())
                                              for _n, p in _gps
                                              if id(p) in _t0) ** 0.5
                                    _na = sum(float(((p.grad.detach()
                                                      - _t0.get(id(p),
                                                                torch.zeros_like(p.grad)))
                                                     .double() ** 2).sum())
                                              for _n, p in _gps
                                              if p.grad is not None) ** 0.5
                                    print("GRADGROUP {} n={} "
                                          "|g_task|={:.4e} "
                                          "|g_rpbe|={:.4e} "
                                          "r={:.4f}".format(
                                              _gn, len(_gps), _nt, _na,
                                              _na / max(_nt, 1e-30)),
                                          flush=True)
                            for p in params:
                                if id(p) not in repr_set:
                                    p.grad = task_snap.get(id(p))
                        else:
                            loss = task_mean / float(len(pending)) + aux
                            _t = time.perf_counter()
                            if loss.requires_grad:
                                # freeze-host fix (see treewise branch):
                                # a k<3 batch carries no COMP/SUM rows, so
                                # Gamma never touches its output and the
                                # task term is graph-less; with no cuts
                                # aux is the detached zero from
                                # pass2_one.  The whole loss is then
                                # graph-less and there is nothing to
                                # backprop for this microbatch.
                                scaler.scale(loss).backward()
                            _pf("pass2_bwd", _t)
                        if os.environ.get("CCM_AUX_DIAG") == "1":
                            named = [(n, p)
                                     for n, p in model.named_parameters()
                                     if p.requires_grad
                                     and p.grad is not None]
                            bad = bad_grad_names(named, k=2)
                            if bad:
                                print("AUXDIAG mb={} BAD after bwd: {}"
                                      .format(i, bad), flush=True)
                        task_sum += float(task_raw.detach())
                        n_tokens += n_valid
                        aux_terms += n_terms
                if aux_terms - aux_before != n_cut_win:
                    raise RuntimeError(
                        "not every cut was replayed exactly once: "
                        "replayed={} cuts={}".format(
                            aux_terms - aux_before, n_cut_win))
                if os.environ.get("CCM_AUX_DIAG") == "1":
                    # Scaled by the fp16 scaler at this point; the ratio
                    # task-only vs +aux is what matters.
                    named = [(n, p) for n, p in model.named_parameters()
                             if p.requires_grad]
                    print("AUXDIAG win-grads norm={:.6f} (scaled) "
                          "scale={:.0f} bad={}".format(
                              repr_grad_norm(params), scaler.get_scale(),
                              bad_grad_names(named, k=3)), flush=True)
                _t = time.perf_counter()
                if treewise and cert_fail:
                    # Final-spec CERT_FAIL: the QP certificate did not
                    # hold at 1e-6 — NO representation update is executed
                    # for this window (TGN: "skipping repr step").  The
                    # window still closes: data stream, step counter,
                    # boundary records and checkpoint cadence advance
                    # identically to a successful close.
                    cert_skip_steps += 1
                elif treewise and args.rpbe_native_compression:
                    # Native proposal-space write-back (review
                    # 2026-09-22): the QP wrote the projected proposal
                    # d* into p.grad for every repr param (== all
                    # trainable params).  theta <- theta + d* directly;
                    # the optimizer moments are updated with the
                    # true-value JOINT gradient g_joint so the next
                    # window's proposal starts from a consistent AdamW
                    # state.  No scaler.step() — the parameter update
                    # is the projected proposal itself.
                    if native_amp_skip:
                        # AMP overflow detected at unscale: back the
                        # scale off and discard the update (mirrors
                        # grad_step's skip semantics).  The unscale_
                        # records the per-device inf check the
                        # GradScaler.update() assertion requires.
                        scaler.unscale_(optimizer)
                        scaler.update()
                        optimizer_steps_executed += 1
                        amp_skipped_steps += 1
                    else:
                        with torch.no_grad():
                            b1, b2 = tuple(
                                optimizer.param_groups[0]["betas"])
                            for p, g in zip(repr_params,
                                            native_g_joint):
                                st = optimizer.state[p]
                                # torch only initializes state["step"]
                                # inside optimizer.step(); the manual
                                # write-back never calls step(), so
                                # maintain it here (state_dict /
                                # resume compatibility, 2026-09-23).
                                st["step"] = torch.tensor(
                                    float(optimizer_steps_executed + 1))
                                if "exp_avg" in st:
                                    st["exp_avg"].mul_(b1).add_(
                                        g, alpha=1.0 - b1)
                                else:
                                    st["exp_avg"] = (
                                        (1.0 - b1) * g).clone()
                                if "exp_avg_sq" in st:
                                    st["exp_avg_sq"].mul_(b2).addcmul_(
                                        g, g, value=1.0 - b2)
                                else:
                                    st["exp_avg_sq"] = (
                                        (1.0 - b2) * g * g).clone()
                            _lr_eff = float(
                                optimizer.param_groups[0]["lr"])
                            for p in repr_params:
                                # theta <- theta + lr_eff * d* (QP
                                # wrote the lr=1 projected proposal;
                                # fp32 -> param dtype for the add; the
                                # real current lr restores the warmup
                                # semantics — 0 at warmup step 0).
                                p.data.add_(
                                    (p.grad.to(p.dtype)) * _lr_eff)
                        # The unscale_ records the per-device inf check
                        # GradScaler.update() asserts on (p.grad holds
                        # the already-consumed d* — its rescale is
                        # harmless; the real AMP check ran on the
                        # g_joint snapshot in the projection branch).
                        scaler.unscale_(optimizer)
                        scaler.update()
                        optimizer_steps_executed += 1
                        scheduler.step()
                        scheduler_steps += 1
                    assert optimizer_steps_executed \
                        == scheduler_steps + amp_skipped_steps, (
                            "step counters diverged: executed={} "
                            "sched={} skips={}".format(
                                optimizer_steps_executed, scheduler_steps,
                                amp_skipped_steps))
                else:
                    grad_step()  # counters live inside grad_step (nonlocal)
                _pf("grad_step", _t)
                # CPU-memory hygiene (2026-09-23): the S4-2Obs window
                # builds ~600 collated mini-batches and the QP streams
                # ~100 block transfers per close; the Python-side cycle
                # garbage (autograd graph wrappers, collator temporaries)
                # is not reclaimed by refcounting alone and the RSS
                # crept 98 -> 113GB across windows (the system OOM
                # trigger sits at ~115GB).  A forced collection at the
                # window boundary returns the allocator cache to the
                # reuse pool.
                gc.collect()
                if profile:
                    n_cut = sum(1 for cr in cut_records for _ in cr)
                    parts = ["profile win={} n_mb={} n_cut_mb={}:".format(
                        kf_closed + 1, len(pending), n_cut)]
                    for key in ("pass1_fwd", "collect_rows", "window_add",
                                "close_replay", "pass2_fwd", "collect_z",
                                "surrogate", "pass2_bwd", "grad_step"):
                        if PROF.get(key, 0.0) > 0:
                            parts.append("{}={:.1f}s".format(
                                key, PROF[key]))
                    parts.append("wall={:.1f}s".format(
                        time.time() - t_win_start))
                    print(" ".join(parts), flush=True)
                    for key in PROF:
                        PROF[key] = 0.0
                    t_win_start = time.time()
                _restore_rng(resume_rng)  # data stream continues correctly
                total_task_sum += task_sum
                total_tokens += n_tokens
                total_microbatches += len(pending)
                total_kf += float(sum(closed.values()))
                kf_closed += 1
                step += 1
                # Per-window boundary record (review): every arm hashes
                # (window ordinal, n_mb, n_cut_mb) into boundary_hash so
                # cadence identity is checkable window by window, not
                # only through the final data-flow hash.  n_cut_win was
                # computed before pass 2 (audit #3 replay integrity).
                # A CERT_FAIL close appends ":SKIP" so the hash honestly
                # reflects that this window executed no update.
                _skip_tag = (":SKIP" if (treewise and cert_fail) else "")
                boundary_hash.update(
                    "w{}:{}:{}{}".format(
                        kf_closed, len(pending), n_cut_win,
                        _skip_tag).encode())
                boundary_records.append(
                    "w{}:{}:{}{}".format(
                        kf_closed, len(pending), n_cut_win, _skip_tag))
                _n_rows = sum(
                    2 if int(v) != int(meta["L"]) - 1 else 1
                    for _rec in cut_records for meta, _oid, v in _rec)
                close_depth_window(len(pending), step, n_cut_win, _n_rows)
                pending = []
                cut_records = []
                pass1_rngs = []
                window_start_state = None
                if args.max_windows and step >= args.max_windows:
                    break
            elif len(pending) >= args.max_pending_mbs:
                raise RuntimeError(
                    "degenerate window: {} microbatch collected fewer "
                    "than {} effective cuts".format(
                        len(pending), args.kf_min_cuts))
        else:
            # Review P0-2: the official arm runs on the same dialogue
            # stream and fires on the same adaptive boundary (accumulated
            # effective cuts) as the RPBE arms.  Previously it stepped
            # every grad_accum=128 microbatches while ours/gamma stepped
            # every ~266 (one 128-cut window), so after 1000 steps the
            # first two arms had consumed ~2.07x the task data of
            # ccm_merge — a silent breach of the frozen cadence
            # requirement.  window-matched restores identical task
            # exposure; --merge-cadence official keeps the fixed cadence
            # for the ccm_merge_official reproduction reference only.
            eff = sum(1 for m in metas
                      if m["ok"] and m["L"] >= 1)
            if args.merge_cadence == "window-matched":
                merge_eff_cuts += eff
            fire = (args.merge_cadence == "window-matched"
                    and merge_eff_cuts >= args.kf_min_cuts) \
                or (args.merge_cadence == "official"
                    and sum(int(b["input_ids"].shape[0])
                            for b, _sid in pending) >= args.grad_accum)
            if fire:
                optimizer.zero_grad(set_to_none=True)
                task_sum = 0.0
                n_tokens = 0
                # micro_batch>1: the window objective is the sum of the
                # per-dialogue mean CEs divided by the window's dialogue
                # count — bit-identical to the batch=1 normalization.
                n_dial_win = sum(int(b["input_ids"].shape[0])
                                 for b, _sid in pending)
                for b, sid in pending:
                    fwd_out = run_forward(model, b, device, grad_enabled=True)
                    if args.micro_batch > 1:
                        row_sum, row_n = task_ce_rows(
                            fwd_out, b["labels"], device)
                        row_mean = row_sum / row_n.clamp(min=1)
                        loss_scale = row_mean.sum() / float(
                            max(n_dial_win, 1))
                        task_sum += float(row_sum.detach().sum())
                        n_tokens += int(row_n.sum())
                    else:
                        task_raw, n_valid = task_ce_shifted(
                            fwd_out, b["labels"], device)
                        # Official Trainer protocol (verified against
                        # accelerate 1.14 Accelerator.backward + HF 4.44
                        # in scripts/ccm_parity.py): the per-microbatch
                        # MEAN loss is divided by the window size BEFORE
                        # backward — accelerate does
                        # `loss = loss / self.gradient_accumulation_steps`
                        # inside backward().  Normalizing by len(pending)
                        # makes the task gradient scale independent of
                        # the window length, identical across arms.
                        task_mean = task_raw / max(n_valid, 1)
                        loss_scale = task_mean / float(len(pending))
                        task_sum += float(task_raw.detach())
                        n_tokens += n_valid
                    scaler.scale(loss_scale).backward()
                grad_step()  # counters live inside grad_step (nonlocal)
                total_task_sum += task_sum
                total_tokens += n_tokens
                total_microbatches += n_dial_win
                step += 1
                if args.merge_cadence == "window-matched":
                    # Same per-window boundary record as the RPBE arms
                    # (review): (ordinal, n_mb, n_cut_mb) per window.
                    print("win={} n_mb={} n_cut_mb={} (matched cadence)"
                          .format(step, len(pending), merge_eff_cuts),
                          flush=True)
                    boundary_hash.update("w{}:{}:{}:".format(
                        step, len(pending), merge_eff_cuts).encode())
                    boundary_records.append("w{}:{}:{}:".format(
                        step, len(pending), merge_eff_cuts))
                    merge_eff_cuts = 0
                close_depth_window(len(pending), step, merge_eff_cuts)
                pending = []
                if args.max_windows and step >= args.max_windows:
                    break
        if (step and step != last_logged_step
                and step % args.log_every == 0):
            # L7 smoke: without the dedup this block re-prints the same
            # row on every non-fire loop iteration (~260 rows/window).
            last_logged_step = step
            elapsed = time.time() - t_start
            row = {"step": step,
                   "optimizer_steps_executed": optimizer_steps_executed,
                   "amp_skipped_steps": amp_skipped_steps,
                   "cert_skip_steps": cert_skip_steps,
                   "scheduler_steps": scheduler_steps,
                   "task_ce_token": total_task_sum / max(total_tokens, 1),
                   "task_microbatches": total_microbatches,
                   "task_valid_tokens": total_tokens,
                   "kf_closed": kf_closed,
                   "kf_score": total_kf / max(kf_closed, 1),
                   "aux_terms": aux_terms, "lambda": lambda_kf,
                   "elapsed": elapsed}
            print("step={} executed={} skips={} cert={} sched={} "
                  "ce_token={:.4f} "
                  "mbs={} kf_closed={} kf_score={:.4f} aux={} sec={:.1f}"
                  .format(step, optimizer_steps_executed,
                          amp_skipped_steps, cert_skip_steps,
                          scheduler_steps,
                          row["task_ce_token"], total_microbatches,
                          kf_closed, row["kf_score"], aux_terms, elapsed),
                  flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        if (args.checkpoint_every and step and step % args.checkpoint_every == 0
                and step != last_saved_step):
            # per-N-step immutable snapshot (eval-safe: no overwrite race);
            # save exactly once per step (review 2026-09-05: the old guard
            # re-saved on every remaining microbatch of the window)
            last_saved_step = step
            save_trainable(out / f"checkpoint_step{step}.pt", model, step=step,
                           optimizer=optimizer.state_dict(),
                           scaler=scaler.state_dict(),
                           scheduler=scheduler.state_dict(),
                           builder_oid=builder.next_oid if builder else 0,
                           lambda_kf=lambda_kf,
                           sample_cursor=sample_cursor,
                           rng=_rng_state(),
                           optimizer_steps_executed=optimizer_steps_executed,
                           amp_skipped_steps=amp_skipped_steps,
                           cert_skip_steps=cert_skip_steps,
                           scheduler_steps=scheduler_steps,
                           total_task_sum=total_task_sum,
                           total_tokens=total_tokens,
                           total_microbatches=total_microbatches,
                           total_kf=total_kf,
                           kf_closed=kf_closed,
                           aux_terms=aux_terms,
                           data_flow_len=data_flow_len,
                           boundary_records=boundary_records)
    save_trainable(out / "final.pt", model, step=step, lambda_kf=lambda_kf)
    summary = {
        "arm": args.arm, "seed": args.seed,
        # Review ruling: "steps" is the global window-attempt counter
        # (HF Trainer global_step semantics: advances on AMP skips too);
        # the frozen 1000-step budget is THIS counter (1000 global steps),
        # not "1000 non-zero-LR updates".  optimizer_steps_executed counts
        # optimizer.step() invocations (includes the lr=0 warmup-first
        # execution), amp_skipped_steps the GradScaler overflows, and
        # scheduler_steps is an INDEPENDENT counter that must equal
        # executed - skips (Trainer 4.44.2 protocol, asserted per step).
        "steps": step,
        "optimizer_steps_executed": optimizer_steps_executed,
        "amp_skipped_steps": amp_skipped_steps,
        "cert_skip_steps": cert_skip_steps,
        "scheduler_steps": scheduler_steps,
        "amp_scale": float(scaler.get_scale()),
        # Token-normalized CE: the only cross-arm comparable task number
        # (L6.5 review; per-microbatch sums vary with the window length).
        "mean_task_ce_per_token": total_task_sum / max(total_tokens, 1),
        "task_microbatches": total_microbatches,
        "task_valid_tokens": total_tokens,
        "kf_closed": kf_closed,
        "mean_kf_score": total_kf / max(kf_closed, 1),
        "aux_terms": aux_terms, "lambda_kf": lambda_kf,
        "rpbe_lr": (float(args.rpbe_lr)
                    if args.rpbe_lr is not None else float(args.lr)),
        "paired_seed_hash": paired_seed_hash(args.seed, model)
        if use_rpbe else "n/a",
        "data_flow_hash": data_flow_hash.hexdigest(),
        "data_flow_len": data_flow_len,
        "boundary_hash": boundary_hash.hexdigest(),
    }
    save_json(out / "summary.json", summary)
    save_json(out / "_SUCCESS.json", {"status": "complete", "steps": step,
                                      "optimizer_steps_executed":
                                      optimizer_steps_executed,
                                      "amp_skipped_steps":
                                      amp_skipped_steps,
                                      "cert_skip_steps": cert_skip_steps,
                                      "scheduler_steps":
                                      scheduler_steps})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
