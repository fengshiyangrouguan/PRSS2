"""Frozen constants for the Meta^n x RPBE port.

Source of truth: TASKBOOK_v4.1.md §1, §1.1, §1.3.
Nothing here may be tuned on results; every value is either a scaled reference
constant or a quantity frozen before any run.
"""

from __future__ import annotations

import torch

# --- data / window (v4.1 §1) -------------------------------------------------
N_RUNS_BUDGET = 64         # Phase A Official-reduction Omega run BUDGET (the
                           # planned sample size). v4.2: NOT a requirement that
                           # all 64 succeed -- a shortfall is labelled 'pilot'.
DATA_WINDOW_PAIRS = 1      # one mutually-exclusive task/protect window pair
TRAIN_STEPS = 300          # Phase B offline Gamma steps
N_TREES_TASK = 32
N_TREES_PROTECT = 32
N_TREE_MIN = 32            # formal-data TARGET = unique tree_id (v4.2: a
                           # readiness status, never a training gate)

# --- measurement (v4.1 §1) ---------------------------------------------------
M_SKETCH = 8               # P dim per LPSE branch
B_V_DIM = 32               # b_v dim; 16 is an ABLATION only (changes d_b:m)
N_BRANCHES = 4             # branches stay independent
D_E = 256                  # fusion dim; never the LLM hidden size
H_LORA = 64                # low-rank bottleneck for W_K / W_V
N_SLOTS = 4
COND_RANK = 32             # Cond is low-rank: B_c A_c
MAX_DEPTH = 10             # from evolutionary_orchestrator.max_depth

# --- proposal-space QP (Stage8 frozen) ---------------------------------------
KAPPA = 0.02               # frozen geometric calibration; never tuned on results
TAU = 3e-4                 # certificate tolerance only
RIDGE_EPS = 1e-3           # numerical-safety ridge inside the CCA Cholesky.
#                            Only keeps the factorization alive: the small-sample
#                            load is carried by OAS shrinkage (kf._oas_shrink),
#                            never by this ridge. Value taken from the frozen
#                            PRSS2 configs/ccm/frozen_method.json (rpbe.ridge_eps).

# --- misc --------------------------------------------------------------------
GAMMA_BUDGET = 1_000_000
TEMPERATURE = 1.0          # fixed; no annealing
GAMMA_INIT_SCALE = 0.02

# --- seeds (v4.1 §1.3) -------------------------------------------------------
GAMMA_INIT_SEED = 0
P_B_SEED = 0
P_RC_SEED = 0
R_S_SEEDS = (0, 1, 2, 3)

# --- optimizer (v4.1 §1.3) ---------------------------------------------------
OPTIMIZER = "AdamW"
LR = 3e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.0
FOREACH = False
GRAD_CLIP = 1.0
SCHEDULER = "cosine"
WARMUP_STEPS = 9
TOTAL_STEPS = 300

# --- frozen text encoder (v4.1 §1.1) -----------------------------------------
ENCODER_MODEL_ID = "microsoft/codebert-base"
ENCODER_REVISION = "3b0952feddeffad0063f274080e3c23d75e7eb39"
ENCODER_HIDDEN = 768
ENCODER_MAX_TOKENS = 256   # per chunk, special tokens included


def gamma_param_count(d_e: int = D_E, h: int = H_LORA, n_slots: int = N_SLOTS,
                      cond_rank: int = COND_RANK) -> int:
    """Exact Gamma parameter count (v4.1 §1.2).

    Cond: A_c [cond_rank, d_e] + B_c [n_slots*d_e, cond_rank]
    W_K = A_K B_K, W_V = A_V B_V, each d_e*h + h*d_e
    w_g [d_e], Q_0 [n_slots, d_e], LayerNorm weight+bias [d_e] each
    """
    cond = cond_rank * d_e + (n_slots * d_e) * cond_rank
    wk = d_e * h + h * d_e
    wv = d_e * h + h * d_e
    return cond + wk + wv + d_e + n_slots * d_e + 2 * d_e


def assert_gamma_budget(gamma, *, expected: int | None = None) -> int:
    """Assert the frozen Gamma parameter budget; return the counted total."""
    if expected is None:
        expected = gamma_param_count()
    n = sum(p.numel() for p in gamma.parameters())
    if n != expected:
        raise AssertionError(
            "Gamma parameter budget violated: counted {} != frozen {}".format(
                n, expected))
    if n >= GAMMA_BUDGET:
        raise AssertionError(
            "Gamma parameter count {} exceeds hard budget {}".format(
                n, GAMMA_BUDGET))
    return n


def assert_initial_state(gamma, *, tol: float = 0.0) -> None:
    """Assert the frozen initialisation (v4.1 §1.3).

    w_g must be exactly 0 (so g = sigmoid(0) = 0.5 and log(g+eps) is a
    constant across items); LayerNorm weight exactly 1, bias exactly 0.
    """
    wg = getattr(gamma, "w_g", None)
    if wg is not None and not bool((wg == 0).all()):
        raise AssertionError("w_g must be initialised to exactly 0")
    ln = getattr(gamma, "ln", None)
    if ln is not None:
        if not bool(torch.allclose(
                ln.weight.detach(),
                torch.ones_like(ln.weight.detach()), atol=tol)):
            raise AssertionError("LayerNorm weight must be exactly 1")
        if not bool(torch.allclose(
                ln.bias.detach(),
                torch.zeros_like(ln.bias.detach()), atol=tol)):
            raise AssertionError("LayerNorm bias must be exactly 0")
