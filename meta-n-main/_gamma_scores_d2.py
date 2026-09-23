"""Raw 6-dim Gamma scores for the d1->d2 decision, and their margin.

The user's question: are assignment / assortment / bin / common_due kept because
Gamma scores them CLEARLY higher, or only barely higher? That single fact decides
which layer to fix:

    large gap  s_(4) - s_(5) >> 0   -> Gamma is confident; the problem is a
                                       learned preference / limited view
    small gap  s_(4) - s_(5) ~ 0    -> hard Top-4 is turning a weak preference
                                       into a fully deterministic routing, and
                                       proposal-level diversification is the
                                       natural fix

It also reports, for SEVERAL DIFFERENT pools, whether the ranking moves at all --
because "Gamma keeps the same 4" is only a defect if the ranking is content-INSENSITIVE.
Across sibling pools the ranking MUST be identical (same pool), so the interesting
variation is across DIFFERENT parents.

Scores reported three ways, because "the score" is not unique here:
    A[k, i]        the raw attention the fusion emits, per slot
    sum_k A[k,i]   total attention mass per trace  (the headline s_i)
    max_k A[k,i]   best single slot's preference
Delta_4_5 is reported for the headline s_i. Entropy is over softmax(s_i) at T=0.
"""
import glob
import json
import os
import sys

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
OUT = "/root/autodl-tmp/sri_r3_seed0"
CKPT = ("/root/autodl-tmp/rpbe-sri/meta-n-main/runs/"
        "_gamma_mt_gem_traceonly_clean_phaseb.pt")
sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")

import torch                                                    # noqa: E402
from meta_n.core.meta_layer import Trace                        # noqa: E402
from meta_n.rpbe.checkpoint import load_gamma_checkpoint        # noqa: E402
from meta_n.rpbe.encoder import FrozenItemEncoder               # noqa: E402
from meta_n.rpbe.selector import PredictiveSelector             # noqa: E402
from meta_n.utils.context_manager import ContextBudget          # noqa: E402


def load_pool(cdir):
    out = []
    for p in sorted(glob.glob(os.path.join(cdir, "traces", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out.append(Trace(**{k: v for k, v in d.items()
                            if k in Trace.model_fields}))
    return out


enc = FrozenItemEncoder(os.environ["CODEBERT_PATH"])
fusion, meta = load_gamma_checkpoint(CKPT)
print("Gamma: params=%s steps=%s checksum=%s"
      % (meta.param_count, meta.steps, meta.checksum))
print()

POOLS = []
for arm in ("official", "predictive"):
    for cid in ("gen0_seed", "gen1_b1_k1", "gen2_b1_k1", "gen3_b0_k0"):
        d = os.path.join(OUT, "arms", arm, "archive", cid)
        if os.path.isdir(d):
            POOLS.append((arm, cid, d))

for arm, cid, cdir in POOLS:
    pool = load_pool(cdir)
    if len(pool) < 4:
        continue
    X = enc.encode(pool)
    q_emb = enc.encode_query(enc.serialize_query(pool))
    with torch.no_grad():
        _, A = fusion(X, q_emb, None)
    A = A.squeeze(0) if A.dim() == 3 else A          # [4, n]
    mass = A.sum(dim=0)                              # per-trace total attention
    best = A.max(dim=0).values
    order = torch.argsort(mass, descending=True)
    names = [t.task_id for t in pool]

    sel = PredictiveSelector().reduce(list(pool), [], A, ContextBudget())
    picked = sorted(t.task_id for t in sel[0])

    print("=" * 96)
    print("POOL %s / %s   (n=%d)" % (arm, cid, len(pool)))
    print("=" * 96)
    print("  raw attention A[k,i] (rows=slots, cols=trace):")
    print("      %s" % " ".join("%14s" % n[:14] for n in names))
    for k in range(A.shape[0]):
        print("   k=%d %s" % (k, " ".join("%14.5f" % float(v) for v in A[k])))
    print("  %-6s %14s %14s" % ("", "sum_k A[k,i]", "max_k A[k,i]"))
    for i, n in enumerate(names):
        print("   %-6s %14.5f %14.5f"
              % (n[:14], float(mass[i]), float(best[i])))
    print()
    print("  ranking by total attention mass:")
    s = [float(mass[i]) for i in range(len(names))]
    ranked = sorted(zip(s, names), reverse=True)
    for r, (v, n) in enumerate(ranked, 1):
        print("    s_(%d) = %.6f   %s" % (r, v, n))
    gap = ranked[3][0] - ranked[4][0]
    keep = ranked[3][0]
    print()
    print("  DELTA_4_5 = s_(4) - s_(5) = %.6f    (%.2f%% of s_(4))"
          % (gap, 100.0 * gap / keep if keep else float("nan")))
    p = torch.softmax(mass, dim=0)
    ent = float(-(p * torch.log(p + 1e-12)).sum())
    print("  entropy(softmax(mass), T=1) = %.4f nats   (ln 6 = %.4f = uniform)"
          % (ent, float(torch.log(torch.tensor(6.0)))))
    print("  selector actually kept : %s" % ",".join(t[:14] for t in picked))
    print("  top-4 by mass         : %s"
          % ",".join(n[:14] for _, n in ranked[:4]))
    print()
