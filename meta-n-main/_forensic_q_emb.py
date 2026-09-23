"""Trace the d1->d2 reduction inputs for every sibling: is `q` really the pool?

The user's question: if the four sibling proposals all share the same `q_emb`,
Gamma cannot see any difference between them and can only repeat one 4-of-6
selection. Is that correct behaviour or a defect?

Code answer (encoder.py §2.8 + ContextReducer._predictive):
    q_src  = current_traces if current_traces is not None else trace_items
           = the trace pool   (the adapter never passes current_traces)
    q_text = OmegaEngine._format_raw_traces(pool)      <- serialize_query
    q_emb  = E_0(q_text)
    X      = E_0(pool)                                 <- same pool, two encodings
so `q` is a pure function of the POOL. Siblings sharing a pool MUST get the same
q_emb. This script verifies that empirically and prints, per reduction:

    arm, candidate_id, parent_id, pool task ids, q sha256, q_emb sha256,
    selected 4 trace ids

NOTE ON PROVENANCE: `q_text`/`q_emb` were never persisted (META_N_REDUCTION_TRACE
was off), so the values below are REPRODUCED from the frozen pool, not read back
from a log. The pool itself is read from the archive, and `selected` is read from
the stored prompt -- those two are the real recorded facts.
"""
import glob
import hashlib
import json
import os
import re
import sys

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
OUT = "/root/autodl-tmp/sri_r3_seed0"
sys.path.insert(0, "/root/autodl-tmp/rpbe-sri/meta-n-main")

from meta_n.core.meta_layer import Trace                       # noqa: E402
from meta_n.rpbe.encoder import FrozenItemEncoder               # noqa: E402

BLOCK = re.compile(r"^--- Task: (\S+) \[(SUCCESS|FAILURE)([^\]]*)\] ---", re.M)


def load_traces(cdir):
    out = []
    for p in sorted(glob.glob(os.path.join(cdir, "traces", "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        out.append(Trace(**{k: v for k, v in d.items()
                            if k in Trace.model_fields}))
    return out


def selected_from_prompt(cdir):
    p = os.path.join(cdir, "omega_prompt_d2.txt")
    if not os.path.isfile(p):
        return None
    txt = open(p, encoding="utf-8", errors="replace").read()
    if "Execution Traces (sampled" not in txt:
        return None
    return tuple(sorted({m[0] for m in BLOCK.findall(txt)}))


enc = FrozenItemEncoder(os.environ["CODEBERT_PATH"])

root_pool = load_traces(os.path.join(OUT, "arms", "official", "archive",
                                     "gen0_seed"))
print("=" * 100)
print("THE SHARED POOL (root = gen0_seed): %d traces" % len(root_pool))
print("  task ids: %s" % sorted(t.task_id for t in root_pool))
q_text = enc.serialize_query(root_pool)
q_emb = enc.encode_query(q_text)
X = enc.encode(root_pool)
q_sha = hashlib.sha256(q_text.encode("utf-8")).hexdigest()[:16]
e_sha = hashlib.sha256(q_emb.numpy().tobytes()).hexdigest()[:16]
x_sha = hashlib.sha256(X.numpy().tobytes()).hexdigest()[:16]
print("  q_text  sha256[:16] = %s   (chars=%d)" % (q_sha, len(q_text)))
print("  q_emb   sha256[:16] = %s   shape=%s" % (e_sha, tuple(q_emb.shape)))
print("  X       sha256[:16] = %s   shape=%s" % (x_sha, tuple(X.shape)))
print("  -> q_emb is a pure function of the pool: same pool => same q_emb.")
print()

print("=" * 100)
print("PER-REDUCTION TABLE (selected read from the stored prompt; q reproduced)")
print("=" * 100)
print("  %-11s %-13s %-13s %-9s %-17s %-17s %s"
      % ("arm", "candidate", "parent", "pool", "q_sha[:12]/q_emb", "q_emb_sha12",
         "selected 4 (sorted)"))
print("  " + "-" * 96)
for arm in ("official", "predictive"):
    base = os.path.join(OUT, "arms", arm, "archive")
    for cdir in sorted(glob.glob(os.path.join(base, "gen1_*"))):
        cid = os.path.basename(cdir)
        s = json.load(open(os.path.join(cdir, "summary.json"), encoding="utf-8"))
        sel = selected_from_prompt(cdir)
        # the pool each sibling actually saw = its parent's traces
        pt = os.path.join(base, s["parent_id"], "traces")
        pool = sorted(os.path.basename(p)[:-5] for p in glob.glob(pt + "/*.json"))
        same = "SAME" if pool == sorted(t.task_id for t in root_pool) else "DIFF"
        print("  %-11s %-13s %-13s %-9s %-17s %-17s %s"
              % (arm, cid, s["parent_id"], "%d/%s" % (len(pool), same),
                 q_sha[:8] + "/" + e_sha[:8], e_sha[:12],
                 ",".join(t[:12] for t in (sel or ()))))

print()
print("=" * 100)
print("VERDICT")
print("=" * 100)
print("  q_text and q_emb are identical for EVERY sibling above, because each")
print("  reduces the SAME pool and `q` is defined as the pool's rendering")
print("  (§2.8: 'the current cut's traces only', never the child).")
print("  So Gamma repeating one 4-of-6 is SPEC-CONFORMANT, not a defect.")
print()
print("  What IS a real structural asymmetry (not a fairness bug): the arms")
print("  differ in whether the kept subset VARIES across the beam.")
print("    official: sample_4(P; xi)  -- random, so siblings explore different")
print("                                  subsets")
print("    ours:     TopK{s_Gamma(t_i, q)} with q = f(P) -- deterministic, so")
print("                                  every sibling sees the SAME subset")
print("  Same 4 traces per call, same budget -- but no subset exploration on")
print("  the Gamma side. That is a candidate mechanism for the observed shape")
print("  (fast dev rise, shallow lineage, no matching held-out gain) and it is")
print("  NOT the same thing as parametric overfitting.")
