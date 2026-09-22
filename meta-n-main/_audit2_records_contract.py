"""Issue #4, decisive: is `_gem6_traceonly.jsonl` really built by the CURRENT
(trace-only) build_records.py contract?

The filename says "traceonly" and the build log says "wrote 17 records", but a
name is not evidence and the log does not print X_v widths. The mtimes are
ambiguous in the other direction too: the records file is 22:01 and
build_records.py is 22:16, so the file was written BEFORE the source's last
edit -- which settles nothing, because mtime says when a file was last written,
not what contract it held.

So: rebuild the records from the same 6 archives with the CURRENT builder and
compare the result to the stored file, field by field. Zero API, zero model.

    identical X_v widths and values        -> the file IS the current contract
    stored widths == n_traces + n_codes    -> the file is the RETIRED contract
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CODEBERT_PATH", "/root/autodl-tmp/models/codebert-base")
TREE = "/root/autodl-tmp/rpbe-sri/meta-n-main"
sys.path.insert(0, TREE)

import torch                                                     # noqa: E402
from meta_n.rpbe.build_records import build_run_records, cut_items  # noqa: E402
from meta_n.rpbe.encoder import FrozenItemEncoder                # noqa: E402
from meta_n.rpbe.future import FutureSketcher                    # noqa: E402

STORED = "/root/autodl-tmp/_gem6_traceonly.jsonl"
OLD = "/root/autodl-tmp/_gem5_records.jsonl"
ARCH = sorted(glob.glob(
    "/root/autodl-tmp/meta-n-main/runs/mtgem_run*/*co_bench_gemini-3.1-pro"))


def stored_rows(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


enc = FrozenItemEncoder(os.environ["CODEBERT_PATH"])
sk = FutureSketcher(enc)

print("=" * 78)
print("A. what the ARCHIVES actually hold (independent of any records file)")
print("=" * 78)
print("  %-30s %-22s %6s %6s %8s" % ("run", "candidate", "traces", "codes",
                                     "items"))
for a in ARCH:
    rid = os.path.basename(a)
    for cd in sorted(glob.glob(os.path.join(a, "archive", "*"))):
        if not os.path.isdir(cd):
            continue
        cid = os.path.basename(cd)
        try:
            items = cut_items(Path(a), cid)
            ntr2 = sum(1 for i in items if type(i).__name__ == "Trace")
            ncd = len(items) - ntr2
        except Exception as e:                                # noqa: BLE001
            print("  %-30s %-22s   ERR %s" % (rid[:30], cid[:22],
                                              type(e).__name__))
            continue
        print("  %-30s %-22s %6d %6d %8d" % (rid[:30], cid[:22], ntr2, ncd,
                                             len(items)))

print()
print("=" * 78)
print("B. stored file vs a FRESH rebuild under the CURRENT builder")
print("=" * 78)

fresh = []
for a in ARCH:
    fresh.extend(build_run_records(a, enc, sk))
print("  rebuilt %d records from %d archives" % (len(fresh), len(ARCH)))

for path in (STORED, OLD):
    if not os.path.isfile(path):
        continue
    rows = stored_rows(path)
    print()
    print("  %s  (%d records)" % (path, len(rows)))
    sw = sorted({len(r["X_v"]) for r in rows})
    print("    stored X_v widths     : %s" % sw)
    print("    fresh  X_v widths     : %s"
          % sorted({int(r.X_v.shape[0]) for r in fresh}))
    # match by (run_id, candidate_id, occurrence_seq)
    fidx = {(r.meta.run_id, r.meta.candidate_id.split(":", 1)[-1],
             r.occurrence_seq): r for r in fresh}
    same_w = same_v = miss = 0
    for r in rows:
        m = r["meta"]
        key = (m["run_id"], m["candidate_id"].split(":", 1)[-1],
               r["occurrence_seq"])
        f = fidx.get(key)
        if f is None:
            miss += 1
            continue
        if int(f.X_v.shape[0]) == len(r["X_v"]):
            same_w += 1
            if torch.allclose(f.X_v, torch.tensor(r["X_v"]), atol=1e-5):
                same_v += 1
    print("    matched keys          : %d (missing %d)" % (len(rows) - miss, miss))
    print("    same X_v WIDTH        : %d / %d" % (same_w, len(rows)))
    print("    same X_v VALUES       : %d / %d" % (same_v, len(rows)))
    verdict = ("IS the CURRENT trace-only contract"
               if same_w == len(rows) and same_v == len(rows)
               else "is NOT the current contract (or not a full match)")
    print("    VERDICT               : %s" % verdict)
