"""Patch LIBERO's get_task_init_states for torch >= 2.6.

torch 2.6 flipped `torch.load`'s `weights_only` default to True, but the
`.pruned_init` files are numpy pickles, so loading them raises:

    _pickle.UnpicklingError: Weights only load failed ... Unsupported global:
    GLOBAL numpy.core.multiarray._reconstruct was not an allowed global

The files are trusted local artefacts, so load them with weights_only=False.

(This path is only reached if the evaluator falls back to the benchmark's own
init-state loader; `tiered_eval_t1.py` normally reads `metainfo.json` instead.
Apply it anyway so the fallback works.)
"""
import pathlib
import shutil
import sys

P = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                 "/root/autodl-tmp/libero-mem/libero/libero/benchmark/__init__.py")

src = P.read_text(encoding="utf-8")
OLD = "        init_states = torch.load(init_states_path)\n"
NEW = "        init_states = torch.load(init_states_path, weights_only=False)\n"

if src.count(NEW) == 1 and src.count(OLD) == 0:
    print("already patched:", P)
    raise SystemExit(0)

n = src.count(OLD)
if n != 1:
    raise SystemExit(f"ABORT: found {n} matches, expected 1 in {P}")

shutil.copy2(P, str(P) + ".bak")
P.write_text(src.replace(OLD, NEW), encoding="utf-8")
print("patched:", P)
print("backup :", str(P) + ".bak")
