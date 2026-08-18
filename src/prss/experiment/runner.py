"""Stdlib-only experiment matrix runner.

An experiment YAML declares scalar ``defaults`` plus a ``matrix`` of lists; the
Cartesian product expands into jobs.  Each job runs ``scripts.train`` in a
subprocess with its own output directory; completion is marked by
``_SUCCESS.json`` (idempotent skip), failures are archived as ``*__failed__*``
and do not abort the matrix.
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_experiment(yaml_path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required for the experiment runner")
    with open(yaml_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if "defaults" not in spec or "matrix" not in spec:
        raise ValueError("experiment YAML needs 'defaults' and 'matrix' sections")
    return spec


def expand(spec: dict):
    """Cartesian product of the matrix lists -> list of job dicts."""
    keys = list(spec["matrix"].keys())
    values = [spec["matrix"][k] for k in keys]
    jobs = []
    for combo in itertools.product(*values):
        job = dict(spec["defaults"])
        job.update(dict(zip(keys, combo)))
        jobs.append(job)
    return jobs


def job_id(job: dict) -> str:
    return "{dataset}__{variant}__seed{seed:03d}".format(
        dataset=job.get("dataset", "tgbl-wiki"),
        variant=job.get("variant", "spectral"),
        seed=int(job.get("seed", 0)))


def run_job(job: dict, yaml_path: str, root: Path, gpu: int) -> int:
    out_dir = root / job_id(job)
    if (out_dir / "_SUCCESS.json").exists():
        print(f"SKIP (complete): {out_dir.name}", flush=True)
        return 0
    cmd = [sys.executable, "-m", "scripts.train", "--config", yaml_path]
    for key, value in job.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    cmd += ["--gpu", str(gpu), "--output", str(out_dir)]
    print(f"RUN: {out_dir.name}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(Path(yaml_path).resolve().parents[1]))
    if proc.returncode == 0 and (out_dir / "_SUCCESS.json").exists():
        print(f"DONE: {out_dir.name} ({time.time() - t0:.0f}s)", flush=True)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    failed_dir = root / f"{out_dir.name}__failed__{stamp}"
    if out_dir.exists():
        out_dir.rename(failed_dir)
    print(f"FAILED: archived to {failed_dir.name} (exit {proc.returncode})", flush=True)
    return 1


def run_matrix(yaml_path: str, gpu: int = 0, only: str = "") -> int:
    spec = load_experiment(yaml_path)
    root = Path(spec.get("root", "outputs"))
    root.mkdir(parents=True, exist_ok=True)
    jobs = expand(spec)
    if only:
        jobs = [j for j in jobs if job_id(j) == only]
    failures = 0
    for job in jobs:
        failures += run_job(job, yaml_path, root, gpu)
    print(f"matrix complete: {len(jobs)} jobs, {failures} failed", flush=True)
    return failures


def main():
    p = argparse.ArgumentParser("PRSS2 experiment matrix runner")
    p.add_argument("--config", required=True, help="experiment YAML")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only", default="", help="run a single job id")
    args = p.parse_args()
    sys.exit(run_matrix(args.config, gpu=args.gpu, only=args.only))


if __name__ == "__main__":
    main()
