"""Build a paired, multi-seed PRSS mechanism-evidence report from completed runs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REQUIRED_VARIANTS = (
  "response_only", "fixed_random", "pca", "direct", "linear_reader_svd",
  "no_nonlinear_lift", "neural_svd_no_spec", "full",
)
PRIMARY_COMPARATORS = ("response_only", "fixed_random", "pca", "direct")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--runs-root", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--bootstrap-samples", type=int, default=10000)
  parser.add_argument("--seed", type=int, default=2026)
  return parser.parse_args()


def mean_std(values):
  values = np.asarray(values, dtype=float)
  return {"mean": float(values.mean()),
          "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
          "n": int(len(values))}


def bootstrap_mean_ci(values, samples, rng):
  values = np.asarray(values, dtype=float)
  if len(values) == 0:
    return None
  draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
  return {
    "mean": float(values.mean()),
    "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    "n_pairs": int(len(values)),
    "win_rate": float((values > 0).mean()),
    "paired_values": values.tolist(),
  }


def load_records(root):
  records = []
  for results_path in sorted(Path(root).rglob("results.json")):
    config_path = results_path.with_name("config.json")
    success_path = results_path.with_name("_SUCCESS.json")
    if not config_path.exists() or not success_path.exists():
      continue
    with open(results_path, encoding="utf-8") as handle:
      results = json.load(handle)
    with open(config_path, encoding="utf-8") as handle:
      config = json.load(handle)
    variant = results.get("variant", config.get("variant"))
    if variant not in REQUIRED_VARIANTS:
      continue
    interface_types = config.get("prss_interface_types") or list(results["spectral"])
    if not interface_types:
      raise ValueError("No PRSS interfaces recorded in {}".format(results_path))
    root_tau = interface_types[-1]
    if root_tau not in results["spectral"]:
      raise KeyError("Root interface {} missing from spectral results {}".format(
        root_tau, results_path))
    spectral = results["spectral"][root_tau]
    host_dim = int(spectral["host_dim"])
    energy_key = "energy@{}".format(host_dim)
    test = results["test"]
    unrestricted = test.get("unrestricted_response_nll")
    structured = test.get("structured_response_nll")
    monitor_path = results_path.parent / "monitor" / "monitor_summary.json"
    monitor = None
    if monitor_path.exists():
      with open(monitor_path, encoding="utf-8") as handle:
        monitor = json.load(handle)
    records.append({
      "path": str(results_path),
      "variant": variant,
      "seed": int(config["seed"]),
      "host_dim": host_dim,
      "candidate_dim": int(spectral["candidate_dim"]),
      "test_ap": float(test["ap"]),
      "test_auc": float(test["auc"]),
      "structured_response_nll": structured,
      "unrestricted_response_nll": unrestricted,
      "structured_unrestricted_ratio": (
        float(structured / max(unrestricted, 1e-12))
        if structured is not None and unrestricted is not None else None),
      "predictive_energy_coverage": test.get(
        "predictive_energy_coverage", {}).get(root_tau),
      "root_interface": root_tau,
      "operator_statistic_energy_at_host_k": float(spectral["energy"][energy_key]),
      "tail_at_host_k": float(spectral["tail"]["tail@{}".format(host_dim)]),
      "gram_updates": int(spectral["reader_gram_updates"]),
      "spectral_updates": int(spectral["spectral_updates"]),
      "inference_contract": results["inference_contract"],
      "monitor_health": monitor.get("health_status") if monitor else "missing",
    })
  return records


def summarize(records, bootstrap_samples=10000, seed=2026):
  by_variant = defaultdict(list)
  for record in records:
    by_variant[record["variant"]].append(record)
  summaries = {}
  for variant, values in by_variant.items():
    summaries[variant] = {
      metric: mean_std([value[metric] for value in values if value[metric] is not None])
      for metric in ("test_ap", "test_auc", "structured_response_nll",
                     "unrestricted_response_nll", "structured_unrestricted_ratio",
                     "predictive_energy_coverage", "operator_statistic_energy_at_host_k",
                     "tail_at_host_k")
      if any(value[metric] is not None for value in values)
    }
    summaries[variant]["seeds"] = sorted(value["seed"] for value in values)

  rng = np.random.default_rng(seed)
  paired = {}
  full_by_seed = {record["seed"]: record for record in by_variant.get("full", [])}
  for comparator in PRIMARY_COMPARATORS:
    comparator_by_seed = {record["seed"]: record for record in by_variant.get(comparator, [])}
    common = sorted(set(full_by_seed) & set(comparator_by_seed))
    paired["full_minus_{}".format(comparator)] = {}
    for metric in ("test_ap", "test_auc", "predictive_energy_coverage"):
      differences = [
        full_by_seed[item][metric] - comparator_by_seed[item][metric]
        for item in common
        if full_by_seed[item][metric] is not None and comparator_by_seed[item][metric] is not None
      ]
      paired["full_minus_{}".format(comparator)][metric] = bootstrap_mean_ci(
        differences, bootstrap_samples, rng)

  missing = [variant for variant in REQUIRED_VARIANTS if variant not in by_variant]
  full = by_variant.get("full", [])
  invariant_checks = {
    "all_required_variants_present": not missing,
    "full_has_reader_gram_updates": bool(full) and all(item["gram_updates"] > 0 for item in full),
    "full_has_spectral_updates": bool(full) and all(item["spectral_updates"] > 0 for item in full),
    "full_structured_reader_is_close_to_unrestricted": bool(full) and all(
      item["structured_unrestricted_ratio"] is not None and
      item["structured_unrestricted_ratio"] <= 2.0 for item in full),
    "full_has_held_out_predictive_coverage": bool(full) and all(
      item["predictive_energy_coverage"] is not None for item in full),
    "full_monitor_has_no_errors": bool(full) and all(
      item["monitor_health"] in ("passed", "passed_with_warnings") for item in full),
    "validation_test_did_not_update_gram_or_svd": bool(full) and all(
      not item["inference_contract"].get("validation_test_gram_updated", True) and
      not item["inference_contract"].get("validation_test_svd_updated", True) for item in full),
    "standard_inference_excludes_outside_and_updates": bool(full) and all(
      not item["inference_contract"].get("standard_inference_outside_reader_used", True) and
      not item["inference_contract"].get("standard_inference_gram_updated", True) and
      not item["inference_contract"].get("standard_inference_svd_updated", True)
      for item in full),
    "full_and_direct_dimensions_match": bool(full) and bool(by_variant.get("direct")) and
      {(item["host_dim"], item["candidate_dim"]) for item in full} ==
      {(item["host_dim"], item["candidate_dim"]) for item in by_variant["direct"]},
  }
  primary_cis = [paired["full_minus_{}".format(name)]["test_ap"]
                 for name in PRIMARY_COMPARATORS]
  energy_cis = [paired["full_minus_{}".format(name)]["predictive_energy_coverage"]
                for name in PRIMARY_COMPARATORS]
  means_positive = all(item is not None and item["mean"] > 0
                       for item in primary_cis + energy_cis)
  cis_positive = all(item is not None and item["ci95"][0] > 0
                     for item in primary_cis + energy_cis)
  health_ok = all(invariant_checks.values())
  if health_ok and cis_positive:
    status = "supported_by_completed_paired_ablations"
  elif health_ok and means_positive:
    status = "promising_but_ci_crosses_zero"
  elif missing:
    status = "incomplete_evidence_missing_runs"
  else:
    status = "not_supported_by_current_runs"
  return {
    "evidence_status": status,
    "required_variants": list(REQUIRED_VARIANTS),
    "missing_variants": missing,
    "variant_summaries": summaries,
    "paired_effects": paired,
    "invariant_checks": invariant_checks,
    "claims_boundary": {
      "health_checks_prove_execution_correctness": True,
      "single_run_proves_method_superiority": False,
      "positive_mean_is_not_significance": True,
      "recommended_minimum": "at least 3 matched seeds; prefer 5+ for stable intervals",
      "vanilla_original_tgn": "must be reported separately in the paper task-performance table",
    },
    "records": records,
  }


def markdown(report):
  lines = [
    "# PRSS multi-seed evidence report", "",
    "Evidence status: **{}**".format(report["evidence_status"]), "",
    "## Variant summary", "",
    "| variant | seeds | AP mean±std | AUC mean±std | held-out predictive coverage | response ratio |", 
    "|---|---:|---:|---:|---:|---:|",
  ]
  for variant in REQUIRED_VARIANTS:
    item = report["variant_summaries"].get(variant)
    if item is None:
      lines.append("| {} | missing | — | — | — | — |".format(variant))
      continue
    def formatted(metric):
      value = item.get(metric)
      return "{:.5f}±{:.5f}".format(value["mean"], value["std"]) if value else "—"
    lines.append("| {} | {} | {} | {} | {} | {} |".format(
      variant, len(item["seeds"]), formatted("test_ap"), formatted("test_auc"),
      formatted("predictive_energy_coverage"), formatted("structured_unrestricted_ratio")))
  lines.extend(["", "## Paired full-model effects", "",
                "| comparison | ΔAP mean [95% CI] | Δpredictive coverage [95% CI] | AP win rate | matched seeds |",
                "|---|---:|---:|---:|---:|"])
  for comparison, values in report["paired_effects"].items():
    value = values["test_ap"]
    energy = values["predictive_energy_coverage"]
    if value is None or energy is None:
      lines.append("| {} | — | — | — | 0 |".format(comparison))
    else:
      lines.append("| {} | {:.5f} [{:.5f}, {:.5f}] | {:.5f} [{:.5f}, {:.5f}] | {:.1%} | {} |".format(
        comparison, value["mean"], value["ci95"][0], value["ci95"][1],
        energy["mean"], energy["ci95"][0], energy["ci95"][1],
        value["win_rate"], value["n_pairs"]))
  lines.extend(["", "## Invariant and evidence gates", ""])
  for name, passed in report["invariant_checks"].items():
    lines.append("- [{}] `{}`".format("x" if passed else " ", name))
  lines.extend(["", "A healthy single run does not prove PRSS superiority. The superiority claim "
                "requires matched seeds, positive paired effects, and intervals that do not cross zero. "
                "Original vanilla TGN remains a separate required task-performance baseline.", ""])
  return "\n".join(lines)


def main():
  args = parse_args()
  records = load_records(args.runs_root)
  report = summarize(records, args.bootstrap_samples, args.seed)
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with open(output, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
  with open(output.with_suffix(".md"), "w", encoding="utf-8") as handle:
    handle.write(markdown(report))
  print(json.dumps({
    "evidence_status": report["evidence_status"],
    "missing_variants": report["missing_variants"],
    "output": str(output.resolve()),
  }, indent=2))


if __name__ == "__main__":
  main()
