"""Compare matched vanilla, response-only, and full PRSS node-classification runs."""
import argparse
import json
import math
from pathlib import Path


def load_run(path):
  with open(Path(path) / "results.json", encoding="utf-8") as handle:
    return json.load(handle)


def metric_delta(a, b):
  """Return a-b with NLL sign left explicit (negative is better for a)."""
  return {
    "test_auc": a["test"]["auc"] - b["test"]["auc"],
    "test_ap": a["test"]["ap"] - b["test"]["ap"],
    "test_nll": a["test"]["nll"] - b["test"]["nll"],
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vanilla", required=True)
  parser.add_argument("--response-only")
  parser.add_argument("--prss", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  vanilla = load_run(args.vanilla)
  full = load_run(args.prss)
  response = load_run(args.response_only) if args.response_only else None
  report = {
    "vanilla": vanilla,
    "prss_full": full,
    "delta_full_minus_vanilla": metric_delta(full, vanilla),
    "interpretation": (
      "Preliminary single-seed matched result. Full-vs-response-only isolates the quotient/spectral "
      "increment from auxiliary deep supervision; it is not multi-seed paper evidence."),
  }
  if response is not None:
    report["response_only"] = response
    report["delta_response_only_minus_vanilla"] = metric_delta(response, vanilla)
    report["delta_full_minus_response_only"] = metric_delta(full, response)
  out = Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2), encoding="utf-8")
  compact = {
    key: value for key, value in report.items() if key.startswith("delta_")
  }
  print(json.dumps(compact, indent=2))


if __name__ == "__main__":
  main()
