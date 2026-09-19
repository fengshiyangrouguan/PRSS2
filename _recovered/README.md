# `_recovered/` — raw recovery scratch

Everything here was scraped out of the working session transcripts
(`~/.claude/projects/D--RPBE/*.jsonl`) after the AutoDL box holding the
Stage8 `outputs/` became unreachable. It is the provenance trail behind the
numbers in `../results/`.

- `scan_rollout.py` — the scanner: walks every transcript, finds a regex, and
  prints each hit with its surrounding context (run name, command, log tail).
  Usage: `python scan_rollout.py "<regex>" <window-chars>`.
- `block_<run>.txt` — per-run concatenations of every transcript line that
  mentions that run.
- `curves_{val,train,by_header}.json` — curves grouped by the header they were
  printed under. **These are positionally grouped, not attribution-safe**: a
  header can capture lines belonging to a different run, so use them only to
  locate values, never to attribute them. `../results/val_curves.csv` is the
  attributed version.
- `ctx_*.txt` — context dumps for specific searches (`ctx_s5r`, `ctx_tier`,
  `ctx_17k`, ...). Scratch; regenerate rather than trust.

**Credentials were redacted** (`<REDACTED>`) before the first commit of this
directory. No secret in here ever entered git history.
