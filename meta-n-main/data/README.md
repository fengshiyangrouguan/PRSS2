# `data/` — benchmark datasets

Benchmark datasets are **not** bundled in this repository. Each integration
expects its data under `./data/<benchmark>` by default (overridable with
`--bench-data-dir`); this directory is populated on demand:

| Path | Source | How to obtain |
| --- | --- | --- |
| `co_bench/` | CO-Bench (36 OR tasks) | download from the upstream CO-Bench release and place here |
| `arc_agi_2/` | [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) | `bash scripts/setup_arc_agi_2.sh` |
| `openevolve/` | [algorithmicsuperintelligence/openevolve](https://github.com/algorithmicsuperintelligence/openevolve) | `bash scripts/setup_openevolve.sh` (experiments used upstream `80945ed`) |
| `AlgoTune/` | [oripress/AlgoTune](https://github.com/oripress/AlgoTune) | clone upstream (experiments used commit `9e27099`) |
| `text_classification/` | Symptom2Disease (HF `gretelai/symptom_to_diagnosis`), LawBench 3-3 ([open-compass/LawBench](https://github.com/open-compass/LawBench)) | auto-downloaded on first run |
| `.cache/` | HuggingFace dataset cache | gitignored, regenerated on first `load_dataset()` |

## Notes

- **Symbolic regression**: after `setup_openevolve.sh`, generate the problem
  set with `python scripts/generate_synthetic_sr.py` (writes to
  `data/openevolve/examples/symbolic_regression/problems/`).
- **TerminalBench 2.0 / SWE-bench Verified** need no files here — the adapter
  self-downloads task packages via `harbor` (requires the `harbor` extra and
  the Docker CLI on PATH).
- Some openevolve scripts expect AlgoTune at `./AlgoTune` when run from
  `data/openevolve/`; if needed: `ln -s ../AlgoTune data/openevolve/AlgoTune`.
  Not required for `meta_n` integrations — they reference `data/AlgoTune`
  directly.
- The HuggingFace cache under `data/.cache/` is safe to delete and regenerate.
