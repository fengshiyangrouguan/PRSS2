# Vendored from snu-mllab/Context-Memory

- Source: https://github.com/snu-mllab/Context-Memory
- Commit: a89dd08e2c9587ec9c6c3ad339bb154c33e6b41a ("add comment")
- License: MIT (see LICENSE)
- Vendored for the CCM-merge × RPBE cross-domain experiment
  (docs/CCM_RPBE_plan_v2.md).  Modifications are kept minimal and
  documented: the Gamma residual hooks the arithmetic merge in
  src/arch/ccm_llama.py, and the dialogue split is restored in
  src/data/dialogue/data.py (see the plan's L0/L1).
