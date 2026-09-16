# results/ — recovered Stage8 artifacts

These files were reconstructed from the working session transcript after the
AutoDL box became unreachable (see `STAGE8_RESULTS.md`).

- `STAGE8_RESULTS.md` — the full record: method, val curves, rollout curve,
  constraint audit, and an explicit list of what is lost.
- `val_curves.csv` — per-run validation (action loss) curves.
- `recovered_rollout.txt` / `recovered_tierdist.txt` — every
  `weighted_success` / `tier_dist` line that was read during the runs. These
  lines carry the arm name inline, so they need no attribution guesswork.

Not recoverable: raw `train.log` files and all checkpoints (they existed only
on the box).
