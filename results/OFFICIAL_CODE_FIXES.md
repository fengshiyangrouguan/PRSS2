# Implementation-口径 fixes made before any comparison was meaningful

Every number in `STAGE8_RESULTS.md` sits downstream of the fixes below.
They are split into three groups on purpose:

- **A — issues in the official code we received** (MemoryVLA / LIBERO-Mem
  harness). These are not our bugs; they are what the released code does.
- **B — bugs we introduced ourselves**, in the RPBE plugin or in our edits to
  the official trainer. Listed because they affect results too, and because
  hiding them would be dishonest.
- **C — reporting definitions** we had to pin down because the official
  harness leaves them implicit.

Each entry names the commit that fixed it.

---

## A. Issues in the official code as received

### A1. The rollout harness never advanced the subgoal machine — every result was 0/3

`tiered_eval.py`. The LIBERO-Mem episode machine only increments its
satisfied-subgoal set when `env._check_success(inc=True)` is called during the
episode. The released rollout path never called it, so `_satisfied_subgoals`
stayed empty and **every demo scored 0 cycles, structurally, regardless of the
policy**. On top of that, the official `reset()` does not clear `_overshot`.

Fix (`638fa05`): call `base._check_success(inc=True)` before every physical
step, `reset_subgoal_progress()` + `_overshot = False` at episode start, and
report the `{0,1,2,3}` tier distribution. Smoke: `gamma-rpbe` demo_1 went
0/3 -> 3/3.

**This is the single largest 口径 issue.** Until it was fixed, the entire
three-arm comparison ("all arms score 0") was an artifact of the evaluator.

### A2. The rollout harness truncated trajectories at intermediate successes

`tiered_eval.py`. `env.step` returns `done=True` when *an intermediate*
subgoal is reached, and the old loop broke on `done`. That cuts the episode
short exactly when the policy is doing well.

Fix (`5a5c6d7`): removed every `done`-break; rollouts run a fixed length.

### A3. Train/inference conditioning mismatch in `predict_action`

`vla/memory_vla.py`. `predict_action` built its prompt *without* the empty gpt
turn, appended manual `[29871, 2]` tokens, ran `generate()`, and took the last
generated token's hidden state as the cognition token. Training instead runs a
plain causal forward over a prompt-only sequence (human + empty gpt turns via
`add_special_tokens`) and gathers the cognition token at the last true prompt
position, after slicing the vision prefix. The two paths therefore conditioned
the action head on **different sequences** — 29 vs 30 input ids.

Fix (`ef70105`): inference now mirrors training. Verified by a conditioning
alignment test (frame 0, eval, empty memory, same image): before the fix
`cog_pre` max abs diff 7.9; after, `input_ids` / `pixel_values` /
`cog_pre` / `cog_post` / `per_pre` / `per_post` are bitwise equal.

Honest scope: this was real but **not the bottleneck** — teacher-forced gripper
transitions moved ~80 -> ~70 and accuracy +1 point, xyz unchanged. Fixed for
correctness, not for gain.

### A4. Dataset windowing did not match official dense training

`vla/datasets/hdf5_dataset.py`. The released dataset code does not produce the
dense stride-1 sliding window (one training sample per frame) that the
official recipe trains with, and it does not pad the tail so that the last
frames still yield a full future-action window.

Fix (`c8c73d1`, `7380730`): stride-1 dense windowing (276 frames -> 276
samples), neutral tail padding with a padded `valid_mask`, `timestep = t`
ordering, and tail padding that writes physical zeros into the first six dims.

### A5. Gripper label convention

The environment emits gripper in `{-1, +1}`; the model is trained in `{0, 1}`.
The released path did not relabel, and it normalised the gripper dimension with
the stats of the other six.

Fix (`c8c73d1`): `env{-1,+1} -> model{0,1}` relabel, `mask[6] = False` so the
gripper is never normalised, stats computed over all training frames.

### A6. `max-steps` did not mean optimizer steps

`--max-steps` counted raw micro-steps, not optimizer updates, so a "15000
step" budget was not comparable across configurations with different
`grad_accum`.

Fix (`c8c73d1`): `max-steps` is optimizer steps; the scheduler denominator is
`ceil(max_steps / grad_accum)`.

### A7. Image augmentation geometry and photometrics were off

`_augment`. The crop kept 81% of the area instead of the intended 90%
(edge length `sqrt(0.9)`, not `0.9`), and brightness was multiplicative
instead of the official additive form.

Fix (`726efe7`): edge length `sqrt(0.9)`, additive brightness in `[-0.2, 0.2]`,
order crop -> brightness -> contrast -> saturation -> hue.

### A8. Random crop position injected spatial label noise

Follow-on from A7. A random crop *window* moves the pixels while the action
label is untransformed, i.e. it adds spatial noise to the regression target.
The actor turned out to be highly pixel-sensitive: a centre crop at eval alone
moved weighted success 5.3% -> 12.3%.

Fix (`493e2b8`): fixed-centre 90%-area crop; photometric jitter kept (still
deterministic per `(seed, epoch, eid, t)`).

### A9. The diffusion action head is zero-initialised, which silently breaks gradient-based calibration

The official DiT head zero-initialises its final layer (a diffusion
convention). Consequence: at `theta_0` the leaf gradients are exactly zero, so
any gradient-based calibration or audit run at step 0 measures **zero** and
looks like a broken graph.

Fix (`740a67a`): diagnostics warm up 20 optimizer steps before measuring
(`0.035 -> 0.14`). Not a code change to the model — a measurement protocol
change, but one that would otherwise have produced a wrong conclusion
("leaf gradients are disconnected").

---

## B. Bugs we introduced ourselves (in the plugin / our trainer edits)

### B1. The avg arm's LoRA was never updated

The macro-boundary counter `episodes_since_boundary` was only incremented
inside `feed_merges_and_futures()`, which we gated on the arm being a Gamma
arm. For the avg arm (`gamma = None`) the function never ran, the counter
stayed 0, the boundary never fired, and `opt_repr.step()` was never called.
Evidence: `param_version = 0` and **all 224 `lora_B` tensors with norm 0** in
the avg checkpoints — the LLM was frozen and only the randomly initialised
action head and memory modules had trained.

Fix (`ff8ed58`): single-point arm semantics (`IS_GAMMA` / `IS_RPBE` /
`IS_DENSE` / `IS_AVG_BOUNDARY`) replacing every scattered `arm == "avg"`;
avg-boundary drives the same macro clock by pure episode counting; LoRA
gradients normalised by `scale = grad_accum / window_micro` before the
boundary; the repr scheduler indexed by real `param_version` rather than batch
step.

### B2. Gamma was frozen by its own initialisation

`gamma_alpha_init = 0.0`. With `alpha = 1 + U = 0`, the update rule reproduces
the official `AvgMerge` — but `dL/dU = alpha * h = 0`, so the MLP gradient
chain is permanently closed and U never leaves zero. Every `final_*` Gamma
result to that point was effectively a frozen merge operator.

Fix (`e9eb1dc`): `alpha_init = 1.0` with `U = 0` (same initial output, open
learning path). Verified by a learning-path test (`1e012e5`).

### B3. The RPBE replay had the wrong sign

The replay loss was minimised where the objective calls for maximising `J`.

Fix (`7380730`): negate the replay term.

### B4. Only the optimizer's own gradients were cleared

Under `--train-scope lora-gamma` the banks/DiT were not in the task optimizer,
so `opt_task.zero_grad()` left their grads accumulating. Two harms: unbounded
growth, and — worse — the Gamma task cotangents are read straight off the
bank-leaf `.grad`, so the cotangents were **polluted by stale accumulated
gradients**.

Fix (`29c1368`): `vla.zero_grad(set_to_none=True)`, a superset of the old call.

### B5. Checkpoint saving poisoned the live optimizer

`_opt_state()` did an in-place `.cpu()` on tensors shared with the live
optimizer, permanently moving `exp_avg` / `exp_avg_sq` to CPU; the next
optimizer step crashed on a device mismatch.

Fix (`7b9a791`): `deepcopy` first, then move.

### B6. Global RNG was consumed by construction and by eval

- `GammaMerger` construction drew from the global RNG, so the three arms were
  **not** identically initialised (`91e92b9`).
- `run_eval` consumed the global CUDA RNG (the diffusion loss uses
  `torch.randn_like` / `torch.randint`), so evaluating perturbed the training
  stream (`ff8ed58`).

Fix: local generator for construction; full snapshot/restore of Python, NumPy,
CPU-torch and CUDA RNG around eval, plus save/restore of the cognition and
per-episode banks, `try/finally` so exceptions restore too.

### B7. Gamma parameters overlapped the task parameter set

Fix (`7380730`): gamma excluded from `task_params` with a disjointness
assertion.

### B8. Merge records leaked GPU memory / crossed epochs

`MergeRecord` tensors were held on GPU; `merge_registry` / `merge_id_map` were
never pruned; `ep_merge_count` accumulated across epochs and shrank tree
weights.

Fix (`7380730`): `.detach().cpu()` on records, prune by surviving episodes,
pop on drain.

---

## C. Reporting definitions the official harness leaves implicit

- **`weighted_success = sum(tier) / (3 * n)`** (`372ed54`) — tier counts
  completed 3-cycle repetitions, so a 2/3 demo scores 2. Without this, "1
  cycle" and "3 cycles" counted the same.
- **`strict_success`** (`5a5c6d7`): tier 3 *and* not overshot *and*
  `check_success(inc=False)`. Reported separately from `tier3_reached`,
  because reaching three cycles after overshooting is not the same result.
- **Per-demo paired seeds** (`5a5c6d7`): the diffusion seed is derived from the
  demo index and reset per demo, so the same demo is comparable across
  checkpoints. Without it, cross-checkpoint comparisons carry rollout noise.
- **Eval state rollback** (`1571858` B6): the cognition and per-episode banks
  are deep-copied and restored around eval. The released harness `reset()`s
  the per bank instead of restoring it, so evaluating mutated training state.

---

## What this means for the results

Two of the official-code items (A1, A2) are **measurement-breaking**: before
them, the whole three-arm rollout comparison sat at a structural 0 and could
not have shown a difference if one existed. One of ours (B1) is
**training-breaking** for the control arm: the "official host" had a frozen
LLM, so it was not the official architecture being compared.

Everything after those fixes is what `STAGE8_RESULTS.md` reports. Items A3, A7
and A8 were verified to matter little or moderately; they are fixed for
correctness, not because they moved the headline number.
