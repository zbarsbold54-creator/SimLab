# Diagnostics

## Nondeterministic multi-worker PPO training (fixed, det-v0)

### Symptom
Same config + same seed produced different weights when training was run twice
in separate processes at `workers > 1`. `workers=1` was fully reproducible.

### Localization (CPU, obs_only, Balanced 5, seed 7, 2 iterations, two runs each)

```text
workers=1   max_abs_diff = 0.000e+00
workers=2   max_abs_diff = 8.131e-04   (before fix)
workers=4   max_abs_diff = 6.796e-03   (before fix)
```

`workers=1` deterministic, `workers>1` not => the cause is in multi-worker
coordination, not in a single worker's rollout or in the PPO update itself.

### Cause
`PPOTrainer._collect` read rollout results from the shared `mp.Queue` with
`results = [self._result_q.get() for _ in self._procs]` and then concatenated
segments in that **arrival order**. `mp.Queue.get()` returns whichever worker
finished first, so the segment concatenation order (and therefore the
advantage array ordering and minibatch chunking) depended on nondeterministic
worker completion timing. Each result carries its `rank`, but it was ignored.

Per-worker rollout streams were already deterministic (world and action-sampling
RNGs are pure functions of worker rank / episode index), so sorting the results
by worker id fully restores cross-process reproducibility.

### Fix
- `_collect`: sort collected results by `rank` before concatenating (fixed
  worker-id order).
- `_rollout_worker`: re-seed the per-iteration action-sampling RNG from
  `_rollout_seed(training_seed, worker_id, iteration)` (a pure function),
  delivered via the broadcast job. Previously the stream was seeded only from
  `TRAIN_SEED_BASE + rank * 1000` and advanced implicitly across iterations.

No algorithm change. Training numbers for a given seed change after this fix
(expected; the RNG streams are reassigned). Checkpoint loading and eval are
unaffected (eval of `tests/fixtures/learner_v0_final.pt` unchanged: cap 5.0%,
median 3705).

### Verification
```text
workers=1   max_abs_diff = 0.000e+00   (after fix)
workers=2   max_abs_diff = 0.000e+00   (after fix)
workers=4   max_abs_diff = 0.000e+00   (after fix)
87 passed  (full suite)  ruff clean
legacy eval: cap_pct=5.0% median_survival=3705
```
