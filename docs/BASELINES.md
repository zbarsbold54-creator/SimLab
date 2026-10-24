# SimLab baselines

Reference scenario ("Balanced") and "Solo", across policies and agent counts,
with 95% bootstrap CIs over seeds. Oracles are **not** an upper bound; they are
kept for reference only (see Policy notes).

## How to compare a new policy

To evaluate a candidate policy against these baselines:

- Run the Balanced rows for 1, 5, and 20 agents, plus Solo, at 20 seeds, cap
  10000 (the same grid as `scripts/baseline_refresh.py`).
- Use the seed list `range(20)` (0..19).
- Report for each cell: cap percent with its 95% bootstrap CI (resampled over
  seeds, not agents), median survival step, and the death-cause split
  (starvation / hazard / truncation).
- A learned policy counts as better only if its CI does not overlap the
  heuristic's CI on the same cell.

## Reference scenario

| Setting | Value |
| --- | --- |
| Arena | 50 x 50 (continuous) |
| Config | `configs/balanced.toml` (same values as `configs/default.toml`) |
| Seed | 42 (world), per-seed sweep `range(20)` |
| Step cap | 10000 |
| Agent count | {1, 5, 20} |
| Hazard count | 4 (nominal), radius 1.5, damage 1/step |
| Hazard detection radius | 16 |
| Food | 20 patches x 20 max_amount (capacity 400) |
| Food regrowth | `supply_ratio = 2.0`, raw `regrow_rate = 0.0`; effective `regrow_rate = 2.0 * n_agents * 0.0336 / 20` |
| Partial drive | `simple` / `fixed_decay` retention |

Effective per-agent regrowth (units/step/agent): 1 agent -> 0.00672; 5 agents
-> 0.0168; 20 agents -> 0.0672. Nominal drain per agent at thrust 0.8 is
0.0336 energy/step.

## Results (20 seeds, cap 10000)

`cap%` = fraction of agents alive at the step cap (truncation). `CI` = 95%
bootstrap interval for cap%, resampled over seeds. `median` = median survival
step. `starve/hazard/trunc` = death-cause split (counts of agents).
`food_left` = mean total food remaining at episode end. Every row is marked
with the commit hash and its reproduce command under Reproduction.

| Policy | Agents | cap% | CI | median | starve | hazard | trunc | food_left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 0.0 | [0.0, 0.0] | 1886 | 0 | 20 | 0 | 400.0 |
| heuristic_noflee | 1 | 0.0 | [0.0, 0.0] | 662 | 0 | 20 | 0 | 388.8 |
| heuristic | 1 | 25.0 | [10.0, 45.0] | 6774 | 0 | 15 | 5 | 385.7 |
| heuristic_v2 | 1 | 20.0 | [5.0, 40.0] | 4449 | 2 | 14 | 4 | 391.7 |
| random | 5 | 0.0 | [0.0, 0.0] | 3242 | 10 | 90 | 0 | 400.0 |
| heuristic_noflee | 5 | 4.0 | [1.0, 8.0] | 1201 | 0 | 96 | 4 | 390.8 |
| heuristic | 5 | 80.0 | [71.0, 88.0] | 10000 | 0 | 20 | 80 | 372.5 |
| heuristic_v2 | 5 | 69.0 | [55.0, 82.0] | 10000 | 4 | 27 | 69 | 363.6 |
| random | 20 | 0.0 | [0.0, 0.0] | 4264 | 189 | 211 | 0 | 400.0 |
| heuristic_noflee | 20 | 8.8 | [5.0, 13.0] | 4138 | 0 | 365 | 35 | 400.0 |
| heuristic | 20 | 91.8 | [88.5, 95.0] | 10000 | 0 | 33 | 367 | 398.7 |
| heuristic_v2 | 20 | 85.8 | [80.5, 90.5] | 10000 | 0 | 57 | 343 | 384.9 |

## Solo scenario (`configs/solo.toml`)

Solo = default/Balanced scenario with 1 agent and 1 hazard (same arena,
detection radius 16, `supply_ratio = 2.0`). 20 seeds, cap 10000.

| Policy | cap% | CI | median | starve | hazard | trunc | food_left |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.0 | [0.0, 0.0] | 4198 | 4 | 16 | 0 | 400.0 |
| heuristic_noflee | 25.0 | [10.0, 45.0] | 1240 | 0 | 15 | 5 | 376.8 |
| heuristic | 50.0 | [30.0, 70.0] | 10000 | 0 | 10 | 10 | 380.0 |
| heuristic_v2 | 35.0 | [15.0, 55.0] | 6417 | 7 | 6 | 7 | 384.3 |

## Hard scenario (`configs/hard.toml`)

Hard = Balanced with a denser hazard field: `hazard.count = 12` (3x the
nominal 4), same detection radius 16, same arena and food count, `supply_ratio
= 2.0` (unchanged). Chosen from a hazard-count sweep at 5 agents (see
`scripts/hazard_hard_sweep.py`) so the sensor heuristic cap lands near 50%
while random stays under 5%, leaving headroom to measure a better policy.
20 seeds, cap 10000.

| Policy | Agents | cap% | CI | median | starve | hazard | trunc | food_left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 1 | 0.0 | [0.0, 0.0] | 728 | 0 | 20 | 0 | 400.0 |
| heuristic_noflee | 1 | 0.0 | [0.0, 0.0] | 432 | 0 | 20 | 0 | 394.2 |
| heuristic | 1 | 5.0 | [0.0, 15.0] | 4237 | 0 | 19 | 1 | 388.5 |
| heuristic_v2 | 1 | 0.0 | [0.0, 0.0] | 1804 | 0 | 20 | 0 | 389.5 |
| random | 5 | 0.0 | [0.0, 0.0] | 1372 | 0 | 100 | 0 | 400.0 |
| heuristic_noflee | 5 | 0.0 | [0.0, 0.0] | 643 | 0 | 100 | 0 | 395.2 |
| heuristic | 5 | 46.0 | [32.0, 60.0] | 9110 | 0 | 54 | 46 | 381.9 |
| heuristic_v2 | 5 | 26.0 | [15.0, 39.0] | 4250 | 0 | 74 | 26 | 385.7 |
| random | 20 | 0.0 | [0.0, 0.0] | 2508 | 11 | 389 | 0 | 400.0 |
| heuristic_noflee | 20 | 0.2 | [0.0, 0.8] | 1726 | 0 | 399 | 1 | 399.9 |
| heuristic | 20 | 71.0 | [62.0, 79.5] | 10000 | 0 | 116 | 284 | 397.9 |
| heuristic_v2 | 20 | 72.0 | [63.0, 80.5] | 10000 | 0 | 112 | 288 | 385.5 |

## Reproduction

All rows above are produced by `scripts/baseline_refresh.py`
(multiprocessing, one process per episode, 20 seeds, cap 10000), which calls
`run_episode(policy, seed, agents, cap=10000, config_path=...)` per episode and
writes the machine-readable table to `docs/baselines.json` (per-scenario
cap%, CI, median, death-cause counts, food left, seeds, cap, commit, date).

| Row | Command | Commit |
| --- | --- | --- |
| Balanced 1 | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Balanced 5 | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Balanced 20 | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Solo | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Hard 1 | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Hard 5 | `python scripts/baseline_refresh.py` | `world-v1.4` |
| Hard 20 | `python scripts/baseline_refresh.py` | `world-v1.4` |

Policies used: `random`, `heuristic_noflee`, `heuristic`, `heuristic_v2` (the
pre-principled-thrust sensor heuristic). Oracle policies are excluded from the
refreshed grid; see Policy notes.

## Policy notes

| Policy | Registered name | Needs observations |
| --- | --- | --- |
| Sensor heuristic (with flee) | `heuristic` | yes |
| Sensor heuristic (no flee) | `heuristic_noflee` | yes |
| Sensor heuristic v1-era (frozen) | `heuristic_v2` | yes |
| Random | `random` | no |
| Oracle heuristic (with flee) | `oracle` | no |
| Oracle heuristic (no flee) | `oracle_noflee` (alias `oracle_heuristic`) | no |
| Oracle heuristic v1-era (frozen) | `oracle_v2` | no |

`heuristic_v2` / `oracle_v2` freeze the world-v1-era control law (eat thrust
0.6, steer /135) so the pre-refresh baseline table remains reproducible; the
current `heuristic` / `oracle` derive near-food thrust from the world's eat
range and max turn rate (min turn radius <= 0.7 * eat_range).

Oracles read world state directly and are **not an upper bound**: in this
environment the sensor heuristic consistently out-survives the oracle because
direct world-state gaze does not avoid the dominant hazard deaths once energy is
tight. The oracle policies are kept for reference/reproducibility only.
