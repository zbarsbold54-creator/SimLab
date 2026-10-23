# SimLab baselines

Reference scenario ("Balanced"), results across policies and agent counts, and
exact reproduction commands. Oracles are **not** an upper bound; they are kept
for reference only (see the note below).

## Reference scenario

| Setting | Value |
| --- | --- |
| Arena | 50 x 50 (continuous) |
| Config | `configs/balanced.toml` (same values now copied into `configs/default.toml`) |
| Seed | 42 (world), per-seed sweep `range(15)` / `range(20)` |
| Step cap | 10000 |
| Agent count | {1, 5, 20} |
| Hazard count | 4 (nominal), radius 1.5, damage 1/step |
| Hazard detection radius | 16 |
| Food | 20 patches x 20 max_amount (capacity 400) |
| Food regrowth | `supply_ratio = 2.0`, raw `regrow_rate = 0.0`; effective `regrow_rate = 2.0 * n_agents * 0.0336 / 20` |
| Partial drive | `simple` / `fixed_decay` retention |

Effective per-agent regrowth (units/step/agent): 1 agent -> 0.00672 g 5 agents -> 0.0168; 20 agents -> 0.0672. Nominal drain per agent at thrust 0.8 is 0.0336 energy/step.

## Results (15 seeds, cap 10000)

`cap%` = fraction of agents alive at the step cap (truncation). `median` =
median survival step. `starve/hazard/trunc` = death-cause split (counts of
agents). `food_left` = mean total food remaining at episode end. `svfood` =
agents that starved while >= 50% of the 400-unit food capacity was uneaten.

| Policy | Agents | cap% | median | starve | hazard | trunc | food_left | svfood |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 1 | 20.0 | 4449 | 2 | 10 | 3 | 391.2 | 2 |
| heuristic_noflee | 1 | 0.0 | 1973 | 1 | 14 | 0 | 378.3 | 1 |
| random | 1 | 0.0 | 1886 | 0 | 15 | 0 | 400.0 | 0 |
| oracle | 1 | 0.0 | 1015 | 1 | 14 | 0 | 394.5 | 1 |
| heuristic | 5 | 74.7 | 10000 | 4 | 15 | 56 | 358.6 | 4 |
| heuristic_noflee | 5 | 12.0 | 4610 | 4 | 62 | 9 | 380.6 | 4 |
| random | 5 | 0.0 | 3242 | 9 | 66 | 0 | 400.0 | 9 |
| oracle | 5 | 4.0 | 1247 | 0 | 72 | 3 | 390.2 | 0 |
| heuristic | 20 | 85.3 | 10000 | 0 | 44 | 256 | 384.2 | 0 |
| heuristic_noflee | 20 | 70.0 | 10000 | 10 | 80 | 210 | 382.5 | 10 |
| random | 20 | 0.0 | 4262 | 145 | 155 | 0 | 400.0 | 145 |
| oracle | 20 | 13.0 | 2744 | 16 | 245 | 39 | 394.4 | 16 |

### Final 20-seed run, 5 agents

| Policy | cap% | median | starve | hazard | trunc | food_left | svfood |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heuristic | 69.0 | 10000 | 4 | 27 | 69 | 363.6 | 4 |
| random | 0.0 | 3242 | 10 | 90 | 0 | 400.0 | 10 |

## Reproduction

The entire grid above (except the final 20-seed row) is produced by:

```text
python scripts/balanced_eval_v2.py
```

That script:

- verifies multiprocessing == serial on 2 seeds,
- runs the 15-seed grid over agents {1, 5, 20} and policies
  {heuristic, heuristic_noflee, oracle, random},
- runs the 20-seed final 5-agent heuristic/random run.

Each table row is the aggregate over `SEEDS` seeds of
`run_episode(policy, seed, agents, cap=10000, config_path="configs/balanced.toml")`
(`scripts/episode_lib.py`). Seeds are `range(15)` (grid) or `range(20)`
(final row).

Reference commit: `world-v1` tag (points at the World v1 baseline commit; see git log).

## Policy notes

| Policy | Registered name | Needs observations |
| --- | --- | --- |
| Sensor heuristic (with flee) | `heuristic` | yes |
| Sensor heuristic (no flee) | `heuristic_noflee` | yes |
| Random | `random` | no |
| Oracle heuristic (with flee) | `oracle` | no |
| Oracle heuristic (no flee) | `oracle_noflee` (alias `oracle_heuristic`) | no |

Oracles read world state directly and are **not an upper bound**: in this
environment the sensor heuristic consistently out-survives the oracle (for
example 74.7% vs 4% cap at 5 agents), because the oracle's direct world-state
gaze does not help it avoid the dominant hazard deaths once energy is tight. The
oracle policies are kept for reference/reproducibility only and are **not**
used in the baselines above beyond the informational rows.
