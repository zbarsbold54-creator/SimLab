# SimLab progress

## Phase 0+1 — deterministic core

Status: Phase 1 gate passed and committed as `Phase 1: verified deterministic core`.

## Phase 1 audit

The initial audit found these real defects and addressed them as follows:

- `python -m simlab.cli --version` was unsupported; the CLI now exposes the package version.
- The world used a grid and global `np.random.seed`; it now uses continuous motion and one `numpy.Generator`.
- Dead agents were removed and state was mirrored in the runner; the World now retains full agent state and owns snapshots.
- `run_episode` referenced a missing registry; it now consumes World state directly.
- Snapshots omitted reliable Generator state; snapshots now carry the complete state and restore exact futures.
- Tests were duplicated under `simlab/tests/` and scratch files were collected; tests now live in root `tests/` with pytest discovery restricted there.

What exists now:

- `World` owns continuous agent, food, hazard, time, and RNG state; it exposes `snapshot`, `restore`, and `state_hash`.
- `Action` uses turn/thrust/eat controls and `StepResult` follows the terminated/truncated/info shape.
- `Runner` seeds one NumPy Generator, records actions and snapshots, and provides headless run and benchmark commands.
- Root `tests/` contains the five Phase 1 gates; pytest is restricted to that directory.
- The GUI and survival hazards, sensing, drives, plugins, and research integrations remain for later phases.

Run:

```text
python -m pytest -q
python -m simlab.cli --version
python -m simlab.cli run --steps 1000 --seed 1
python -m simlab.cli bench --steps 1000
```

Open decisions: none for Phase 1.

## Phase 2 — survival, sensing, drives, plugins

Status: Phase 2 gate passed from `.venv`.

- Hazards chase nearby agents, wander through the seeded generator otherwise, and damage health on contact.
- Observations contain 16 ray distances, type channels for food/hazard/wall/agent, and proprioception.
- `BasicDriveCore` exposes the fixed eight placeholder drives and routes updates through `RetentionRule`.
- `FixedDecay`, `PassthroughRetention`, `HeuristicPolicy`, `RandomPolicy`, task, metric, and example plugin are registry components.
- Registry startup imports every module in `simlab/plugins/` automatically.

Gate output:

```text
17 passed in 2.45s
steps=1000 steps_per_second=3841.2
heuristic: mean=3750.15 spread(std)=131.53 min=3680 max=4315
random: mean=852.90 spread(std)=804.24 min=277 max=3551
```

The GUI remains for Phase 3; no GUI code was added in this phase.

## Phase 2 verification

Verification from `.venv`:

- `python -m pytest -v`: 17 tests passed, including exact ray distances/types, 2000-step drive bounds, TOML-selected retention behavior, and auto-discovery/selectability of a temporary plugin.
- `python -m simlab.cli bench --steps 1000 --agents 20`: 20 agents, 286.8 steps/s.
- `Snapshot.terrain` was unused and has been removed. `Snapshot.world_state` remains a derived occupancy raster used by `World.get_world_state()` and stored in `.npz` as a display/inspection preview.

Survival diagnosis (`Runner._max_steps = 100000`; cause is starvation if energy reached zero before death, hazard damage otherwise):

```text
heuristic seed=0 survival_steps=3767 cause=starvation food_eaten=88 configured_max_steps=100000
heuristic seed=1 survival_steps=3767 cause=starvation food_eaten=87 configured_max_steps=100000
heuristic seed=2 survival_steps=3767 cause=starvation food_eaten=85 configured_max_steps=100000
heuristic seed=3 survival_steps=3768 cause=starvation food_eaten=86 configured_max_steps=100000
heuristic seed=4 survival_steps=3768 cause=starvation food_eaten=87 configured_max_steps=100000
heuristic seed=5 survival_steps=3766 cause=starvation food_eaten=84 configured_max_steps=100000
heuristic seed=6 survival_steps=3784 cause=starvation food_eaten=89 configured_max_steps=100000
heuristic seed=7 survival_steps=3767 cause=starvation food_eaten=85 configured_max_steps=100000
heuristic seed=8 survival_steps=3767 cause=starvation food_eaten=84 configured_max_steps=100000
heuristic seed=9 survival_steps=4959 cause=starvation food_eaten=490 configured_max_steps=100000
heuristic seed=10 survival_steps=3767 cause=starvation food_eaten=87 configured_max_steps=100000
heuristic seed=11 survival_steps=3767 cause=starvation food_eaten=86 configured_max_steps=100000
heuristic seed=12 survival_steps=3766 cause=starvation food_eaten=86 configured_max_steps=100000
heuristic seed=13 survival_steps=3803 cause=starvation food_eaten=93 configured_max_steps=100000
heuristic seed=14 survival_steps=3767 cause=starvation food_eaten=79 configured_max_steps=100000
heuristic seed=15 survival_steps=3767 cause=starvation food_eaten=85 configured_max_steps=100000
heuristic seed=16 survival_steps=3767 cause=starvation food_eaten=85 configured_max_steps=100000
heuristic seed=17 survival_steps=3767 cause=starvation food_eaten=85 configured_max_steps=100000
heuristic seed=18 survival_steps=4361 cause=starvation food_eaten=223 configured_max_steps=100000
heuristic seed=19 survival_steps=3768 cause=starvation food_eaten=86 configured_max_steps=100000
random seed=0 survival_steps=547 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=1 survival_steps=801 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=2 survival_steps=864 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=3 survival_steps=677 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=4 survival_steps=2052 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=5 survival_steps=1093 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=6 survival_steps=1311 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=7 survival_steps=1818 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=8 survival_steps=493 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=9 survival_steps=959 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=10 survival_steps=1032 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=11 survival_steps=1854 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=12 survival_steps=1402 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=13 survival_steps=1065 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=14 survival_steps=782 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=15 survival_steps=2076 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=16 survival_steps=816 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=17 survival_steps=648 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=18 survival_steps=573 cause=hazard_damage food_eaten=0 configured_max_steps=100000
random seed=19 survival_steps=1190 cause=hazard_damage food_eaten=0 configured_max_steps=100000
```

The heuristic runs consume food but the current finite food supply runs out, after which energy reaches zero and health drains. Random runs do not find food reliably and usually die from chasing hazards before starvation. Neither policy reached the 100000-step limit.
