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

## Phase 3 — GUI shell

Status: GUI shell implemented; Phase 3 gate verified from `.venv`.

- Added a dark Qt window, toolbar, antialiased world renderer with zoom/pan, and a simulation worker hosted in a `QThread`.
- The worker publishes deeply immutable `Snapshot` instances; the GUI timer renders the newest snapshot at no more than 60 FPS.
- Added speed, play/pause, single-step, reset/seed controls and status for step, sim time, render FPS, sim steps/s, and seed.
- `python -m simlab.cli gui` lazily imports the GUI; a subprocess test confirms `import simlab.core` does not load PySide6.
- Added an offscreen `QApplication` worker smoke test and `scripts/render_screenshot.py`.

Gate output:

```text
20 passed in 1.28s
agents=20 speed=10x render_fps=55.4 sim_steps_per_second=100.8
agents=20 speed=Max render_fps=48.6 sim_steps_per_second=125.0
```

Screenshot: `docs/screenshots/phase3.png`.

## Phase 3 visual fixes

- The speed selector and worker now share `theme.DEFAULT_SPEED`; selecting a speed updates the target rate.
- The default 50x50 arena uses a configured 30-unit hazard detection radius and 1.5-unit contact radius. The old renderer drew the 30-unit detection ring. Detection configuration is unchanged; a smaller 10-unit default was proposed and awaits user direction.
- Arena contents are clipped to the arena. The faint default hazard ring shows contact radius; detection radius appears on hazard hover or when a selected agent is within detection range.
- Food is amount-scaled and desaturated; agents use a solid energy-interpolated fill and render above food and hazards.
- Default GUI agent count is read from `configs/default.toml` (`[agent] count = 5`).
- Toolbar/status spacing, seed label, title, minimum size, arena fill/border, and combo styling were polished.
- Added speed, food-radius, detection-ring, and TOML agent-count tests.

Verification output:

```text
selector_default=1x worker_default=1x
speed=0.25x target_steps_per_second=2.5
speed=0.5x target_steps_per_second=5
speed=1x target_steps_per_second=10
speed=2x target_steps_per_second=20
speed=5x target_steps_per_second=50
speed=10x target_steps_per_second=100
configured_detection_radius=30
contact_damage_radius=1.5
previous_ring_drawn_radius=30
arena=50x50 default_agent_count=5

food_patch_sample=0 amount=2.00 drawn_radius=0.680
food_patch_sample=1 amount=5.00 drawn_radius=0.800
food_patch_sample=2 amount=10.00 drawn_radius=1.000
food_patch_sample=3 amount=15.00 drawn_radius=1.200
food_patch_sample=4 amount=20.00 drawn_radius=1.400

24 passed in 1.18s
All checks passed!
```

Screenshots: `docs/screenshots/phase3_polish_5agents.png` and `docs/screenshots/phase3_polish_20agents.png`.

## Phase 3 polish 2

- Agent body length is 1.6 radii (5.28 world units), about 1.55 times the default food diameter (3.4 units). Geometry is generated in local coordinates, then rotated and translated; a pure-function test checks the heading axis and mirrored rear vertices.
- Toolbar and status text insets are verified in an offscreen window: first toolbar widget x=8, status label x=16.
- Seed label background is transparent. The speed selector uses the SVG chevron icon and themed popup list.
- Screenshot script now advances to the requested screenshot step and then prints actual patch sizes at its food diagnostic step.

Raw screenshot and food output:

```text
agents=5 steps=1500 screenshot=docs/screenshots/phase3_polish2_5agents.png
food_radius_diagnostics_after_step=2000
food_patch=0 amount=20.00 drawn_radius=1.700
food_patch=1 amount=20.00 drawn_radius=1.700
food_patch=2 amount=0.00 drawn_radius=0.600
food_patch=3 amount=20.00 drawn_radius=1.700
food_patch=4 amount=20.00 drawn_radius=1.700
agents=20 steps=1500 screenshot=docs/screenshots/phase3_polish2_20agents.png
food_radius_diagnostics_after_step=2000
food_patch=0 amount=20.00 drawn_radius=1.700
food_patch=1 amount=0.00 drawn_radius=0.600
food_patch=2 amount=17.92 drawn_radius=1.585
food_patch=3 amount=0.00 drawn_radius=0.600
food_patch=4 amount=18.42 drawn_radius=1.613

27 passed in 1.20s
All checks passed!
```

## Phase 4 — Inspection

- Added selectable agent inspection, follow camera, rays and vision overlays, trails, grid, and a drive-colored halo.
- Added a right-side inspector with live agent state, current action, sensor summary, and eight drive bars; a bottom pyqtgraph history plot and timeline; and a registry-backed scenario dock with presets, sliders, policy and DriveCore selectors.
- Scenario apply resets the worker on its QThread with the selected seed. The worker publishes frozen per-step `SimulationFrame` data; core imports remain independent of Qt.
- Verification: selecting `agent_0` displays inspector data and live drive series; applying a 4-agent, 2-hazard scenario retains seed 29 and applies food regrowth 0.17.

Raw Phase 4 gate output:

```text
29 passed in 1.74s
All checks passed!
gui_cli_exit=0
agents=5 steps=1504 screenshot=C:\\AI\\SimulationCode\\docs\\screenshots\\phase4.png
```

Screenshot: `docs/screenshots/phase4.png`.

## Phase 5 — Timeline, snapshots, replay

- The worker now retains a bounded ring of per-step snapshots and batches them to the GUI. The timeline can seek to an exact stored step; Play resumes from that checkpoint using the recorded actions, preserving the original future.
- Added JSON+NPZ snapshot save/load controls and an in-memory Record/Replay toolbar. `ActionLog` can also serialize action records as JSON.
- Split worker frame data, component execution, history, and thread controls into focused modules; `simlab/gui/sim_worker.py` is 261 lines.
- Run the studio with `python -m simlab.cli gui`.

Raw Phase 5 gate output:

```text
test_scrub_back_500_steps_and_replay_matches_original_future PASSED
test_action_log_round_trips_json_and_groups_by_step PASSED
test_phase5_gui_saves_and_loads_json_npz_snapshot PASSED
33 passed in 2.77s
All checks passed!
gui_cli_exit=0
agents=5 steps=1500 screenshot=C:\\AI\\SimulationCode\\docs\\screenshots\\phase5.png
```

Screenshot: `docs/screenshots/phase5.png`.

## Phase 6 — Polish and docs

- Finished the shared dark Qt styling for checkbox indicators, hover/pressed/checked/disabled/focus states, themed combo popup rows, scrollbars, and tooltips. Existing controls use text labels and the shared chevron asset; layout and palette remain in `simlab/gui/theme.py` and `simlab/gui/style.qss`.
- Added a fresh-checkout README with PowerShell virtual-environment, install, run, test, and lint commands. Added `docs/EXTENDING.md` with short decorator-based drive, policy, and retention examples; noted that retention registry selection is not yet wired into GUI drive construction.
- Added `scripts/phase6_performance.py` to measure a visible 20-agent GUI at 10x and Max.
- Ruff formatting corrected the existing NumPy tutorial file so repository-wide `ruff format --check .` passes.
- Fresh-clone packaging validation exposed setuptools flat-layout discovery of `configs` and `learning`; `pyproject.toml` now limits installed package discovery to `simlab*`.

Raw Phase 6 verification output:

```text
33 passed in 2.92s
All checks passed!
50 files already formatted
agents=20 speed=10x render_fps=58.2 sim_steps_per_second=99.7
agents=20 speed=Max render_fps=58.0 sim_steps_per_second=113.3
font_check: Segoe UI available=False; Arial available=False; Qt font families=0
fresh_clone_install=success
fresh_clone_tests=33 passed in 2.80s
fresh_clone_cli_run=steps=1 seed=1
fresh_clone_cli_bench=agents=20 steps=1000 steps_per_second=291.5
```

Offscreen screenshot review: panel and control styling is consistent; this Qt runtime exposes no font families, so its screenshot renders text as missing-glyph boxes. This environment limitation does not establish how text renders in the user's interactive Windows session.

## Verification pass — Phase 5 replay and GUI throughput

- Added root `AGENTS.md` with the workspace boundaries and implementation gates; committed separately as `Add AGENTS.md` (`c1d2044`).
- Confirmed Phase 5 replay coverage: `tests/test_phase5.py::test_scrub_back_500_steps_and_replay_matches_original_future` stores 700 action/result hashes, restores the step-200 checkpoint, replays the final 500 recorded actions, and asserts every resulting hash. `tests/test_hardening.py::test_snapshot_disk_round_trip_replays_future` saves NPZ+JSON, restores a fresh World, checks the starting hash, and checks 200 subsequent hashes.
- Throughput work batches Max stepping, omits per-step observation/snapshot construction, publishes the newest immutable frame for GUI-timer polling, stores timeline checkpoints every 25 steps, batches heuristic food-location preparation, and yields once per 32-step Max batch. Simulation action/state rules are unchanged. Diagnostic and measurement scripts: `scripts/verification_episodes.py` and `scripts/verification_throughput.py`.
- Throughput measurement used 20 agents, all overlays on, `agent_0` selected, 3-second samples (0.5-second Max sample), and p95 timer-frame intervals:

```text
agents=20 overlays=all selected=agent_0 speed=1x render_fps=59.1 sim_steps_per_second=10.0 p95_frame_time_ms=23.43
agents=20 overlays=all selected=agent_0 speed=10x render_fps=58.3 sim_steps_per_second=100.3 p95_frame_time_ms=22.53
agents=20 overlays=all selected=agent_0 speed=Max render_fps=21.7 sim_steps_per_second=1512.3 p95_frame_time_ms=63.36
```

  Before optimization, the same script measured Max at 96.3 steps/s, 0.6 FPS, and 1817.49 ms p95. The uninstrumented final Max sample clears 1500 steps/s; the UI remains less responsive in Max than at capped speeds.
- Instrumented final profile: 1540.7 steps/s, 15.4 FPS, 71.45 ms p95; 135 snapshots cost 25.4 ms total, 67 scheduler yields cost 5.2 ms, worker CPU fraction was 0.810, 441.5 ms was in action generation, 451.5 ms in Runner stepping, and 334.3 ms across 62 paint events. Instrumentation reduces throughput and is not the primary reported rate.
- Episode diagnostics used the default 50x50 configuration, one agent, seeds 1-20, and the 100000-step Runner cap. Raw per-seed rows were printed by `scripts/verification_episodes.py`: heuristic episodes ended by starvation after 3766-4959 steps and ate 79-490 patches; random episodes ended by hazard damage after 423-3081 steps and ate none. No run reached the cap. For comparison on a 100-unit square, detection radius 30 is 30% of the side (60% diameter), and contact radius 1.5 is 1.5% of the side (3% diameter). The configured default arena is 50x50, so those radii are 60% and 3% of its side respectively.
- `python -m simlab.cli bench --agents 20`: final rerun printed `agents=20 steps=1000 steps_per_second=1073.7` (a prior run printed 1119.3).
- Full `.venv` test run: 33 passed. Ruff check and format check both pass. `where.exe python` listed `.venv` first and `sys.prefix` was `C:\AI\SimulationCode\.venv`.
- Real-window inspection could not be completed: the desktop automation inventory exposed no application/native window, including after starting the CLI GUI. The offscreen Qt smoke test passed, but the real interactive rendering has not been visually verified in this session.

## Verification 2 and world tuning sweep

- Added `tests/test_fast_path.py::test_max_batch_matches_headless_hashes_for_1000_steps`. It drives the same `_advance_one_step` helper used by the GUI worker in 32-step Max batches and compares the 20-agent worker and ordinary Runner hashes after every one of 1000 steps. It passed with the Phase 5 replay and disk restore tests (`3 passed in 3.36s`). This also checks the world RNG state because it is included in `state_hash()`.
- Before the bench fast path, an uninstrumented 20-agent run measured 1073 steps/s. cProfile (profiler overhead included) showed 1000 `World.step` calls at 1.437 s cumulative, snapshots at 0.605 s, observations at 0.368 s, ray casts at 0.277 s, and `Runner._default_action` at 0.412 s. The CLI bench does not call registered drives or policy classes; it uses the Runner's built-in heuristic action. `run_bench` now skips unused observations, snapshots, and AgentState views while retaining the same state transitions and action log. After: 2593.4 steps/s in the final uninstrumented run; a cProfile run measured 1000 `World.step` calls at 0.414 s and default action generation at 0.393 s. Sensing, drive, and registered-policy calls are absent from this benchmark path.
- The CLI bench and GUI Max rates measure different workloads. CLI bench now measures headless world stepping and Runner actions without rendering, observations, or per-step snapshots. GUI Max adds policy components, history/action storage, checkpoint snapshots every 25 steps, GUI frame construction and rendering, and a batch yield; its earlier measured rate was 1512.3 steps/s.
- The default arena is set to 50x50 in `configs/default.toml` (and `WorldConfig` in `simlab/core/config.py`). `git log -S 'width = 50' -- configs/default.toml` and `git blame` identify commit `ab3b0b6 Initial project structure: pyproject.toml, README.md, PROGRESS.md, folder layout, core and interfaces stubs` as the commit that introduced the 50-unit dimensions. Added `configs/harsh_original.toml` as an unchanged named copy of the original config; `configs/default.toml` was not modified.
- Energy-budget check, seed 1, one agent: initial energy 100; heuristic thrust 0.8 gives speed 6.4 and drain `(0.08 + 0.04*6.4)*0.1 = 0.0336` energy/step; energy alone lasts about 2976 steps, followed by 200 health-drain steps. There are 20 patches × 20 units = 400 initial food units, but no regrowth. The run consumed 20 units in 87 successful eat ticks, yielding about 20 / 0.0336 = 595 additional energy-equivalent steps and explaining the typical ~3770-step outcomes. `food_eaten` counts successful bite ticks, not whole patches. The 17 ordinary seeds cluster near the same duration because they obtain about one patch's 20 energy units before starvation; a few seeds happen to intersect more food.
- No world food-regrowth mechanic exists: `food.spawn_rate` only determines initial patch count, capped by `food.max_food=20`. Therefore the requested 1x/3x/10x sweep uses the only existing food knob and reports the cap; those factors all produce 20 initial patches. This is not a regrowth experiment, and the script reports that limitation instead of changing world behavior. All 12 cells are below:

```text
detection food_factor initial_patches heuristic_median heuristic_cap_pct heuristic_starve/hazard/trunc random_median random_starve/hazard/trunc
        8           1             20             3770               0.0   90/  10/   0          3629   42/  58/   0
        8           3             20             3770               0.0   90/  10/   0          3629   42/  58/   0
        8          10             20             3770               0.0   90/  10/   0          3629   42/  58/   0
       12           1             20             3770               0.0   90/  10/   0          3500   34/  66/   0
       12           3             20             3770               0.0   90/  10/   0          3500   34/  66/   0
       12          10             20             3770               0.0   90/  10/   0          3500   34/  66/   0
       16           1             20             3770               0.0   87/  13/   0          3238   27/  73/   0
       16           3             20             3770               0.0   87/  13/   0          3238   27/  73/   0
       16          10             20             3770               0.0   87/  13/   0          3238   27/  73/   0
       30           1             20             3770               0.0   87/  13/   0          3599   39/  61/   0
       30           3             20             3770               0.0   87/  13/   0          3599   39/  61/   0
       30          10             20             3770               0.0   87/  13/   0          3599   39/  61/   0
one_agent seed=1 policy=heuristic detection=30 food_factor=1 initial_patches=20 survival=3767 cause=starvation
one_agent seed=1 policy=random detection=30 food_factor=1 initial_patches=20 survival=1343 cause=hazard
current_food_budget=initial_energy:100 food_patches:20 food_amount_per_patch:20 total_food:400 regrowth_per_step:0
energy_budget seed=1 survival=3767 starting_energy=100 heuristic_speed=6.4 drain_per_step=0.0336 energy_only_steps=2976.2 starvation_health_steps=200 food_consumed=20.0000 food_eaten_ticks=87 initial_food=400.0 regrowth_per_step=0
```

- Every five-agent cell had 0% heuristic cap survivors. With finite food and positive drain, indefinite survival is impossible in the current implementation; no parameter recommendation can meet the requested indefinite-survival condition without a real regrowth mechanism. Conditional candidates after a regrowth rate is defined are detection radius 12 or 16; hazard deaths were 10-13% in those cells. A moving heuristic agent needs about 0.0336 energy per step, or 0.168 per step for five agents before accounting for food availability. No default was changed.

## Heuristic and fast-path fixes (tasks 1-6)

### 1. Starvation anomaly (fixed)

Root cause: target selection (both `Runner._default_action` and the old world-state heuristic) picked the nearest FoodPatch even when its `amount` was 0.0, so an agent orbited a depleted patch forever and starved with ~380/400 food still present. Fixed by filtering patches with `amount > 0.0` in `Runner._default_action` and in the sensor heuristic. Before (one agent, seeds 1-5): deaths at 3771-4366 with 20-40 food units eaten. After: seeds 1,2,3,5 eat all 400 units and survive to ~16266; seed 4 dies at 7902 by hazard (158.8 eaten).

### 2. Sensor-based heuristic (done)

`HeuristicPolicy` now uses only `obs` (rays + ray_types + proprioception) via `Policy.act`; it no longer touches World state and no longer implements `select_action`. World-state version preserved as `OracleHeuristicPolicy`, registered as `oracle_heuristic`. Survival report over 10 seeds x 5 agents (defaults, step cap 50000):
- sensor heuristic: median 6421, spread 4504-7567, all hazard deaths
- oracle heuristic: median 4100, spread 3472-5554
- random: median 4270, spread 3562-4346

### 3. Fast-path correctness (done)

Added `needs_observations` to the `Policy` and `DriveCore` protocols (default True); `pipeline.needs_observations()` computes the combined flag. `tests/test_fast_path_plugins.py` registers a recurrent `RecurrentTestDriveCore` (FixedDecay retention) and a `StatefulTestPolicy` that reads obs + drive_state and keeps internal state; it asserts GUI Max path (`_advance_one_step`) and headless Runner give identical per-step world hashes, identical drive-state arrays, and identical policy internal state over 1000 steps x 20 agents (passes). It also shows that skipping a drive step (a broken fast path) diverges the world state.

### 4. Bench (done)

`python -m simlab.cli bench --agents 20 --steps 1000` now prints both numbers:
- world_only_steps_per_second=5817.9
- full_pipeline_steps_per_second=829.6

### 5. Food regrowth (done)

Each FoodPatch now carries `max_amount` and regrows by `food.regrow_rate` units/step up to it. Added `regrow_rate=0.0` to TOML (default unchanged behavior), included in the state hash (snapshot metadata), and covered by `tests/test_food_regrowth.py` (determinism, cap, hash participation, snapshot/restore preservation). Scenario dock: the old "Food regrowth" slider actually controlled initial patch density (`spawn_rate`); I relabeled it "Food density (initial patches)" and added a real "Food regrowth" slider wired to `regrow_rate`. Added the "Harsh (original)" preset that loads `configs/harsh_original.toml`.

### 6. World sweep (report only)

`scripts/world_sweep_regrowth.py`, 5 agents, 5 seeds (reduced from 10 to stay under the runtime budget), cap 10000, detection radius in {12,16,30}, regrow_rate in {0.005,0.01,0.02,0.05}. Ran in 481.6s.

Break-even arithmetic (5 moving agents): drain 0.0336/step/agent, total 0.168/step; energy per food unit 1.0 (1:1 transfer); 20 initial patches; supply-equals-drain regrow_rate = 0.168/20 = 0.00840 units/step/patch.

```text
radius regrow_rate heuristic_median heuristic_cap_pct heuristic_starve/hazard/trunc random_median random_starve/hazard/trunc
    12      0.005              5378             12.0   22/   0/   3          3181    9/  16/   0
    12      0.010              7739             28.0   18/   0/   7          3181    9/  16/   0
    12      0.020              9383             48.0   13/   0/  12          3181    9/  16/   0
    12      0.050             10000             52.0   12/   0/  13          3181    9/  16/   0
    16      0.005              5986              8.0   23/   0/   2          2842    6/  19/   0
    16      0.010              7681             40.0   15/   0/  10          2842    6/  19/   0
    16      0.020             10000             52.0   12/   0/  13          2842    6/  19/   0
    16      0.050             10000             52.0   12/   0/  13          2842    6/  19/   0
    30      0.005              6585              4.0   23/   1/   1          3719   11/  14/   0
    30      0.010              9222             44.0   14/   0/  11          3719   11/  14/   0
    30      0.020             10000             60.0   10/   0/  15          3719   11/  14/   0
    30      0.050             10000             60.0   10/   0/  15          3719   11/  14/   0
```

Conclusion: sensors alone (random) never reach the cap and mostly die to hazards; the heuristic's survival scales strongly with regrow_rate. Agents do compete for food (shared finite patches; food is the limiting resource). Recommended parameter sets:
- detection_radius=16, regrow_rate=0.02: heuristic median hits the 10000 cap, 52% of agents reach it, starvation split reduced to 12/25.
- detection_radius=30, regrow_rate=0.02 (or 0.05): highest cap rate (60%).

Defaults unchanged; no `default.toml` modification.

### Verified
- Full suite: `40 passed` (was 34 before this work). Ruff clean.
- Bench numbers above; sweep table above.

### Open
- Awaiting your choice on adopting a recommended parameter set (detection radius / regrow rate) into a config or named preset.
- Random policy and oracle/sensor comparisons are report-only; no default changes made.
- Whether to keep `scripts/world_sweep_regrowth.py` as a long-term tool.

## Balanced preset and baseline verification

### Death-cause classifier (verified and fixed)
Old classifier labeled a death "hazard" unless energy hit zero at the exact death step, so a random walker that starved while wandering into a hazard field was mislabeled. New classifier in `scripts/episode_lib.py` measures cumulative health loss by source (hazard contact damage vs the 0.5/step starvation drain) and assigns the cause to whichever actually consumed health; survivors at the cap are truncation. Verified on random policy, 5 agents, 10 seeds:

```text
random total agents=50 median_survival=3546 starvation/hazard/truncation=5/45/0
```

Random walkers eat nothing (food_eaten=0, food_left=400). Their energy budget (mean thrust 0.5 -> speed 4 -> drain 0.024/step) empties around step 4000-4200; health then drains 0.5/step, so pure starvers die ~step 4300. Many instead die earlier inside hazard fields (energy still positive, hazard_loss ~100), which is why a naive "hazard" label was common and the median ends up ~3546 rather than near 4300.

### Oracle baseline (verified and aligned)
OracleHeuristicPolicy now mirrors the sensor HeuristicPolicy control law exactly: it filters depleted patches (amount <= 0) and uses the same turn/0.8 thrust/eat-within-1.5 mapping, sourced from world geometry instead of rays. Neither policy has flee logic (both only seek food and otherwise wander), so they are behaviorally equivalent modulo how they sense food. Medians are per agent (pooled across agents x seeds). Re-run on default config, 10 seeds, 5 agents, cap 10000:

```text
heuristic           median_survival=5808 starve/hazard/trunc=40/10/0
oracle_heuristic   median_survival=3188 starve/hazard/trunc=41/9/0
random             median_survival=3546 starve/hazard/trunc=5/45/0
```

Sensor beats oracle because the ray-sampled sensor spends much of its time in low-thrust wander (0.5, drain 0.024) rather than constant full-speed pursuit (0.8, drain 0.0336), and its quantized turning disperses agents so they exploit far more of the map; it converts far more of the 400 food units into survival time before energy runs out. (Earlier reported 6421/4100 came from the old classifier; ordering is unchanged.)

### Depleted-patch sensor fix
Observations now exclude food patches with amount <= 0, so ray-based policies cannot chase empty patches (this closes the original starvation anomaly for the registered sensor heuristic, not just `_default_action`). Oracle applies the same filter in `select_action`.

### Balanced preset (added)
- `configs/balanced.toml`: default config with `hazard.detection_radius = 16` and `food.regrow_rate = 0.02`. `configs/harsh_original.toml` unchanged.
- Scenario dock: added a "Balanced" preset (sets the sliders and detection radius).
- `tests/test_balanced.py`: qualifies preset loads (radius 16, regrow 0.02), determinism (two identical seeds -> equal state hash over 2000 steps), and regrowth is active and capped.

### Balanced evaluation (decision gate)
`scripts/balanced_eval.py`, config balanced.toml, cap 10000. Seeds reduced 20->15 to stay under the 15-minute budget (858s estimated; 793.6s actual). Break-even regrow rates (full-speed pursuit drain 0.0336/agent/step, 20 patches): 1 agent = 0.00168, 5 agents = 0.00840, 20 agents = 0.03360.

```text
policy agents median_survival cap_pct mean_food_left starve/hazard/trunc
heuristic          1  10000 73.3 394.0  2/2/11
heuristic          5   6906 41.3 383.2 40/4/31
heuristic         20   6766 37.0 206.6 185/4/111
oracle_heuristic   1   3200 26.7 399.8  9/2/4
oracle_heuristic   5   3199 22.7 388.9 38/20/17
oracle_heuristic  20   3186 10.3 338.0 176/93/31
random              1   1886  0.0 400.0  0/15/0
random              5   3229  0.0 400.0  8/67/0
random             20   4262  0.0 400.0 141/159/0
```

Decision rule: at 5 agents sensor reaches cap 41.3% (in 40-70), random 0% (<10), but starvation is 40/44 = 90.9% of heuristic non-truncation deaths (>5%). Rule fails, so `configs/default.toml` was NOT changed.

1-agent vs 20-agent: a single agent reaches the cap most often (73%) but leaves ~394 food because it cannot cover the map; 20 agents consume far more (food_left 207) but cap drops to 37% from competition/clustering thinning patches faster. Regrowth should scale with agent count: the break-even rate is linear in agent count (0.0017 / 0.0084 / 0.0336), so one fixed rate cannot be optimal for all counts; the current 0.02 covers 5 agents but is above 1-agent need and below 20-agent need. Recommend making regrow_rate a per-scenario value and/or scaling it with agent count, plus better foraging/hazard avoidance to cut the dominant starvation mode.

### Verified
Full suite `44 passed`, ruff clean.

## Foraging diagnosis and heuristic v2

### Step 1 + 4 — starvation root cause and foraging fix
Diagnosis (`scripts/foraging_diag.py`, Balanced, 5 agents, 10 seeds, `heuristic_noflee`): starved agents spent ~99.4% of steps in `seek` mode with only ~14-20 eat ticks total, and never reached eat range — the nearest-food-ray distance was pinned at ~2.86-2.88 (just above the 1.5 threshold) while the agent orbited the patch. The defect is a convergence/orbit problem, not out-of-range eating.

Fix in `HeuristicPolicyNoflee`/`OracleHeuristicPolicyNoflee`: a near-food branch (within 5.0 units) that reduces thrust to 0.5 and steers harder (turn gain /60) so the agent converges instead of orbiting. After the fix, starved agents eat properly (eat ticks ~1900-2700, food ~55-240 units, eat-mode ~30-34%); starvation dropped from 22/30 (pinned) to 4/50 noflee and 3/50 with flee.

### Step 2 — sensor vs oracle reconciliation
On `heuristic_noflee` vs `oracle_noflee` (6 seeds, 5 agents), mean thrust (0.799 vs 0.800) and drain/step (0.0336) are identical, so the survival gap is not explained by thrust/drain. Sensor ate 196.2 food units/agent vs oracle 90.4 and survived median 8406 vs 3183; the oracle dies earlier to hazards and therefore converts fewer total units per agent. Oracle is not currently a meaningful upper bound.

### Step 3 — hazard flee
Added flee to both sensor (`HeuristicPolicy`, default `heuristic`) and oracle (`OracleHeuristicPolicy`, default `oracle`): flee when hazard threat > 0.3, turn away from the nearest hazard ray/bearing. Historic non-flee versions remain registered as `heuristic_noflee` and `oracle_noflee` (alias `oracle_heuristic`).

### Step 5 — scenario-level supply scaling
Added optional `food.supply_ratio`. When set, `Runner` computes `regrow_rate = supply_ratio * n_agents * 0.0336 / n_patches` (0.0336 = nominal drain per agent at thrust 0.8). Raw `regrow_rate` stays available (slider now labeled "Food regrowth (raw)"). `configs/balanced.toml` now uses `supply_ratio = 2.0` (+ radius 16). Tests: rate scales with agent count, determinism holds, snapshot/restore preserves the effective rate.

### Step 6 — Balanced re-evaluation (decision gate)
`scripts/balanced_eval_v2.py`, multiprocessing (one process per seed, stdlib). Verified `mp == serial` on 2 seeds. Grid: 15 seeds, cap 10000, agents {1,5,20}, policies {heuristic, heuristic_noflee, oracle, random}:

```text
policy             ag   cap%  median starve hazard  trunc food_left svfood
heuristic           1   20.0    4449      2     10      3    391.2      2
heuristic_noflee    1    0.0    1973      1     14      0    378.3      1
oracle              1    0.0    1015      1     14      0    394.5      1
random              1    0.0    1886      0     15      0    400.0      0
heuristic           5   74.7   10000      4     15     56    358.6      4
heuristic_noflee    5   12.0    4610      4     62      9    380.6      4
oracle              5    4.0    1247      0     72      3    390.2      0
random              5    0.0    3242      9     66      0    400.0      9
heuristic          20   85.3   10000      0     44    256    384.2      0
heuristic_noflee   20   70.0   10000     10     80    210    382.5     10
oracle             20   13.0    2744     16    245     39    394.4     16
random             20    0.0    4262    145    155      0    400.0    145
```

Final 20-seed run, 5 agents: heuristic cap 69.0%, median 10000, split (4,27,69), food_left 363.6, starved_with_food 4; random cap 0%, median 3242, split (10,90,0).

Revised decision rule: sensor cap 40-70% at all counts, random <10%, <10% of heuristic deaths starved with >=50% food uneaten, heuristic hazard deaths non-zero and <40% of deaths. Result: NOT met. Sensor cap is 20% at 1 agent (too low) and 74.7-85.3% at 5/20 agents (above 70); the deaths that occur are dominated by hazard contact (79-100% of heuristic deaths). Random is 0% everywhere. `configs/default.toml` was NOT changed. Regrowth already scales with agent count via `supply_ratio`; at 5/20 agents supply exceeds drain so agents reach the cap, while at 1 agent the lone agent cannot cover the map and dies to hazards.

### Verified
Full suite `47 passed`, ruff clean.

---

## Round 4 - "World v1 baseline"

### Step 1 - decision rule and default copy
New per-agent rule applied to the 5-agent reference only, 20 seeds, cap 10000
(`scripts/rule_check_5agent.py`): heuristic cap 69.0% (PASS 40-75), random cap
0.0% (PASS <10), starved-with->=50%-food-uneaten 4% (PASS <10), heuristic hazard
deaths 27% (PASS 5-50). All criteria passed, so balanced settings were copied into
`configs/default.toml` (hazard detection_radius 16, food supply_ratio 2.0, raw
regrow_rate 0.0). `configs/balanced.toml` and `configs/harsh_original.toml` kept.

### Snapshot/restore regression (fixed)
Copying the defaults enabled nonzero regrowth, which exposed a genuine ordering
bug in `World.step`: the snapshot was captured **before** `_regrow_food`, so a
snapshot at step N omitted step N's regrowth while `state_hash()` (computed
after the step) included it. Restoring such a checkpoint systematically dropped
that regrowth tick, so `test_scrub_back_500_steps_and_replay...` failed for any
nonzero regrowth rate. Fixed by running `_regrow_food` before snapshot capture,
making the snapshot reflect the full post-step state consistent with
`state_hash()`. This was a code fix, not a stale-default test change; no test
needed updating. Full suite: 47 passed; ruff clean.

### Step 2 - hazard scaling diagnosis (no config change)
`scripts/hazard_sweep.py`, 15 seeds, cap 10000. Confirmed the chase rule: each
hazard independently targets the nearest alive agent within its
detection_radius (`World._move_hazards`).

```text
policy      ag  haz   cap% starve hazard  trunc
heuristic    1    1   33.3      6      4      5
random       1    1    0.0      3     12      0
heuristic    1    2   26.7      3      8      4
random       1    2    0.0      1     14      0
heuristic    1    4   20.0      2     10      3
random       1    4    0.0      0     15      0
heuristic    5    4   74.7      4     15     56
random       5    4    0.0      9     66      0
heuristic    5    8   70.7      1     21     53
random       5    8    0.0      0     75      0
heuristic   20    4   85.3      0     44    256
random      20    4    0.0    145    155      0
heuristic   20    8   79.0      0     63    237
random      20    8    0.0     32    268      0
heuristic   20   16   55.3      0    134    166
random      20   16    0.0      1    299      0
```

The data support the hypothesis: at 1 agent each hazard chases the single agent,
so increasing hazard count monotonically lowers the cap (33 -> 27 -> 20%) and
raises hazard deaths. Proposed (not implemented): optional `hazard.per_agent`
(unset by default) analogous to `food.supply_ratio` so hazard pressure scales
with agent count.

### Step 3 - docs/BASELINES.md
Written: reference scenario, results table (random / heuristic_noflee /
heuristic / oracle at 1/5/20 agents), reproduce command per row, commit hash
(`world-v1`), oracle policies marked "not an upper bound; kept for reference
only", and which policies need observations.
