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
