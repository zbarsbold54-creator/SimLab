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
