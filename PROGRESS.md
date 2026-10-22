# SimLab progress

## Phase 0+1 — deterministic core

Status: Phase 1 gate passed and committed as `Phase 1: verified deterministic core`.

Audit findings before repair:

- `python -m simlab.cli --version` was unsupported.
- The world used a grid, global `np.random.seed`, and mutable dict state that removed dead agents.
- The runner created policy actions and duplicated state outside the World; `run_episode` referenced a missing registry.
- Snapshots did not reliably include the Generator state and were assembled partly by the runner.
- Tests were duplicated under `simlab/tests/` and scratch files were collected by pytest.

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
