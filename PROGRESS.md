# SimLab Progress

## Phase 0: Project Setup

- [x] Initialize git repository
- [x] Create project structure (pyproject.toml, README.md, PROGRESS.md)
- [x] Create folder layout as specified
- [x] Create core module stubs (types.py, registry.py, config.py, rng.py, world.py, entities.py, physics.py, sensors.py, snapshot.py, runner.py)
- [x] Create interfaces module stubs (drive_core.py, policy.py)
- [x] Create configs\default.toml
- [x] Create docs\EXTENDING.md
- [x] Implement core types (Snapshot, AgentState, Action, Observation dataclasses)
- [x] Implement RNG module with seeded Generator and child streams
- [x] Implement config module with TOML loading/saving
- [x] Implement World class with reset/step
- [x] Implement Entity classes (Agent, FoodPatch, Hazard)
- [x] Implement physics system
- [x] Implement sensor system
- [x] Implement snapshot/restore module
- [x] Implement headless runner
- [x] Implement registry pattern
- [x] Implement DriveCore and Policy protocols

### What exists:
- Git repo initialized, branch main
- pyproject.toml with dependencies (numpy, PySide6, pyqtgraph)
- README.md with getting started guide
- PROGRESS.md tracking progress
- All folder structure created as specified
- Core module stubs created
- Interfaces module stubs created
- Config file created
- Extending documentation created
- Types module implemented with full dataclasses
- RNG module implemented with SimRNG class supporting child streams
- Config module implemented with TOML loading/saving and dataclass configuration
- World class implemented with reset/step functionality, agent management, food/hazard handling
- Entity classes implemented with Agent, FoodPatch, Hazard dataclasses and EntityRegistry
- Physics module implemented with movement, collision detection, arena bounds, and reflection
- Sensor module implemented with ray-cast vision and proprioception
- Snapshot/restore module implemented with ActionLogEntry and SnapshotManager
- Headless runner implemented with Runner class supporting run_step, run_episode, run_bench
- Registry pattern implemented with Registry class and global registry
- DriveCore and Policy protocols implemented with RetentionRule and example implementations

### How to run:
N/A - project structure only at this point

### Key decisions:
- Using numpy for world state management
- PySide6 + pyqtgraph for GUI
- Plugin-style architecture with decorator-based registration
- Clean separation between core and GUI
- Immutable Snapshot pattern for deterministic state
- numpy Generator for reproducible RNG with child stream support
- TOML-based configuration with dataclass mapping
- Deterministic world simulation with snapshot/restore capability
- Entity-component pattern with registry for managing agents, food, and hazards
- Physics system with boundary checking, collision detection, and toroidal wrapping support
- Sensor system with ray-casting vision and proprioception
- Snapshot system with action-log recording and replay capability
- Headless runner for benchmarking and headless operation
- Registry pattern with decorator-based registration and lookup
- DriveCore and Policy protocols for plugin-style drive and policy systems

### Phase 1: Core Modules Functional ✅

- [x] Same seed produces identical trajectory
- [x] Snapshot restore reproduces exact future
- [x] Action log replay matches original run
- [x] Rename venv to .venv (collides with simlab\ package)
- [x] .venv added to .gitignore
- [x] Removed _entity_registry from runner (reads state from world directly)
- [x] CLI entry point created (python -m simlab.cli run --steps 1000 --seed 1)
- [x] All 3 Phase 1 tests pass
- [x] Snapshot design fixed: World produces its own snapshot with ALL state (agent fields, food, step counter, RNG state)
- [x] Runner no longer passes agent_positions/agent_energies into snapshot
- [x] Simulation evolves with different seeds (verified: seed 1 ≠ seed 2)
- [x] Run summary includes: steps, seed, per-agent final position, energy, food eaten, final state hash

### Open issues:
- Full integration testing of all modules together
- Type hints completeness across all files
- CLI --steps 1000 performance tuning
