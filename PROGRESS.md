# SimLab Progress

## Phase 0: Project Setup

- [x] Initialize git repository
- [x] Create project structure (pyproject.toml, README.md, PROGRESS.md)
- [x] Create folder layout as specified
- [x] Create core module stubs (types.py, registry.py, config.py, rng.py, world.py, entities.py, physics.py, sensors.py, snapshot.py, runner.py)
- [x] Create interfaces module stubs (drive_core.py, policy.py)
- [x] Create configs\default.toml
- [x] Create docs\EXTENDING.md

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

### How to run:
N/A - project structure only at this point

### Key decisions:
- Using numpy for world state management
- PySide6 + pyqtgraph for GUI
- Plugin-style architecture with decorator-based registration
- Clean separation between core and GUI

### Open issues:
- Implement full functionality in each module
- Add type hints and dataclasses
- Implement registry pattern
- Implement RNG management
- Implement World class with reset/step
- Implement Entity classes
- Implement physics system
- Implement sensor system
- Implement snapshot/restore
- Implement headless runner
- Implement DriveCore and Policy protocols
