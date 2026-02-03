# Changelog

All notable changes to this project will be documented in this file.

Format: Agents append entries at the top (below this header) with the date, files modified, and a brief summary.

---

<!-- Agents: add new entries below this line, above previous entries -->

## 2026-02-03
### Refactor to src Layout

Reorganized the project from a flat layout to the standard Python `src/` layout.

#### Directory Structure Changes
- Created `src/aquacal/` package structure with all subpackages
- Added `__init__.py` files to all packages (aquacal, calibration, config, core, io, triangulation, utils, validation)
- Moved `config/example_config.yaml` to `src/aquacal/config/example_config.yaml`
- Deleted old top-level package directories (calibration/, config/, core/, io/, triangulation/, utils/, validation/)

#### [DEPENDENCIES.yaml]
- Updated all module paths from flat layout (e.g., `config/schema.py`) to src layout (e.g., `src/aquacal/config/schema.py`)

#### [docs/development_plan.md]
- Updated architecture diagram to reflect new src/aquacal/ structure

#### [docs/agent_implementation_spec.md]
- Updated implementation order diagram paths
- Updated all section headings to new paths
- Updated all import statements to use `aquacal.` prefix

#### [STRUCTURE.md]
- Completely rewrote to reflect new src layout
- Added import convention documentation

#### [CLAUDE.md]
- Updated test file path example to new src layout

---

## 2026-02-03
### Documentation Review and Consistency Fixes

#### [docs/agent_implementation_spec.md]
- Fixed world Z direction: "Z points down (into water)" (was incorrectly "up")
- Fixed interface normal to [0,0,-1] in all examples and specs (was [0,0,1])
- Updated Snell's law examples and tests for Z-down convention
- Added note that TASKS.md is authoritative for phase numbering
- Added InterfaceParams docstring clarifying per-camera distances stored in CameraCalibration
- Updated detect_all_frames to accept VideoSet in addition to dict
- Fixed utils/transforms.py to import Vec3/Mat3 from schema instead of redefining
- Added custom exception classes to schema section
- Confirmed board frame convention matches OpenCV 4.6+ (Z into board)
- Added TODO for refractive_project algorithm details
- Added holdout_fraction comment specifying random frame selection

#### [docs/development_plan.md]
- Fixed interface normal references from [0,0,1] to [0,0,-1]
- Added note that TASKS.md is authoritative for phase numbering
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [docs/COORDINATES.md]
- No changes needed (was already correct)

#### [TASKS.md]
- Added header noting this is authoritative source for task IDs
- Added Phase 0 (Project Setup) with tasks 0.1-0.3 for pyproject.toml, __init__.py, requirements.txt
- Updated interface file references to new names

#### [DEPENDENCIES.yaml]
- Added Phase 0 comment about project setup files
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [STRUCTURE.md]
- Renamed core/interface.py -> core/interface_model.py
- Renamed calibration/interface.py -> calibration/interface_estimation.py

#### [CLAUDE.md]
- Added Testing Conventions section (test file naming, classes, fixtures)

#### [config/example_config.yaml] (new file)
- Created example configuration template with all options documented

#### [tasks/current.md]
- Cleaned up duplicate template text

#### [tests/synthetic/README.md] (new file)
- Added TODO specification for synthetic data generation requirements
