# Directory Structure

This file documents the expected directory structure. The project uses the standard Python `src/` layout.

```
AquaCal/
├── CLAUDE.md                      # Agent instructions (read every session)
├── CHANGELOG.md                   # Agent appends changes here
├── TASKS.md                       # Task status (orchestrator maintains)
├── DEPENDENCIES.yaml              # Module dependency graph
│
├── tasks/
│   ├── template.md                # Task file template
│   ├── current.md                 # Active task (replace for each session)
│   └── handoff.md                 # Created if task spans sessions (delete when done)
│
├── docs/
│   ├── COORDINATES.md             # Coordinate frame quick reference
│   ├── development_plan.md        # Architecture and module specs
│   └── agent_implementation_spec.md  # Detailed signatures and types
│
├── src/
│   └── aquacal/                   # Main package
│       ├── __init__.py            # Package init with version
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── schema.py          # Type definitions and dataclasses
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── board.py           # ChArUco board geometry
│       │   ├── camera.py          # Camera model (no refraction)
│       │   ├── interface_model.py # Refractive interface model
│       │   └── refractive_geometry.py  # Ray tracing and Snell's law
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── video.py           # Video loading
│       │   ├── detection.py       # ChArUco detection
│       │   └── serialization.py   # Save/load calibration
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   ├── intrinsics.py      # Stage 1
│       │   ├── extrinsics.py      # Stage 2
│       │   ├── interface_estimation.py  # Stage 3
│       │   ├── refinement.py      # Stage 4
│       │   └── pipeline.py        # Orchestration
│       │
│       ├── triangulation/
│       │   ├── __init__.py
│       │   └── triangulate.py     # Refractive triangulation
│       │
│       ├── validation/
│       │   ├── __init__.py
│       │   ├── reprojection.py    # Reprojection error
│       │   ├── reconstruction.py  # 3D metrics
│       │   └── diagnostics.py     # Error analysis
│       │
│       └── utils/
│           ├── __init__.py
│           ├── transforms.py      # Rotation/pose utilities
│           └── visualization.py   # Plotting
│
└── tests/                         # Test suite (stays at top level)
    ├── unit/                      # Per-module unit tests
    ├── integration/               # Multi-module tests
    └── synthetic/                 # Full pipeline with synthetic data
```

## Import Convention

After this layout, imports use the `aquacal` package name:

```python
# Correct imports
from aquacal.config.schema import CalibrationResult
from aquacal.core.camera import Camera
from aquacal.core.refractive_geometry import snells_law_3d

# The src/ directory is NOT part of the import path
```
