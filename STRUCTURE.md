# Directory Structure

This file documents the expected directory structure. Empty directories include a `.gitkeep` file to ensure they're tracked by git.

```
repo/
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
├── config/
│   └── schema.py                  # Type definitions and dataclasses
│
├── core/
│   ├── board.py                   # ChArUco board geometry
│   ├── camera.py                  # Camera model (no refraction)
│   ├── interface.py               # Refractive interface model
│   └── refractive_geometry.py     # Ray tracing and Snell's law
│
├── io/
│   ├── video.py                   # Video loading
│   ├── detection.py               # ChArUco detection
│   └── serialization.py           # Save/load calibration
│
├── calibration/
│   ├── intrinsics.py              # Stage 1
│   ├── extrinsics.py              # Stage 2
│   ├── interface.py               # Stage 3
│   ├── refinement.py              # Stage 4
│   └── pipeline.py                # Orchestration
│
├── triangulation/
│   └── triangulate.py             # Refractive triangulation
│
├── validation/
│   ├── reprojection.py            # Reprojection error
│   ├── reconstruction.py          # 3D metrics
│   └── diagnostics.py             # Error analysis
│
├── utils/
│   ├── transforms.py              # Rotation/pose utilities
│   └── visualization.py           # Plotting
│
└── tests/
    ├── unit/                      # Per-module unit tests
    ├── integration/               # Multi-module tests
    └── synthetic/                 # Full pipeline with synthetic data
```
