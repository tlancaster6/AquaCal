# AquaCal

## What This Is

AquaCal is a Python library for calibrating multi-camera arrays that view underwater scenes through a flat water surface. It models Snell's law refraction at the air-water interface to jointly optimize camera extrinsics, water surface position, and calibration board poses. Target audience is the underwater computer vision research community.

## Core Value

Accurate refractive camera calibration from standard ChArUco board observations — researchers can `pip install aquacal`, point it at their videos, and get a calibration result they trust.

## Requirements

### Validated

<!-- Inferred from existing codebase -->

- ✓ Four-stage calibration pipeline (intrinsics, extrinsics init, joint optimization, optional refinement) — existing
- ✓ Refractive projection through flat air-water interface using Snell's law — existing
- ✓ Pinhole and fisheye (equidistant) camera model support — existing
- ✓ ChArUco board detection with filtering (min corners, collinearity) — existing
- ✓ BFS pose graph for extrinsic initialization with rotation averaging — existing
- ✓ Sparse Jacobian optimization with Newton-Raphson fast projection — existing
- ✓ Auxiliary camera registration (cameras not seeing shared frames) — existing
- ✓ Optional interface tilt estimation — existing
- ✓ Validation with held-out frames (reprojection errors, 3D distance errors) — existing
- ✓ Diagnostic report generation with visualizations — existing
- ✓ Cross-run calibration comparison (CSV + PNG) — existing
- ✓ CLI with calibrate, init, and compare commands — existing
- ✓ JSON serialization of calibration results — existing
- ✓ YAML-based configuration — existing
- ✓ Public API: run_calibration(), load/save_calibration(), core types — existing

### Active

- [ ] Clean pip-installable package on PyPI
- [ ] Getting started tutorial (end-to-end: collect images to calibration result)
- [ ] Theory/math background documentation (refractive geometry, coordinate conventions, optimizer)
- [ ] Example datasets (real calibration data + synthetic)
- [ ] Jupyter notebook examples demonstrating the pipeline
- [ ] Cleanup of legacy development artifacts (old agent configs, task files)
- [ ] CI/CD pipeline (GitHub Actions for tests, linting, publishing)

### Out of Scope

- Web interface or REST API — this is a library/CLI tool
- GPU acceleration — CPU-only NumPy/SciPy is sufficient for calibration workloads
- Real-time calibration or streaming — batch processing only
- Non-flat interface models (curved surfaces, waves) — flat plane approximation is the scope
- Support for non-ChArUco calibration targets — ChArUco only for now

## Context

AquaCal is a mature codebase with a working four-stage pipeline, comprehensive validation, and diagnostics. The core calibration engine is solid — this milestone is about packaging and presentation for the research community.

The existing `pyproject.toml` already defines the package structure with setuptools, entry points, and dependency declarations. The public API (`__init__.py` exports) is considered stable.

Key technical context:
- Z-down world frame, OpenCV camera convention
- Global `water_z` parameter (not per-camera interface distances)
- Sparse Jacobian with column grouping for optimization efficiency
- No CI/CD exists yet — no automated testing or publishing pipeline

## Constraints

- **Python compatibility**: 3.10, 3.11, 3.12 — already declared in pyproject.toml
- **Dependencies**: Must remain lightweight (NumPy, SciPy, OpenCV, Matplotlib, Pandas, PyYAML)
- **License**: TBD — needs decision before public release
- **Example data size**: Real datasets must be small enough for reasonable download (or hosted externally)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyPI + GitHub distribution | Standard for research Python libraries | — Pending |
| Real + synthetic example data | Real data builds trust, synthetic demonstrates correctness | — Pending |
| Jupyter notebooks for examples | Interactive, visual — ideal for research audience | — Pending |
| License choice | Needed before public release | — Pending |

---
*Last updated: 2026-02-14 after initialization*
