# AquaCal

## What This Is

AquaCal is a Python library for calibrating multi-camera arrays that view underwater scenes through a flat water surface. It models Snell's law refraction at the air-water interface to jointly optimize camera extrinsics, water surface position, and calibration board poses. Available on PyPI with Sphinx documentation, example datasets, and Jupyter tutorials.

## Core Value

Accurate refractive camera calibration from standard ChArUco board observations — researchers can `pip install aquacal`, point it at their videos, and get a calibration result they trust.

## Requirements

### Validated

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
- ✓ Clean pip-installable package on PyPI — v1.2
- ✓ Getting started tutorial (end-to-end: collect images to calibration result) — v1.2
- ✓ Theory/math background documentation (refractive geometry, coordinate conventions, optimizer) — v1.2
- ✓ Example datasets (real calibration data + synthetic) — v1.2
- ✓ Jupyter notebook examples demonstrating the pipeline — v1.2
- ✓ Cleanup of legacy development artifacts — v1.2
- ✓ CI/CD pipeline (GitHub Actions for tests, linting, publishing) — v1.2

### Active

## Current Milestone: v1.3 QA & Polish

**Goal:** Human-in-the-loop QA of CLI workflows with real rig data, documentation audit and polish, and shipping pending infrastructure todos.

**Target features:**
- Human QA of calibrate, init, and compare CLI workflows with real data
- Documentation audit (inconsistencies, redundancy, factual errors)
- Improved visualizations and ASCII diagrams in docs
- Tutorial verification with correct embedded outputs
- Visual abstract / hero image creation
- Ship pending todos (Read the Docs, DOI badge, RELEASE_TOKEN)

### Out of Scope

- Web interface or REST API — this is a library/CLI tool
- GPU acceleration — CPU-only NumPy/SciPy is sufficient for calibration workloads
- Real-time calibration or streaming — batch processing only
- Non-flat interface models (curved surfaces, waves) — flat plane approximation is the scope
- Support for non-ChArUco calibration targets — ChArUco only for now
- Conda-forge recipe — defer until PyPI adoption validates demand
- Docker container — defer until reproducibility requests arrive
- Multi-language bindings — Python-only for now

## Context

Shipped v1.2 with ~39,900 LOC Python.
Tech stack: NumPy, SciPy, OpenCV, Matplotlib, Pandas, PyYAML, Sphinx (Furo), GitHub Actions.
Published on PyPI as `aquacal` v1.2.0. Sphinx docs with theory pages, API reference, and Jupyter tutorials.
Zenodo DOI pending webhook setup.

Known issues:
- Read the Docs deployment not yet configured (docs build locally)
- Zenodo DOI badge placeholder until webhook is set up
- RELEASE_TOKEN PAT needed for semantic-release to trigger publish workflow

## Constraints

- **Python compatibility**: 3.10, 3.11, 3.12
- **Dependencies**: Lightweight (NumPy, SciPy, OpenCV, Matplotlib, Pandas, PyYAML)
- **License**: MIT

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyPI + GitHub distribution | Standard for research Python libraries | ✓ Good — v1.2.0 live on PyPI |
| Real + synthetic example data | Real data builds trust, synthetic demonstrates correctness | ✓ Good — both available |
| Jupyter notebooks for examples | Interactive, visual — ideal for research audience | ✓ Good — 3 notebooks shipped |
| MIT license | Maximizes adoption in research community | ✓ Good |
| Ruff over Black/mypy | Faster, all-in-one linting and formatting | ✓ Good |
| Trusted Publishing (OIDC) | No API tokens needed for PyPI | ✓ Good |
| python-semantic-release | Automates version bumping from conventional commits | ✓ Good |
| Sphinx + Furo theme | Clean, modern docs with MyST Markdown | ✓ Good |
| FrameSet Protocol | Structural subtyping for image/video input flexibility | ✓ Good |
| Pre-execute notebooks | Reproducible docs builds without runtime dependencies | ✓ Good |

---
*Last updated: 2026-02-15 after v1.3 milestone started*
