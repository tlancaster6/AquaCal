# AquaCal Documentation

[![Build Status](https://github.com/tlancaster6/AquaCal/actions/workflows/test.yml/badge.svg)](https://github.com/tlancaster6/AquaCal/actions/workflows/test.yml)
[![Coverage](https://codecov.io/gh/tlancaster6/AquaCal/branch/main/graph/badge.svg)](https://codecov.io/gh/tlancaster6/AquaCal)
[![PyPI version](https://badge.fury.io/py/aquacal.svg)](https://badge.fury.io/py/aquacal)
[![Python](https://img.shields.io/pypi/pyversions/aquacal.svg)](https://pypi.org/project/aquacal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AquaCal is a refractive multi-camera calibration library for underwater arrays. It calibrates air cameras viewing through a flat water surface, modeling Snell's law refraction for accurate 3D reconstruction.

**All values are in meters.** This includes camera positions, water surface position, board dimensions, and 3D reconstruction outputs.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📖 Overview
:link: overview
:link-type: doc

Learn what refractive calibration is and why standard calibration fails for underwater scenarios.
:::

:::{grid-item-card} 🧭 User Guide
:link: guide/index
:link-type: doc

Understand the theory: refractive geometry, coordinate conventions, and optimizer pipeline.
:::

:::{grid-item-card} 📚 API Reference
:link: api/index
:link-type: doc

Detailed API documentation for all public modules and functions.
:::

:::{grid-item-card} 🎓 Tutorials
:link: tutorials/index
:link-type: doc

Interactive Jupyter notebook examples: full pipeline, diagnostics, and synthetic validation.
:::

::::

## Quick Start

```python
from aquacal import run_calibration, load_calibration

# Run calibration from YAML config
result = run_calibration("config.yaml")

# Load saved calibration
calib = load_calibration("output/calibration.yaml")
```

:::{toctree}
:hidden:
:maxdepth: 2

overview
guide/index
api/index
tutorials/index
contributing
changelog
:::
