---
created: 2026-02-15T14:46:42.474Z
title: Design better hero image for README
area: docs
files:
  - README.md
  - docs/_static/hero_ray_trace.png
---

## Problem

The current hero image (`hero_ray_trace.png`) is a refractive ray diagram. While technically accurate, it doesn't convey "innovative" or make the project visually compelling at first glance. The README hero image is the first thing researchers see and should immediately communicate what AquaCal does.

## Solution

Two candidate approaches:

1. **3D calibration rig visualization** — Render a 3D scene showing cameras mounted above a water surface viewing a submerged ChArUco board, with refracted rays visible. Would convey the physical setup at a glance.

2. **Graphical abstract of the pipeline** — A polished diagram showing the full workflow: multi-camera input -> detection -> refractive bundle adjustment -> calibration output, with visual icons/illustrations at each stage. Common in academic software to communicate scope quickly.

Either approach should be visually striking enough to stop a researcher scrolling through GitHub.
