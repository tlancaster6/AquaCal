# User Guide

This section explains the theory behind AquaCal's calibration approach and provides practical guidance for using the tools. Each page is self-contained — you can jump to any topic without reading others first.

## Theory Pages

- [Refractive Geometry](refractive_geometry.md) — How light rays refract at the air-water interface, refractive projection, and ray tracing
- [Coordinate Conventions](coordinates.md) — World frame, camera frame, pixel coordinates, and transformations
- [Optimizer Pipeline](optimizer.md) — Four-stage calibration pipeline, bundle adjustment structure, and camera models

## Practical Guides

- [CLI Reference](cli.md) — Command-line interface documentation for `calibrate`, `init`, and `compare`
- [Troubleshooting](troubleshooting.md) — Common issues and solutions
- [Glossary](glossary.md) — Definitions of key terms

:::{toctree}
:hidden:
:maxdepth: 2

refractive_geometry
coordinates
optimizer
cli
troubleshooting
glossary
:::
