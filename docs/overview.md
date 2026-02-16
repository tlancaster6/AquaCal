# What is Refractive Calibration?

## The Problem

Standard multi-camera calibration assumes cameras and targets occupy the same optical medium. For underwater scenarios, this assumption breaks down: cameras are in air, viewing targets underwater through a flat water surface. Light rays from underwater targets refract at the air-water interface following Snell's law. Standard calibration methods ignore refraction, leading to systematic errors in 3D reconstruction.

## What AquaCal Does

AquaCal jointly optimizes:
- Camera extrinsics (position and orientation)
- Water surface position
- Calibration board poses

using refractive ray tracing to accurately model light paths through the interface. The calibration process uses standard ChArUco board observations but accounts for refraction during optimization.

## Why It Matters

Without refractive modeling, underwater 3D reconstructions exhibit:
- Biased depth estimates (targets appear shallower than actual position)
- Systematic position errors increasing with distance from camera
- Breaking of geometric constraints (parallel lines appear non-parallel)

AquaCal provides accurate calibration for research applications requiring precise underwater measurements: behavioral tracking, environmental monitoring, volumetric capture.

## Learn More

For detailed explanations of theory and implementation:

- [Refractive Geometry](guide/refractive_geometry.md) — ray tracing through the water interface
- [Coordinate Conventions](guide/coordinates.md) — world frame, camera frame, transforms
- [Optimizer Pipeline](guide/optimizer.md) — bundle adjustment structure, parameters, and camera models

For practical usage:

- [CLI Reference](guide/cli.md) — command-line tools and options
- [Troubleshooting](guide/troubleshooting.md) — common issues and solutions
- [User Guide](guide/index) — complete theory and practical guides
- [API Reference](api/index) — function and class documentation
- [Tutorials](tutorials/index) — interactive notebook examples
