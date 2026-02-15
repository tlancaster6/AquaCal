API Reference
=============

Complete reference for all public modules, classes, and functions.

AquaCal's API is organized into seven main subpackages:

- **core**: Refractive geometry, camera models, and board/interface models
- **calibration**: Four-stage calibration pipeline (intrinsics, extrinsics, interface estimation, refinement)
- **config**: Configuration schema and calibration result types
- **io**: Serialization, detection loading, and video processing
- **validation**: Reprojection error analysis, diagnostics, and reconstruction validation
- **triangulation**: Multi-camera 3D reconstruction through refractive interface
- **datasets**: Example data loaders and synthetic data generation

.. toctree::
   :maxdepth: 2

   core
   calibration
   config
   io
   validation
   triangulation
   datasets
