# Tutorials

Interactive Jupyter notebook tutorials demonstrating AquaCal's calibration pipeline with real and synthetic data.

Each tutorial is self-contained and can be run locally or on Google Colab.

## Tutorial 01: Calibrate Your Rig

End-to-end calibration from data loading to validated 3D results. Covers ChArUco detection, intrinsic/extrinsic initialization, joint refractive bundle adjustment, and a built-in diagnostics section for interpreting reprojection errors, checking interface distance recovery, and troubleshooting common issues.

**Start here** if you want to calibrate a real or synthetic underwater multi-camera rig.

## Tutorial 02: Why Refractive Calibration Matters

Controlled synthetic experiments that quantify what you gain from modeling Snell's law refraction. Compares refractive vs non-refractive calibration on the same data — showing how non-refractive models introduce systematic bias in focal length and camera position even when reprojection error looks acceptable.

**Start here** if you want to understand when the refractive model is essential and how to validate parameter recovery accuracy.

:::{toctree}
:maxdepth: 1
:hidden:

01_full_pipeline
02_synthetic_validation
:::
