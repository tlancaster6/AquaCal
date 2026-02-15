# Tutorials

Interactive Jupyter notebook tutorials demonstrating AquaCal's calibration pipeline with real and synthetic data.

Each tutorial is a self-contained notebook you can run locally or on Google Colab to explore calibration workflows, diagnose optimization issues, and validate results.

## Available Tutorials

**Full Pipeline Walkthrough** — End-to-end calibration from video files to 3D reconstruction results. Covers ChArUco detection, intrinsic/extrinsic initialization, joint bundle adjustment, and result validation.

**Diagnostics and Troubleshooting** — Understanding optimizer convergence, reprojection error patterns, and quality metrics. Learn how to identify problematic frames, interpret residual plots, and tune optimization settings.

**Synthetic Data Testing** — Generate ground truth calibration data with known parameters, run the pipeline, and measure accuracy. Demonstrates validation methodology and parameter recovery analysis.

:::{note}
Notebooks will be added in Phase 6 Plans 03-04. The toctree below is prepared for upcoming content.
:::

:::{toctree}
:maxdepth: 1
:hidden:

01_full_pipeline
02_diagnostics
03_synthetic_validation
:::
