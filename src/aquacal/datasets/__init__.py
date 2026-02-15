"""Synthetic and example datasets for AquaCal.

This module provides utilities for generating synthetic calibration data with
known ground truth. Useful for testing, validation, and demonstration purposes.

The main entry point is `generate_synthetic_rig()`, which produces complete
calibration scenarios with fixed presets.

Examples:
    Quick smoke test scenario::

        from aquacal.datasets import generate_synthetic_rig

        scenario = generate_synthetic_rig('small')
        print(f"{len(scenario.intrinsics)} cameras")
        print(f"{len(scenario.board_poses)} frames")

    Generate with rendered images::

        scenario = generate_synthetic_rig('small', include_images=True)
        img = scenario.images['cam0'][0]  # First frame from cam0

    Add pixel noise for robustness testing::

        scenario = generate_synthetic_rig('medium', noisy=True)
        print(f"Noise std: {scenario.noise_std}px")

Available presets:
    - 'small': 2 cameras, 10 frames - quick smoke test
    - 'medium': 6 cameras, 80 frames - integration testing
    - 'large': 13 cameras (real rig), 300 frames - full-scale realistic scenario

All scenarios use a 12x9 ChArUco board (60mm squares, 45mm markers, DICT_5X5_100).
"""

from aquacal.datasets.synthetic import SyntheticScenario, generate_synthetic_rig

__all__ = ["generate_synthetic_rig", "SyntheticScenario"]
