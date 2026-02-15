"""Synthetic and example datasets for AquaCal.

This module provides utilities for generating synthetic calibration data with
known ground truth, and loading example datasets for testing and validation.

Synthetic Data Generation
-------------------------
Use `generate_synthetic_rig()` to create synthetic scenarios with fixed presets::

    from aquacal.datasets import generate_synthetic_rig

    scenario = generate_synthetic_rig('small')
    print(f"{len(scenario.intrinsics)} cameras")
    print(f"{len(scenario.board_poses)} frames")

Example Dataset Loading
------------------------
Use `load_example()` to load pre-packaged or downloadable example datasets::

    from aquacal.datasets import load_example

    # Load small preset (included, no download)
    ds = load_example('small')
    print(f"{len(ds.ground_truth.intrinsics)} cameras")

    # Access detections
    frame0 = ds.detections.frames[0]
    print(frame0.detections.keys())

Available Datasets
------------------
- 'small': 2 cameras, 10 frames (included with package)
- 'medium': 6 cameras, 80 frames (Zenodo download)
- 'large': 13 cameras, 300 frames (Zenodo download)
- 'real-rig': Real hardware calibration (Zenodo download)

Cache Management
----------------
Downloaded datasets are cached in `./aquacal_data/`::

    from aquacal.datasets import get_cache_info, clear_cache

    # Check cache status
    info = get_cache_info()
    print(f"Cached datasets: {info['cached_datasets']}")

    # Clear specific dataset
    clear_cache('medium')

    # Clear entire cache
    clear_cache()
"""

from aquacal.datasets._manifest import list_datasets
from aquacal.datasets.download import clear_cache, get_cache_info
from aquacal.datasets.loader import ExampleDataset, load_example
from aquacal.datasets.synthetic import SyntheticScenario, generate_synthetic_rig

__all__ = [
    "generate_synthetic_rig",
    "SyntheticScenario",
    "load_example",
    "ExampleDataset",
    "list_datasets",
    "clear_cache",
    "get_cache_info",
]
