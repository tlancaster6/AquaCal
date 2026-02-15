"""Dataset manifest loading and registry."""

from __future__ import annotations

import json
from importlib.resources import files


def get_manifest() -> dict:
    """Load the dataset manifest from package data.

    Returns:
        Dict containing dataset metadata including Zenodo URLs and checksums.

    Raises:
        FileNotFoundError: If manifest.json is missing from package data
        ValueError: If manifest format is invalid
    """
    manifest_path = files("aquacal.datasets").joinpath("data/manifest.json")

    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(manifest_text)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Dataset manifest not found. The manifest.json file should be "
            "included in the package data."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid manifest.json format: {e}")

    return manifest


def get_dataset_info(name: str) -> dict:
    """Get metadata for a specific dataset.

    Args:
        name: Dataset name (e.g., 'small', 'medium', 'large', 'real-rig')

    Returns:
        Dict containing dataset metadata

    Raises:
        ValueError: If dataset name is not found in manifest
    """
    manifest = get_manifest()
    datasets = manifest.get("datasets", {})

    if name not in datasets:
        available = list(datasets.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available datasets: {available}")

    return datasets[name]


def list_datasets() -> list[str]:
    """List all available dataset names.

    Returns:
        List of dataset names from the manifest
    """
    manifest = get_manifest()
    return list(manifest.get("datasets", {}).keys())
