"""Example dataset loading and management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np

from aquacal.config.schema import (
    BoardConfig,
    BoardPose,
    CalibrationResult,
    CameraExtrinsics,
    CameraIntrinsics,
    Detection,
    DetectionResult,
    FrameDetections,
)
from aquacal.datasets._manifest import get_dataset_info
from aquacal.datasets.download import download_and_extract
from aquacal.datasets.synthetic import SyntheticScenario


@dataclass
class ExampleDataset:
    """Example calibration dataset with detections and optional ground truth.

    Attributes:
        name: Dataset name (e.g., 'small', 'medium', 'large', 'real-rig')
        type: Dataset type ('synthetic' or 'real')
        detections: Detection results for all cameras and frames
        ground_truth: Ground truth scenario (synthetic datasets only)
        reference_calibration: Optional reference calibration result
        metadata: Additional metadata about the dataset
        cache_path: Path to cached dataset files (None for included datasets)
    """

    name: str
    type: str
    detections: DetectionResult
    ground_truth: SyntheticScenario | None = None
    reference_calibration: CalibrationResult | None = None
    metadata: dict = field(default_factory=dict)
    cache_path: Path | None = None


def _deserialize_detections(data: dict) -> DetectionResult:
    """Deserialize DetectionResult from JSON dict.

    Args:
        data: Serialized detection data

    Returns:
        DetectionResult object
    """
    frames = {}
    for frame_idx_str, frame_data in data["frames"].items():
        frame_idx = int(frame_idx_str)
        detections_dict = {}

        for cam_name, det_data in frame_data["detections"].items():
            detections_dict[cam_name] = Detection(
                corner_ids=np.array(det_data["corner_ids"], dtype=np.int32),
                corners_2d=np.array(det_data["corners_2d"], dtype=np.float64),
            )

        frames[frame_idx] = FrameDetections(
            frame_idx=frame_idx,
            detections=detections_dict,
        )

    return DetectionResult(
        frames=frames,
        camera_names=data["camera_names"],
        total_frames=data["total_frames"],
    )


def _deserialize_ground_truth(data: dict) -> SyntheticScenario:
    """Deserialize SyntheticScenario from JSON dict.

    Args:
        data: Serialized ground truth data

    Returns:
        SyntheticScenario object
    """
    # Deserialize board config
    bc = data["board_config"]
    board_config = BoardConfig(
        squares_x=bc["squares_x"],
        squares_y=bc["squares_y"],
        square_size=bc["square_size"],
        marker_size=bc["marker_size"],
        dictionary=bc["dictionary"],
    )

    # Deserialize intrinsics
    intrinsics = {}
    for cam_name, intr_data in data["intrinsics"].items():
        intrinsics[cam_name] = CameraIntrinsics(
            K=np.array(intr_data["K"], dtype=np.float64),
            dist_coeffs=np.array(intr_data["dist_coeffs"], dtype=np.float64),
            image_size=tuple(intr_data["image_size"]),
        )

    # Deserialize extrinsics
    extrinsics = {}
    for cam_name, extr_data in data["extrinsics"].items():
        extrinsics[cam_name] = CameraExtrinsics(
            R=np.array(extr_data["R"], dtype=np.float64),
            t=np.array(extr_data["t"], dtype=np.float64),
        )

    # Deserialize interface distances
    water_zs = {cam_name: float(dist) for cam_name, dist in data["water_zs"].items()}

    # Deserialize board poses
    board_poses = []
    for pose_data in data["board_poses"]:
        board_poses.append(
            BoardPose(
                frame_idx=pose_data["frame_idx"],
                rvec=np.array(pose_data["rvec"], dtype=np.float64),
                tvec=np.array(pose_data["tvec"], dtype=np.float64),
            )
        )

    return SyntheticScenario(
        name=data["name"],
        board_config=board_config,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        water_zs=water_zs,
        board_poses=board_poses,
        noise_std=data["noise_std"],
        description=data["description"],
        images=None,
    )


def load_example(name: str) -> ExampleDataset:
    """Load an example calibration dataset.

    This function provides access to example datasets for testing and validation.
    The 'small' preset is included with the package and loads instantly. Larger
    datasets are downloaded from Zenodo on first use and cached locally.

    Args:
        name: Dataset name. Available options:
            - 'small': 2 cameras, 10 frames (included, no download)
            - 'medium': 6 cameras, 80 frames (Zenodo download)
            - 'large': 13 cameras, 300 frames (Zenodo download)
            - 'real-rig': Real hardware calibration (Zenodo download)

    Returns:
        ExampleDataset with detections and ground truth (if synthetic)

    Raises:
        ValueError: If dataset name is not recognized
        NotImplementedError: If dataset is not yet available for download

    Examples:
        >>> from aquacal.datasets import load_example
        >>> # Load small preset (instant, no download)
        >>> ds = load_example('small')
        >>> print(f"{len(ds.ground_truth.intrinsics)} cameras")
        2 cameras
        >>>
        >>> # Access detections
        >>> frame0 = ds.detections.frames[0]
        >>> print(frame0.detections.keys())
        dict_keys(['cam0', 'cam1'])
        >>>
        >>> # Access ground truth (synthetic datasets only)
        >>> gt_intrinsics = ds.ground_truth.intrinsics['cam0']
        >>> print(gt_intrinsics.K[0, 0])  # Focal length
    """
    # Get dataset metadata from manifest
    dataset_info = get_dataset_info(name)

    if dataset_info["included"]:
        # Load from package data
        data_path = files("aquacal.datasets").joinpath(f"data/{name}")

        # Load detections
        detections_file = data_path / "detections.json"
        detections_text = detections_file.read_text(encoding="utf-8")
        detections_data = json.loads(detections_text)
        detections = _deserialize_detections(detections_data)

        # Load ground truth (synthetic datasets only)
        ground_truth = None
        if dataset_info["type"] == "synthetic":
            gt_file = data_path / "ground_truth.json"
            gt_text = gt_file.read_text(encoding="utf-8")
            gt_data = json.loads(gt_text)
            ground_truth = _deserialize_ground_truth(gt_data)

        return ExampleDataset(
            name=name,
            type=dataset_info["type"],
            detections=detections,
            ground_truth=ground_truth,
            metadata={"description": dataset_info["description"]},
            cache_path=None,
        )

    else:
        # Dataset requires download
        if dataset_info.get("zenodo_record_id") is None:
            raise NotImplementedError(
                f"Dataset '{name}' is not yet available for download. "
                f"Use generate_synthetic_rig('{name}') to generate it locally, "
                f"or check back later when it's uploaded to Zenodo."
            )

        # Download and extract (cached)
        _cache_path = download_and_extract(name, dataset_info)

        # Handle nested directory structure (Zenodo archives often have top-level folder)
        if (_cache_path / name).exists():
            actual_path = _cache_path / name
        else:
            actual_path = _cache_path

        # Load based on dataset type
        if dataset_info["type"] == "real":
            # Real datasets have config.yaml and optional reference_calibration.json
            # but no pre-serialized detections
            reference_calibration = None
            ref_calib_file = actual_path / "reference_calibration.json"
            if ref_calib_file.exists():
                with open(ref_calib_file, encoding="utf-8") as f:
                    ref_data = json.load(f)
                # TODO: Deserialize CalibrationResult from JSON
                # For now, just store the raw dict
                reference_calibration = ref_data

            # Read config for metadata
            import yaml

            config_file = actual_path / "config.yaml"
            if config_file.exists():
                with open(config_file, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                camera_names = config_data.get("cameras", [])
            else:
                camera_names = []

            return ExampleDataset(
                name=name,
                type=dataset_info["type"],
                detections=DetectionResult(
                    frames={}, camera_names=camera_names, total_frames=0
                ),
                ground_truth=None,
                reference_calibration=reference_calibration,
                metadata={
                    "description": dataset_info["description"],
                    "dataset_path": str(actual_path),
                    "has_reference_calibration": ref_calib_file.exists(),
                },
                cache_path=actual_path,
            )

        else:
            # Synthetic datasets have serialized detections.json and ground_truth.json
            detections_file = actual_path / "detections.json"
            if not detections_file.exists():
                raise FileNotFoundError(
                    f"Expected detections.json in {actual_path} for synthetic dataset"
                )

            with open(detections_file, encoding="utf-8") as f:
                detections_data = json.load(f)
            detections = _deserialize_detections(detections_data)

            ground_truth = None
            gt_file = actual_path / "ground_truth.json"
            if gt_file.exists():
                with open(gt_file, encoding="utf-8") as f:
                    gt_data = json.load(f)
                ground_truth = _deserialize_ground_truth(gt_data)

            return ExampleDataset(
                name=name,
                type=dataset_info["type"],
                detections=detections,
                ground_truth=ground_truth,
                metadata={"description": dataset_info["description"]},
                cache_path=actual_path,
            )
