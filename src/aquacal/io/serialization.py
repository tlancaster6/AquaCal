"""Save and load calibration results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import (
    BoardConfig,
    CalibrationMetadata,
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    DiagnosticsData,
    InterfaceParams,
)

# Current serialization format version
SERIALIZATION_VERSION = "1.0"


def _ndarray_to_list(arr: NDArray) -> list:
    """Convert numpy array to nested Python list."""
    return arr.tolist()


def _list_to_ndarray(lst: list, dtype: type = np.float64) -> NDArray:
    """Convert nested list to numpy array."""
    return np.array(lst, dtype=dtype)


def _serialize_camera_intrinsics(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    """Serialize CameraIntrinsics to dict."""
    result = {
        "K": _ndarray_to_list(intrinsics.K),
        "dist_coeffs": _ndarray_to_list(intrinsics.dist_coeffs),
        "image_size": list(intrinsics.image_size),
    }
    if intrinsics.is_fisheye:
        result["is_fisheye"] = True
    return result


def _deserialize_camera_intrinsics(data: dict[str, Any]) -> CameraIntrinsics:
    """Deserialize dict to CameraIntrinsics."""
    return CameraIntrinsics(
        K=_list_to_ndarray(data["K"]),
        dist_coeffs=_list_to_ndarray(data["dist_coeffs"]),
        image_size=tuple(data["image_size"]),
        is_fisheye=data.get("is_fisheye", False),
    )


def _serialize_camera_extrinsics(extrinsics: CameraExtrinsics) -> dict[str, Any]:
    """Serialize CameraExtrinsics to dict."""
    return {
        "R": _ndarray_to_list(extrinsics.R),
        "t": _ndarray_to_list(extrinsics.t),
    }


def _deserialize_camera_extrinsics(data: dict[str, Any]) -> CameraExtrinsics:
    """Deserialize dict to CameraExtrinsics."""
    return CameraExtrinsics(
        R=_list_to_ndarray(data["R"]),
        t=_list_to_ndarray(data["t"]),
    )


def _serialize_camera_calibration(cam: CameraCalibration) -> dict[str, Any]:
    """Serialize CameraCalibration to dict."""
    result = {
        "name": cam.name,
        "intrinsics": _serialize_camera_intrinsics(cam.intrinsics),
        "extrinsics": _serialize_camera_extrinsics(cam.extrinsics),
        "water_z": cam.water_z,
    }
    if cam.is_auxiliary:
        result["is_auxiliary"] = True
    return result


def _deserialize_camera_calibration(data: dict[str, Any]) -> CameraCalibration:
    """Deserialize dict to CameraCalibration.

    Supports backward compatibility: accepts both 'water_z' (new) and
    'interface_distance' (legacy).
    """
    # Backward compatibility: accept both water_z and interface_distance
    if "water_z" in data:
        water_z = data["water_z"]
    elif "interface_distance" in data:
        water_z = data["interface_distance"]
    else:
        raise ValueError(
            "Missing 'water_z' or 'interface_distance' field in camera calibration"
        )

    return CameraCalibration(
        name=data["name"],
        intrinsics=_deserialize_camera_intrinsics(data["intrinsics"]),
        extrinsics=_deserialize_camera_extrinsics(data["extrinsics"]),
        water_z=water_z,
        is_auxiliary=data.get("is_auxiliary", False),
    )


def _serialize_interface_params(interface: InterfaceParams) -> dict[str, Any]:
    """Serialize InterfaceParams to dict."""
    return {
        "normal": _ndarray_to_list(interface.normal),
        "n_air": interface.n_air,
        "n_water": interface.n_water,
    }


def _deserialize_interface_params(data: dict[str, Any]) -> InterfaceParams:
    """Deserialize dict to InterfaceParams."""
    return InterfaceParams(
        normal=_list_to_ndarray(data["normal"]),
        n_air=data["n_air"],
        n_water=data["n_water"],
    )


def _serialize_board_config(board: BoardConfig) -> dict[str, Any]:
    """Serialize BoardConfig to dict."""
    return {
        "squares_x": board.squares_x,
        "squares_y": board.squares_y,
        "square_size": board.square_size,
        "marker_size": board.marker_size,
        "dictionary": board.dictionary,
    }


def _deserialize_board_config(data: dict[str, Any]) -> BoardConfig:
    """Deserialize dict to BoardConfig."""
    return BoardConfig(
        squares_x=data["squares_x"],
        squares_y=data["squares_y"],
        square_size=data["square_size"],
        marker_size=data["marker_size"],
        dictionary=data["dictionary"],
    )


def _serialize_diagnostics(diag: DiagnosticsData) -> dict[str, Any]:
    """Serialize DiagnosticsData to dict. Omits None fields."""
    result = {
        "reprojection_error_rms": diag.reprojection_error_rms,
        "reprojection_error_per_camera": diag.reprojection_error_per_camera,
        "validation_3d_error_mean": diag.validation_3d_error_mean,
        "validation_3d_error_std": diag.validation_3d_error_std,
    }
    # Only include optional fields if not None
    if diag.per_corner_residuals is not None:
        result["per_corner_residuals"] = _ndarray_to_list(diag.per_corner_residuals)
    if diag.per_frame_errors is not None:
        # Convert int keys to strings for JSON compatibility
        result["per_frame_errors"] = {
            str(k): v for k, v in diag.per_frame_errors.items()
        }
    return result


def _deserialize_diagnostics(data: dict[str, Any]) -> DiagnosticsData:
    """Deserialize dict to DiagnosticsData."""
    per_corner_residuals = None
    if "per_corner_residuals" in data:
        per_corner_residuals = _list_to_ndarray(data["per_corner_residuals"])

    per_frame_errors = None
    if "per_frame_errors" in data:
        # Convert string keys back to int
        per_frame_errors = {int(k): v for k, v in data["per_frame_errors"].items()}

    return DiagnosticsData(
        reprojection_error_rms=data["reprojection_error_rms"],
        reprojection_error_per_camera=data["reprojection_error_per_camera"],
        validation_3d_error_mean=data["validation_3d_error_mean"],
        validation_3d_error_std=data["validation_3d_error_std"],
        per_corner_residuals=per_corner_residuals,
        per_frame_errors=per_frame_errors,
    )


def _serialize_metadata(meta: CalibrationMetadata) -> dict[str, Any]:
    """Serialize CalibrationMetadata to dict."""
    return {
        "calibration_date": meta.calibration_date,
        "software_version": meta.software_version,
        "config_hash": meta.config_hash,
        "num_frames_used": meta.num_frames_used,
        "num_frames_holdout": meta.num_frames_holdout,
    }


def _deserialize_metadata(data: dict[str, Any]) -> CalibrationMetadata:
    """Deserialize dict to CalibrationMetadata."""
    return CalibrationMetadata(
        calibration_date=data["calibration_date"],
        software_version=data["software_version"],
        config_hash=data["config_hash"],
        num_frames_used=data["num_frames_used"],
        num_frames_holdout=data["num_frames_holdout"],
    )


def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    """
    Save calibration result to JSON file.

    Args:
        result: Complete calibration result to save
        path: Output file path (should end in .json)

    Raises:
        OSError: If file cannot be written

    Example:
        >>> save_calibration(result, "calibration.json")
    """
    data = {
        "version": SERIALIZATION_VERSION,
        "cameras": {
            name: _serialize_camera_calibration(cam)
            for name, cam in result.cameras.items()
        },
        "interface": _serialize_interface_params(result.interface),
        "board": _serialize_board_config(result.board),
        "diagnostics": _serialize_diagnostics(result.diagnostics),
        "metadata": _serialize_metadata(result.metadata),
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: str | Path) -> CalibrationResult:
    """
    Load calibration result from JSON file.

    Args:
        path: Path to calibration JSON file

    Returns:
        CalibrationResult object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is invalid or version mismatch

    Example:
        >>> result = load_calibration("calibration.json")
        >>> print(result.diagnostics.reprojection_error_rms)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Version check
    version = data.get("version")
    if version != SERIALIZATION_VERSION:
        raise ValueError(
            f"Unsupported calibration file version: {version}. "
            f"Expected: {SERIALIZATION_VERSION}"
        )

    return CalibrationResult(
        cameras={
            name: _deserialize_camera_calibration(cam_data)
            for name, cam_data in data["cameras"].items()
        },
        interface=_deserialize_interface_params(data["interface"]),
        board=_deserialize_board_config(data["board"]),
        diagnostics=_deserialize_diagnostics(data["diagnostics"]),
        metadata=_deserialize_metadata(data["metadata"]),
    )
