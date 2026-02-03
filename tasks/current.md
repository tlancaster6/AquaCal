# Task: 3.3 Serialization (io/serialization.py)

## Objective

Implement save/load functions for `CalibrationResult` using JSON format with numpy arrays converted to nested lists.

## Context Files

Read these files before starting (in order):

1. `CLAUDE.md` — project conventions and error handling rules
2. `docs/development_plan.md` (lines 283-296) — serialization responsibilities
3. `src/aquacal/config/schema.py` (lines 26-168) — all dataclasses that need serialization
4. `src/aquacal/__init__.py` — for `__version__` string

## Dependencies

- Task 1.1 (`config/schema.py`) — all dataclasses

## Modify

Files to create or edit:

- `src/aquacal/io/serialization.py` (create)
- `tests/unit/test_serialization.py` (create)
- `src/aquacal/io/__init__.py` (add serialization imports/exports)

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/config/schema.py`
- Any other io/ modules

---

## Function Signatures

Implement the following in `src/aquacal/io/serialization.py`:

```python
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
    pass


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
    pass
```

---

## Implementation Details

### JSON Structure

The output JSON should have this structure:

```json
{
  "version": "1.0",
  "cameras": {
    "cam0": {
      "name": "cam0",
      "intrinsics": {
        "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "dist_coeffs": [k1, k2, p1, p2, k3],
        "image_size": [width, height]
      },
      "extrinsics": {
        "R": [[r11, r12, r13], ...],
        "t": [tx, ty, tz]
      },
      "interface_distance": 0.15
    },
    "cam1": { ... }
  },
  "interface": {
    "normal": [0, 0, -1],
    "n_air": 1.0,
    "n_water": 1.333
  },
  "board": {
    "squares_x": 8,
    "squares_y": 6,
    "square_size": 0.03,
    "marker_size": 0.022,
    "dictionary": "DICT_4X4_50"
  },
  "diagnostics": {
    "reprojection_error_rms": 0.45,
    "reprojection_error_per_camera": {"cam0": 0.42, "cam1": 0.48},
    "validation_3d_error_mean": 0.0015,
    "validation_3d_error_std": 0.0008
  },
  "metadata": {
    "calibration_date": "2024-01-15T10:30:00",
    "software_version": "0.1.0",
    "config_hash": "abc123...",
    "num_frames_used": 50,
    "num_frames_holdout": 10
  }
}
```

### Serialization Helpers

```python
def _ndarray_to_list(arr: NDArray) -> list:
    """Convert numpy array to nested Python list."""
    return arr.tolist()


def _list_to_ndarray(lst: list, dtype: type = np.float64) -> NDArray:
    """Convert nested list to numpy array."""
    return np.array(lst, dtype=dtype)


def _serialize_camera_intrinsics(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    """Serialize CameraIntrinsics to dict."""
    return {
        "K": _ndarray_to_list(intrinsics.K),
        "dist_coeffs": _ndarray_to_list(intrinsics.dist_coeffs),
        "image_size": list(intrinsics.image_size),
    }


def _deserialize_camera_intrinsics(data: dict[str, Any]) -> CameraIntrinsics:
    """Deserialize dict to CameraIntrinsics."""
    return CameraIntrinsics(
        K=_list_to_ndarray(data["K"]),
        dist_coeffs=_list_to_ndarray(data["dist_coeffs"]),
        image_size=tuple(data["image_size"]),
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
    return {
        "name": cam.name,
        "intrinsics": _serialize_camera_intrinsics(cam.intrinsics),
        "extrinsics": _serialize_camera_extrinsics(cam.extrinsics),
        "interface_distance": cam.interface_distance,
    }


def _deserialize_camera_calibration(data: dict[str, Any]) -> CameraCalibration:
    """Deserialize dict to CameraCalibration."""
    return CameraCalibration(
        name=data["name"],
        intrinsics=_deserialize_camera_intrinsics(data["intrinsics"]),
        extrinsics=_deserialize_camera_extrinsics(data["extrinsics"]),
        interface_distance=data["interface_distance"],
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
        result["per_frame_errors"] = {str(k): v for k, v in diag.per_frame_errors.items()}
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
```

### Main Functions

```python
def save_calibration(result: CalibrationResult, path: str | Path) -> None:
    """Save calibration result to JSON file."""
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
    """Load calibration result from JSON file."""
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
```

---

## Acceptance Criteria

- [ ] `save_calibration` writes valid JSON file
- [ ] `load_calibration` reconstructs identical `CalibrationResult`
- [ ] Round-trip test: save then load produces equivalent object
- [ ] Numpy arrays are correctly serialized/deserialized with correct dtypes
- [ ] Optional fields (`per_corner_residuals`, `per_frame_errors`) handled correctly
- [ ] `per_frame_errors` dict keys converted int↔str for JSON compatibility
- [ ] Version field is written and checked on load
- [ ] `FileNotFoundError` raised for missing file
- [ ] `ValueError` raised for version mismatch
- [ ] Accepts both `str` and `Path` arguments
- [ ] Tests pass: `pytest tests/unit/test_serialization.py -v`
- [ ] Type check passes: `mypy src/aquacal/io/serialization.py --ignore-missing-imports`
- [ ] `src/aquacal/io/__init__.py` exports `save_calibration` and `load_calibration`

---

## Testing Strategy

### Test Fixtures

```python
import pytest
import numpy as np
from pathlib import Path

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
from aquacal.io.serialization import (
    save_calibration,
    load_calibration,
    SERIALIZATION_VERSION,
)


@pytest.fixture
def sample_intrinsics() -> CameraIntrinsics:
    """Sample camera intrinsics."""
    return CameraIntrinsics(
        K=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float64),
        image_size=(640, 480),
    )


@pytest.fixture
def sample_extrinsics() -> CameraExtrinsics:
    """Sample camera extrinsics."""
    return CameraExtrinsics(
        R=np.eye(3, dtype=np.float64),
        t=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )


@pytest.fixture
def sample_camera(sample_intrinsics, sample_extrinsics) -> CameraCalibration:
    """Sample camera calibration."""
    return CameraCalibration(
        name="cam0",
        intrinsics=sample_intrinsics,
        extrinsics=sample_extrinsics,
        interface_distance=0.15,
    )


@pytest.fixture
def sample_interface() -> InterfaceParams:
    """Sample interface parameters."""
    return InterfaceParams(
        normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        n_air=1.0,
        n_water=1.333,
    )


@pytest.fixture
def sample_board() -> BoardConfig:
    """Sample board configuration."""
    return BoardConfig(
        squares_x=8,
        squares_y=6,
        square_size=0.03,
        marker_size=0.022,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def sample_diagnostics() -> DiagnosticsData:
    """Sample diagnostics without optional fields."""
    return DiagnosticsData(
        reprojection_error_rms=0.45,
        reprojection_error_per_camera={"cam0": 0.42, "cam1": 0.48},
        validation_3d_error_mean=0.0015,
        validation_3d_error_std=0.0008,
    )


@pytest.fixture
def sample_diagnostics_full() -> DiagnosticsData:
    """Sample diagnostics with all optional fields."""
    return DiagnosticsData(
        reprojection_error_rms=0.45,
        reprojection_error_per_camera={"cam0": 0.42, "cam1": 0.48},
        validation_3d_error_mean=0.0015,
        validation_3d_error_std=0.0008,
        per_corner_residuals=np.array([[0.1, 0.2], [-0.1, 0.15]], dtype=np.float64),
        per_frame_errors={0: 0.4, 5: 0.5, 10: 0.45},
    )


@pytest.fixture
def sample_metadata() -> CalibrationMetadata:
    """Sample metadata."""
    return CalibrationMetadata(
        calibration_date="2024-01-15T10:30:00",
        software_version="0.1.0",
        config_hash="abc123def456",
        num_frames_used=50,
        num_frames_holdout=10,
    )


@pytest.fixture
def sample_calibration_result(
    sample_camera, sample_interface, sample_board, sample_diagnostics, sample_metadata
) -> CalibrationResult:
    """Complete sample calibration result."""
    # Create a second camera with different extrinsics
    cam1 = CameraCalibration(
        name="cam1",
        intrinsics=sample_camera.intrinsics,
        extrinsics=CameraExtrinsics(
            R=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
            t=np.array([0.1, 0.0, 0.0], dtype=np.float64),
        ),
        interface_distance=0.16,
    )
    return CalibrationResult(
        cameras={"cam0": sample_camera, "cam1": cam1},
        interface=sample_interface,
        board=sample_board,
        diagnostics=sample_diagnostics,
        metadata=sample_metadata,
    )
```

### Test Cases

```python
class TestSaveCalibration:
    def test_creates_file(self, tmp_path, sample_calibration_result):
        """save_calibration creates a JSON file."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        assert path.exists()

    def test_accepts_string_path(self, tmp_path, sample_calibration_result):
        """Accepts string path argument."""
        path = str(tmp_path / "calibration.json")
        save_calibration(sample_calibration_result, path)
        assert Path(path).exists()

    def test_valid_json(self, tmp_path, sample_calibration_result):
        """Output is valid JSON."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)

        import json
        with open(path) as f:
            data = json.load(f)

        assert "version" in data
        assert "cameras" in data
        assert "interface" in data

    def test_includes_version(self, tmp_path, sample_calibration_result):
        """Output includes version field."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)

        import json
        with open(path) as f:
            data = json.load(f)

        assert data["version"] == SERIALIZATION_VERSION


class TestLoadCalibration:
    def test_round_trip(self, tmp_path, sample_calibration_result):
        """Save then load produces equivalent result."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(path)

        # Check cameras
        assert set(loaded.cameras.keys()) == set(sample_calibration_result.cameras.keys())
        for name in loaded.cameras:
            orig = sample_calibration_result.cameras[name]
            load = loaded.cameras[name]
            assert load.name == orig.name
            np.testing.assert_allclose(load.intrinsics.K, orig.intrinsics.K)
            np.testing.assert_allclose(load.intrinsics.dist_coeffs, orig.intrinsics.dist_coeffs)
            assert load.intrinsics.image_size == orig.intrinsics.image_size
            np.testing.assert_allclose(load.extrinsics.R, orig.extrinsics.R)
            np.testing.assert_allclose(load.extrinsics.t, orig.extrinsics.t)
            assert load.interface_distance == orig.interface_distance

        # Check interface
        np.testing.assert_allclose(loaded.interface.normal, sample_calibration_result.interface.normal)
        assert loaded.interface.n_air == sample_calibration_result.interface.n_air
        assert loaded.interface.n_water == sample_calibration_result.interface.n_water

        # Check board
        assert loaded.board.squares_x == sample_calibration_result.board.squares_x
        assert loaded.board.dictionary == sample_calibration_result.board.dictionary

        # Check diagnostics
        assert loaded.diagnostics.reprojection_error_rms == sample_calibration_result.diagnostics.reprojection_error_rms

        # Check metadata
        assert loaded.metadata.calibration_date == sample_calibration_result.metadata.calibration_date

    def test_accepts_string_path(self, tmp_path, sample_calibration_result):
        """Accepts string path argument."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(str(path))
        assert loaded is not None

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "nonexistent.json")

    def test_version_mismatch(self, tmp_path):
        """Raises ValueError for version mismatch."""
        path = tmp_path / "calibration.json"
        import json
        with open(path, "w") as f:
            json.dump({"version": "0.0"}, f)

        with pytest.raises(ValueError, match="version"):
            load_calibration(path)


class TestOptionalFields:
    def test_diagnostics_without_optional(self, tmp_path, sample_calibration_result):
        """Handles diagnostics without optional fields."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(path)

        assert loaded.diagnostics.per_corner_residuals is None
        assert loaded.diagnostics.per_frame_errors is None

    def test_diagnostics_with_optional(
        self, tmp_path, sample_calibration_result, sample_diagnostics_full
    ):
        """Handles diagnostics with optional fields."""
        # Replace diagnostics with full version
        result = CalibrationResult(
            cameras=sample_calibration_result.cameras,
            interface=sample_calibration_result.interface,
            board=sample_calibration_result.board,
            diagnostics=sample_diagnostics_full,
            metadata=sample_calibration_result.metadata,
        )

        path = tmp_path / "calibration.json"
        save_calibration(result, path)
        loaded = load_calibration(path)

        assert loaded.diagnostics.per_corner_residuals is not None
        np.testing.assert_allclose(
            loaded.diagnostics.per_corner_residuals,
            sample_diagnostics_full.per_corner_residuals,
        )

        assert loaded.diagnostics.per_frame_errors is not None
        assert loaded.diagnostics.per_frame_errors == sample_diagnostics_full.per_frame_errors


class TestNumpyArrays:
    def test_array_dtypes(self, tmp_path, sample_calibration_result):
        """Numpy arrays have correct dtypes after round-trip."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(path)

        cam = loaded.cameras["cam0"]
        assert cam.intrinsics.K.dtype == np.float64
        assert cam.intrinsics.dist_coeffs.dtype == np.float64
        assert cam.extrinsics.R.dtype == np.float64
        assert cam.extrinsics.t.dtype == np.float64
        assert loaded.interface.normal.dtype == np.float64

    def test_array_shapes(self, tmp_path, sample_calibration_result):
        """Numpy arrays have correct shapes after round-trip."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(path)

        cam = loaded.cameras["cam0"]
        assert cam.intrinsics.K.shape == (3, 3)
        assert cam.intrinsics.dist_coeffs.shape == (5,)
        assert cam.extrinsics.R.shape == (3, 3)
        assert cam.extrinsics.t.shape == (3,)
        assert loaded.interface.normal.shape == (3,)


class TestPerFrameErrorsIntKeys:
    def test_int_keys_preserved(self, tmp_path, sample_calibration_result, sample_diagnostics_full):
        """per_frame_errors dict keys are converted to int on load."""
        result = CalibrationResult(
            cameras=sample_calibration_result.cameras,
            interface=sample_calibration_result.interface,
            board=sample_calibration_result.board,
            diagnostics=sample_diagnostics_full,
            metadata=sample_calibration_result.metadata,
        )

        path = tmp_path / "calibration.json"
        save_calibration(result, path)
        loaded = load_calibration(path)

        # Keys should be integers, not strings
        for key in loaded.diagnostics.per_frame_errors.keys():
            assert isinstance(key, int)
```

---

## Import Structure

Update `src/aquacal/io/__init__.py`:

```python
"""Input/output modules."""

from aquacal.io.video import VideoSet
from aquacal.io.detection import detect_charuco, detect_all_frames
from aquacal.io.serialization import save_calibration, load_calibration

__all__ = [
    "VideoSet",
    "detect_charuco",
    "detect_all_frames",
    "save_calibration",
    "load_calibration",
]
```

---

## Notes

- JSON `indent=2` for human readability
- Numpy `tolist()` automatically handles nested arrays
- JSON doesn't support integer dict keys, so `per_frame_errors` keys are converted str↔int
- `image_size` tuple is serialized as list, converted back to tuple on load
- No need for `export_for_downstream` in this task; can be added later if needed
