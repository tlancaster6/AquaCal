"""Tests for io/serialization.py."""

import json
from pathlib import Path

import numpy as np
import pytest

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
    SERIALIZATION_VERSION,
    load_calibration,
    save_calibration,
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

        with open(path) as f:
            data = json.load(f)

        assert "version" in data
        assert "cameras" in data
        assert "interface" in data

    def test_includes_version(self, tmp_path, sample_calibration_result):
        """Output includes version field."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)

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
