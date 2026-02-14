"""Unit tests for calibration pipeline orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

import numpy as np
import pytest
import yaml

from aquacal.calibration.pipeline import (
    load_config,
    split_detections,
    run_calibration,
    run_calibration_from_config,
    _build_calibration_result,
    _compute_config_hash,
    _save_board_reference_images,
)
from aquacal.config.schema import (
    BoardConfig,
    BoardPose,
    CalibrationConfig,
    CalibrationMetadata,
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    Detection,
    DetectionResult,
    DiagnosticsData,
    FrameDetections,
    InterfaceParams,
)


# --- Fixtures ---


@pytest.fixture
def sample_board_config():
    """Create a sample BoardConfig."""
    return BoardConfig(
        squares_x=7,
        squares_y=5,
        square_size=0.03,
        marker_size=0.022,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def sample_intrinsics():
    """Create sample CameraIntrinsics."""
    return CameraIntrinsics(
        K=np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        image_size=(1280, 720),
    )


@pytest.fixture
def sample_extrinsics():
    """Create sample CameraExtrinsics."""
    return CameraExtrinsics(
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
    )


@pytest.fixture
def sample_detection_result():
    """Create a sample DetectionResult with 10 frames."""
    frames = {}
    for i in range(10):
        detections = {}
        for cam in ["cam0", "cam1"]:
            detections[cam] = Detection(
                corner_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                corners_2d=np.array(
                    [[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float64
                ),
            )
        frames[i] = FrameDetections(frame_idx=i, detections=detections)

    return DetectionResult(
        frames=frames,
        camera_names=["cam0", "cam1"],
        total_frames=10,
    )


@pytest.fixture
def valid_config_yaml():
    """Generate valid YAML config content."""
    return {
        "board": {
            "squares_x": 7,
            "squares_y": 5,
            "square_size": 0.03,
            "marker_size": 0.022,
            "dictionary": "DICT_4X4_50",
        },
        "cameras": ["cam0", "cam1"],
        "paths": {
            "intrinsic_videos": {
                "cam0": "/path/to/cam0_inair.mp4",
                "cam1": "/path/to/cam1_inair.mp4",
            },
            "extrinsic_videos": {
                "cam0": "/path/to/cam0_uw.mp4",
                "cam1": "/path/to/cam1_uw.mp4",
            },
            "output_dir": "/path/to/output",
        },
        "interface": {
            "n_air": 1.0,
            "n_water": 1.333,
            "normal_fixed": False,
        },
        "optimization": {
            "robust_loss": "huber",
            "loss_scale": 1.0,
        },
        "detection": {
            "min_corners": 8,
            "min_cameras": 2,
        },
        "validation": {
            "holdout_fraction": 0.2,
            "save_detailed_residuals": True,
        },
    }


# --- Test load_config ---


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_valid(self, valid_config_yaml):
        """Test loading a valid config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.board.squares_x == 7
        assert config.board.squares_y == 5
        assert config.board.square_size == 0.03
        assert config.board.marker_size == 0.022
        assert config.board.dictionary == "DICT_4X4_50"
        assert config.camera_names == ["cam0", "cam1"]
        assert config.n_air == 1.0
        assert config.n_water == 1.333
        assert config.robust_loss == "huber"
        assert config.loss_scale == 1.0
        assert config.min_corners_per_frame == 8
        assert config.min_cameras_per_frame == 2
        assert config.holdout_fraction == 0.2
        assert config.save_detailed_residuals is True

    def test_load_config_missing_file(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_missing_board_section(self, valid_config_yaml):
        """Test that missing 'board' section raises ValueError."""
        del valid_config_yaml["board"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="Missing required config section: board"
            ):
                load_config(f.name)

    def test_load_config_missing_cameras_section(self, valid_config_yaml):
        """Test that missing 'cameras' section raises ValueError."""
        del valid_config_yaml["cameras"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="Missing required config section: cameras"
            ):
                load_config(f.name)

    def test_load_config_missing_paths_section(self, valid_config_yaml):
        """Test that missing 'paths' section raises ValueError."""
        del valid_config_yaml["paths"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="Missing required config section: paths"
            ):
                load_config(f.name)

    def test_load_config_defaults(self):
        """Test that defaults are applied for optional sections."""
        minimal_config = {
            "board": {
                "squares_x": 7,
                "squares_y": 5,
                "square_size": 0.03,
                "marker_size": 0.022,
            },
            "cameras": ["cam0"],
            "paths": {
                "intrinsic_videos": {"cam0": "/path/cam0.mp4"},
                "extrinsic_videos": {"cam0": "/path/cam0_uw.mp4"},
                "output_dir": "/output",
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(minimal_config, f)
            f.flush()
            config = load_config(f.name)

        # Check defaults
        assert config.board.dictionary == "DICT_4X4_50"
        assert config.intrinsic_board is None  # Should default to None
        assert config.n_air == 1.0
        assert config.n_water == 1.333
        assert config.robust_loss == "huber"
        assert config.loss_scale == 1.0
        assert config.min_corners_per_frame == 8
        assert config.min_cameras_per_frame == 2
        assert config.holdout_fraction == 0.2
        assert config.save_detailed_residuals is True
        assert config.initial_interface_distances is None  # Should default to None
        assert config.refine_intrinsics is False

    def test_load_config_with_intrinsic_board(self, valid_config_yaml):
        """Test loading config with separate intrinsic_board section."""
        # Add intrinsic_board to the config
        valid_config_yaml["intrinsic_board"] = {
            "squares_x": 12,
            "squares_y": 9,
            "square_size": 0.025,
            "marker_size": 0.018,
            "dictionary": "DICT_4X4_100",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        # Check extrinsic board (main board)
        assert config.board.squares_x == 7
        assert config.board.squares_y == 5
        assert config.board.square_size == 0.03
        assert config.board.marker_size == 0.022
        assert config.board.dictionary == "DICT_4X4_50"

        # Check intrinsic board
        assert config.intrinsic_board is not None
        assert config.intrinsic_board.squares_x == 12
        assert config.intrinsic_board.squares_y == 9
        assert config.intrinsic_board.square_size == 0.025
        assert config.intrinsic_board.marker_size == 0.018
        assert config.intrinsic_board.dictionary == "DICT_4X4_100"

    def test_load_config_with_refine_intrinsics(self, valid_config_yaml):
        """Test loading config with refine_intrinsics: true."""
        valid_config_yaml["optimization"]["refine_intrinsics"] = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.refine_intrinsics is True

    def test_load_config_without_intrinsic_board(self, valid_config_yaml):
        """Test that intrinsic_board is None when section is absent."""
        # Make sure intrinsic_board is not in the config
        if "intrinsic_board" in valid_config_yaml:
            del valid_config_yaml["intrinsic_board"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        # intrinsic_board should be None (backward compatible)
        assert config.intrinsic_board is None

    def test_load_config_without_initial_distances(self, valid_config_yaml):
        """Test that initial_interface_distances is None when not provided."""
        # Ensure initial_distances is not in config
        if "initial_distances" in valid_config_yaml.get("interface", {}):
            del valid_config_yaml["interface"]["initial_distances"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        # initial_interface_distances should be None (backward compatible)
        assert config.initial_interface_distances is None

    def test_load_config_with_per_camera_initial_distances(self, valid_config_yaml):
        """Test loading config with per-camera initial_distances."""
        valid_config_yaml["interface"]["initial_distances"] = {
            "cam0": 0.25,
            "cam1": 0.28,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.initial_interface_distances is not None
        assert config.initial_interface_distances == {"cam0": 0.25, "cam1": 0.28}

    def test_load_config_with_scalar_initial_distance(self, valid_config_yaml):
        """Test loading config with scalar initial_distance (expanded to all cameras)."""
        valid_config_yaml["interface"]["initial_distances"] = 0.3

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.initial_interface_distances is not None
        assert config.initial_interface_distances == {"cam0": 0.3, "cam1": 0.3}

    def test_load_config_with_incomplete_initial_distances_dict(
        self, valid_config_yaml
    ):
        """Test that incomplete initial_distances dict raises ValueError."""
        # Only provide distance for cam0, not cam1
        valid_config_yaml["interface"]["initial_distances"] = {"cam0": 0.25}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="initial_distances dict must cover all cameras"
            ):
                load_config(f.name)

    def test_load_config_with_negative_scalar_initial_distance(self, valid_config_yaml):
        """Test that negative scalar initial_distance raises ValueError."""
        valid_config_yaml["interface"]["initial_distances"] = -0.15

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(ValueError, match="initial_distances must be positive"):
                load_config(f.name)

    def test_load_config_with_negative_dict_initial_distance(self, valid_config_yaml):
        """Test that negative value in initial_distances dict raises ValueError."""
        valid_config_yaml["interface"]["initial_distances"] = {
            "cam0": 0.25,
            "cam1": -0.15,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="initial_distances\\['cam1'\\] must be positive"
            ):
                load_config(f.name)

    def test_load_config_with_extra_cameras_in_initial_distances(
        self, valid_config_yaml, capsys
    ):
        """Test that extra cameras in initial_distances dict produce a warning."""
        valid_config_yaml["interface"]["initial_distances"] = {
            "cam0": 0.25,
            "cam1": 0.28,
            "cam2": 0.30,  # Extra camera not in cameras list
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        # Check warning was printed to stderr
        captured = capsys.readouterr()
        assert "cam2" in captured.err
        assert "not in cameras list" in captured.err

        # Config should still load successfully with all cameras
        assert config.initial_interface_distances is not None
        assert "cam0" in config.initial_interface_distances
        assert "cam1" in config.initial_interface_distances
        assert "cam2" in config.initial_interface_distances

    def test_load_config_with_invalid_type_initial_distance(self, valid_config_yaml):
        """Test that invalid type for initial_distances raises ValueError."""
        valid_config_yaml["interface"]["initial_distances"] = "invalid"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(
                ValueError, match="initial_distances must be a number or dict"
            ):
                load_config(f.name)

    def test_load_config_with_frame_step(self, valid_config_yaml):
        """Test loading config with frame_step specified."""
        valid_config_yaml["detection"]["frame_step"] = 5

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.frame_step == 5

    def test_load_config_without_frame_step(self, valid_config_yaml):
        """Test loading config without frame_step defaults to 1."""
        # Make sure frame_step is not in the config
        if "frame_step" in valid_config_yaml.get("detection", {}):
            del valid_config_yaml["detection"]["frame_step"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.frame_step == 1

    def test_load_config_with_legacy_pattern_true(self, valid_config_yaml):
        """Test loading config with legacy_pattern: true."""
        valid_config_yaml["board"]["legacy_pattern"] = True

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.board.legacy_pattern is True

    def test_load_config_with_legacy_pattern_false(self, valid_config_yaml):
        """Test loading config with legacy_pattern: false."""
        valid_config_yaml["board"]["legacy_pattern"] = False

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.board.legacy_pattern is False

    def test_load_config_without_legacy_pattern_defaults_false(self, valid_config_yaml):
        """Test that legacy_pattern defaults to False when omitted."""
        # Ensure legacy_pattern is not in the config
        if "legacy_pattern" in valid_config_yaml["board"]:
            del valid_config_yaml["board"]["legacy_pattern"]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.board.legacy_pattern is False

    def test_load_config_with_intrinsic_board_legacy_pattern(self, valid_config_yaml):
        """Test loading config with legacy_pattern in intrinsic_board section."""
        valid_config_yaml["intrinsic_board"] = {
            "squares_x": 12,
            "squares_y": 9,
            "square_size": 0.025,
            "marker_size": 0.018,
            "dictionary": "DICT_4X4_100",
            "legacy_pattern": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.intrinsic_board.legacy_pattern is True

    def test_load_config_intrinsic_board_legacy_pattern_defaults_false(
        self, valid_config_yaml
    ):
        """Test that intrinsic_board legacy_pattern defaults to False when omitted."""
        valid_config_yaml["intrinsic_board"] = {
            "squares_x": 12,
            "squares_y": 9,
            "square_size": 0.025,
            "marker_size": 0.018,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.intrinsic_board.legacy_pattern is False

    def test_load_config_fisheye_cameras_valid(self, valid_config_yaml):
        """Fisheye cameras load correctly when subset of auxiliary_cameras."""
        valid_config_yaml["auxiliary_cameras"] = ["aux_cam"]
        valid_config_yaml["fisheye_cameras"] = ["aux_cam"]
        valid_config_yaml["paths"]["intrinsic_videos"]["aux_cam"] = (
            "/path/to/aux_inair.mp4"
        )
        valid_config_yaml["paths"]["extrinsic_videos"]["aux_cam"] = (
            "/path/to/aux_uw.mp4"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.fisheye_cameras == ["aux_cam"]

    def test_load_config_fisheye_cameras_not_in_auxiliary_raises(
        self, valid_config_yaml
    ):
        """ValueError if fisheye_cameras entry is not in auxiliary_cameras."""
        valid_config_yaml["fisheye_cameras"] = [
            "cam0"
        ]  # cam0 is primary, not auxiliary

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(ValueError, match="subset of auxiliary_cameras"):
                load_config(f.name)

    def test_load_config_fisheye_rational_overlap_raises(self, valid_config_yaml):
        """ValueError if fisheye_cameras overlaps with rational_model_cameras."""
        valid_config_yaml["auxiliary_cameras"] = ["aux_cam"]
        valid_config_yaml["fisheye_cameras"] = ["aux_cam"]
        valid_config_yaml["rational_model_cameras"] = ["aux_cam"]
        valid_config_yaml["paths"]["intrinsic_videos"]["aux_cam"] = (
            "/path/to/aux_inair.mp4"
        )
        valid_config_yaml["paths"]["extrinsic_videos"]["aux_cam"] = (
            "/path/to/aux_uw.mp4"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            with pytest.raises(ValueError, match="disjoint"):
                load_config(f.name)

    def test_load_config_fisheye_cameras_defaults_empty(self, valid_config_yaml):
        """fisheye_cameras defaults to empty list when not in config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()
            config = load_config(f.name)

        assert config.fisheye_cameras == []


# --- Test split_detections ---


class TestSplitDetections:
    """Tests for split_detections function."""

    def test_split_detections_reproducible(self, sample_detection_result):
        """Test that same seed produces same split."""
        cal1, val1 = split_detections(sample_detection_result, 0.2, seed=42)
        cal2, val2 = split_detections(sample_detection_result, 0.2, seed=42)

        assert set(cal1.frames.keys()) == set(cal2.frames.keys())
        assert set(val1.frames.keys()) == set(val2.frames.keys())

    def test_split_detections_different_seed(self, sample_detection_result):
        """Test that different seeds produce different splits."""
        cal1, val1 = split_detections(sample_detection_result, 0.3, seed=42)
        cal2, val2 = split_detections(sample_detection_result, 0.3, seed=123)

        # With 10 frames and 30% holdout, different seeds should give different results
        assert set(cal1.frames.keys()) != set(cal2.frames.keys())

    def test_split_detections_fraction(self, sample_detection_result):
        """Test that holdout fraction is approximately respected."""
        cal, val = split_detections(sample_detection_result, 0.2, seed=42)

        # With 10 frames and 0.2 holdout, expect ~2 in validation
        total = len(cal.frames) + len(val.frames)
        assert total == 10
        assert len(val.frames) == 2
        assert len(cal.frames) == 8

    def test_split_detections_preserves_frames(self, sample_detection_result):
        """Test that all frames are in exactly one of the two sets."""
        cal, val = split_detections(sample_detection_result, 0.3, seed=42)

        cal_indices = set(cal.frames.keys())
        val_indices = set(val.frames.keys())
        original_indices = set(sample_detection_result.frames.keys())

        # No overlap
        assert cal_indices.isdisjoint(val_indices)

        # Union equals original
        assert cal_indices.union(val_indices) == original_indices

    def test_split_detections_zero_holdout(self, sample_detection_result):
        """Test with zero holdout fraction."""
        cal, val = split_detections(sample_detection_result, 0.0, seed=42)

        assert len(cal.frames) == 10
        assert len(val.frames) == 0

    def test_split_detections_full_holdout(self, sample_detection_result):
        """Test with full holdout fraction."""
        cal, val = split_detections(sample_detection_result, 1.0, seed=42)

        assert len(cal.frames) == 0
        assert len(val.frames) == 10


# --- Test _build_calibration_result ---


class TestBuildCalibrationResult:
    """Tests for _build_calibration_result function."""

    def test_build_calibration_result(
        self, sample_intrinsics, sample_extrinsics, sample_board_config
    ):
        """Test that components are assembled correctly."""
        intrinsics = {"cam0": sample_intrinsics, "cam1": sample_intrinsics}
        extrinsics = {"cam0": sample_extrinsics, "cam1": sample_extrinsics}
        interface_distances = {"cam0": 0.15, "cam1": 0.16}
        interface_params = InterfaceParams(
            normal=np.array([0, 0, -1], dtype=np.float64),
            n_air=1.0,
            n_water=1.333,
        )
        diagnostics = DiagnosticsData(
            reprojection_error_rms=0.5,
            reprojection_error_per_camera={"cam0": 0.4, "cam1": 0.6},
            validation_3d_error_mean=0.001,
            validation_3d_error_std=0.0005,
        )
        metadata = CalibrationMetadata(
            calibration_date="2025-01-01T00:00:00",
            software_version="0.1.0",
            config_hash="abc123",
            num_frames_used=80,
            num_frames_holdout=20,
        )

        result = _build_calibration_result(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            interface_distances=interface_distances,
            board_config=sample_board_config,
            interface_params=interface_params,
            diagnostics=diagnostics,
            metadata=metadata,
        )

        assert len(result.cameras) == 2
        assert "cam0" in result.cameras
        assert "cam1" in result.cameras

        # Check camera calibration assembly
        cam0 = result.cameras["cam0"]
        assert cam0.name == "cam0"
        assert cam0.interface_distance == 0.15
        assert np.allclose(cam0.intrinsics.K, sample_intrinsics.K)
        assert np.allclose(cam0.extrinsics.R, sample_extrinsics.R)

        # Check other fields
        assert result.board.squares_x == 7
        assert result.interface.n_water == 1.333
        assert result.diagnostics.reprojection_error_rms == 0.5
        assert result.metadata.num_frames_used == 80


# --- Test _compute_config_hash ---


class TestSaveBoardReferenceImages:
    """Tests for _save_board_reference_images function."""

    def test_save_board_reference_images_with_separate_intrinsic_board(
        self, sample_board_config
    ):
        """Test that both board images are saved when intrinsic board differs."""
        from aquacal.core.board import BoardGeometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create two different board configurations
            board = BoardGeometry(sample_board_config)
            intrinsic_board_config = BoardConfig(
                squares_x=12,
                squares_y=9,
                square_size=0.025,
                marker_size=0.018,
                dictionary="DICT_4X4_100",
            )
            intrinsic_board = BoardGeometry(intrinsic_board_config)

            # Call the function
            _save_board_reference_images(board, intrinsic_board, output_dir)

            # Verify both images exist
            extrinsic_path = output_dir / "board_extrinsic.png"
            intrinsic_path = output_dir / "board_intrinsic.png"

            assert extrinsic_path.exists(), "board_extrinsic.png not saved"
            assert intrinsic_path.exists(), "board_intrinsic.png not saved"

            # Verify images are not empty
            assert extrinsic_path.stat().st_size > 0
            assert intrinsic_path.stat().st_size > 0

    def test_save_board_reference_images_without_separate_intrinsic_board(
        self, sample_board_config
    ):
        """Test that only extrinsic image is saved when intrinsic board is same."""
        from aquacal.core.board import BoardGeometry

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create a single board used for both intrinsic and extrinsic
            board = BoardGeometry(sample_board_config)
            intrinsic_board = board  # Same object

            # Call the function
            _save_board_reference_images(board, intrinsic_board, output_dir)

            # Verify only extrinsic image exists
            extrinsic_path = output_dir / "board_extrinsic.png"
            intrinsic_path = output_dir / "board_intrinsic.png"

            assert extrinsic_path.exists(), "board_extrinsic.png not saved"
            assert not intrinsic_path.exists(), (
                "board_intrinsic.png should not be saved when boards are identical"
            )

            # Verify extrinsic image is not empty
            assert extrinsic_path.stat().st_size > 0


class TestComputeConfigHash:
    """Tests for _compute_config_hash function."""

    def test_compute_config_hash_deterministic(self, sample_board_config):
        """Test that same config produces same hash."""
        config = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0", "cam1"],
            intrinsic_video_paths={"cam0": Path("/a"), "cam1": Path("/b")},
            extrinsic_video_paths={"cam0": Path("/c"), "cam1": Path("/d")},
            output_dir=Path("/out"),
        )

        hash1 = _compute_config_hash(config)
        hash2 = _compute_config_hash(config)

        assert hash1 == hash2
        assert len(hash1) == 12  # Truncated to 12 hex chars

    def test_compute_config_hash_different_config(self, sample_board_config):
        """Test that different configs produce different hashes."""
        config1 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0"],
            intrinsic_video_paths={"cam0": Path("/a")},
            extrinsic_video_paths={"cam0": Path("/c")},
            output_dir=Path("/out"),
            n_water=1.333,
        )

        config2 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0"],
            intrinsic_video_paths={"cam0": Path("/a")},
            extrinsic_video_paths={"cam0": Path("/c")},
            output_dir=Path("/out"),
            n_water=1.4,  # Different refractive index
        )

        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)

        assert hash1 != hash2

    def test_compute_config_hash_includes_intrinsic_board(self, sample_board_config):
        """Test that intrinsic_board is included in hash when provided."""
        intrinsic_board_config = BoardConfig(
            squares_x=12,
            squares_y=9,
            square_size=0.025,
            marker_size=0.018,
            dictionary="DICT_4X4_100",
        )

        config1 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0"],
            intrinsic_video_paths={"cam0": Path("/a")},
            extrinsic_video_paths={"cam0": Path("/c")},
            output_dir=Path("/out"),
            intrinsic_board=None,
        )

        config2 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0"],
            intrinsic_video_paths={"cam0": Path("/a")},
            extrinsic_video_paths={"cam0": Path("/c")},
            output_dir=Path("/out"),
            intrinsic_board=intrinsic_board_config,
        )

        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)

        # Different intrinsic_board should produce different hash
        assert hash1 != hash2

    def test_compute_config_hash_includes_initial_interface_distances(
        self, sample_board_config
    ):
        """Test that initial_interface_distances is included in hash when provided."""
        config1 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0", "cam1"],
            intrinsic_video_paths={"cam0": Path("/a"), "cam1": Path("/b")},
            extrinsic_video_paths={"cam0": Path("/c"), "cam1": Path("/d")},
            output_dir=Path("/out"),
            initial_interface_distances=None,
        )

        config2 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0", "cam1"],
            intrinsic_video_paths={"cam0": Path("/a"), "cam1": Path("/b")},
            extrinsic_video_paths={"cam0": Path("/c"), "cam1": Path("/d")},
            output_dir=Path("/out"),
            initial_interface_distances={"cam0": 0.25, "cam1": 0.28},
        )

        config3 = CalibrationConfig(
            board=sample_board_config,
            camera_names=["cam0", "cam1"],
            intrinsic_video_paths={"cam0": Path("/a"), "cam1": Path("/b")},
            extrinsic_video_paths={"cam0": Path("/c"), "cam1": Path("/d")},
            output_dir=Path("/out"),
            initial_interface_distances={"cam0": 0.30, "cam1": 0.28},
        )

        hash1 = _compute_config_hash(config1)
        hash2 = _compute_config_hash(config2)
        hash3 = _compute_config_hash(config3)

        # Different initial_interface_distances should produce different hashes
        assert hash1 != hash2
        assert hash2 != hash3
        assert hash1 != hash3


# --- Test run_calibration ---


class TestRunCalibration:
    """Tests for run_calibration function."""

    def test_run_calibration_loads_config_and_delegates(self, valid_config_yaml):
        """Test that run_calibration loads config and calls run_calibration_from_config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_yaml, f)
            f.flush()

            with patch(
                "aquacal.calibration.pipeline.run_calibration_from_config"
            ) as mock_run:
                mock_run.return_value = MagicMock(spec=CalibrationResult)

                _result = run_calibration(f.name)

                # Verify run_calibration_from_config was called
                mock_run.assert_called_once()

                # Verify the config was passed
                called_config = mock_run.call_args[0][0]
                assert isinstance(called_config, CalibrationConfig)
                assert called_config.board.squares_x == 7


# --- Test run_calibration_from_config (integration with mocks) ---


class TestRunCalibrationFromConfig:
    """Integration tests for run_calibration_from_config with mocked stages."""

    @pytest.fixture
    def mock_calibration_stages(
        self, sample_intrinsics, sample_extrinsics, sample_detection_result
    ):
        """Create mocks for all calibration stage functions."""
        with (
            patch("aquacal.calibration.pipeline.calibrate_intrinsics_all") as mock_intr,
            patch("aquacal.calibration.pipeline.detect_all_frames") as mock_detect,
            patch("aquacal.calibration.pipeline.build_pose_graph") as mock_pose_graph,
            patch("aquacal.calibration.pipeline.estimate_extrinsics") as mock_ext,
            patch("aquacal.calibration.pipeline.optimize_interface") as mock_opt,
            patch(
                "aquacal.calibration.pipeline.compute_reprojection_errors"
            ) as mock_reproj,
            patch("aquacal.calibration.pipeline.compute_3d_distance_errors") as mock_3d,
            patch(
                "aquacal.calibration.pipeline.generate_diagnostic_report"
            ) as mock_diag,
            patch(
                "aquacal.calibration.pipeline.save_diagnostic_report"
            ) as mock_save_diag,
            patch("aquacal.calibration.pipeline.save_calibration") as mock_save_cal,
        ):
            # Setup return values
            mock_intr.return_value = {
                "cam0": (sample_intrinsics, 0.5),
                "cam1": (sample_intrinsics, 0.6),
            }
            mock_detect.return_value = sample_detection_result
            mock_pose_graph.return_value = MagicMock()
            mock_ext.return_value = {
                "cam0": sample_extrinsics,
                "cam1": sample_extrinsics,
            }
            mock_opt.return_value = (
                {"cam0": sample_extrinsics, "cam1": sample_extrinsics},  # extrinsics
                {"cam0": 0.15, "cam1": 0.16},  # distances
                [BoardPose(0, np.zeros(3), np.array([0, 0, 0.5]))],  # poses
                0.8,  # rms
            )

            # Mock reprojection errors
            mock_reproj_result = MagicMock()
            mock_reproj_result.rms = 0.7
            mock_reproj_result.per_camera = {"cam0": 0.6, "cam1": 0.8}
            mock_reproj_result.per_frame = {0: 0.7}
            mock_reproj_result.residuals = np.array([[0.1, 0.2]])
            mock_reproj.return_value = mock_reproj_result

            # Mock 3D errors
            mock_3d_result = MagicMock()
            mock_3d_result.mean = 0.001
            mock_3d_result.std = 0.0005
            mock_3d_result.signed_mean = 0.0002
            mock_3d_result.rmse = 0.0011
            mock_3d_result.percent_error = 2.5
            mock_3d_result.num_frames = 8
            mock_3d.return_value = mock_3d_result

            # Mock diagnostic report
            mock_diag.return_value = MagicMock()
            mock_save_diag.return_value = {"json": Path("/out/diagnostics.json")}

            yield {
                "intrinsics": mock_intr,
                "detect": mock_detect,
                "pose_graph": mock_pose_graph,
                "extrinsics": mock_ext,
                "optimize": mock_opt,
                "reproj": mock_reproj,
                "3d": mock_3d,
                "diag": mock_diag,
                "save_diag": mock_save_diag,
                "save_cal": mock_save_cal,
            }

    def test_run_calibration_from_config_stages_order(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that all stages are called in correct order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            result = run_calibration_from_config(config)

            # Verify all stages were called
            mock_calibration_stages["intrinsics"].assert_called_once()
            mock_calibration_stages["detect"].assert_called_once()
            mock_calibration_stages["pose_graph"].assert_called_once()
            mock_calibration_stages["extrinsics"].assert_called_once()
            mock_calibration_stages["optimize"].assert_called_once()
            mock_calibration_stages["reproj"].assert_called_once()
            mock_calibration_stages["3d"].assert_called_once()
            mock_calibration_stages["diag"].assert_called_once()
            mock_calibration_stages["save_diag"].assert_called_once()
            # save_cal called twice: once for calibration_initial.json, once for calibration.json
            assert mock_calibration_stages["save_cal"].call_count == 2

            # Verify result
            assert isinstance(result, CalibrationResult)
            assert len(result.cameras) == 2

    def test_run_calibration_from_config_saves_calibration(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that calibration is saved to output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            # Verify save_calibration was called twice
            assert mock_calibration_stages["save_cal"].call_count == 2
            # First call: calibration_initial.json
            initial_call_args = mock_calibration_stages["save_cal"].call_args_list[0]
            assert str(initial_call_args[0][1]).endswith("calibration_initial.json")
            # Second call: calibration.json
            final_call_args = mock_calibration_stages["save_cal"].call_args_list[1]
            assert str(final_call_args[0][1]).endswith("calibration.json")

    def test_run_calibration_from_config_saves_diagnostics(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that diagnostics are saved to output_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            # Verify save_diagnostic_report was called
            mock_calibration_stages["save_diag"].assert_called_once()
            call_args = mock_calibration_stages["save_diag"].call_args
            # Fourth positional arg is output_dir (report, calibration, detections, output_dir)
            assert call_args[0][3] == Path(tmpdir)
            # save_images should be True
            assert call_args[1]["save_images"] is True

    def test_run_calibration_from_config_prints_progress(
        self, mock_calibration_stages, sample_board_config, capsys
    ):
        """Test that progress is printed to stdout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            captured = capsys.readouterr()

            # Check for key progress messages
            assert "AquaCal Calibration Pipeline" in captured.out
            assert "[Stage 1]" in captured.out
            assert "[Stage 2]" in captured.out
            assert "[Stage 3]" in captured.out
            assert "[Validation]" in captured.out
            assert "[Diagnostics]" in captured.out
            assert "[Save]" in captured.out
            assert "Calibration complete!" in captured.out

    def test_run_calibration_from_config_uses_intrinsic_board(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that intrinsic_board is passed to calibrate_intrinsics_all when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a different board for intrinsics
            intrinsic_board_config = BoardConfig(
                squares_x=12,
                squares_y=9,
                square_size=0.025,
                marker_size=0.018,
                dictionary="DICT_4X4_100",
            )

            config = CalibrationConfig(
                board=sample_board_config,
                intrinsic_board=intrinsic_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            # Verify calibrate_intrinsics_all was called with intrinsic board
            mock_calibration_stages["intrinsics"].assert_called_once()
            call_args = mock_calibration_stages["intrinsics"].call_args

            # Check that the board parameter has the intrinsic board config
            board_arg = call_args[1]["board"]
            assert board_arg.config.squares_x == 12
            assert board_arg.config.squares_y == 9
            assert board_arg.config.square_size == 0.025
            assert board_arg.config.marker_size == 0.018

    def test_run_calibration_from_config_falls_back_to_extrinsic_board(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that extrinsic board is used for intrinsics when intrinsic_board is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                intrinsic_board=None,  # No separate intrinsic board
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            # Verify calibrate_intrinsics_all was called with extrinsic board
            mock_calibration_stages["intrinsics"].assert_called_once()
            call_args = mock_calibration_stages["intrinsics"].call_args

            # Check that the board parameter has the extrinsic board config
            board_arg = call_args[1]["board"]
            assert board_arg.config.squares_x == 7
            assert board_arg.config.squares_y == 5
            assert board_arg.config.square_size == 0.03
            assert board_arg.config.marker_size == 0.022

    def test_run_calibration_from_config_passes_initial_interface_distances(
        self, mock_calibration_stages, sample_board_config
    ):
        """Test that initial_interface_distances is passed to optimize_interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_distances = {"cam0": 0.25, "cam1": 0.28}
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
                initial_interface_distances=initial_distances,
            )

            run_calibration_from_config(config)

            # Verify optimize_interface was called with initial_interface_distances
            mock_calibration_stages["optimize"].assert_called_once()
            call_args = mock_calibration_stages["optimize"].call_args

            # Check that initial_interface_distances was passed
            assert call_args[1]["initial_interface_distances"] == initial_distances

    def test_run_calibration_from_config_estimates_validation_poses(
        self, mock_calibration_stages, sample_board_config, capsys
    ):
        """Test that validation frame board poses are estimated and reported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
                holdout_fraction=0.2,
            )

            run_calibration_from_config(config)

            captured = capsys.readouterr()

            # Check that validation pose estimation message is printed
            assert (
                "[Validation] Estimating board poses for held-out frames"
                in captured.out
            )
            assert "Estimated" in captured.out
            assert "validation frame poses" in captured.out


# --- Test Auxiliary Camera Separation ---


class TestAuxiliaryCameraSeparation:
    """Tests for auxiliary camera metric separation from primary cameras."""

    @pytest.fixture
    def mock_calibration_stages_with_aux(
        self, sample_intrinsics, sample_extrinsics, sample_detection_result
    ):
        """Create mocks for all calibration stage functions, including auxiliary camera."""
        with (
            patch("aquacal.calibration.pipeline.calibrate_intrinsics_all") as mock_intr,
            patch("aquacal.calibration.pipeline.detect_all_frames") as mock_detect,
            patch("aquacal.calibration.pipeline.build_pose_graph") as mock_pose_graph,
            patch("aquacal.calibration.pipeline.estimate_extrinsics") as mock_ext,
            patch("aquacal.calibration.pipeline.optimize_interface") as mock_opt,
            patch("aquacal.calibration.pipeline.register_auxiliary_camera") as mock_aux,
            patch(
                "aquacal.calibration.pipeline.compute_reprojection_errors"
            ) as mock_reproj,
            patch("aquacal.calibration.pipeline.compute_3d_distance_errors") as mock_3d,
            patch(
                "aquacal.calibration.pipeline.generate_diagnostic_report"
            ) as mock_diag,
            patch(
                "aquacal.calibration.pipeline.save_diagnostic_report"
            ) as mock_save_diag,
            patch("aquacal.calibration.pipeline.save_calibration") as mock_save_cal,
        ):
            # Setup return values
            mock_intr.return_value = {
                "cam0": (sample_intrinsics, 0.5),
                "cam1": (sample_intrinsics, 0.6),
                "aux_cam": (sample_intrinsics, 0.7),
            }

            # Create detection result with auxiliary camera
            frames = {}
            for i in range(10):
                detections = {}
                for cam in ["cam0", "cam1", "aux_cam"]:
                    detections[cam] = Detection(
                        corner_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                        corners_2d=np.array(
                            [[100, 100], [200, 100], [100, 200], [200, 200]],
                            dtype=np.float64,
                        ),
                    )
                frames[i] = FrameDetections(frame_idx=i, detections=detections)

            detection_result_with_aux = DetectionResult(
                frames=frames,
                camera_names=["cam0", "cam1", "aux_cam"],
                total_frames=10,
            )

            mock_detect.return_value = detection_result_with_aux
            mock_pose_graph.return_value = MagicMock()
            mock_ext.return_value = {
                "cam0": sample_extrinsics,
                "cam1": sample_extrinsics,
            }
            mock_opt.return_value = (
                {"cam0": sample_extrinsics, "cam1": sample_extrinsics},  # extrinsics
                {"cam0": 0.15, "cam1": 0.16},  # distances
                [BoardPose(0, np.zeros(3), np.array([0, 0, 0.5]))],  # poses
                0.8,  # rms
            )

            # Mock auxiliary camera registration
            mock_aux.return_value = (
                sample_extrinsics,  # extrinsics
                0.17,  # distance
                1.5,  # rms (higher than primary)
            )

            # Mock reprojection errors - will be called multiple times
            def reproj_side_effect(calibration, detections, poses):
                # Count cameras to determine if primary or auxiliary
                num_cams = len(calibration.cameras)
                if num_cams == 2:
                    # Primary cameras only
                    result = MagicMock()
                    result.rms = 0.7
                    result.per_camera = {"cam0": 0.6, "cam1": 0.8}
                    result.per_frame = {0: 0.7}
                    result.residuals = np.array([[0.1, 0.2]])
                    result.num_observations = 100
                    return result
                elif num_cams == 1:
                    # Auxiliary camera only
                    result = MagicMock()
                    result.rms = 1.5
                    result.per_camera = {"aux_cam": 1.5}
                    result.per_frame = {0: 1.5}
                    result.residuals = np.array([[0.3, 0.4]])
                    result.num_observations = 50
                    return result
                else:
                    # Full result (shouldn't be used for metrics)
                    result = MagicMock()
                    result.rms = 1.0
                    result.per_camera = {"cam0": 0.6, "cam1": 0.8, "aux_cam": 1.5}
                    result.per_frame = {0: 1.0}
                    result.residuals = np.array([[0.2, 0.3]])
                    result.num_observations = 150
                    return result

            mock_reproj.side_effect = reproj_side_effect

            # Mock 3D errors
            mock_3d_result = MagicMock()
            mock_3d_result.mean = 0.001
            mock_3d_result.std = 0.0005
            mock_3d_result.signed_mean = 0.0002
            mock_3d_result.rmse = 0.0011
            mock_3d_result.percent_error = 2.5
            mock_3d_result.num_frames = 8
            mock_3d.return_value = mock_3d_result

            # Mock diagnostic report
            mock_diag.return_value = MagicMock()
            mock_save_diag.return_value = {"json": Path("/out/diagnostics.json")}

            yield {
                "intrinsics": mock_intr,
                "detect": mock_detect,
                "pose_graph": mock_pose_graph,
                "extrinsics": mock_ext,
                "optimize": mock_opt,
                "aux": mock_aux,
                "reproj": mock_reproj,
                "3d": mock_3d,
                "diag": mock_diag,
                "save_diag": mock_save_diag,
                "save_cal": mock_save_cal,
            }

    def test_auxiliary_cameras_excluded_from_diagnostics_data(
        self, mock_calibration_stages_with_aux, sample_board_config
    ):
        """Test that DiagnosticsData contains only primary camera metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                auxiliary_cameras=["aux_cam"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                    "aux_cam": Path("/path/aux_cam.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                    "aux_cam": Path("/path/aux_cam_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            result = run_calibration_from_config(config)

            # Verify DiagnosticsData uses primary-only metrics
            assert result.diagnostics.reprojection_error_rms == 0.7  # Primary only
            assert "aux_cam" not in result.diagnostics.reprojection_error_per_camera
            assert "cam0" in result.diagnostics.reprojection_error_per_camera
            assert "cam1" in result.diagnostics.reprojection_error_per_camera

    def test_auxiliary_cameras_in_final_result(
        self, mock_calibration_stages_with_aux, sample_board_config
    ):
        """Test that auxiliary cameras are in final CalibrationResult with is_auxiliary=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                auxiliary_cameras=["aux_cam"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                    "aux_cam": Path("/path/aux_cam.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                    "aux_cam": Path("/path/aux_cam_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            result = run_calibration_from_config(config)

            # Verify all cameras in final result
            assert len(result.cameras) == 3
            assert "aux_cam" in result.cameras
            assert result.cameras["aux_cam"].is_auxiliary is True
            assert result.cameras["cam0"].is_auxiliary is False
            assert result.cameras["cam1"].is_auxiliary is False

    def test_auxiliary_cameras_saved_in_diagnostics_json(
        self, mock_calibration_stages_with_aux, sample_board_config
    ):
        """Test that auxiliary camera metrics appear in diagnostics.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                auxiliary_cameras=["aux_cam"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                    "aux_cam": Path("/path/aux_cam.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                    "aux_cam": Path("/path/aux_cam_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            # Verify save_diagnostic_report was called with auxiliary_reprojection
            save_diag_calls = mock_calibration_stages_with_aux["save_diag"].call_args
            assert "auxiliary_reprojection" in save_diag_calls[1]
            aux_reproj = save_diag_calls[1]["auxiliary_reprojection"]
            assert aux_reproj is not None
            assert "aux_cam" in aux_reproj.per_camera

    def test_no_auxiliary_cameras_no_regression(
        self, mock_calibration_stages_with_aux, sample_board_config
    ):
        """Test that pipeline still works when no auxiliary cameras are configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                auxiliary_cameras=[],  # No auxiliary cameras
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            result = run_calibration_from_config(config)

            # Verify result is valid
            assert isinstance(result, CalibrationResult)
            assert len(result.cameras) == 2
            assert all(not cam.is_auxiliary for cam in result.cameras.values())

    def test_auxiliary_cameras_printed_separately(
        self, mock_calibration_stages_with_aux, sample_board_config, capsys
    ):
        """Test that auxiliary camera metrics are printed separately in console output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CalibrationConfig(
                board=sample_board_config,
                camera_names=["cam0", "cam1"],
                auxiliary_cameras=["aux_cam"],
                intrinsic_video_paths={
                    "cam0": Path("/path/cam0.mp4"),
                    "cam1": Path("/path/cam1.mp4"),
                    "aux_cam": Path("/path/aux_cam.mp4"),
                },
                extrinsic_video_paths={
                    "cam0": Path("/path/cam0_uw.mp4"),
                    "cam1": Path("/path/cam1_uw.mp4"),
                    "aux_cam": Path("/path/aux_cam_uw.mp4"),
                },
                output_dir=Path(tmpdir),
            )

            run_calibration_from_config(config)

            captured = capsys.readouterr()

            # Check that primary and auxiliary are printed separately
            assert "Primary cameras:" in captured.out
            assert "Auxiliary cameras:" in captured.out
            assert "aux_cam: RMS" in captured.out
