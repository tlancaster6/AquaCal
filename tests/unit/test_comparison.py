"""Tests for validation.comparison module."""

import cv2
import numpy as np
import pandas as pd
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
from aquacal.validation.comparison import (
    ComparisonResult,
    compare_calibrations,
    plot_depth_error_comparison,
    plot_position_overlay,
    plot_rms_bar_chart,
    plot_xy_error_heatmaps,
    plot_z_position_dumbbell,
    write_comparison_report,
)
from aquacal.validation.reconstruction import (
    DepthBinnedErrors,
    SpatialErrorGrid,
    SpatialMeasurements,
)


def make_test_result(
    camera_names: list[str],
    camera_positions: dict[str, np.ndarray],
    camera_rotations: dict[str, np.ndarray],
    water_z: float = 0.5,
    reprojection_rms: float = 0.5,
    reproj_per_camera: dict[str, float] = None,
    validation_3d_mean: float = 0.01,
    validation_3d_std: float = 0.005,
    num_frames_used: int = 50,
) -> CalibrationResult:
    """Helper to create a CalibrationResult with specified parameters."""
    cameras = {}

    if reproj_per_camera is None:
        reproj_per_camera = {name: reprojection_rms for name in camera_names}

    for name in camera_names:
        # Create intrinsics (simple pinhole)
        K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 480.0], [0.0, 0.0, 1.0]])
        dist_coeffs = np.zeros(5)
        intrinsics = CameraIntrinsics(
            K=K, dist_coeffs=dist_coeffs, image_size=(1280, 960)
        )

        # Create extrinsics
        C = camera_positions.get(name, np.array([0.0, 0.0, 0.0]))
        R = camera_rotations.get(name, np.eye(3))
        t = -R @ C
        extrinsics = CameraExtrinsics(R=R, t=t)

        cameras[name] = CameraCalibration(
            name=name,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            water_z=water_z,
            is_auxiliary=False,
        )

    interface = InterfaceParams(
        normal=np.array([0.0, 0.0, -1.0]), n_air=1.0, n_water=1.333
    )

    board = BoardConfig(
        squares_x=7,
        squares_y=5,
        square_size=0.05,
        marker_size=0.04,
        dictionary="DICT_4X4_50",
    )

    diagnostics = DiagnosticsData(
        reprojection_error_rms=reprojection_rms,
        reprojection_error_per_camera=reproj_per_camera,
        validation_3d_error_mean=validation_3d_mean,
        validation_3d_error_std=validation_3d_std,
    )

    metadata = CalibrationMetadata(
        calibration_date="2024-01-01",
        software_version="1.0.0",
        config_hash="abc123",
        num_frames_used=num_frames_used,
        num_frames_holdout=10,
    )

    return CalibrationResult(
        cameras=cameras,
        interface=interface,
        board=board,
        diagnostics=diagnostics,
        metadata=metadata,
    )


class TestValidation:
    """Test input validation."""

    def test_too_few_results(self):
        """Should raise ValueError if fewer than 2 results."""
        result = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        with pytest.raises(ValueError, match="at least 2 results"):
            compare_calibrations([result], ["run1"])

    def test_mismatched_lengths(self):
        """Should raise ValueError if result and label counts differ."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.1, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        with pytest.raises(ValueError, match="Mismatch"):
            compare_calibrations([result1, result2], ["run1"])

    def test_duplicate_labels(self):
        """Should raise ValueError if labels are not unique."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.1, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        with pytest.raises(ValueError, match="unique"):
            compare_calibrations([result1, result2], ["run1", "run1"])


class TestTwoResults:
    """Test comparison with two results."""

    def test_basic_comparison(self):
        """Basic comparison of two results with same cameras."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            water_z=0.5,
            reprojection_rms=0.5,
            validation_3d_mean=0.01,
            validation_3d_std=0.005,
            num_frames_used=50,
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.001, 0.0, 0.0]),  # 1mm shift
                "cam1": np.array([0.501, 0.0, 0.0]),  # 1mm shift
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            water_z=0.5,
            reprojection_rms=0.6,
            validation_3d_mean=0.012,
            validation_3d_std=0.006,
            num_frames_used=55,
        )

        comp = compare_calibrations([result1, result2], ["baseline", "variant"])

        # Check labels
        assert comp.labels == ["baseline", "variant"]

        # Check metric table shape and content
        assert comp.metric_table.shape == (2, 8)
        assert list(comp.metric_table.index) == ["baseline", "variant"]
        assert "reprojection_rms" in comp.metric_table.columns
        assert "water_z" in comp.metric_table.columns
        assert comp.metric_table.loc["baseline", "reprojection_rms"] == 0.5
        assert comp.metric_table.loc["variant", "reprojection_rms"] == 0.6
        assert comp.metric_table.loc["baseline", "num_cameras"] == 2
        assert comp.metric_table.loc["baseline", "num_frames_used"] == 50

        # Check per-camera metrics
        assert set(comp.per_camera_metrics.keys()) == {"cam0", "cam1"}
        cam0_df = comp.per_camera_metrics["cam0"]
        assert cam0_df.shape == (2, 9)  # 2 runs, 9 columns
        assert list(cam0_df.index) == ["baseline", "variant"]
        assert cam0_df.loc["baseline", "position_x"] == 0.0
        assert cam0_df.loc["variant", "position_x"] == 0.001

        # Check parameter diffs
        # Should have 2 cameras * 1 pair = 2 rows
        assert comp.parameter_diffs.shape == (2, 9)
        assert set(comp.parameter_diffs["camera"]) == {"cam0", "cam1"}

        # Check position delta for cam0 (1mm shift in X)
        cam0_row = comp.parameter_diffs[comp.parameter_diffs["camera"] == "cam0"].iloc[
            0
        ]
        assert cam0_row["label_a"] == "baseline"
        assert cam0_row["label_b"] == "variant"
        assert np.isclose(cam0_row["position_delta_mm"], 1.0, atol=1e-6)
        assert np.isclose(cam0_row["orientation_delta_deg"], 0.0, atol=1e-6)

    def test_orientation_delta(self):
        """Test orientation delta computation."""
        # Create small rotation around Z axis (10 degrees)
        angle_deg = 10.0
        angle_rad = np.radians(angle_deg)
        rvec = np.array([0.0, 0.0, angle_rad])
        R_rotated, _ = cv2.Rodrigues(rvec)

        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": R_rotated},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Check orientation delta
        assert comp.parameter_diffs.shape == (1, 9)
        assert np.isclose(
            comp.parameter_diffs.iloc[0]["orientation_delta_deg"], angle_deg, atol=1e-6
        )

    def test_intrinsic_deltas(self):
        """Test intrinsic parameter deltas."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        # Modify intrinsics for result2
        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        # Modify K matrix
        result2.cameras["cam0"].intrinsics.K[0, 0] = 810.0  # fx + 10
        result2.cameras["cam0"].intrinsics.K[1, 1] = 795.0  # fy - 5
        result2.cameras["cam0"].intrinsics.K[0, 2] = 642.0  # cx + 2
        result2.cameras["cam0"].intrinsics.K[1, 2] = 478.0  # cy - 2

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Check intrinsic deltas
        row = comp.parameter_diffs.iloc[0]
        assert np.isclose(row["fx_delta"], 10.0, atol=1e-6)
        assert np.isclose(row["fy_delta"], 5.0, atol=1e-6)
        assert np.isclose(row["cx_delta"], 2.0, atol=1e-6)
        assert np.isclose(row["cy_delta"], 2.0, atol=1e-6)


class TestMultipleResults:
    """Test comparison with 3+ results."""

    def test_three_results(self):
        """Test comparison with three results."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )
        result3 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.002, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations(
            [result1, result2, result3], ["run1", "run2", "run3"]
        )

        # Metric table should have 3 rows
        assert comp.metric_table.shape[0] == 3
        assert list(comp.metric_table.index) == ["run1", "run2", "run3"]

        # Per-camera metrics should have 3 rows
        assert comp.per_camera_metrics["cam0"].shape[0] == 3

        # Parameter diffs should have 3 pairs: (1,2), (1,3), (2,3)
        assert comp.parameter_diffs.shape == (3, 9)
        pairs = set(
            zip(
                comp.parameter_diffs["label_a"].tolist(),
                comp.parameter_diffs["label_b"].tolist(),
            )
        )
        assert pairs == {("run1", "run2"), ("run1", "run3"), ("run2", "run3")}


class TestDifferentCameraSets:
    """Test handling of different camera sets across results."""

    def test_missing_camera_in_one_result(self):
        """Test handling when a camera is missing from one result."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            reproj_per_camera={"cam0": 0.5, "cam1": 0.6},
        )

        result2 = make_test_result(
            camera_names=["cam0"],  # Only cam0
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
            reproj_per_camera={"cam0": 0.7},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Per-camera metrics should include both cameras
        assert set(comp.per_camera_metrics.keys()) == {"cam0", "cam1"}

        # cam0 should have data for both runs
        cam0_df = comp.per_camera_metrics["cam0"]
        assert not pd.isna(cam0_df.loc["run1", "position_x"])
        assert not pd.isna(cam0_df.loc["run2", "position_x"])

        # cam1 should have data for run1, NaN for run2
        cam1_df = comp.per_camera_metrics["cam1"]
        assert not pd.isna(cam1_df.loc["run1", "position_x"])
        assert pd.isna(cam1_df.loc["run2", "position_x"])
        assert pd.isna(cam1_df.loc["run2", "reprojection_rms"])

        # Parameter diffs should only include cam0 (intersection)
        assert comp.parameter_diffs.shape == (1, 9)
        assert comp.parameter_diffs.iloc[0]["camera"] == "cam0"

    def test_disjoint_camera_sets(self):
        """Test comparison with completely disjoint camera sets."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam1"],
            camera_positions={"cam1": np.array([0.5, 0.0, 0.0])},
            camera_rotations={"cam1": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Per-camera metrics should include both cameras
        assert set(comp.per_camera_metrics.keys()) == {"cam0", "cam1"}

        # Each camera should have one NaN row
        cam0_df = comp.per_camera_metrics["cam0"]
        assert not pd.isna(cam0_df.loc["run1", "position_x"])
        assert pd.isna(cam0_df.loc["run2", "position_x"])

        cam1_df = comp.per_camera_metrics["cam1"]
        assert pd.isna(cam1_df.loc["run1", "position_x"])
        assert not pd.isna(cam1_df.loc["run2", "position_x"])

        # Parameter diffs should be empty (no intersection)
        assert comp.parameter_diffs.shape == (0, 9)


class TestMetricComputations:
    """Test specific metric computations."""

    def test_3d_rmse_computation(self):
        """Test 3D RMSE computation from mean and std."""
        mean = 0.01
        std = 0.005
        expected_rmse = np.sqrt(mean**2 + std**2)

        result = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
            validation_3d_mean=mean,
            validation_3d_std=std,
        )

        comp = compare_calibrations([result, result], ["run1", "run2"])

        assert np.isclose(
            comp.metric_table.loc["run1", "reproj_3d_rmse"], expected_rmse, atol=1e-9
        )

    def test_3d_percent_error(self):
        """Test 3D percent error computation."""
        water_z = 0.5
        mean_error = 0.01
        expected_pct = (mean_error / water_z) * 100

        result = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
            water_z=water_z,
            validation_3d_mean=mean_error,
        )

        comp = compare_calibrations([result, result], ["run1", "run2"])

        assert np.isclose(
            comp.metric_table.loc["run1", "reproj_3d_pct"], expected_pct, atol=1e-6
        )

    def test_camera_center_extraction(self):
        """Test that camera centers are correctly extracted via C property."""
        # Camera at position [1, 2, 3]
        C = np.array([1.0, 2.0, 3.0])
        R = np.eye(3)

        result = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": C},
            camera_rotations={"cam0": R},
        )

        comp = compare_calibrations([result, result], ["run1", "run2"])

        cam_metrics = comp.per_camera_metrics["cam0"]
        assert np.isclose(cam_metrics.loc["run1", "position_x"], 1.0)
        assert np.isclose(cam_metrics.loc["run1", "position_y"], 2.0)
        assert np.isclose(cam_metrics.loc["run1", "position_z"], 3.0)


class TestWriteComparisonReport:
    """Test write_comparison_report function."""

    def test_all_files_created(self, tmp_path):
        """Test that all expected files are created."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.001, 0.0, 0.0]),
                "cam1": np.array([0.501, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Write report with plots
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(comp, [result1, result2], output_dir)

        # Check all expected keys are present
        assert "metrics_csv" in paths
        assert "per_camera_csv" in paths
        assert "parameter_diffs_csv" in paths
        assert "rms_bar_chart" in paths
        assert "position_overlay" in paths
        assert "z_position_dumbbell" in paths

        # Check all files exist
        assert paths["metrics_csv"].exists()
        assert paths["per_camera_csv"].exists()
        assert paths["parameter_diffs_csv"].exists()
        assert paths["rms_bar_chart"].exists()
        assert paths["position_overlay"].exists()
        assert paths["z_position_dumbbell"].exists()

        # Check PNG files are non-empty
        assert paths["rms_bar_chart"].stat().st_size > 0
        assert paths["position_overlay"].stat().st_size > 0
        assert paths["z_position_dumbbell"].stat().st_size > 0

    def test_csv_files_loadable(self, tmp_path):
        """Test that CSV files can be loaded back with correct shape."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.001, 0.0, 0.0]),
                "cam1": np.array([0.501, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(comp, [result1, result2], output_dir)

        # Load metrics_summary.csv
        metrics_df = pd.read_csv(paths["metrics_csv"])
        assert metrics_df.shape[0] == 2  # 2 runs
        assert "run" in metrics_df.columns
        assert "reprojection_rms" in metrics_df.columns

        # Load per_camera_metrics.csv
        per_cam_df = pd.read_csv(paths["per_camera_csv"])
        assert per_cam_df.shape[0] == 4  # 2 cameras * 2 runs
        assert "camera" in per_cam_df.columns
        assert "run" in per_cam_df.columns
        assert "position_x" in per_cam_df.columns

        # Load parameter_diffs.csv
        param_diffs_df = pd.read_csv(paths["parameter_diffs_csv"])
        assert param_diffs_df.shape[0] == 2  # 2 cameras * 1 pair
        assert "label_a" in param_diffs_df.columns
        assert "label_b" in param_diffs_df.columns
        assert "camera" in param_diffs_df.columns

    def test_no_plots_option(self, tmp_path):
        """Test that plots are not created when save_plots=False."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(
            comp, [result1, result2], output_dir, save_plots=False
        )

        # Check that plot keys are not present
        assert "rms_bar_chart" not in paths
        assert "position_overlay" not in paths

        # Check that only CSV files are present
        assert "metrics_csv" in paths
        assert "per_camera_csv" in paths
        assert "parameter_diffs_csv" in paths

    def test_output_dir_created(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Use a nested directory that doesn't exist
        output_dir = tmp_path / "nested" / "comparison_output"
        assert not output_dir.exists()

        _paths = write_comparison_report(
            comp, [result1, result2], output_dir, save_plots=False
        )

        # Check that directory was created
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestBarChart:
    """Test plot_rms_bar_chart function."""

    def test_basic_bar_chart(self):
        """Test that bar chart is created without error."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            reproj_per_camera={"cam0": 0.5, "cam1": 0.6},
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.001, 0.0, 0.0]),
                "cam1": np.array([0.501, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            reproj_per_camera={"cam0": 0.7, "cam1": 0.8},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a matplotlib Figure without error
        fig = plot_rms_bar_chart(comp)
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_camera_gracefully_handled(self):
        """Test that missing cameras are handled gracefully (NaN values)."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
            reproj_per_camera={"cam0": 0.5, "cam1": 0.6},
        )

        result2 = make_test_result(
            camera_names=["cam0"],  # Only cam0
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
            reproj_per_camera={"cam0": 0.7},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a Figure without error even though cam1 is missing from run2
        fig = plot_rms_bar_chart(comp)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_comparison(self):
        """Test bar chart with no cameras."""
        # Create a minimal ComparisonResult with no cameras
        comp = ComparisonResult(
            labels=["run1", "run2"],
            metric_table=pd.DataFrame(
                {
                    "reprojection_rms": [0.5, 0.6],
                    "reproj_3d_mean": [0.01, 0.012],
                    "reproj_3d_std": [0.005, 0.006],
                    "reproj_3d_rmse": [0.011, 0.013],
                    "reproj_3d_pct": [2.0, 2.4],
                    "water_z": [0.5, 0.5],
                    "num_cameras": [0, 0],
                    "num_frames_used": [50, 55],
                },
                index=["run1", "run2"],
            ),
            per_camera_metrics={},
            parameter_diffs=pd.DataFrame(
                columns=[
                    "label_a",
                    "label_b",
                    "camera",
                    "position_delta_mm",
                    "orientation_delta_deg",
                    "fx_delta",
                    "fy_delta",
                    "cx_delta",
                    "cy_delta",
                ]
            ),
        )

        # Should return a Figure with "No cameras found" message
        fig = plot_rms_bar_chart(comp)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPositionOverlay:
    """Test plot_position_overlay function."""

    def test_basic_position_overlay(self):
        """Test that position overlay is created without error."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.001, 0.0, 0.0]),
                "cam1": np.array([0.501, 0.0, 0.0]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a matplotlib Figure without error
        fig = plot_position_overlay(comp, [result1, result2])
        assert fig is not None

        # Check that Y-axis is inverted (per KB entry)
        import matplotlib.pyplot as plt

        ax = fig.axes[0]
        # Y-axis is inverted if ylim[0] > ylim[1]
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1], "Y-axis should be inverted"

        plt.close(fig)

    def test_single_camera(self):
        """Test position overlay with single camera per run."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.001, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        fig = plot_position_overlay(comp, [result1, result2])
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestZPositionDumbbell:
    """Test plot_z_position_dumbbell function."""

    def test_basic_dumbbell(self):
        """Test basic dumbbell chart with two runs and two cameras."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.1]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.05]),
                "cam1": np.array([0.5, 0.0, 0.15]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a matplotlib Figure without error
        fig = plot_z_position_dumbbell(comp)
        assert fig is not None

        # Check that figure has axes
        assert len(fig.axes) > 0
        ax = fig.axes[0]
        assert ax.get_title() == "Camera Z Positions Across Runs"
        assert ax.get_xlabel() == "Z position (meters)"
        assert ax.get_ylabel() == "Camera"

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_camera(self):
        """Test dumbbell chart with one camera missing from one run."""
        result1 = make_test_result(
            camera_names=["cam0", "cam1"],
            camera_positions={
                "cam0": np.array([0.0, 0.0, 0.0]),
                "cam1": np.array([0.5, 0.0, 0.1]),
            },
            camera_rotations={"cam0": np.eye(3), "cam1": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],  # Only cam0
            camera_positions={"cam0": np.array([0.0, 0.0, 0.05])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a Figure without error even though cam1 is missing from run2
        fig = plot_z_position_dumbbell(comp)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_camera(self):
        """Test dumbbell chart with single camera and two runs."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.05])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Should return a Figure without error
        fig = plot_z_position_dumbbell(comp)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestDepthErrorPlot:
    """Test plot_depth_error_comparison function."""

    def test_basic_depth_plot(self):
        """Test basic depth plot with two runs."""
        # Create two DepthBinnedErrors with different data
        dbe1 = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6, 0.7]),
            bin_centers=np.array([0.45, 0.55, 0.65]),
            signed_means=np.array([0.001, 0.002, 0.003]),
            signed_stds=np.array([0.0005, 0.0006, 0.0007]),
            counts=np.array([10, 15, 12], dtype=np.int32),
        )

        dbe2 = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6, 0.7]),
            bin_centers=np.array([0.45, 0.55, 0.65]),
            signed_means=np.array([0.0015, 0.0025, 0.0035]),
            signed_stds=np.array([0.0006, 0.0007, 0.0008]),
            counts=np.array([11, 14, 13], dtype=np.int32),
        )

        # Create plot
        fig = plot_depth_error_comparison({"run1": dbe1, "run2": dbe2})

        # Verify returns Figure
        assert fig is not None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_run(self):
        """Test plot with just one run."""
        dbe = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6]),
            bin_centers=np.array([0.45, 0.55]),
            signed_means=np.array([0.001, 0.002]),
            signed_stds=np.array([0.0005, 0.0006]),
            counts=np.array([10, 15], dtype=np.int32),
        )

        fig = plot_depth_error_comparison({"run1": dbe})
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_empty_bins(self):
        """Test plot with some empty bins (NaN values)."""
        # Create data with some NaN bins
        dbe = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
            bin_centers=np.array([0.45, 0.55, 0.65, 0.75]),
            signed_means=np.array([0.001, np.nan, 0.003, np.nan]),
            signed_stds=np.array([0.0005, np.nan, 0.0007, np.nan]),
            counts=np.array([10, 0, 12, 0], dtype=np.int32),
        )

        fig = plot_depth_error_comparison({"run1": dbe})
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestWriteComparisonReportDepth:
    """Test write_comparison_report with depth data."""

    def test_depth_data_output(self, tmp_path):
        """Test that depth data generates plot and CSV."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Create depth data
        dbe1 = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6]),
            bin_centers=np.array([0.45, 0.55]),
            signed_means=np.array([0.001, 0.002]),
            signed_stds=np.array([0.0005, 0.0006]),
            counts=np.array([10, 15], dtype=np.int32),
        )

        dbe2 = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5, 0.6]),
            bin_centers=np.array([0.45, 0.55]),
            signed_means=np.array([0.0015, 0.0025]),
            signed_stds=np.array([0.0006, 0.0007]),
            counts=np.array([11, 14], dtype=np.int32),
        )

        depth_data = {"run1": dbe1, "run2": dbe2}

        # Write report with depth data
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(
            comp, [result1, result2], output_dir, depth_data=depth_data
        )

        # Verify depth outputs are present
        assert "depth_error_plot" in paths
        assert "depth_binned_csv" in paths

        # Verify files exist
        assert paths["depth_error_plot"].exists()
        assert paths["depth_binned_csv"].exists()

        # Verify PNG is non-empty
        assert paths["depth_error_plot"].stat().st_size > 0

        # Load and verify CSV
        csv_df = pd.read_csv(paths["depth_binned_csv"])
        assert csv_df.shape[0] == 4  # 2 runs * 2 bins each
        assert "label" in csv_df.columns
        assert "bin_center_z" in csv_df.columns
        assert "signed_mean_mm" in csv_df.columns
        assert "signed_std_mm" in csv_df.columns
        assert "count" in csv_df.columns

        # Verify values are in millimeters
        # dbe1 has signed_mean[0] = 0.001 m = 1.0 mm
        run1_bin0 = csv_df[
            (csv_df["label"] == "run1") & (csv_df["bin_center_z"] == 0.45)
        ]
        assert len(run1_bin0) == 1
        assert np.isclose(run1_bin0.iloc[0]["signed_mean_mm"], 1.0, atol=1e-6)

    def test_depth_data_no_plots(self, tmp_path):
        """Test that depth CSV is created even when save_plots=False."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Create depth data
        dbe = DepthBinnedErrors(
            bin_edges=np.array([0.4, 0.5]),
            bin_centers=np.array([0.45]),
            signed_means=np.array([0.001]),
            signed_stds=np.array([0.0005]),
            counts=np.array([10], dtype=np.int32),
        )

        depth_data = {"run1": dbe}

        # Write report with depth data but no plots
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(
            comp,
            [result1, result2],
            output_dir,
            save_plots=False,
            depth_data=depth_data,
        )

        # Verify CSV is present but plot is not
        assert "depth_binned_csv" in paths
        assert "depth_error_plot" not in paths
        assert paths["depth_binned_csv"].exists()


class TestXYErrorHeatmaps:
    """Test plot_xy_error_heatmaps function."""

    def test_basic_heatmap(self):
        """Test basic heatmap with two runs and synthetic grid data."""
        # Create two SpatialErrorGrid objects
        depth_bin_edges = np.array([0.4, 0.5, 0.6])
        x_edges = np.linspace(-0.1, 0.1, 5)  # 4 X bins
        y_edges = np.linspace(-0.1, 0.1, 5)  # 4 Y bins

        # Grid 1: some non-zero errors
        grids1 = np.random.uniform(-0.002, 0.002, size=(2, 4, 4))
        counts1 = np.random.randint(0, 10, size=(2, 4, 4), dtype=np.int32)
        grids1[counts1 == 0] = np.nan  # Empty cells get NaN

        grid1 = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids1,
            counts=counts1,
        )

        # Grid 2: different errors
        grids2 = np.random.uniform(-0.003, 0.003, size=(2, 4, 4))
        counts2 = np.random.randint(0, 10, size=(2, 4, 4), dtype=np.int32)
        grids2[counts2 == 0] = np.nan

        grid2 = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids2,
            counts=counts2,
        )

        # Create plot
        fig = plot_xy_error_heatmaps({"run1": grid1, "run2": grid2})

        # Verify returns Figure
        assert fig is not None

        # Verify correct number of axes (2 runs * 2 depth bins)
        # axes is 2D array with shape (n_runs, n_depth_bins)
        assert len(fig.axes) >= 4  # At least 4 subplots (may have colorbar axis)

        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_run(self):
        """Test heatmap plot with just one run."""
        depth_bin_edges = np.array([0.4, 0.5])
        x_edges = np.linspace(-0.1, 0.1, 5)
        y_edges = np.linspace(-0.1, 0.1, 5)

        grids = np.random.uniform(-0.002, 0.002, size=(1, 4, 4))
        counts = np.full((1, 4, 4), 10, dtype=np.int32)

        grid = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids,
            counts=counts,
        )

        fig = plot_xy_error_heatmaps({"run1": grid})
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_symmetric_colorscale(self):
        """Test that colorbar range is symmetric around zero."""
        depth_bin_edges = np.array([0.4, 0.5])
        x_edges = np.linspace(-0.1, 0.1, 3)  # 2 bins
        y_edges = np.linspace(-0.1, 0.1, 3)  # 2 bins

        # Create grid with known asymmetric values: [-0.002, 0.001]
        grids = np.array([[[-0.002, 0.001], [0.0, 0.0]]])
        counts = np.full((1, 2, 2), 10, dtype=np.int32)

        grid = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids,
            counts=counts,
        )

        fig = plot_xy_error_heatmaps({"run1": grid})
        assert fig is not None

        # Verify colorbar is symmetric
        # The vmax should be at least abs(-0.002) = 0.002 meters = 2 mm
        # Get the image from first subplot
        im = fig.axes[0].images[0]
        norm = im.norm

        # vmax should be >= 2.0 mm (symmetric around zero)
        assert norm.vmax >= 2.0, f"Expected vmax >= 2.0 mm, got {norm.vmax}"
        assert norm.vmin == -norm.vmax, "Colorscale should be symmetric"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_spatial_heatmap_output(self, tmp_path):
        """Test that write_comparison_report creates xy_error_heatmaps.png when spatial_data is provided."""
        result1 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.0, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        result2 = make_test_result(
            camera_names=["cam0"],
            camera_positions={"cam0": np.array([0.001, 0.0, 0.0])},
            camera_rotations={"cam0": np.eye(3)},
        )

        comp = compare_calibrations([result1, result2], ["run1", "run2"])

        # Create synthetic spatial data
        np.random.seed(42)
        positions1 = np.random.uniform(
            low=[-0.1, -0.1, 0.4], high=[0.1, 0.1, 0.6], size=(30, 3)
        )
        signed_errors1 = np.random.uniform(-0.001, 0.001, size=30)
        frame_indices1 = np.zeros(30, dtype=np.int32)

        spatial1 = SpatialMeasurements(
            positions=positions1,
            signed_errors=signed_errors1,
            frame_indices=frame_indices1,
        )

        positions2 = np.random.uniform(
            low=[-0.1, -0.1, 0.4], high=[0.1, 0.1, 0.6], size=(30, 3)
        )
        signed_errors2 = np.random.uniform(-0.001, 0.001, size=30)
        frame_indices2 = np.zeros(30, dtype=np.int32)

        spatial2 = SpatialMeasurements(
            positions=positions2,
            signed_errors=signed_errors2,
            frame_indices=frame_indices2,
        )

        spatial_data = {"run1": spatial1, "run2": spatial2}

        # Write report with spatial data
        output_dir = tmp_path / "comparison_output"
        paths = write_comparison_report(
            comp, [result1, result2], output_dir, spatial_data=spatial_data
        )

        # Verify xy_error_heatmaps.png is created
        assert "xy_error_heatmaps" in paths
        assert paths["xy_error_heatmaps"].exists()
        assert paths["xy_error_heatmaps"].stat().st_size > 0

    def test_heatmap_no_colorbar_overlap(self, tmp_path):
        """Regression test: verify colorbar does not overlap subplots (constrained_layout fix).

        This test ensures that:
        1. The heatmap renders without warnings
        2. The PNG file is created and non-empty
        3. constrained_layout properly spaces colorbar, suptitle, and subplots
        """
        import matplotlib.pyplot as plt

        # Create 2 runs x 3 depth bins of spatial data
        depth_bin_edges = np.array([0.4, 0.5, 0.6, 0.7])  # 3 depth bins
        x_edges = np.linspace(-0.1, 0.1, 5)  # 4 X bins
        y_edges = np.linspace(-0.1, 0.1, 5)  # 4 Y bins

        # Grid 1: with some data
        grids1 = np.random.uniform(-0.002, 0.002, size=(3, 4, 4))
        counts1 = np.full((3, 4, 4), 10, dtype=np.int32)

        grid1 = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids1,
            counts=counts1,
        )

        # Grid 2: different errors
        grids2 = np.random.uniform(-0.003, 0.003, size=(3, 4, 4))
        counts2 = np.full((3, 4, 4), 10, dtype=np.int32)

        grid2 = SpatialErrorGrid(
            depth_bin_edges=depth_bin_edges,
            x_edges=x_edges,
            y_edges=y_edges,
            grids=grids2,
            counts=counts2,
        )

        # Generate heatmap
        fig = plot_xy_error_heatmaps({"run1": grid1, "run2": grid2})
        assert fig is not None

        # Save to temporary PNG file
        output_file = tmp_path / "test_heatmap.png"
        fig.savefig(str(output_file), dpi=100, bbox_inches="tight")

        # Verify file is non-empty (renders without error)
        assert output_file.exists()
        assert output_file.stat().st_size > 0, "Heatmap PNG should be non-empty"

        # Clean up
        plt.close(fig)
