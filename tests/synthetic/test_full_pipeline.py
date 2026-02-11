"""Full pipeline integration tests using synthetic data with known ground truth."""

import numpy as np
import pytest

from aquacal.calibration.extrinsics import build_pose_graph, estimate_extrinsics
from aquacal.calibration.interface_estimation import optimize_interface
from aquacal.config.schema import (
    BoardConfig,
    CalibrationMetadata,
    CalibrationResult,
    CameraCalibration,
    DiagnosticsData,
    InterfaceParams,
)
from aquacal.core.board import BoardGeometry

from .ground_truth import (
    SyntheticScenario,
    compute_calibration_errors,
    create_scenario,
    generate_camera_array,
    generate_camera_intrinsics,
    generate_real_rig_array,
    generate_synthetic_detections,
)


def _run_calibration_stages(
    scenario: SyntheticScenario,
    noise_std: float | None = None,
) -> CalibrationResult:
    """
    Run calibration stages 2-3 using synthetic detections.

    Skips Stage 1 (intrinsic calibration from video) and uses ground truth intrinsics.

    Args:
        scenario: Synthetic scenario with ground truth
        noise_std: Override scenario's noise_std if provided

    Returns:
        CalibrationResult from the calibration pipeline
    """
    # Use scenario noise unless overridden
    actual_noise = noise_std if noise_std is not None else scenario.noise_std

    # Create board geometry
    board = BoardGeometry(scenario.board_config)

    # Generate synthetic detections
    detections = generate_synthetic_detections(
        intrinsics=scenario.intrinsics,
        extrinsics=scenario.extrinsics,
        interface_distances=scenario.interface_distances,
        board=board,
        board_poses=scenario.board_poses,
        noise_std=actual_noise,
        seed=42,
    )

    # Stage 2: Build pose graph and estimate extrinsics
    reference_camera = "cam0"
    pose_graph = build_pose_graph(detections, min_cameras=2)
    initial_extrinsics = estimate_extrinsics(
        pose_graph, scenario.intrinsics, board, reference_camera
    )

    # Stage 3: Interface optimization
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    opt_extrinsics, opt_distances, opt_poses, rms = optimize_interface(
        detections=detections,
        intrinsics=scenario.intrinsics,
        initial_extrinsics=initial_extrinsics,
        board=board,
        reference_camera=reference_camera,
        interface_normal=interface_normal,
        n_air=1.0,
        n_water=1.333,
        loss="huber",
        loss_scale=1.0,
        min_corners=4,
    )

    # Build CalibrationResult
    cameras = {}
    for cam_name in scenario.intrinsics:
        cameras[cam_name] = CameraCalibration(
            name=cam_name,
            intrinsics=scenario.intrinsics[cam_name],  # Use ground truth intrinsics
            extrinsics=opt_extrinsics[cam_name],
            interface_distance=opt_distances[cam_name],
        )

    interface_params = InterfaceParams(
        normal=interface_normal,
        n_air=1.0,
        n_water=1.333,
    )

    diagnostics = DiagnosticsData(
        reprojection_error_rms=rms,
        reprojection_error_per_camera={},
        validation_3d_error_mean=0.0,
        validation_3d_error_std=0.0,
    )

    metadata = CalibrationMetadata(
        calibration_date="synthetic",
        software_version="test",
        config_hash="synthetic",
        num_frames_used=len(opt_poses),
        num_frames_holdout=0,
    )

    return CalibrationResult(
        cameras=cameras,
        interface=interface_params,
        board=scenario.board_config,
        diagnostics=diagnostics,
        metadata=metadata,
    )


@pytest.fixture(scope="class")
def ideal_result(scenario_ideal):
    """Run calibration once for all ideal scenario tests."""
    result = _run_calibration_stages(scenario_ideal, noise_std=0.0)
    errors = compute_calibration_errors(result, scenario_ideal)
    return result, errors


@pytest.fixture(scope="class")
def realistic_result(scenario_realistic):
    """Run calibration once for all realistic scenario tests."""
    result = _run_calibration_stages(scenario_realistic)
    errors = compute_calibration_errors(result, scenario_realistic)
    return result, errors


@pytest.fixture(scope="class")
def minimal_result(scenario_minimal):
    """Run calibration once for all minimal scenario tests."""
    result = _run_calibration_stages(scenario_minimal)
    errors = compute_calibration_errors(result, scenario_minimal)
    return result, errors


class TestGenerateCameraIntrinsics:
    """Tests for generate_camera_intrinsics function."""

    def test_default_intrinsics(self):
        """Default intrinsics should have correct dimensions."""
        intr = generate_camera_intrinsics()

        assert intr.image_size == (1920, 1080)
        assert intr.K.shape == (3, 3)
        assert len(intr.dist_coeffs) == 5

        # Principal point at center
        np.testing.assert_allclose(intr.K[0, 2], 960.0)
        np.testing.assert_allclose(intr.K[1, 2], 540.0)

    def test_custom_fov(self):
        """Custom FOV should produce correct focal length."""
        # 90 degree horizontal FOV with 1000 pixel width
        intr = generate_camera_intrinsics(image_size=(1000, 500), fov_horizontal_deg=90.0)

        # At 90 deg FOV: fx = width / (2 * tan(45)) = 1000 / 2 = 500
        np.testing.assert_allclose(intr.K[0, 0], 500.0, rtol=1e-6)
        np.testing.assert_allclose(intr.K[1, 1], 500.0, rtol=1e-6)

    def test_principal_point_offset(self):
        """Principal point offset should shift from center."""
        intr = generate_camera_intrinsics(
            image_size=(1000, 800),
            principal_point_offset=(10.0, -5.0),
        )

        np.testing.assert_allclose(intr.K[0, 2], 510.0)  # 1000/2 + 10
        np.testing.assert_allclose(intr.K[1, 2], 395.0)  # 800/2 - 5


class TestGenerateCameraArray:
    """Tests for generate_camera_array function."""

    def test_cam0_at_origin(self):
        """cam0 should be at origin with identity rotation."""
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=4, layout="grid", seed=42
        )

        assert "cam0" in extrinsics
        np.testing.assert_allclose(extrinsics["cam0"].t, [0, 0, 0])
        np.testing.assert_allclose(extrinsics["cam0"].R, np.eye(3))

    def test_correct_camera_count(self):
        """Should create the requested number of cameras."""
        for n in [2, 4, 8, 14]:
            intrinsics, extrinsics, distances = generate_camera_array(
                n_cameras=n, seed=42
            )
            assert len(intrinsics) == n
            assert len(extrinsics) == n
            assert len(distances) == n

    def test_grid_layout(self):
        """Grid layout should arrange cameras in square grid."""
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=4, layout="grid", spacing=0.1, seed=42
        )

        # 4 cameras should form 2x2 grid
        positions = [extrinsics[f"cam{i}"].C for i in range(4)]

        # cam0 at origin
        np.testing.assert_allclose(positions[0], [0, 0, 0], atol=1e-10)

        # Check spacing
        assert len(set([tuple(p) for p in positions])) == 4  # All unique

    def test_line_layout(self):
        """Line layout should arrange cameras in a row."""
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=3, layout="line", spacing=0.1, seed=42
        )

        # Cameras should be along X axis
        for i, cam in enumerate(["cam0", "cam1", "cam2"]):
            C = extrinsics[cam].C
            np.testing.assert_allclose(C[1], 0.0, atol=1e-10)  # Y=0
            np.testing.assert_allclose(C[2], 0.0, atol=1e-10)  # Z=0

    def test_ring_layout(self):
        """Ring layout should arrange cameras in a circle."""
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=4, layout="ring", spacing=0.1, seed=42
        )

        # All cameras should be at same radius from center
        radii = []
        for i in range(4):
            C = extrinsics[f"cam{i}"].C
            radii.append(np.sqrt(C[0] ** 2 + C[1] ** 2))

        # cam0 centered, others on ring (after centering)
        assert radii[0] == 0.0  # cam0 at origin after centering


class TestGenerateRealRigArray:
    """Tests for generate_real_rig_array function."""

    def test_creates_13_cameras(self):
        """Should create exactly 13 cameras."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        assert len(intrinsics) == 13
        assert len(extrinsics) == 13
        assert len(distances) == 13

    def test_cam0_at_origin(self):
        """cam0 should be at origin with identity rotation."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        np.testing.assert_allclose(extrinsics["cam0"].C, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(extrinsics["cam0"].R, np.eye(3))

    def test_inner_ring_radius(self):
        """Inner ring cameras (cam1-cam6) should be at 300mm radius."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        for i in range(1, 7):
            C = extrinsics[f"cam{i}"].C
            radius = np.sqrt(C[0] ** 2 + C[1] ** 2)
            np.testing.assert_allclose(radius, 0.300, atol=1e-6)

    def test_outer_ring_radius(self):
        """Outer ring cameras (cam7-cam12) should be at 600mm radius."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        for i in range(7, 13):
            C = extrinsics[f"cam{i}"].C
            radius = np.sqrt(C[0] ** 2 + C[1] ** 2)
            np.testing.assert_allclose(radius, 0.600, atol=1e-6)

    def test_camera_specs(self):
        """Camera specs should match real hardware."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        for cam_name in intrinsics:
            intr = intrinsics[cam_name]
            assert intr.image_size == (1600, 1200)

    def test_roll_angles(self):
        """Roll angles should make camera X-axis tangent to circle."""
        intrinsics, extrinsics, distances = generate_real_rig_array(seed=42)

        ANGULAR_SPACING = np.pi / 3  # 60 degrees

        # Inner ring
        for i in range(1, 7):
            theta = (i - 1) * ANGULAR_SPACING
            expected_roll = theta + np.pi / 2

            R = extrinsics[f"cam{i}"].R
            # Roll is rotation around Z axis, check the Z component of the rotation
            # For a Z-rotation by angle phi: R = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
            actual_cos = R[0, 0]
            actual_sin = R[1, 0]
            actual_roll = np.arctan2(actual_sin, actual_cos)

            # Normalize to same range
            expected_normalized = expected_roll % (2 * np.pi)
            actual_normalized = actual_roll % (2 * np.pi)

            diff = min(
                abs(expected_normalized - actual_normalized),
                abs(expected_normalized - actual_normalized + 2 * np.pi),
                abs(expected_normalized - actual_normalized - 2 * np.pi),
            )
            assert diff < 0.01, f"Roll mismatch for cam{i}"


class TestCreateScenario:
    """Tests for create_scenario function."""

    def test_ideal_scenario(self):
        """Ideal scenario should have 0 noise and 4 cameras."""
        scenario = create_scenario("ideal")

        assert scenario.name == "ideal"
        assert scenario.noise_std == 0.0
        assert len(scenario.intrinsics) == 4
        assert len(scenario.board_poses) == 20

    def test_minimal_scenario(self):
        """Minimal scenario should have 2 cameras."""
        scenario = create_scenario("minimal")

        assert scenario.name == "minimal"
        assert len(scenario.intrinsics) == 2

    def test_realistic_scenario(self):
        """Realistic scenario should have 13 cameras matching real hardware."""
        scenario = create_scenario("realistic")

        assert scenario.name == "realistic"
        assert len(scenario.intrinsics) == 13
        assert len(scenario.extrinsics) == 13
        assert len(scenario.interface_distances) == 13
        assert len(scenario.board_poses) == 30
        assert scenario.noise_std == 0.5

    def test_unknown_scenario_raises(self):
        """Unknown scenario name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            create_scenario("nonexistent")


class TestGenerateSyntheticDetections:
    """Tests for generate_synthetic_detections function."""

    def test_produces_valid_detection_result(self, scenario_ideal):
        """Should produce valid DetectionResult format."""
        board = BoardGeometry(scenario_ideal.board_config)
        detections = generate_synthetic_detections(
            intrinsics=scenario_ideal.intrinsics,
            extrinsics=scenario_ideal.extrinsics,
            interface_distances=scenario_ideal.interface_distances,
            board=board,
            board_poses=scenario_ideal.board_poses,
            noise_std=0.0,
        )

        assert hasattr(detections, "frames")
        assert hasattr(detections, "camera_names")
        assert hasattr(detections, "total_frames")
        assert len(detections.frames) > 0

    def test_detections_match_poses(self, scenario_ideal):
        """Each detection should correspond to a board pose."""
        board = BoardGeometry(scenario_ideal.board_config)
        detections = generate_synthetic_detections(
            intrinsics=scenario_ideal.intrinsics,
            extrinsics=scenario_ideal.extrinsics,
            interface_distances=scenario_ideal.interface_distances,
            board=board,
            board_poses=scenario_ideal.board_poses,
            noise_std=0.0,
        )

        pose_indices = {bp.frame_idx for bp in scenario_ideal.board_poses}
        for frame_idx in detections.frames:
            assert frame_idx in pose_indices


@pytest.mark.slow
class TestIdealScenario:
    """Test with zero noise - should recover ground truth exactly."""

    def test_rotation_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["rotation_error_deg"] < 0.5

    def test_translation_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["translation_error_mm"] < 5.0

    def test_interface_distance_accuracy(self, ideal_result):
        result, errors = ideal_result
        assert errors["interface_distance_error_mm"] < 10.0

    def test_rms_reprojection_error(self, ideal_result):
        result, errors = ideal_result
        assert result.diagnostics.reprojection_error_rms < 1.0


@pytest.mark.slow
class TestRealisticScenario:
    """Test with 13-camera rig matching actual hardware."""

    def test_rotation_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["rotation_error_deg"] < 1.5

    def test_translation_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["translation_error_mm"] < 15.0

    def test_interface_distance_accuracy(self, realistic_result):
        result, errors = realistic_result
        assert errors["interface_distance_error_mm"] < 20.0

    def test_rms_reprojection_error(self, realistic_result):
        result, errors = realistic_result
        assert result.diagnostics.reprojection_error_rms < 2.0


@pytest.mark.slow
class TestMinimalScenario:
    """Test edge case: minimum viable configuration (2 cameras)."""

    def test_rotation_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["rotation_error_deg"] < 3.0

    def test_translation_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["translation_error_mm"] < 30.0

    def test_interface_distance_accuracy(self, minimal_result):
        result, errors = minimal_result
        assert errors["interface_distance_error_mm"] < 30.0

    def test_rms_reprojection_error(self, minimal_result):
        result, errors = minimal_result
        assert result.diagnostics.reprojection_error_rms < 3.0


class TestComputeCalibrationErrors:
    """Tests for compute_calibration_errors function."""

    def test_perfect_match_gives_zero_errors(self, scenario_ideal):
        """Perfect match should give all zero errors."""
        # Create a "result" that exactly matches ground truth
        cameras = {}
        for cam_name in scenario_ideal.intrinsics:
            cameras[cam_name] = CameraCalibration(
                name=cam_name,
                intrinsics=scenario_ideal.intrinsics[cam_name],
                extrinsics=scenario_ideal.extrinsics[cam_name],
                interface_distance=scenario_ideal.interface_distances[cam_name],
            )

        result = CalibrationResult(
            cameras=cameras,
            interface=InterfaceParams(
                normal=np.array([0.0, 0.0, -1.0]),
                n_air=1.0,
                n_water=1.333,
            ),
            board=scenario_ideal.board_config,
            diagnostics=DiagnosticsData(
                reprojection_error_rms=0.0,
                reprojection_error_per_camera={},
                validation_3d_error_mean=0.0,
                validation_3d_error_std=0.0,
            ),
            metadata=CalibrationMetadata(
                calibration_date="test",
                software_version="test",
                config_hash="test",
                num_frames_used=0,
                num_frames_holdout=0,
            ),
        )

        errors = compute_calibration_errors(result, scenario_ideal)

        np.testing.assert_allclose(errors["focal_length_error_percent"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["principal_point_error_px"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["rotation_error_deg"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["translation_error_mm"], 0.0, atol=1e-10)
        np.testing.assert_allclose(errors["interface_distance_error_mm"], 0.0, atol=1e-10)
