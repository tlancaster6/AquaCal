"""Tests for 3D reconstruction validation metrics."""

import pytest
import numpy as np

from aquacal.config.schema import (
    BoardConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraCalibration,
    CalibrationResult,
    InterfaceParams,
    DiagnosticsData,
    CalibrationMetadata,
    Detection,
    FrameDetections,
    DetectionResult,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.validation.reconstruction import (
    triangulate_charuco_corners,
    compute_3d_distance_errors,
    compute_board_planarity_error,
    get_adjacent_corner_pairs,
    DistanceErrors,
)


@pytest.fixture
def board_config():
    """Standard ChArUco board config."""
    return BoardConfig(
        squares_x=5,
        squares_y=4,
        square_size=0.04,  # 4 cm squares
        marker_size=0.03,  # 3 cm markers
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def board_geometry(board_config):
    """BoardGeometry instance."""
    return BoardGeometry(board_config)


@pytest.fixture
def interface_params():
    """Standard interface parameters."""
    return InterfaceParams(
        normal=np.array([0.0, 0.0, -1.0]),
        n_air=1.0,
        n_water=1.333,
    )


@pytest.fixture
def camera_intrinsics():
    """Standard camera intrinsics."""
    return CameraIntrinsics(
        K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros(5),
        image_size=(640, 480),
    )


def create_camera_calibration(
    name: str,
    intrinsics: CameraIntrinsics,
    R: np.ndarray,
    t: np.ndarray,
    interface_distance: float,
) -> CameraCalibration:
    """Helper to create CameraCalibration."""
    extrinsics = CameraExtrinsics(R=R, t=t)
    return CameraCalibration(
        name=name,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        interface_distance=interface_distance,
    )


def create_calibration_result(
    cameras: dict[str, CameraCalibration],
    interface: InterfaceParams,
    board: BoardConfig,
) -> CalibrationResult:
    """Helper to create CalibrationResult."""
    diagnostics = DiagnosticsData(
        reprojection_error_rms=0.0,
        reprojection_error_per_camera={},
        validation_3d_error_mean=0.0,
        validation_3d_error_std=0.0,
    )
    metadata = CalibrationMetadata(
        calibration_date="2025-01-01",
        software_version="0.1.0",
        config_hash="test",
        num_frames_used=0,
        num_frames_holdout=0,
    )
    return CalibrationResult(
        cameras=cameras,
        interface=interface,
        board=board,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def project_corner_to_pixel(
    corner_3d: np.ndarray,
    camera: Camera,
    interface: Interface,
) -> np.ndarray | None:
    """Helper to project 3D corner to pixel using refractive projection."""
    return refractive_project(camera, interface, corner_3d)


def create_synthetic_detections(
    calibration: CalibrationResult,
    board: BoardGeometry,
    corner_ids: list[int],
    corner_positions_3d: dict[int, np.ndarray],
    frame_idx: int,
    noise_std: float = 0.0,
) -> DetectionResult:
    """
    Create synthetic detections by projecting 3D corners through refractive model.

    Args:
        calibration: Calibration with cameras
        board: Board geometry
        corner_ids: List of corner IDs to include
        corner_positions_3d: Dict mapping corner_id to 3D position in world frame
        frame_idx: Frame index
        noise_std: Standard deviation of Gaussian pixel noise

    Returns:
        DetectionResult with synthetic detections
    """
    frame_detections = {}

    for cam_name, cam_calib in calibration.cameras.items():
        camera = Camera(cam_name, cam_calib.intrinsics, cam_calib.extrinsics)
        interface = Interface(
            normal=calibration.interface.normal,
            base_height=0.0,
            camera_offsets={cam_name: cam_calib.interface_distance},
            n_air=calibration.interface.n_air,
            n_water=calibration.interface.n_water,
        )

        detected_ids = []
        detected_pixels = []

        for corner_id in corner_ids:
            corner_3d = corner_positions_3d[corner_id]
            pixel = project_corner_to_pixel(corner_3d, camera, interface)

            if pixel is not None:
                if noise_std > 0:
                    pixel = pixel + np.random.normal(0, noise_std, size=2)
                detected_ids.append(corner_id)
                detected_pixels.append(pixel)

        if detected_ids:
            detection = Detection(
                corner_ids=np.array(detected_ids, dtype=np.int32),
                corners_2d=np.array(detected_pixels, dtype=np.float64),
            )
            frame_detections[cam_name] = detection

    frame_det = FrameDetections(frame_idx=frame_idx, detections=frame_detections)

    return DetectionResult(
        frames={frame_idx: frame_det},
        camera_names=list(calibration.cameras.keys()),
        total_frames=1,
    )


# --- Test Cases ---


def test_get_adjacent_corner_pairs(board_geometry):
    """Test that get_adjacent_corner_pairs returns correct pairs."""
    # Board is 5x4 squares, so (5-1)x(4-1) = 4x3 = 12 corners
    # Horizontal pairs: (cols-1) * rows = 3 * 3 = 9
    # Vertical pairs: cols * (rows-1) = 4 * 2 = 8
    # Total: 17 pairs
    pairs = get_adjacent_corner_pairs(board_geometry)

    assert len(pairs) == 17

    # Check that all pairs are adjacent (differ by 1 horizontally or cols vertically)
    cols = board_geometry.config.squares_x - 1  # 4
    for id1, id2 in pairs:
        assert id1 < id2  # Lower ID first
        diff = id2 - id1
        # Either horizontal neighbor (diff=1) or vertical neighbor (diff=cols=4)
        assert diff == 1 or diff == cols

    # Check specific pairs exist (spot check)
    assert (0, 1) in pairs  # Horizontal pair in first row
    assert (0, 4) in pairs  # Vertical pair in first column
    assert (8, 9) in pairs  # Horizontal pair in third row
    assert (7, 11) in pairs  # Vertical pair in fourth column


def test_triangulate_charuco_corners_basic(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test triangulation with synthetic detections from known 3D corners."""
    # Create two cameras looking at the board
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    # Place board underwater at known position
    corner_ids = [0, 1, 2, 3, 4]
    corner_positions_3d = {}
    for cid in corner_ids:
        # Board corners in their original positions (Z=0), shifted down in Z
        board_pos = board_geometry.corner_positions[cid]
        corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6])

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    # Triangulate
    result = triangulate_charuco_corners(calibration, detections, frame_idx=0)

    # Verify all corners were triangulated
    assert len(result) == len(corner_ids)
    for cid in corner_ids:
        assert cid in result

    # Verify triangulated positions match original (within 1mm)
    for cid in corner_ids:
        triangulated = result[cid]
        expected = corner_positions_3d[cid]
        error = np.linalg.norm(triangulated - expected)
        assert error < 0.001, f"Corner {cid}: error {error*1000:.3f}mm > 1mm"


def test_triangulate_charuco_corners_empty_frame(
    board_config, interface_params, camera_intrinsics
):
    """Test that requesting a non-existent frame returns empty dict."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1},
        interface=interface_params,
        board=board_config,
    )

    # Create empty detection result
    detections = DetectionResult(frames={}, camera_names=["cam1"], total_frames=0)

    result = triangulate_charuco_corners(calibration, detections, frame_idx=0)

    assert result == {}


def test_triangulate_charuco_corners_single_camera(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that corners visible in only one camera are not triangulated."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1},
        interface=interface_params,
        board=board_config,
    )

    # Create detections with only one camera
    corner_ids = [0, 1, 2]
    corner_positions_3d = {}
    for cid in corner_ids:
        board_pos = board_geometry.corner_positions[cid]
        corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6])

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    result = triangulate_charuco_corners(calibration, detections, frame_idx=0)

    # Should return empty dict since all corners only visible in one camera
    assert result == {}


def test_compute_3d_distance_errors_perfect(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that perfect synthetic data yields near-zero error."""
    # Create two cameras
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    # Create perfect detections (corners 0,1,2,3 are in first row, so 3 adjacent pairs)
    corner_ids = [0, 1, 2, 3]
    corner_positions_3d = {}
    for cid in corner_ids:
        board_pos = board_geometry.corner_positions[cid]
        corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6])

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    # Compute errors
    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    # Should have low error (< 0.1mm)
    assert errors.mean < 0.0001, f"Mean error {errors.mean*1000:.3f}mm > 0.1mm"
    assert errors.max_error < 0.001, f"Max error {errors.max_error*1000:.3f}mm > 1mm"
    assert errors.num_comparisons == 3  # Adjacent pairs: (0,1), (1,2), (2,3)
    assert errors.num_frames == 1
    # Check new fields
    assert abs(errors.signed_mean) < 0.0001  # Should be near zero for perfect data
    assert errors.rmse < 0.0001
    assert errors.percent_error < 0.01  # Less than 0.01%


def test_compute_3d_distance_errors_with_noise(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that noisy synthetic data produces proportional error."""
    np.random.seed(42)

    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    corner_ids = [0, 1, 2, 3]
    corner_positions_3d = {}
    for cid in corner_ids:
        board_pos = board_geometry.corner_positions[cid]
        corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6])

    # Add pixel noise
    detections = create_synthetic_detections(
        calibration,
        board_geometry,
        corner_ids,
        corner_positions_3d,
        frame_idx=0,
        noise_std=1.0,
    )

    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    # With noise, error should be non-zero but reasonable
    assert errors.mean > 0.0
    assert errors.mean < 0.020  # Less than 2 cm (reasonable for 1px noise at this distance)
    assert errors.num_comparisons == 3  # Adjacent pairs: (0,1), (1,2), (2,3)
    assert errors.num_frames == 1
    # Check new fields exist and are reasonable
    assert errors.rmse > 0.0
    assert errors.percent_error > 0.0


def test_compute_3d_distance_errors_multiple_frames(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test aggregation across multiple frames."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    # Create detections for multiple frames (corners 0,1,2 give 2 adjacent pairs)
    corner_ids = [0, 1, 2]
    all_frames = {}

    for frame_idx in range(3):
        corner_positions_3d = {}
        for cid in corner_ids:
            board_pos = board_geometry.corner_positions[cid]
            corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6 + frame_idx * 0.1])

        frame_det = create_synthetic_detections(
            calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx
        )
        all_frames[frame_idx] = frame_det.frames[frame_idx]

    detections = DetectionResult(
        frames=all_frames,
        camera_names=["cam1", "cam2"],
        total_frames=3,
    )

    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    # Should have comparisons from all frames
    # 3 corners = 2 adjacent pairs per frame (0-1, 1-2), 3 frames = 6 pairs
    assert errors.num_comparisons == 6
    assert errors.num_frames == 3
    assert errors.mean < 0.001


def test_compute_3d_distance_errors_signed_error(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that signed error is positive when distances are overestimated."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    # Create detections with artificially scaled-up distances (1% larger)
    corner_ids = [0, 1, 2, 3]
    corner_positions_3d = {}
    for cid in corner_ids:
        board_pos = board_geometry.corner_positions[cid]
        # Scale up by 1% to simulate overestimation
        corner_positions_3d[cid] = (board_pos * 1.01) + np.array([0.0, 0.0, 0.6])

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    # With 1% scale, distances should be ~1% larger
    assert errors.signed_mean > 0, "Signed mean should be positive for overestimated distances"
    expected_signed_error = 0.01 * board_config.square_size  # ~0.4mm
    assert abs(errors.signed_mean - expected_signed_error) < 0.0002  # Within 0.2mm


def test_compute_3d_distance_errors_empty(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that empty detections return NaN-valued DistanceErrors."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1},
        interface=interface_params,
        board=board_config,
    )

    detections = DetectionResult(frames={}, camera_names=["cam1"], total_frames=0)

    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    assert np.isnan(errors.mean)
    assert np.isnan(errors.std)
    assert np.isnan(errors.max_error)
    assert np.isnan(errors.signed_mean)
    assert np.isnan(errors.rmse)
    assert np.isnan(errors.percent_error)
    assert errors.num_comparisons == 0
    assert errors.num_frames == 0


def test_compute_3d_distance_errors_include_per_pair(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that include_per_pair flag populates per_corner_pair dict."""
    cam1 = create_camera_calibration(
        "cam1",
        camera_intrinsics,
        R=np.eye(3),
        t=np.array([0.0, 0.0, 0.5]),
        interface_distance=0.3,
    )
    cam2 = create_camera_calibration(
        "cam2",
        camera_intrinsics,
        R=np.array([[0.866, 0.0, 0.5], [0.0, 1.0, 0.0], [-0.5, 0.0, 0.866]]),
        t=np.array([0.2, 0.0, 0.5]),
        interface_distance=0.3,
    )

    calibration = create_calibration_result(
        cameras={"cam1": cam1, "cam2": cam2},
        interface=interface_params,
        board=board_config,
    )

    corner_ids = [0, 1, 2]
    corner_positions_3d = {}
    for cid in corner_ids:
        board_pos = board_geometry.corner_positions[cid]
        corner_positions_3d[cid] = board_pos + np.array([0.0, 0.0, 0.6])

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    errors = compute_3d_distance_errors(
        calibration, detections, board_geometry, include_per_pair=True
    )

    assert errors.per_corner_pair is not None
    # 3 corners in first row = 2 adjacent pairs: (0,1), (1,2)
    assert len(errors.per_corner_pair) == 2
    assert (0, 1) in errors.per_corner_pair
    assert (1, 2) in errors.per_corner_pair
    # Values should be signed errors (near zero for perfect data)
    for pair, signed_error in errors.per_corner_pair.items():
        assert abs(signed_error) < 0.001, f"Pair {pair} error too large"


def test_compute_board_planarity_error_coplanar():
    """Test that perfectly coplanar points yield near-zero error."""
    # Create coplanar points (Z=0 plane)
    corners = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
        3: np.array([1.0, 1.0, 0.0]),
    }

    error = compute_board_planarity_error(corners)

    assert error is not None
    assert error < 1e-10, f"Planarity error {error} should be near zero"


def test_compute_board_planarity_error_noisy():
    """Test that points with Z perturbation produce proportional error."""
    np.random.seed(42)

    # Create points on Z=0 plane with small Z noise
    z_noise_std = 0.001  # 1mm
    corners = {}
    for i in range(4):
        x = (i % 2) * 0.04
        y = (i // 2) * 0.04
        z = np.random.normal(0, z_noise_std)
        corners[i] = np.array([x, y, z])

    error = compute_board_planarity_error(corners)

    assert error is not None
    # Error should be on the order of the noise magnitude
    assert error > 0.0
    assert error < 0.01  # Should be less than 1cm


def test_compute_board_planarity_error_insufficient():
    """Test that fewer than 3 corners returns None."""
    # Test with 0 corners
    assert compute_board_planarity_error({}) is None

    # Test with 1 corner
    corners_1 = {0: np.array([0.0, 0.0, 0.0])}
    assert compute_board_planarity_error(corners_1) is None

    # Test with 2 corners
    corners_2 = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
    }
    assert compute_board_planarity_error(corners_2) is None

    # Test with 3 corners (should work)
    corners_3 = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([0.0, 1.0, 0.0]),
    }
    error = compute_board_planarity_error(corners_3)
    assert error is not None
    assert error < 1e-10
