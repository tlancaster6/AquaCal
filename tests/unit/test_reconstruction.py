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
    SpatialMeasurements,
    DepthBinnedErrors,
    SpatialErrorGrid,
    bin_by_depth,
    compute_xy_error_grids,
    save_spatial_measurements,
    load_spatial_measurements,
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
            camera_distances={cam_name: cam_calib.interface_distance},
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
        assert error < 0.001, f"Corner {cid}: error {error * 1000:.3f}mm > 1mm"


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
    assert errors.mean < 0.0001, f"Mean error {errors.mean * 1000:.3f}mm > 0.1mm"
    assert errors.max_error < 0.001, f"Max error {errors.max_error * 1000:.3f}mm > 1mm"
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
    assert (
        errors.mean < 0.020
    )  # Less than 2 cm (reasonable for 1px noise at this distance)
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
            corner_positions_3d[cid] = board_pos + np.array(
                [0.0, 0.0, 0.6 + frame_idx * 0.1]
            )

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
    assert errors.signed_mean > 0, (
        "Signed mean should be positive for overestimated distances"
    )
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

    with pytest.warns(UserWarning, match="No valid 3D distance comparisons"):
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


def test_spatial_measurements_basic(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that include_spatial=True populates spatial measurements correctly."""
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

    # Compute errors with spatial data
    errors = compute_3d_distance_errors(
        calibration, detections, board_geometry, include_spatial=True
    )

    # Verify spatial field is populated
    assert errors.spatial is not None
    assert isinstance(errors.spatial, SpatialMeasurements)

    # 4 corners in first row = 3 adjacent pairs: (0,1), (1,2), (2,3)
    assert errors.spatial.positions.shape == (3, 3)
    assert errors.spatial.signed_errors.shape == (3,)
    assert errors.spatial.frame_indices.shape == (3,)

    # All signed errors should be near zero for perfect data
    assert np.all(np.abs(errors.spatial.signed_errors) < 0.001)

    # Verify midpoints are reasonable (all should be near Z=0.6)
    for i in range(3):
        assert abs(errors.spatial.positions[i, 2] - 0.6) < 0.001


def test_spatial_measurements_midpoint_values(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that midpoints are the arithmetic mean of corner positions."""
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

    # Use known corner positions at regular intervals
    corner_ids = [0, 1, 2]
    corner_positions_3d = {
        0: np.array([0.0, 0.0, 0.6]),
        1: np.array([0.04, 0.0, 0.6]),
        2: np.array([0.08, 0.0, 0.6]),
    }

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    errors = compute_3d_distance_errors(
        calibration, detections, board_geometry, include_spatial=True
    )

    # Triangulate the corners to get actual 3D positions
    triangulated = triangulate_charuco_corners(calibration, detections, frame_idx=0)

    # Verify midpoints for pairs (0,1) and (1,2)
    # The pairs are ordered by get_adjacent_corner_pairs, which returns (0,1), (1,2) for this setup
    expected_midpoint_01 = (triangulated[0] + triangulated[1]) / 2.0
    expected_midpoint_12 = (triangulated[1] + triangulated[2]) / 2.0

    # The spatial measurements are collected in order
    assert np.allclose(errors.spatial.positions[0], expected_midpoint_01, atol=0.001)
    assert np.allclose(errors.spatial.positions[1], expected_midpoint_12, atol=0.001)


def test_spatial_measurements_frame_indices(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that frame_indices are correctly recorded for multiple frames."""
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
            corner_positions_3d[cid] = board_pos + np.array(
                [0.0, 0.0, 0.6 + frame_idx * 0.1]
            )

        frame_det = create_synthetic_detections(
            calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx
        )
        all_frames[frame_idx] = frame_det.frames[frame_idx]

    detections = DetectionResult(
        frames=all_frames,
        camera_names=["cam1", "cam2"],
        total_frames=3,
    )

    errors = compute_3d_distance_errors(
        calibration, detections, board_geometry, include_spatial=True
    )

    # 3 corners = 2 adjacent pairs per frame, 3 frames = 6 total measurements
    assert errors.spatial.positions.shape == (6, 3)
    assert errors.spatial.signed_errors.shape == (6,)
    assert errors.spatial.frame_indices.shape == (6,)

    # Verify frame indices are correct
    # First 2 measurements from frame 0, next 2 from frame 1, last 2 from frame 2
    expected_frames = [0, 0, 1, 1, 2, 2]
    assert np.array_equal(errors.spatial.frame_indices, expected_frames)


def test_spatial_measurements_disabled_by_default(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that spatial field is None when include_spatial is not set."""
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

    detections = create_synthetic_detections(
        calibration, board_geometry, corner_ids, corner_positions_3d, frame_idx=0
    )

    # Call without include_spatial (default False)
    errors = compute_3d_distance_errors(calibration, detections, board_geometry)

    # Verify spatial field is None
    assert errors.spatial is None


def test_spatial_measurements_empty_data(
    board_config, board_geometry, interface_params, camera_intrinsics
):
    """Test that empty detections with include_spatial=True return empty arrays."""
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

    with pytest.warns(UserWarning, match="No valid 3D distance comparisons"):
        errors = compute_3d_distance_errors(
            calibration, detections, board_geometry, include_spatial=True
        )

    # Verify spatial field has empty arrays with correct shapes
    assert errors.spatial is not None
    assert errors.spatial.positions.shape == (0, 3)
    assert errors.spatial.signed_errors.shape == (0,)
    assert errors.spatial.frame_indices.shape == (0,)


class TestBinByDepth:
    """Test bin_by_depth function."""

    def test_basic_binning(self):
        """Test basic binning with 20 measurements spanning Z=[0.4, 0.8]."""
        # Create synthetic spatial measurements
        np.random.seed(42)
        n_measurements = 20
        positions = np.random.uniform(
            low=[0.0, 0.0, 0.4], high=[0.1, 0.1, 0.8], size=(n_measurements, 3)
        )
        signed_errors = np.random.uniform(-0.001, 0.001, size=n_measurements)
        frame_indices = np.zeros(n_measurements, dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Bin into 4 bins
        binned = bin_by_depth(spatial, n_bins=4)

        # Verify shapes
        assert binned.bin_edges.shape == (5,)
        assert binned.bin_centers.shape == (4,)
        assert binned.signed_means.shape == (4,)
        assert binned.signed_stds.shape == (4,)
        assert binned.counts.shape == (4,)

        # Verify total count
        assert binned.counts.sum() == 20

        # Verify bin edges span the data
        z_min = positions[:, 2].min()
        z_max = positions[:, 2].max()
        assert np.isclose(binned.bin_edges[0], z_min, atol=1e-10)
        assert np.isclose(binned.bin_edges[-1], z_max, atol=1e-10)

    def test_signed_mean_correctness(self):
        """Test that signed mean is computed correctly."""
        # Create measurements with known values
        positions = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
        signed_errors = np.array([0.001, -0.001])
        frame_indices = np.array([0, 0], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Single bin should contain all measurements
        binned = bin_by_depth(spatial, n_bins=1)

        # Mean should be zero
        assert np.isclose(binned.signed_means[0], 0.0, atol=1e-10)
        assert binned.counts[0] == 2

    def test_empty_bins(self):
        """Test that empty bins have NaN for signed_means and signed_stds."""
        # Create measurements that leave some bins empty
        # Put all measurements in the first half of the range
        positions = np.array(
            [
                [0.0, 0.0, 0.4],
                [0.0, 0.0, 0.41],
                [0.0, 0.0, 0.42],
                [0.0, 0.0, 0.43],
            ]
        )
        signed_errors = np.array([0.001, 0.002, 0.003, 0.004])
        frame_indices = np.array([0, 0, 0, 0], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Use 10 bins - most will be empty since all data is clustered
        binned = bin_by_depth(spatial, n_bins=10)

        # Should have some bins with zero counts
        assert np.any(binned.counts == 0)

        # Empty bins should have NaN for signed_means and signed_stds
        empty_bins = binned.counts == 0
        assert np.all(np.isnan(binned.signed_means[empty_bins]))
        assert np.all(np.isnan(binned.signed_stds[empty_bins]))

        # Non-empty bins should not have NaN
        non_empty_bins = binned.counts > 0
        assert np.all(~np.isnan(binned.signed_means[non_empty_bins]))
        assert np.all(~np.isnan(binned.signed_stds[non_empty_bins]))

    def test_empty_spatial_raises(self):
        """Test that empty spatial measurements raise ValueError."""
        spatial = SpatialMeasurements(
            positions=np.empty((0, 3), dtype=np.float64),
            signed_errors=np.empty((0,), dtype=np.float64),
            frame_indices=np.empty((0,), dtype=np.int32),
        )

        with pytest.raises(ValueError, match="Cannot bin empty"):
            bin_by_depth(spatial)


class TestSpatialIO:
    """Test save and load functions for SpatialMeasurements."""

    def test_roundtrip(self, tmp_path):
        """Test save and load roundtrip."""
        # Create spatial measurements
        positions = np.array([[0.1, 0.2, 0.4], [0.3, 0.4, 0.5]])
        signed_errors = np.array([0.001, -0.002])
        frame_indices = np.array([0, 1], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Save
        csv_path = tmp_path / "spatial.csv"
        save_spatial_measurements(spatial, csv_path)

        # Verify file exists
        assert csv_path.exists()

        # Load
        loaded = load_spatial_measurements(csv_path)

        # Verify data matches
        assert np.allclose(loaded.positions, positions)
        assert np.allclose(loaded.signed_errors, signed_errors)
        assert np.array_equal(loaded.frame_indices, frame_indices)

    def test_load_missing_file(self, tmp_path):
        """Test that loading a missing file raises FileNotFoundError."""
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(FileNotFoundError):
            load_spatial_measurements(missing_path)


class TestComputeXYErrorGrids:
    """Test compute_xy_error_grids function."""

    def test_basic_gridding(self):
        """Test basic gridding with 50 measurements across depth and XY space."""
        # Create synthetic spatial measurements
        np.random.seed(42)
        n_measurements = 50
        positions = np.random.uniform(
            low=[-0.1, -0.1, 0.4], high=[0.1, 0.1, 0.8], size=(n_measurements, 3)
        )
        signed_errors = np.random.uniform(-0.001, 0.001, size=n_measurements)
        frame_indices = np.zeros(n_measurements, dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Create depth bin edges (2 bins)
        depth_bin_edges = np.array([0.4, 0.6, 0.8])

        # Compute XY error grids (4x4 XY grid)
        grid = compute_xy_error_grids(
            spatial, depth_bin_edges=depth_bin_edges, xy_grid_size=(4, 4)
        )

        # Verify output shapes
        assert grid.grids.shape == (2, 4, 4)  # (2 depth bins, 4 Y bins, 4 X bins)
        assert grid.counts.shape == (2, 4, 4)
        assert grid.x_edges.shape == (5,)  # 4 bins + 1
        assert grid.y_edges.shape == (5,)  # 4 bins + 1
        assert grid.depth_bin_edges.shape == (3,)  # 2 bins + 1

        # Verify total count matches
        assert grid.counts.sum() == 50

    def test_known_values(self):
        """Test that grid computes correct mean when all measurements fall in one cell."""
        # Create measurements all at the same XY location but different depths
        positions = np.array(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.51],
                [0.0, 0.0, 0.52],
            ]
        )
        signed_errors = np.array([0.001, 0.002, 0.003])
        frame_indices = np.array([0, 0, 0], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Single depth bin containing all measurements
        depth_bin_edges = np.array([0.4, 0.6])

        # Small XY grid (2x2)
        grid = compute_xy_error_grids(
            spatial, depth_bin_edges=depth_bin_edges, xy_grid_size=(2, 2)
        )

        # All measurements should fall in the same depth bin (0) and same XY cell
        # Find the non-NaN cell
        non_nan_mask = ~np.isnan(grid.grids[0])
        assert np.sum(non_nan_mask) == 1, "Should have exactly one non-NaN cell"

        # Verify the mean is correct
        yi, xi = np.where(non_nan_mask)
        assert grid.counts[0, yi[0], xi[0]] == 3
        expected_mean = np.mean(signed_errors)
        assert np.isclose(grid.grids[0, yi[0], xi[0]], expected_mean, atol=1e-10)

    def test_empty_cells_are_nan(self):
        """Test that empty cells have NaN in grids and 0 in counts."""
        # Create measurements that only occupy some cells
        # Place 4 measurements in corners of XY space
        positions = np.array(
            [
                [-0.1, -0.1, 0.5],
                [0.1, -0.1, 0.5],
                [-0.1, 0.1, 0.5],
                [0.1, 0.1, 0.5],
            ]
        )
        signed_errors = np.array([0.001, 0.002, 0.003, 0.004])
        frame_indices = np.array([0, 0, 0, 0], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Single depth bin
        depth_bin_edges = np.array([0.4, 0.6])

        # 3x3 grid - center cells should be empty
        grid = compute_xy_error_grids(
            spatial, depth_bin_edges=depth_bin_edges, xy_grid_size=(3, 3)
        )

        # Verify there are empty cells
        empty_cells = grid.counts[0] == 0
        assert np.any(empty_cells), "Should have some empty cells"

        # Empty cells should have NaN in grids
        assert np.all(np.isnan(grid.grids[0][empty_cells]))

        # Non-empty cells should not have NaN
        non_empty_cells = grid.counts[0] > 0
        assert np.all(~np.isnan(grid.grids[0][non_empty_cells]))

    def test_custom_xy_range(self):
        """Test that custom XY range is respected."""
        # Create measurements
        positions = np.array([[0.0, 0.0, 0.5]])
        signed_errors = np.array([0.001])
        frame_indices = np.array([0], dtype=np.int32)

        spatial = SpatialMeasurements(
            positions=positions,
            signed_errors=signed_errors,
            frame_indices=frame_indices,
        )

        # Single depth bin
        depth_bin_edges = np.array([0.4, 0.6])

        # Custom XY range
        custom_xy_range = ((-0.5, 0.5), (-0.3, 0.3))

        grid = compute_xy_error_grids(
            spatial,
            depth_bin_edges=depth_bin_edges,
            xy_grid_size=(4, 4),
            xy_range=custom_xy_range,
        )

        # Verify edges match custom range
        assert np.isclose(grid.x_edges[0], -0.5, atol=1e-10)
        assert np.isclose(grid.x_edges[-1], 0.5, atol=1e-10)
        assert np.isclose(grid.y_edges[0], -0.3, atol=1e-10)
        assert np.isclose(grid.y_edges[-1], 0.3, atol=1e-10)

    def test_empty_spatial_raises(self):
        """Test that empty spatial measurements raise ValueError."""
        spatial = SpatialMeasurements(
            positions=np.empty((0, 3), dtype=np.float64),
            signed_errors=np.empty((0,), dtype=np.float64),
            frame_indices=np.empty((0,), dtype=np.int32),
        )

        depth_bin_edges = np.array([0.4, 0.6])

        with pytest.raises(ValueError, match="Cannot compute XY grids for empty"):
            compute_xy_error_grids(spatial, depth_bin_edges=depth_bin_edges)
