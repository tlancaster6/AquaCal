"""Synthetic data generation for testing and validation.

This module provides functions to generate synthetic calibration data with known
ground truth. The main entry point is generate_synthetic_rig() which returns
complete scenarios with detections and optionally rendered images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project


@dataclass
class SyntheticScenario:
    """Complete synthetic test scenario with ground truth.

    Attributes:
        name: Scenario name
        board_config: ChArUco board specification
        intrinsics: Per-camera intrinsics
        extrinsics: Per-camera extrinsics
        water_zs: Per-camera interface distances (Z-coordinate of water surface)
        board_poses: List of board poses for all frames
        noise_std: Gaussian noise standard deviation applied to detections (pixels)
        description: Human-readable description
        images: Optional dict of rendered images (camera_name -> frame_idx -> image)
    """

    name: str
    board_config: BoardConfig
    intrinsics: dict[str, CameraIntrinsics]
    extrinsics: dict[str, CameraExtrinsics]
    water_zs: dict[str, float]
    board_poses: list[BoardPose]
    noise_std: float
    description: str
    images: dict[str, dict[int, NDArray]] | None = None


def generate_camera_intrinsics(
    image_size: tuple[int, int] = (1920, 1080),
    fov_horizontal_deg: float = 60.0,
    principal_point_offset: tuple[float, float] = (0.0, 0.0),
    distortion_k1: float = 0.0,
    distortion_k2: float = 0.0,
) -> CameraIntrinsics:
    """
    Generate camera intrinsics with specified parameters.

    Args:
        image_size: (width, height) in pixels
        fov_horizontal_deg: Horizontal field of view in degrees
        principal_point_offset: Offset from image center (pixels)
        distortion_k1: First radial distortion coefficient
        distortion_k2: Second radial distortion coefficient

    Returns:
        CameraIntrinsics with computed K matrix and distortion
    """
    width, height = image_size

    # Compute focal length from horizontal FOV
    # fov = 2 * atan(width / (2 * fx))
    # => fx = width / (2 * tan(fov/2))
    fov_rad = np.deg2rad(fov_horizontal_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Square pixels

    # Principal point at image center plus offset
    cx = width / 2 + principal_point_offset[0]
    cy = height / 2 + principal_point_offset[1]

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Distortion coefficients: [k1, k2, p1, p2, k3]
    dist_coeffs = np.array(
        [distortion_k1, distortion_k2, 0.0, 0.0, 0.0], dtype=np.float64
    )

    return CameraIntrinsics(K=K, dist_coeffs=dist_coeffs, image_size=image_size)


def _rotation_z(angle: float) -> NDArray[np.float64]:
    """Create rotation matrix for rotation around Z axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def generate_camera_array(
    n_cameras: int,
    layout: str = "grid",
    spacing: float = 0.1,
    height_above_water: float = 0.15,
    height_variation: float = 0.005,
    image_size: tuple[int, int] = (1920, 1080),
    fov_deg: float = 60.0,
    seed: int = 42,
) -> tuple[dict[str, CameraIntrinsics], dict[str, CameraExtrinsics], dict[str, float]]:
    """
    Generate a realistic camera array with known ground truth.

    Args:
        n_cameras: Number of cameras (2-14)
        layout: Camera arrangement - "grid", "line", or "ring"
        spacing: Distance between adjacent cameras (meters)
        height_above_water: Mean interface distance (meters)
        height_variation: Std dev of per-camera height variation (meters)
        image_size: Image dimensions (width, height)
        fov_deg: Horizontal field of view
        seed: Random seed for reproducibility

    Returns:
        Tuple of (intrinsics, extrinsics, water_zs) dicts keyed by camera name.
        Camera "cam0" is always the reference camera at origin with identity rotation.
    """
    rng = np.random.default_rng(seed)

    intrinsics: dict[str, CameraIntrinsics] = {}
    extrinsics: dict[str, CameraExtrinsics] = {}
    distances: dict[str, float] = {}

    # Generate camera positions based on layout
    positions: list[NDArray[np.float64]] = []

    if layout == "grid":
        # Arrange in rough square grid
        side = int(np.ceil(np.sqrt(n_cameras)))
        for i in range(n_cameras):
            row, col = i // side, i % side
            x = (col - (side - 1) / 2) * spacing
            y = (row - (side - 1) / 2) * spacing
            positions.append(np.array([x, y, 0.0], dtype=np.float64))
    elif layout == "line":
        for i in range(n_cameras):
            positions.append(np.array([i * spacing, 0.0, 0.0], dtype=np.float64))
    elif layout == "ring":
        angles = np.linspace(0, 2 * np.pi, n_cameras, endpoint=False)
        radius = spacing * n_cameras / (2 * np.pi)
        for a in angles:
            positions.append(
                np.array(
                    [radius * np.cos(a), radius * np.sin(a), 0.0], dtype=np.float64
                )
            )
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Center positions so cam0 is at origin
    offset = positions[0].copy()
    positions = [p - offset for p in positions]

    for i, pos in enumerate(positions):
        cam_name = f"cam{i}"

        # Intrinsics: same for all cameras
        intrinsics[cam_name] = generate_camera_intrinsics(
            image_size=image_size,
            fov_horizontal_deg=fov_deg,
        )

        # Extrinsics: cameras look straight down
        # Add small random roll for realism (but not for reference camera)
        if i == 0:
            roll = 0.0
        else:
            roll = rng.uniform(-0.1, 0.1)  # radians

        R = _rotation_z(roll)
        t = -R @ pos

        extrinsics[cam_name] = CameraExtrinsics(R=R, t=t)

        # Interface distance with small variation
        if i == 0:
            dist = height_above_water  # Reference camera: exact height
        else:
            dist = height_above_water + rng.normal(0, height_variation)
        distances[cam_name] = dist

    return intrinsics, extrinsics, distances


def generate_real_rig_array(
    height_above_water: float = 0.75,
    height_variation: float = 0.002,
    seed: int = 42,
) -> tuple[dict[str, CameraIntrinsics], dict[str, CameraExtrinsics], dict[str, float]]:
    """
    Generate camera array matching the real-world 13-camera rig.

    Rig geometry:
    - 13 cameras total:
      - cam0: center camera at origin (reference)
      - cam1-cam6: inner circle, radius 300mm, evenly spaced at 60 deg intervals
      - cam7-cam12: outer circle, radius 600mm, evenly spaced at 60 deg intervals
    - All cameras point straight down toward water surface
    - Roll angles: image width (horizontal axis) is tangent to the circle
      - Center camera (cam0): roll = 0 deg (defines coordinate system)
      - Circle cameras: roll = theta + 90 deg, where theta is angular position on circle
      - Inner and outer cameras at same angular position have same roll

    Camera specs:
    - Image size: 1600 x 1200 pixels (4:3 aspect ratio)
    - FOV: 56 deg horizontal, 43 deg vertical
    - Height above water: ~750mm (with small per-camera variation)

    Args:
        height_above_water: Mean interface distance (meters), default 0.75
        height_variation: Std dev of per-camera height variation (meters)
        seed: Random seed for height variations

    Returns:
        Tuple of (intrinsics, extrinsics, water_zs) dicts keyed by camera name.
    """
    rng = np.random.default_rng(seed)

    # Real rig camera specs
    IMAGE_SIZE = (1600, 1200)
    FOV_HORIZONTAL_DEG = 56.0
    INNER_RADIUS = 0.300  # 300mm
    OUTER_RADIUS = 0.600  # 600mm
    N_CAMERAS_PER_RING = 6
    ANGULAR_SPACING = 2 * np.pi / N_CAMERAS_PER_RING  # 60 degrees

    intrinsics: dict[str, CameraIntrinsics] = {}
    extrinsics: dict[str, CameraExtrinsics] = {}
    distances: dict[str, float] = {}

    # Camera 0: center, reference
    intrinsics["cam0"] = generate_camera_intrinsics(
        image_size=IMAGE_SIZE,
        fov_horizontal_deg=FOV_HORIZONTAL_DEG,
    )
    extrinsics["cam0"] = CameraExtrinsics(
        R=np.eye(3, dtype=np.float64),
        t=np.zeros(3, dtype=np.float64),
    )
    distances["cam0"] = height_above_water

    # Inner ring: cam1-cam6 at radius 300mm
    for i in range(N_CAMERAS_PER_RING):
        cam_name = f"cam{i + 1}"
        theta = i * ANGULAR_SPACING  # Angular position: 0, 60, 120, 180, 240, 300 deg

        # Position in world XY plane
        x = INNER_RADIUS * np.cos(theta)
        y = INNER_RADIUS * np.sin(theta)
        C = np.array([x, y, 0.0], dtype=np.float64)

        # Roll angle: image width tangent to circle
        # Tangent direction at theta is perpendicular to radial direction
        # Roll = theta + 90 deg makes camera X-axis point along tangent
        roll = theta + np.pi / 2

        R = _rotation_z(roll)
        t = -R @ C

        intrinsics[cam_name] = generate_camera_intrinsics(
            image_size=IMAGE_SIZE,
            fov_horizontal_deg=FOV_HORIZONTAL_DEG,
        )
        extrinsics[cam_name] = CameraExtrinsics(R=R, t=t)
        distances[cam_name] = height_above_water + rng.normal(0, height_variation)

    # Outer ring: cam7-cam12 at radius 600mm
    for i in range(N_CAMERAS_PER_RING):
        cam_name = f"cam{i + 7}"
        theta = i * ANGULAR_SPACING  # Same angular positions as inner ring

        # Position in world XY plane
        x = OUTER_RADIUS * np.cos(theta)
        y = OUTER_RADIUS * np.sin(theta)
        C = np.array([x, y, 0.0], dtype=np.float64)

        # Roll angle: same as inner ring at same angular position
        roll = theta + np.pi / 2

        R = _rotation_z(roll)
        t = -R @ C

        intrinsics[cam_name] = generate_camera_intrinsics(
            image_size=IMAGE_SIZE,
            fov_horizontal_deg=FOV_HORIZONTAL_DEG,
        )
        extrinsics[cam_name] = CameraExtrinsics(R=R, t=t)
        distances[cam_name] = height_above_water + rng.normal(0, height_variation)

    return intrinsics, extrinsics, distances


def generate_board_trajectory(
    n_frames: int,
    camera_positions: dict[str, NDArray[np.float64]],
    water_zs: dict[str, float],
    depth_range: tuple[float, float] = (0.3, 0.6),
    xy_extent: float = 0.15,
    rotation_range_deg: float = 15.0,
    min_cameras_per_frame: int = 2,
    seed: int = 42,
) -> list[BoardPose]:
    """
    Generate board poses ensuring pose graph connectivity.

    Creates a trajectory that ensures:
    - Each frame is visible by at least min_cameras_per_frame cameras
    - The pose graph is connected (can chain from reference to all cameras)
    - Board stays within reasonable depth range underwater

    Args:
        n_frames: Number of frames to generate
        camera_positions: Dict of camera center positions (from extrinsics)
        water_zs: Per-camera interface distances
        depth_range: (min_z, max_z) for board center in world coords
        xy_extent: Maximum XY offset from origin
        rotation_range_deg: Maximum board tilt from horizontal
        min_cameras_per_frame: Minimum cameras that must see board
        seed: Random seed

    Returns:
        List of BoardPose objects with frame indices 0 to n_frames-1
    """
    rng = np.random.default_rng(seed)

    poses: list[BoardPose] = []
    for frame_idx in range(n_frames):
        # Position: random within extent, random depth
        x = rng.uniform(-xy_extent, xy_extent)
        y = rng.uniform(-xy_extent, xy_extent)
        z = rng.uniform(depth_range[0], depth_range[1])
        tvec = np.array([x, y, z], dtype=np.float64)

        # Rotation: small tilts, full in-plane rotation
        max_tilt = np.deg2rad(rotation_range_deg)
        rx = rng.uniform(-max_tilt, max_tilt)
        ry = rng.uniform(-max_tilt, max_tilt)
        rz = rng.uniform(-np.pi, np.pi)
        rvec = np.array([rx, ry, rz], dtype=np.float64)

        poses.append(BoardPose(frame_idx=frame_idx, rvec=rvec, tvec=tvec))

    return poses


def generate_real_rig_trajectory(
    n_frames: int = 100,
    depth_range: tuple[float, float] = (0.9, 1.5),
    seed: int = 42,
) -> list[BoardPose]:
    """
    Generate board trajectory appropriate for the real rig geometry.

    The real rig has cameras at 750mm above water, so the board should be
    deeper underwater (depth_range default 0.9-1.5m from camera, i.e.,
    ~150-750mm below water surface).

    Trajectory covers the full field of view:
    - Positions sweep across the ~1.2m diameter footprint of the outer ring
    - Ensures connectivity by visiting regions seen by multiple cameras

    Args:
        n_frames: Number of frames to generate
        depth_range: (min_z, max_z) for board center in world coords
        seed: Random seed

    Returns:
        List of BoardPose objects
    """
    rng = np.random.default_rng(seed)

    # The rig spans ~1.2m diameter (outer ring at 600mm radius)
    # Board should move throughout this area to ensure all cameras see it
    XY_EXTENT = 0.5  # +/-500mm from center to ensure coverage
    ROTATION_RANGE_DEG = 20.0

    poses: list[BoardPose] = []
    for frame_idx in range(n_frames):
        # Position: random within footprint, random depth
        x = rng.uniform(-XY_EXTENT, XY_EXTENT)
        y = rng.uniform(-XY_EXTENT, XY_EXTENT)
        z = rng.uniform(depth_range[0], depth_range[1])
        tvec = np.array([x, y, z], dtype=np.float64)

        # Rotation: small tilts, full in-plane rotation
        max_tilt = np.deg2rad(ROTATION_RANGE_DEG)
        rx = rng.uniform(-max_tilt, max_tilt)
        ry = rng.uniform(-max_tilt, max_tilt)
        rz = rng.uniform(-np.pi, np.pi)
        rvec = np.array([rx, ry, rz], dtype=np.float64)

        poses.append(BoardPose(frame_idx=frame_idx, rvec=rvec, tvec=tvec))

    return poses


def generate_dense_xy_grid(
    depth: float,
    n_grid: int = 7,
    xy_extent: float = 0.5,
    tilt_deg: float = 3.0,
    frame_offset: int = 0,
    seed: int = 42,
) -> list[BoardPose]:
    """
    Generate board poses at a regular XY grid at a fixed depth.

    Used for dense spatial coverage in reconstruction evaluation and heatmaps.
    Each grid position has a small random tilt and random in-plane rotation.

    Args:
        depth: Z coordinate for all board poses (meters)
        n_grid: Number of grid positions per axis (total poses = n_grid^2)
        xy_extent: Grid spans from -xy_extent to +xy_extent in X and Y (meters)
        tilt_deg: Maximum random tilt from horizontal (degrees)
        frame_offset: Starting frame index (default 0)
        seed: Random seed for reproducible tilts and rotations

    Returns:
        List of n_grid^2 BoardPose objects with frame indices starting from frame_offset
    """
    rng = np.random.default_rng(seed)

    # Generate grid positions
    x_values = np.linspace(-xy_extent, xy_extent, n_grid)
    y_values = np.linspace(-xy_extent, xy_extent, n_grid)

    poses: list[BoardPose] = []
    frame_idx = frame_offset

    for x in x_values:
        for y in y_values:
            tvec = np.array([x, y, depth], dtype=np.float64)

            # Small random tilt + random in-plane rotation
            max_tilt = np.deg2rad(tilt_deg)
            rx = rng.uniform(-max_tilt, max_tilt)
            ry = rng.uniform(-max_tilt, max_tilt)
            rz = rng.uniform(-np.pi, np.pi)
            rvec = np.array([rx, ry, rz], dtype=np.float64)

            poses.append(BoardPose(frame_idx=frame_idx, rvec=rvec, tvec=tvec))
            frame_idx += 1

    return poses


def generate_synthetic_detections(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    water_zs: dict[str, float],
    board: BoardGeometry,
    board_poses: list[BoardPose],
    noise_std: float = 0.0,
    min_corners: int = 8,
    seed: int = 42,
) -> DetectionResult:
    """
    Generate synthetic detections by projecting through refractive interface.

    For each board pose and camera:
    1. Transform board corners to world coordinates
    2. Project each corner through refractive interface
    3. Add Gaussian noise to pixel coordinates
    4. Filter corners outside image bounds
    5. Only include camera if >= min_corners visible

    Args:
        intrinsics: Per-camera intrinsics
        extrinsics: Per-camera extrinsics
        water_zs: Per-camera interface distances
        board: Board geometry
        board_poses: List of board poses
        noise_std: Gaussian noise standard deviation (pixels)
        min_corners: Minimum corners for valid detection
        seed: Random seed for noise

    Returns:
        DetectionResult matching format from real detection pipeline
    """
    rng = np.random.default_rng(seed)
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    frames: dict[int, FrameDetections] = {}

    for bp in board_poses:
        corners_3d = board.transform_corners(bp.rvec, bp.tvec)
        detections_dict: dict[str, Detection] = {}

        for cam_name in intrinsics:
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
            interface = Interface(
                normal=interface_normal,
                camera_distances={cam_name: water_zs[cam_name]},
            )

            corner_ids: list[int] = []
            corners_2d: list[NDArray[np.float64]] = []

            for corner_id in range(board.num_corners):
                point_3d = corners_3d[corner_id]
                projected = refractive_project(camera, interface, point_3d)

                if projected is not None:
                    # Check if within image bounds
                    w, h = intrinsics[cam_name].image_size
                    if 0 <= projected[0] < w and 0 <= projected[1] < h:
                        corner_ids.append(corner_id)
                        px = projected.copy()
                        if noise_std > 0:
                            px += rng.normal(0, noise_std, 2)
                        corners_2d.append(px)

            if len(corner_ids) >= min_corners:
                detections_dict[cam_name] = Detection(
                    corner_ids=np.array(corner_ids, dtype=np.int32),
                    corners_2d=np.array(corners_2d, dtype=np.float64),
                )

        if detections_dict:
            frames[bp.frame_idx] = FrameDetections(
                frame_idx=bp.frame_idx,
                detections=detections_dict,
            )

    return DetectionResult(
        frames=frames,
        camera_names=list(intrinsics.keys()),
        total_frames=len(board_poses),
    )


def compute_calibration_errors(
    result: CalibrationResult,
    ground_truth: SyntheticScenario,
) -> dict[str, float]:
    """
    Compare calibration result to ground truth.

    Computes:
    - focal_length_error_percent: Max relative error in fx, fy
    - principal_point_error_px: Max error in cx, cy
    - rotation_error_deg: Max rotation error across cameras
    - translation_error_mm: Max translation error across cameras
    - water_z_error_mm: Max interface distance error

    Args:
        result: Calibration result from pipeline
        ground_truth: Synthetic scenario with known truth

    Returns:
        Dict of error metrics
    """
    max_focal_error_pct = 0.0
    max_pp_error_px = 0.0
    max_rotation_error_deg = 0.0
    max_translation_error_mm = 0.0
    max_interface_error_mm = 0.0

    for cam_name in ground_truth.intrinsics:
        if cam_name not in result.cameras:
            continue

        gt_intr = ground_truth.intrinsics[cam_name]
        gt_extr = ground_truth.extrinsics[cam_name]
        gt_dist = ground_truth.water_zs[cam_name]

        cal = result.cameras[cam_name]
        cal_intr = cal.intrinsics
        cal_extr = cal.extrinsics
        cal_dist = cal.water_z

        # Focal length error (relative)
        fx_gt, fy_gt = gt_intr.K[0, 0], gt_intr.K[1, 1]
        fx_cal, fy_cal = cal_intr.K[0, 0], cal_intr.K[1, 1]
        fx_err = abs(fx_cal - fx_gt) / fx_gt * 100
        fy_err = abs(fy_cal - fy_gt) / fy_gt * 100
        max_focal_error_pct = max(max_focal_error_pct, fx_err, fy_err)

        # Principal point error (absolute, pixels)
        cx_gt, cy_gt = gt_intr.K[0, 2], gt_intr.K[1, 2]
        cx_cal, cy_cal = cal_intr.K[0, 2], cal_intr.K[1, 2]
        pp_err = np.sqrt((cx_cal - cx_gt) ** 2 + (cy_cal - cy_gt) ** 2)
        max_pp_error_px = max(max_pp_error_px, pp_err)

        # Rotation error
        # Compute relative rotation: R_err = R_cal @ R_gt.T
        R_err = cal_extr.R @ gt_extr.R.T
        # Rotation angle from rotation matrix: angle = arccos((trace(R) - 1) / 2)
        trace = np.trace(R_err)
        # Clamp to valid range for arccos
        cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.rad2deg(angle_rad)
        max_rotation_error_deg = max(max_rotation_error_deg, angle_deg)

        # Translation error (in mm)
        # Camera center position difference
        C_gt = gt_extr.C
        C_cal = cal_extr.C
        trans_err_mm = np.linalg.norm(C_cal - C_gt) * 1000
        max_translation_error_mm = max(max_translation_error_mm, trans_err_mm)

        # Interface distance error (in mm)
        dist_err_mm = abs(cal_dist - gt_dist) * 1000
        max_interface_error_mm = max(max_interface_error_mm, dist_err_mm)

    return {
        "focal_length_error_percent": max_focal_error_pct,
        "principal_point_error_px": max_pp_error_px,
        "rotation_error_deg": max_rotation_error_deg,
        "translation_error_mm": max_translation_error_mm,
        "water_z_error_mm": max_interface_error_mm,
    }


def generate_synthetic_rig(
    preset: str, *, include_images: bool = False, noisy: bool = False
) -> SyntheticScenario:
    """
    Generate a complete synthetic calibration scenario with fixed presets.

    This function produces synthetic multi-camera calibration data with known
    ground truth. It's designed for testing and validating the AquaCal pipeline.

    Available presets:
    - 'small': 2 cameras (line layout), 10 frames, clean detections by default.
      Good for quick smoke tests and debugging.
    - 'medium': 6 cameras (grid layout), 80 frames, clean detections by default.
      Balanced size for integration testing.
    - 'large': 13 cameras (real rig geometry), 300 frames, clean detections by default.
      Full-scale realistic scenario matching actual hardware.

    All presets use the same ChArUco board:
    - 12x9 squares (11x8 corners)
    - 60mm square size, 45mm marker size
    - DICT_5X5_100 ArUco dictionary

    Args:
        preset: Preset name ('small', 'medium', or 'large')
        include_images: If True, render synthetic ChArUco images for all cameras and frames.
            Increases generation time but provides visual validation data.
        noisy: If True, add Gaussian pixel noise to detections. Noise levels:
            small=0.3px, medium=0.5px, large=0.5px. Default False (clean detections).

    Returns:
        SyntheticScenario with complete ground truth (intrinsics, extrinsics,
        interface distances, board poses, detections). If include_images=True,
        scenario.images contains rendered ChArUco frames.

    Raises:
        ValueError: If preset name is not recognized.

    Examples:
        >>> from aquacal.datasets import generate_synthetic_rig
        >>> # Quick test scenario
        >>> scenario = generate_synthetic_rig('small')
        >>> print(f"{len(scenario.intrinsics)} cameras, {len(scenario.board_poses)} frames")
        2 cameras, 10 frames
        >>>
        >>> # With rendered images for visual validation
        >>> scenario = generate_synthetic_rig('small', include_images=True)
        >>> img = scenario.images['cam0'][0]  # First frame from cam0
        >>> print(img.shape, img.dtype)
        (1080, 1920) uint8
        >>>
        >>> # Noisy detections for robustness testing
        >>> scenario = generate_synthetic_rig('medium', noisy=True)
        >>> print(f"Noise std: {scenario.noise_std}px")
        Noise std: 0.5px
    """
    # Common board config for all presets (matches real hardware)
    board_config = BoardConfig(
        squares_x=12,
        squares_y=9,
        square_size=0.060,
        marker_size=0.045,
        dictionary="DICT_5X5_100",
    )

    # Preset configurations
    if preset == "small":
        seed = 42
        noise_std = 0.3 if noisy else 0.0
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=2,
            layout="line",
            spacing=0.15,
            height_above_water=0.15,
            height_variation=0.003,
            seed=seed,
        )
        camera_positions = {cam: ext.C for cam, ext in extrinsics.items()}
        board_poses = generate_board_trajectory(
            n_frames=10,
            camera_positions=camera_positions,
            water_zs=distances,
            depth_range=(0.25, 0.40),
            xy_extent=0.06,
            seed=seed,
        )
        description = f"Small: 2 cameras, 10 frames, {noise_std}px noise"

    elif preset == "medium":
        seed = 123
        noise_std = 0.5 if noisy else 0.0
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=6,
            layout="grid",
            spacing=0.12,
            height_above_water=0.15,
            height_variation=0.005,
            seed=seed,
        )
        camera_positions = {cam: ext.C for cam, ext in extrinsics.items()}
        board_poses = generate_board_trajectory(
            n_frames=80,
            camera_positions=camera_positions,
            water_zs=distances,
            depth_range=(0.25, 0.45),
            xy_extent=0.10,
            seed=seed,
        )
        description = f"Medium: 6 cameras, 80 frames, {noise_std}px noise"

    elif preset == "large":
        seed = 456
        noise_std = 0.5 if noisy else 0.0
        intrinsics, extrinsics, distances = generate_real_rig_array(
            height_above_water=0.75,
            height_variation=0.002,
            seed=seed,
        )
        board_poses = generate_real_rig_trajectory(
            n_frames=300,
            depth_range=(0.9, 1.5),
            seed=seed,
        )
        description = f"Large: 13 cameras (real rig), 300 frames, {noise_std}px noise"

    else:
        valid_presets = ["small", "medium", "large"]
        raise ValueError(f"Unknown preset: '{preset}'. Valid presets: {valid_presets}")

    # Generate detections
    board = BoardGeometry(board_config)
    detection_result = generate_synthetic_detections(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        water_zs=distances,
        board=board,
        board_poses=board_poses,
        noise_std=noise_std,
        min_corners=8,
        seed=seed,
    )

    # Create scenario
    scenario = SyntheticScenario(
        name=preset,
        board_config=board_config,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        water_zs=distances,
        board_poses=board_poses,
        noise_std=noise_std,
        description=description,
        images=None,
    )

    # Optionally render images
    if include_images:
        from aquacal.datasets.rendering import render_scenario_images

        scenario.images = render_scenario_images(scenario, detection_result)

    return scenario
