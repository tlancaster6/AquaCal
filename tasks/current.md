# Task: 4.4 Joint Refinement

## Objective

Implement Stage 4 optional joint refinement that re-optimizes all parameters from Stage 3, with the option to also refine camera intrinsics (focal length and principal point).

## Context Files

Read these files before starting (in order):

1. `src/aquacal/config/schema.py` — All dataclasses (CameraIntrinsics, CameraExtrinsics, BoardPose, DetectionResult, ConvergenceError)
2. `src/aquacal/calibration/interface_estimation.py` — Stage 3 implementation (reference for cost function structure)
3. `src/aquacal/core/camera.py` — Camera class
4. `src/aquacal/core/interface_model.py` — Interface class
5. `src/aquacal/core/refractive_geometry.py` — `refractive_project()` function
6. `src/aquacal/core/board.py` — BoardGeometry class
7. `docs/development_plan.md` (lines 446-465) — Stage 4 description

## Modify

Files to create or edit:

- `src/aquacal/calibration/refinement.py`
- `tests/unit/test_refinement.py`
- `src/aquacal/calibration/__init__.py` (add exports)

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/calibration/interface_estimation.py` (do not import private functions)
- `src/aquacal/config/schema.py` (no new dataclasses needed)

---

## Main Function Signature

```python
def joint_refinement(
    stage3_result: tuple[
        dict[str, CameraExtrinsics],
        dict[str, float],
        list[BoardPose],
        float,
    ],
    detections: DetectionResult,
    intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str,
    refine_intrinsics: bool = False,
    interface_normal: Vec3 | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
    loss: str = "huber",
    loss_scale: float = 1.0,
    min_corners: int = 4,
) -> tuple[
    dict[str, CameraExtrinsics],
    dict[str, float],
    list[BoardPose],
    dict[str, CameraIntrinsics],
    float,
]:
    """
    Jointly refine all calibration parameters, optionally including intrinsics.

    This is Stage 4 of the calibration pipeline. It takes the output of Stage 3
    and performs additional optimization. When refine_intrinsics=True, it also
    optimizes focal lengths and principal points.

    Args:
        stage3_result: Output tuple from optimize_interface:
            (extrinsics, interface_distances, board_poses, rms_error)
        detections: Underwater ChArUco detections
        intrinsics: Per-camera intrinsic parameters (used as initial values)
        board: ChArUco board geometry
        reference_camera: Camera name fixed at origin
        refine_intrinsics: If True, also optimize fx, fy, cx, cy per camera
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
        n_air: Refractive index of air
        n_water: Refractive index of water
        loss: Robust loss function ("linear", "huber", "soft_l1", "cauchy")
        loss_scale: Scale parameter for robust loss in pixels
        min_corners: Minimum corners per detection to include

    Returns:
        Tuple of:
        - dict[str, CameraExtrinsics]: Refined extrinsics for all cameras
        - dict[str, float]: Refined interface distances per camera
        - list[BoardPose]: Refined board poses
        - dict[str, CameraIntrinsics]: Refined intrinsics (modified if refine_intrinsics=True,
          otherwise copies of input)
        - float: Final RMS reprojection error in pixels

    Raises:
        ConvergenceError: If optimization fails to converge
        ValueError: If reference_camera not in stage3_result extrinsics

    Notes:
        - When refine_intrinsics=False, this is essentially re-running Stage 3
          optimization from the Stage 3 solution (useful for verifying convergence)
        - Distortion coefficients are NOT refined (kept fixed)
        - Intrinsic bounds: fx, fy in [0.5*initial, 2.0*initial],
          cx, cy in [0, image_width] and [0, image_height]
    """
```

---

## Helper Functions

```python
def _pack_params_with_intrinsics(
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board_poses: dict[int, BoardPose],
    intrinsics: dict[str, CameraIntrinsics],
    reference_camera: str,
    camera_order: list[str],
    frame_order: list[int],
    refine_intrinsics: bool,
) -> NDArray[np.float64]:
    """
    Pack optimization parameters into a 1D array.

    Parameter layout:
    - For each non-reference camera (in camera_order, skipping reference):
        cam_rvec (3), cam_tvec (3)
    - For each camera (in camera_order, including reference):
        interface_distance (1)
    - For each frame (in frame_order):
        board_rvec (3), board_tvec (3)
    - If refine_intrinsics, for each camera (in camera_order):
        fx (1), fy (1), cx (1), cy (1)

    Total length:
    - Without intrinsics: 6*(N_cams-1) + N_cams + 6*N_frames
    - With intrinsics: above + 4*N_cams
    """


def _unpack_params_with_intrinsics(
    params: NDArray[np.float64],
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    base_intrinsics: dict[str, CameraIntrinsics],
    camera_order: list[str],
    frame_order: list[int],
    refine_intrinsics: bool,
) -> tuple[
    dict[str, CameraExtrinsics],
    dict[str, float],
    dict[int, BoardPose],
    dict[str, CameraIntrinsics],
]:
    """
    Unpack 1D parameter array into structured objects.

    Args:
        params: 1D parameter vector
        reference_camera: Name of reference camera
        reference_extrinsics: Fixed extrinsics for reference camera
        base_intrinsics: Base intrinsics (for distortion coeffs and image_size)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        refine_intrinsics: Whether intrinsics are included in params

    Returns:
        Tuple of (extrinsics_dict, distances_dict, board_poses_dict, intrinsics_dict)
    """


def _cost_function_with_intrinsics(
    params: NDArray[np.float64],
    detections: DetectionResult,
    base_intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    interface_normal: Vec3,
    n_air: float,
    n_water: float,
    camera_order: list[str],
    frame_order: list[int],
    min_corners: int,
    refine_intrinsics: bool,
) -> NDArray[np.float64]:
    """
    Compute reprojection residuals for all observations.

    Similar to Stage 3 cost function, but optionally uses refined intrinsics
    from the parameter vector instead of fixed intrinsics.

    Returns:
        1D array of residuals [r0_x, r0_y, r1_x, r1_y, ...] in pixels
    """
```

---

## Algorithm Details

### Parameter Packing with Intrinsics

```python
def _pack_params_with_intrinsics(...) -> NDArray[np.float64]:
    params = []

    # Camera extrinsics (skip reference)
    for cam_name in camera_order:
        if cam_name == reference_camera:
            continue
        ext = extrinsics[cam_name]
        rvec = matrix_to_rvec(ext.R)
        params.extend(rvec)
        params.extend(ext.t)

    # Interface distances (all cameras)
    for cam_name in camera_order:
        params.append(interface_distances[cam_name])

    # Board poses
    for frame_idx in frame_order:
        bp = board_poses[frame_idx]
        params.extend(bp.rvec)
        params.extend(bp.tvec)

    # Intrinsics (if refining)
    if refine_intrinsics:
        for cam_name in camera_order:
            K = intrinsics[cam_name].K
            params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])  # fx, fy, cx, cy

    return np.array(params, dtype=np.float64)
```

### Unpacking with Intrinsics

```python
def _unpack_params_with_intrinsics(...) -> tuple[...]:
    idx = 0
    n_cams = len(camera_order)
    n_frames = len(frame_order)

    # Extrinsics
    extrinsics_out = {}
    for cam_name in camera_order:
        if cam_name == reference_camera:
            extrinsics_out[cam_name] = reference_extrinsics
        else:
            rvec = params[idx:idx+3]
            tvec = params[idx+3:idx+6]
            idx += 6
            R = rvec_to_matrix(rvec)
            extrinsics_out[cam_name] = CameraExtrinsics(R=R, t=tvec)

    # Distances
    distances_out = {}
    for cam_name in camera_order:
        distances_out[cam_name] = params[idx]
        idx += 1

    # Board poses
    board_poses_out = {}
    for frame_idx in frame_order:
        rvec = params[idx:idx+3]
        tvec = params[idx+3:idx+6]
        idx += 6
        board_poses_out[frame_idx] = BoardPose(frame_idx=frame_idx, rvec=rvec, tvec=tvec)

    # Intrinsics
    intrinsics_out = {}
    if refine_intrinsics:
        for cam_name in camera_order:
            fx, fy, cx, cy = params[idx:idx+4]
            idx += 4
            base = base_intrinsics[cam_name]
            K_new = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float64)
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=K_new,
                dist_coeffs=base.dist_coeffs.copy(),  # Keep original distortion
                image_size=base.image_size,
            )
    else:
        # Return copies of base intrinsics
        for cam_name in camera_order:
            base = base_intrinsics[cam_name]
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=base.K.copy(),
                dist_coeffs=base.dist_coeffs.copy(),
                image_size=base.image_size,
            )

    return extrinsics_out, distances_out, board_poses_out, intrinsics_out
```

### Parameter Bounds

```python
def _build_bounds(
    camera_order: list[str],
    frame_order: list[int],
    reference_camera: str,
    base_intrinsics: dict[str, CameraIntrinsics],
    refine_intrinsics: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build lower and upper bounds for optimization."""
    n_cams = len(camera_order)
    n_frames = len(frame_order)
    n_extrinsic_params = 6 * (n_cams - 1)
    n_distance_params = n_cams
    n_pose_params = 6 * n_frames
    n_intrinsic_params = 4 * n_cams if refine_intrinsics else 0
    total = n_extrinsic_params + n_distance_params + n_pose_params + n_intrinsic_params

    lower = np.full(total, -np.inf)
    upper = np.full(total, np.inf)

    # Interface distances: [0.01, 2.0]
    dist_start = n_extrinsic_params
    dist_end = dist_start + n_distance_params
    lower[dist_start:dist_end] = 0.01
    upper[dist_start:dist_end] = 2.0

    # Intrinsic bounds
    if refine_intrinsics:
        intr_start = n_extrinsic_params + n_distance_params + n_pose_params
        for i, cam_name in enumerate(camera_order):
            base = base_intrinsics[cam_name]
            fx, fy = base.K[0, 0], base.K[1, 1]
            w, h = base.image_size
            offset = intr_start + i * 4

            # fx, fy: [0.5*initial, 2.0*initial]
            lower[offset] = 0.5 * fx
            upper[offset] = 2.0 * fx
            lower[offset + 1] = 0.5 * fy
            upper[offset + 1] = 2.0 * fy

            # cx: [0, width], cy: [0, height]
            lower[offset + 2] = 0
            upper[offset + 2] = w
            lower[offset + 3] = 0
            upper[offset + 3] = h

    return lower, upper
```

### Main Function Implementation

```python
def joint_refinement(...) -> tuple[...]:
    # Validate inputs
    extrinsics_in, distances_in, poses_in, _ = stage3_result
    if reference_camera not in extrinsics_in:
        raise ValueError(f"reference_camera '{reference_camera}' not in stage3_result")

    # Setup
    if interface_normal is None:
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    camera_order = sorted(extrinsics_in.keys())
    board_poses_dict = {bp.frame_idx: bp for bp in poses_in}
    frame_order = sorted(board_poses_dict.keys())

    if not frame_order:
        raise ConvergenceError("No board poses from Stage 3")

    reference_extrinsics = extrinsics_in[reference_camera]

    # Pack initial parameters
    initial_params = _pack_params_with_intrinsics(
        extrinsics_in, distances_in, board_poses_dict, intrinsics,
        reference_camera, camera_order, frame_order, refine_intrinsics,
    )

    # Build bounds
    lower, upper = _build_bounds(
        camera_order, frame_order, reference_camera,
        intrinsics, refine_intrinsics,
    )

    # Run optimization
    result = least_squares(
        _cost_function_with_intrinsics,
        x0=initial_params,
        args=(
            detections, intrinsics, board, reference_camera,
            reference_extrinsics, interface_normal, n_air, n_water,
            camera_order, frame_order, min_corners, refine_intrinsics,
        ),
        method='trf',
        loss=loss,
        f_scale=loss_scale,
        bounds=(lower, upper),
        verbose=0,
    )

    if result.status <= 0:
        raise ConvergenceError(f"Optimization failed: {result.message}")

    # Unpack results
    ext_out, dist_out, poses_out, intr_out = _unpack_params_with_intrinsics(
        result.x, reference_camera, reference_extrinsics, intrinsics,
        camera_order, frame_order, refine_intrinsics,
    )

    # Convert board poses dict to sorted list
    poses_list = [poses_out[idx] for idx in sorted(poses_out.keys())]

    rms_error = np.sqrt(np.mean(result.fun**2))

    return ext_out, dist_out, poses_list, intr_out, rms_error
```

---

## Acceptance Criteria

- [ ] `joint_refinement` returns refined extrinsics, distances, poses, intrinsics, and RMS error
- [ ] Reference camera extrinsics remain fixed at input values
- [ ] When `refine_intrinsics=False`, returned intrinsics are copies of input
- [ ] When `refine_intrinsics=True`, fx/fy/cx/cy may change, dist_coeffs unchanged
- [ ] Interface distances stay within bounds [0.01, 2.0]
- [ ] Intrinsic bounds enforced: focal lengths within [0.5x, 2x], principal point within image
- [ ] Raises `ConvergenceError` when optimization fails
- [ ] Raises `ValueError` for invalid reference_camera
- [ ] With `refine_intrinsics=False`, achieves similar or better RMS than Stage 3 input
- [ ] With `refine_intrinsics=True`, can recover perturbed intrinsics in synthetic tests
- [ ] Tests pass: `pytest tests/unit/test_refinement.py -v`
- [ ] No modifications to files outside "Modify" list

---

## Testing Strategy

### Test Fixtures

```python
import pytest
import numpy as np

from aquacal.config.schema import (
    BoardConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    Detection,
    FrameDetections,
    DetectionResult,
    BoardPose,
    ConvergenceError,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.calibration.refinement import (
    joint_refinement,
    _pack_params_with_intrinsics,
    _unpack_params_with_intrinsics,
)


@pytest.fixture
def board_config() -> BoardConfig:
    return BoardConfig(
        squares_x=6, squares_y=5,
        square_size=0.04, marker_size=0.03,
        dictionary="DICT_4X4_50"
    )


@pytest.fixture
def board(board_config) -> BoardGeometry:
    return BoardGeometry(board_config)


@pytest.fixture
def intrinsics() -> dict[str, CameraIntrinsics]:
    """Ground truth intrinsics for 3 cameras."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return {
        'cam0': CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        'cam1': CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        'cam2': CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
    }


@pytest.fixture
def ground_truth_extrinsics() -> dict[str, CameraExtrinsics]:
    """Ground truth camera extrinsics."""
    return {
        'cam0': CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
        'cam1': CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.3, 0.0, 0.0], dtype=np.float64),
        ),
        'cam2': CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.0, 0.3, 0.0], dtype=np.float64),
        ),
    }


@pytest.fixture
def ground_truth_distances() -> dict[str, float]:
    return {'cam0': 0.15, 'cam1': 0.16, 'cam2': 0.14}


@pytest.fixture
def synthetic_board_poses() -> list[BoardPose]:
    """Board poses for 10 frames underwater."""
    poses = []
    for i in range(10):
        x_offset = 0.05 * (i % 4 - 1.5)
        y_offset = 0.05 * (i // 4 - 1)
        poses.append(BoardPose(
            frame_idx=i,
            rvec=np.array([0.1 * (i % 3), 0.1 * (i % 2), 0.0], dtype=np.float64),
            tvec=np.array([x_offset, y_offset, 0.4], dtype=np.float64),
        ))
    return poses


def generate_synthetic_detections(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board: BoardGeometry,
    board_poses: list[BoardPose],
    noise_std: float = 0.0,
) -> DetectionResult:
    """Generate synthetic detections using refractive_project."""
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    frames = {}

    for bp in board_poses:
        corners_3d = board.transform_corners(bp.rvec, bp.tvec)
        detections_dict = {}

        for cam_name in intrinsics:
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
            interface = Interface(
                normal=interface_normal,
                base_height=0.0,
                camera_offsets={cam_name: interface_distances[cam_name]},
            )

            corner_ids = []
            corners_2d = []

            for corner_id in range(board.num_corners):
                point_3d = corners_3d[corner_id]
                projected = refractive_project(camera, interface, point_3d)

                if projected is not None:
                    w, h = intrinsics[cam_name].image_size
                    if 0 <= projected[0] < w and 0 <= projected[1] < h:
                        corner_ids.append(corner_id)
                        px = projected.copy()
                        if noise_std > 0:
                            px += np.random.normal(0, noise_std, 2)
                        corners_2d.append(px)

            if len(corner_ids) >= 4:
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


@pytest.fixture
def stage3_result(ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses):
    """Simulated Stage 3 output."""
    return (
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
        0.5,  # RMS error
    )
```

### Test Cases

```python
class TestPackUnpackWithIntrinsics:
    def test_round_trip_without_intrinsics(
        self, ground_truth_extrinsics, ground_truth_distances,
        synthetic_board_poses, intrinsics
    ):
        """Pack/unpack round-trip without refining intrinsics."""
        camera_order = ['cam0', 'cam1', 'cam2']
        frame_order = [0, 1, 2]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses[:3]}

        packed = _pack_params_with_intrinsics(
            ground_truth_extrinsics, ground_truth_distances,
            board_poses_dict, intrinsics, 'cam0',
            camera_order, frame_order, refine_intrinsics=False,
        )

        ext_out, dist_out, poses_out, intr_out = _unpack_params_with_intrinsics(
            packed, 'cam0', ground_truth_extrinsics['cam0'], intrinsics,
            camera_order, frame_order, refine_intrinsics=False,
        )

        for cam in camera_order:
            np.testing.assert_allclose(ext_out[cam].R, ground_truth_extrinsics[cam].R)
            np.testing.assert_allclose(ext_out[cam].t, ground_truth_extrinsics[cam].t)
            assert abs(dist_out[cam] - ground_truth_distances[cam]) < 1e-10
            np.testing.assert_allclose(intr_out[cam].K, intrinsics[cam].K)

    def test_round_trip_with_intrinsics(
        self, ground_truth_extrinsics, ground_truth_distances,
        synthetic_board_poses, intrinsics
    ):
        """Pack/unpack round-trip with intrinsics."""
        camera_order = ['cam0', 'cam1', 'cam2']
        frame_order = [0, 1, 2]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses[:3]}

        packed = _pack_params_with_intrinsics(
            ground_truth_extrinsics, ground_truth_distances,
            board_poses_dict, intrinsics, 'cam0',
            camera_order, frame_order, refine_intrinsics=True,
        )

        ext_out, dist_out, poses_out, intr_out = _unpack_params_with_intrinsics(
            packed, 'cam0', ground_truth_extrinsics['cam0'], intrinsics,
            camera_order, frame_order, refine_intrinsics=True,
        )

        for cam in camera_order:
            np.testing.assert_allclose(intr_out[cam].K, intrinsics[cam].K)
            # Distortion coeffs should be preserved
            np.testing.assert_allclose(
                intr_out[cam].dist_coeffs, intrinsics[cam].dist_coeffs
            )


class TestJointRefinement:
    def test_without_intrinsics_maintains_quality(
        self, board, intrinsics, ground_truth_extrinsics,
        ground_truth_distances, synthetic_board_poses, stage3_result
    ):
        """Refinement without intrinsics maintains or improves RMS."""
        detections = generate_synthetic_detections(
            intrinsics, ground_truth_extrinsics, ground_truth_distances,
            board, synthetic_board_poses, noise_std=0.5,
        )

        ext_opt, dist_opt, poses_opt, intr_opt, rms = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=intrinsics,
            board=board,
            reference_camera='cam0',
            refine_intrinsics=False,
        )

        assert rms < 2.0
        # Intrinsics should be unchanged
        for cam in intrinsics:
            np.testing.assert_allclose(intr_opt[cam].K, intrinsics[cam].K)

    def test_with_intrinsics_recovers_perturbed(
        self, board, intrinsics, ground_truth_extrinsics,
        ground_truth_distances, synthetic_board_poses
    ):
        """Refinement with intrinsics can recover perturbed focal lengths."""
        # Create detections with ground truth intrinsics
        detections = generate_synthetic_detections(
            intrinsics, ground_truth_extrinsics, ground_truth_distances,
            board, synthetic_board_poses, noise_std=0.3,
        )

        # Perturb intrinsics (5% error in focal length)
        perturbed_intrinsics = {}
        for cam, intr in intrinsics.items():
            K_perturbed = intr.K.copy()
            K_perturbed[0, 0] *= 1.05  # fx + 5%
            K_perturbed[1, 1] *= 0.95  # fy - 5%
            perturbed_intrinsics[cam] = CameraIntrinsics(
                K=K_perturbed,
                dist_coeffs=intr.dist_coeffs.copy(),
                image_size=intr.image_size,
            )

        stage3_result = (
            ground_truth_extrinsics,
            ground_truth_distances,
            synthetic_board_poses,
            1.0,
        )

        ext_opt, dist_opt, poses_opt, intr_opt, rms = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=perturbed_intrinsics,
            board=board,
            reference_camera='cam0',
            refine_intrinsics=True,
        )

        # Should achieve good RMS
        assert rms < 2.0

        # Refined intrinsics should be closer to ground truth than perturbed
        for cam in intrinsics:
            gt_fx = intrinsics[cam].K[0, 0]
            perturbed_fx = perturbed_intrinsics[cam].K[0, 0]
            refined_fx = intr_opt[cam].K[0, 0]

            perturbed_error = abs(perturbed_fx - gt_fx)
            refined_error = abs(refined_fx - gt_fx)
            assert refined_error < perturbed_error, f"fx not improved for {cam}"

    def test_reference_camera_unchanged(
        self, board, intrinsics, ground_truth_extrinsics,
        ground_truth_distances, synthetic_board_poses, stage3_result
    ):
        """Reference camera extrinsics remain fixed."""
        detections = generate_synthetic_detections(
            intrinsics, ground_truth_extrinsics, ground_truth_distances,
            board, synthetic_board_poses, noise_std=0.5,
        )

        ext_opt, _, _, _, _ = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=intrinsics,
            board=board,
            reference_camera='cam0',
        )

        np.testing.assert_allclose(
            ext_opt['cam0'].R, ground_truth_extrinsics['cam0'].R, atol=1e-10
        )
        np.testing.assert_allclose(
            ext_opt['cam0'].t, ground_truth_extrinsics['cam0'].t, atol=1e-10
        )

    def test_raises_for_invalid_reference(
        self, board, intrinsics, stage3_result
    ):
        """Raises ValueError for invalid reference camera."""
        detections = DetectionResult(frames={}, camera_names=['cam0'], total_frames=0)

        with pytest.raises(ValueError, match="reference"):
            joint_refinement(
                stage3_result=stage3_result,
                detections=detections,
                intrinsics=intrinsics,
                board=board,
                reference_camera='camX',
            )

    def test_distances_within_bounds(
        self, board, intrinsics, ground_truth_extrinsics,
        ground_truth_distances, synthetic_board_poses, stage3_result
    ):
        """Interface distances stay within bounds."""
        detections = generate_synthetic_detections(
            intrinsics, ground_truth_extrinsics, ground_truth_distances,
            board, synthetic_board_poses, noise_std=0.5,
        )

        _, dist_opt, _, _, _ = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=intrinsics,
            board=board,
            reference_camera='cam0',
        )

        for cam, dist in dist_opt.items():
            assert 0.01 <= dist <= 2.0

    def test_intrinsic_bounds_enforced(
        self, board, intrinsics, ground_truth_extrinsics,
        ground_truth_distances, synthetic_board_poses, stage3_result
    ):
        """Intrinsic parameters stay within bounds."""
        detections = generate_synthetic_detections(
            intrinsics, ground_truth_extrinsics, ground_truth_distances,
            board, synthetic_board_poses, noise_std=0.5,
        )

        _, _, _, intr_opt, _ = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=intrinsics,
            board=board,
            reference_camera='cam0',
            refine_intrinsics=True,
        )

        for cam in intrinsics:
            base = intrinsics[cam]
            opt = intr_opt[cam]

            # Focal lengths within [0.5x, 2x]
            assert 0.5 * base.K[0, 0] <= opt.K[0, 0] <= 2.0 * base.K[0, 0]
            assert 0.5 * base.K[1, 1] <= opt.K[1, 1] <= 2.0 * base.K[1, 1]

            # Principal point within image
            w, h = base.image_size
            assert 0 <= opt.K[0, 2] <= w
            assert 0 <= opt.K[1, 2] <= h
```

---

## Import Structure

Update `src/aquacal/calibration/__init__.py`:

```python
"""Calibration pipeline modules."""

from aquacal.calibration.intrinsics import (
    calibrate_intrinsics_single,
    calibrate_intrinsics_all,
)
from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    estimate_board_pose,
    build_pose_graph,
    estimate_extrinsics,
)
from aquacal.calibration.interface_estimation import (
    optimize_interface,
)
from aquacal.calibration.refinement import (
    joint_refinement,
)

__all__ = [
    # intrinsics
    "calibrate_intrinsics_single",
    "calibrate_intrinsics_all",
    # extrinsics
    "Observation",
    "PoseGraph",
    "estimate_board_pose",
    "build_pose_graph",
    "estimate_extrinsics",
    # interface_estimation
    "optimize_interface",
    # refinement
    "joint_refinement",
]
```

---

## Notes

1. **Code similarity to interface_estimation.py**: The cost function structure is nearly identical. This is intentional - keeping the module self-contained avoids coupling and makes each stage independently testable.

2. **Intrinsic parameterization**: Only fx, fy, cx, cy are refined. Distortion coefficients are kept fixed because:
   - They're well-determined by Stage 1 in-air calibration
   - They rarely drift
   - Adding 5+ more params per camera increases risk of overfitting

3. **When to use Stage 4**:
   - If Stage 3 RMS is higher than expected
   - If you suspect intrinsics may have changed since Stage 1
   - For maximum accuracy (diminishing returns if Stage 3 converged well)

4. **Performance**: Similar to Stage 3. With intrinsics, adds 4*N_cams parameters (~12 for 3 cameras), negligible impact.

5. **Imports needed**:
   ```python
   import numpy as np
   from numpy.typing import NDArray
   from scipy.optimize import least_squares

   from aquacal.config.schema import (
       CameraIntrinsics, CameraExtrinsics, BoardPose,
       DetectionResult, ConvergenceError, Vec3,
   )
   from aquacal.core.board import BoardGeometry
   from aquacal.core.camera import Camera
   from aquacal.core.interface_model import Interface
   from aquacal.core.refractive_geometry import refractive_project
   from aquacal.utils.transforms import rvec_to_matrix, matrix_to_rvec
   ```
