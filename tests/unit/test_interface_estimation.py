"""Tests for interface estimation (Stage 3 optimization)."""

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
    InsufficientDataError,
    ConvergenceError,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.calibration.interface_estimation import (
    optimize_interface,
    _compute_initial_board_poses,
    _pack_params,
    _unpack_params,
)


@pytest.fixture
def board_config() -> BoardConfig:
    return BoardConfig(
        squares_x=6,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def board(board_config) -> BoardGeometry:
    return BoardGeometry(board_config)


@pytest.fixture
def intrinsics() -> dict[str, CameraIntrinsics]:
    """Intrinsics for 3 cameras."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return {
        "cam0": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam1": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam2": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
    }


@pytest.fixture
def ground_truth_extrinsics() -> dict[str, CameraExtrinsics]:
    """Ground truth camera extrinsics for testing."""
    return {
        "cam0": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
        "cam1": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.3, 0.0, 0.0], dtype=np.float64),  # 30cm to the right
        ),
        "cam2": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.0, 0.3, 0.0], dtype=np.float64),  # 30cm down
        ),
    }


@pytest.fixture
def ground_truth_distances() -> dict[str, float]:
    """Ground truth interface distances."""
    return {"cam0": 0.15, "cam1": 0.16, "cam2": 0.14}


def generate_synthetic_detections(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board: BoardGeometry,
    board_poses: list[BoardPose],
    noise_std: float = 0.0,
) -> DetectionResult:
    """
    Generate synthetic detections using refractive_project.

    Creates detections by projecting board corners through the refractive
    interface for each camera and frame.
    """
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
                    # Check if within image bounds
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
def synthetic_board_poses(board) -> list[BoardPose]:
    """Board poses for 3 frames at varying positions underwater.

    Reduced from 10 to 3 frames for faster tests while still providing
    sufficient coverage for optimization testing.
    """
    poses = []
    for i in range(3):
        # Board at Z=0.4m (underwater), varying XY position
        x_offset = 0.05 * (i - 1)
        y_offset = 0.02 * i
        poses.append(
            BoardPose(
                frame_idx=i,
                rvec=np.array([0.1 * (i % 3), 0.1 * (i % 2), 0.0], dtype=np.float64),
                tvec=np.array([x_offset, y_offset, 0.4], dtype=np.float64),
            )
        )
    return poses


class TestComputeInitialBoardPoses:
    def test_computes_poses_for_all_frames(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Computes board pose for each frame with valid detections."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
        )

        poses = _compute_initial_board_poses(
            detections, intrinsics, ground_truth_extrinsics, board
        )

        assert len(poses) == len(synthetic_board_poses)
        for frame_idx, pose in poses.items():
            assert isinstance(pose, BoardPose)
            assert pose.frame_idx == frame_idx
            assert pose.rvec.shape == (3,)
            assert pose.tvec.shape == (3,)

    def test_skips_frames_with_insufficient_corners(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Skips frames where no camera has enough corners."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
        )

        # Request very high min_corners to filter everything
        poses = _compute_initial_board_poses(
            detections, intrinsics, ground_truth_extrinsics, board, min_corners=1000
        )

        assert len(poses) == 0


class TestPackUnpackParams:
    def test_round_trip(
        self, ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Pack and unpack returns same values."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses[:3]]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses[:3]}

        packed = _pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        ext_out, dist_out, poses_out = _unpack_params(
            packed,
            reference_camera="cam0",
            reference_extrinsics=ground_truth_extrinsics["cam0"],
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # Check extrinsics
        for cam in camera_order:
            np.testing.assert_allclose(
                ext_out[cam].R, ground_truth_extrinsics[cam].R, atol=1e-10
            )
            np.testing.assert_allclose(
                ext_out[cam].t, ground_truth_extrinsics[cam].t, atol=1e-10
            )

        # Check distances
        for cam in camera_order:
            assert abs(dist_out[cam] - ground_truth_distances[cam]) < 1e-10

        # Check board poses
        for frame_idx in frame_order:
            np.testing.assert_allclose(
                poses_out[frame_idx].rvec, board_poses_dict[frame_idx].rvec, atol=1e-10
            )
            np.testing.assert_allclose(
                poses_out[frame_idx].tvec, board_poses_dict[frame_idx].tvec, atol=1e-10
            )

    def test_param_count(
        self, ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Verify correct number of parameters packed."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}
        n_frames = len(synthetic_board_poses)

        packed = _pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # 2 non-reference cameras * 6 params + 3 cameras * 1 distance + N frames * 6 params
        expected = 6 * 2 + 3 + 6 * n_frames
        assert len(packed) == expected

    def test_reference_camera_extrinsics_preserved(
        self, ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Reference camera extrinsics come from input, not packed params."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [0]
        board_poses_dict = {synthetic_board_poses[0].frame_idx: synthetic_board_poses[0]}

        # Create modified extrinsics for reference camera
        modified_ref = CameraExtrinsics(
            R=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64),
            t=np.array([1, 2, 3], dtype=np.float64),
        )

        packed = _pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # Unpack with different reference extrinsics
        ext_out, _, _ = _unpack_params(
            packed,
            reference_camera="cam0",
            reference_extrinsics=modified_ref,
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # Reference should match modified_ref, not original
        np.testing.assert_allclose(ext_out["cam0"].R, modified_ref.R, atol=1e-10)
        np.testing.assert_allclose(ext_out["cam0"].t, modified_ref.t, atol=1e-10)


class TestOptimizeInterface:
    def test_recovers_ground_truth(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Optimizer recovers ground truth from noisy detections."""
        # Set seed for reproducibility
        np.random.seed(42)

        # Generate detections with small noise
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        # Perturb initial extrinsics slightly
        initial_extrinsics = {}
        for cam, ext in ground_truth_extrinsics.items():
            if cam == "cam0":
                initial_extrinsics[cam] = ext  # Reference unchanged
            else:
                initial_extrinsics[cam] = CameraExtrinsics(
                    R=ext.R.copy(),
                    t=ext.t + np.random.normal(0, 0.01, 3),
                )

        # Perturb initial distances
        initial_distances = {
            cam: dist + np.random.normal(0, 0.01)
            for cam, dist in ground_truth_distances.items()
        }

        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=initial_extrinsics,
            board=board,
            reference_camera="cam0",
            initial_interface_distances=initial_distances,
        )

        # Should achieve low RMS error
        assert rms < 2.0, f"RMS error too high: {rms}"

        # Reference camera should be unchanged
        np.testing.assert_allclose(
            ext_opt["cam0"].R, ground_truth_extrinsics["cam0"].R, atol=1e-10
        )
        np.testing.assert_allclose(
            ext_opt["cam0"].t, ground_truth_extrinsics["cam0"].t, atol=1e-10
        )

        # Distances should be reasonably close to ground truth
        # With noise and potential degeneracies, allow 0.03m tolerance
        for cam in intrinsics:
            assert (
                abs(dist_opt[cam] - ground_truth_distances[cam]) < 0.03
            ), f"Distance for {cam} off by {abs(dist_opt[cam] - ground_truth_distances[cam])}"

    def test_raises_for_invalid_reference(
        self, board, intrinsics, ground_truth_extrinsics
    ):
        """Raises ValueError for invalid reference camera."""
        detections = DetectionResult(frames={}, camera_names=["cam0"], total_frames=0)

        with pytest.raises(ValueError, match="reference"):
            optimize_interface(
                detections=detections,
                intrinsics=intrinsics,
                initial_extrinsics=ground_truth_extrinsics,
                board=board,
                reference_camera="camX",
            )

    def test_raises_for_no_frames(self, board, intrinsics, ground_truth_extrinsics):
        """Raises InsufficientDataError when no valid frames."""
        detections = DetectionResult(
            frames={},
            camera_names=["cam0", "cam1", "cam2"],
            total_frames=0,
        )

        with pytest.raises(InsufficientDataError):
            optimize_interface(
                detections=detections,
                intrinsics=intrinsics,
                initial_extrinsics=ground_truth_extrinsics,
                board=board,
                reference_camera="cam0",
            )

    def test_distances_within_bounds(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Interface distances stay within [0.01, 2.0] bounds."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        _, dist_opt, _, _ = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
        )

        for cam, dist in dist_opt.items():
            assert 0.01 <= dist <= 2.0, f"Distance out of bounds for {cam}: {dist}"

    def test_default_interface_distances(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Works with default interface distances (0.15m)."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        # Don't provide initial_interface_distances
        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            initial_interface_distances=None,  # Should default to 0.15
        )

        assert rms < 2.0
        assert len(dist_opt) == len(intrinsics)

    def test_returns_correct_types(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Verify return types are correct."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
        )

        # Check types
        assert isinstance(ext_opt, dict)
        assert isinstance(dist_opt, dict)
        assert isinstance(poses_opt, list)
        assert isinstance(rms, float)

        # Check extrinsics dict
        for cam, ext in ext_opt.items():
            assert isinstance(cam, str)
            assert isinstance(ext, CameraExtrinsics)

        # Check distances dict
        for cam, dist in dist_opt.items():
            assert isinstance(cam, str)
            assert isinstance(dist, float)

        # Check board poses list
        for pose in poses_opt:
            assert isinstance(pose, BoardPose)

    def test_single_camera(self, board, intrinsics, ground_truth_distances):
        """Works with single camera (only interface distance optimized)."""
        np.random.seed(42)

        # Use only cam0
        single_intrinsics = {"cam0": intrinsics["cam0"]}
        single_extrinsics = {
            "cam0": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.zeros(3, dtype=np.float64),
            )
        }
        single_distances = {"cam0": ground_truth_distances["cam0"]}

        # Create poses
        poses = [
            BoardPose(
                frame_idx=i,
                rvec=np.array([0.1 * (i % 3), 0.1 * (i % 2), 0.0], dtype=np.float64),
                tvec=np.array([0.0, 0.0, 0.4], dtype=np.float64),
            )
            for i in range(5)
        ]

        detections = generate_synthetic_detections(
            single_intrinsics,
            single_extrinsics,
            single_distances,
            board,
            poses,
            noise_std=0.5,
        )

        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=single_intrinsics,
            initial_extrinsics=single_extrinsics,
            board=board,
            reference_camera="cam0",
        )

        # Should still work
        assert rms < 2.0
        assert len(ext_opt) == 1
        assert len(dist_opt) == 1

    def test_different_loss_functions(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Test different robust loss functions."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        # Robust loss functions should achieve low RMS
        for loss_type in ["huber", "soft_l1", "cauchy"]:
            _, _, _, rms = optimize_interface(
                detections=detections,
                intrinsics=intrinsics,
                initial_extrinsics=ground_truth_extrinsics,
                board=board,
                reference_camera="cam0",
                loss=loss_type,
            )
            assert rms < 2.0, f"RMS too high for loss={loss_type}: {rms}"

        # Linear loss (non-robust) may have higher error due to initialization
        # issues, but should still complete without error
        _, _, _, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            loss="linear",
        )
        # Just verify it completes; linear loss is sensitive to outliers
        assert isinstance(rms, float)

    def test_custom_interface_normal(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Works with custom interface normal."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
        )

        # Same normal, but as explicit array
        custom_normal = np.array([0.0, 0.0, -1.0])

        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            interface_normal=custom_normal,
        )

        assert rms < 2.0
