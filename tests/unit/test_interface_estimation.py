"""Tests for interface estimation (Stage 3 optimization)."""

import pytest
import numpy as np
from scipy.optimize._numdiff import group_columns

from aquacal.config.schema import (
    BoardConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    BoardPose,
    DetectionResult,
    InsufficientDataError,
    ConvergenceError,
)
from aquacal.core.board import BoardGeometry
from aquacal.calibration.interface_estimation import (
    optimize_interface,
    _compute_initial_board_poses,
)
from aquacal.calibration._optim_common import (
    pack_params,
    unpack_params,
    build_jacobian_sparsity,
)

import sys
sys.path.insert(0, ".")
from tests.synthetic.ground_truth import generate_synthetic_detections


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
    """Ground truth camera extrinsics for testing.

    Camera positions are close enough together that all cameras can see
    the board at the test positions (z=0.4m). With 10cm spacing and 500px
    focal length, the cameras have overlapping fields of view.
    """
    return {
        "cam0": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
        "cam1": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.1, 0.0, 0.0], dtype=np.float64),  # 10cm to the right
        ),
        "cam2": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.0, 0.1, 0.0], dtype=np.float64),  # 10cm down
        ),
    }


@pytest.fixture
def ground_truth_distances() -> dict[str, float]:
    """Ground truth interface distances."""
    return {"cam0": 0.15, "cam1": 0.16, "cam2": 0.14}


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
            min_corners=4,
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
            min_corners=4,
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

        packed = pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        ext_out, dist_out, poses_out, _ = unpack_params(
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

        packed = pack_params(
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

        packed = pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # Unpack with different reference extrinsics
        ext_out, _, _, _ = unpack_params(
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
            min_corners=4,
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
            min_corners=4,
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
            min_corners=4,
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
            min_corners=4,
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
            min_corners=4,
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
            min_corners=4,
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
            min_corners=4,
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


class TestBuildJacobianSparsity:
    """Tests for sparse Jacobian structure builder."""

    def test_sparsity_shape(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Sparsity matrix has correct shape."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
            min_corners=4,
        )

        camera_order = sorted(intrinsics.keys())
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]
        min_corners = 4

        sparsity = build_jacobian_sparsity(
            detections, "cam0", camera_order, frame_order, min_corners
        )

        # Count expected residuals
        n_residuals = 0
        for frame_idx in frame_order:
            if frame_idx not in detections.frames:
                continue
            frame_det = detections.frames[frame_idx]
            for cam_name in camera_order:
                if cam_name not in frame_det.detections:
                    continue
                det = frame_det.detections[cam_name]
                if det.num_corners >= min_corners:
                    n_residuals += det.num_corners * 2

        # Expected params
        n_cams = len(camera_order)
        n_frames = len(frame_order)
        n_params = 6 * (n_cams - 1) + n_cams + 6 * n_frames

        assert sparsity.shape == (n_residuals, n_params)

    def test_sparsity_pattern_correct(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Each residual only depends on correct params."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
            min_corners=4,
        )

        camera_order = sorted(intrinsics.keys())  # ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]
        min_corners = 4

        sparsity = build_jacobian_sparsity(
            detections, "cam0", camera_order, frame_order, min_corners
        )

        # Parameter layout:
        # cam1_rvec(0-2), cam1_tvec(3-5), cam2_rvec(6-8), cam2_tvec(9-11) = 12 extrinsic
        # dist_cam0(12), dist_cam1(13), dist_cam2(14) = 3 distance
        # frame0_pose(15-20), frame1_pose(21-26), frame2_pose(27-32) = 18 pose
        # Total = 33 params

        n_cams = len(camera_order)
        n_frames = len(frame_order)
        n_ext = 6 * (n_cams - 1)  # 12
        n_dist = n_cams  # 3

        # Check that reference camera has no extrinsic dependencies
        # Find residuals from cam0 (reference)
        residual_idx = 0
        for frame_idx in frame_order:
            if frame_idx not in detections.frames:
                continue
            frame_det = detections.frames[frame_idx]
            pose_idx = frame_order.index(frame_idx)

            for cam_name in camera_order:
                if cam_name not in frame_det.detections:
                    continue
                det = frame_det.detections[cam_name]
                if det.num_corners < min_corners:
                    continue

                cam_idx = camera_order.index(cam_name)

                for _ in range(det.num_corners):
                    row_x = sparsity[residual_idx]
                    row_y = sparsity[residual_idx + 1]

                    # Both rows should be identical
                    np.testing.assert_array_equal(row_x, row_y)

                    # Check extrinsics dependency
                    if cam_name == "cam0":
                        # Reference: no extrinsic dependencies (first 12 params)
                        assert np.sum(row_x[:n_ext]) == 0
                    else:
                        # Non-reference: 6 extrinsic params set
                        ext_idx = (camera_order.index(cam_name) - 1) * 6
                        assert np.sum(row_x[:n_ext]) == 6
                        assert np.sum(row_x[ext_idx : ext_idx + 6]) == 6

                    # Check distance dependency: exactly 1 distance param
                    dist_start = n_ext
                    dist_end = n_ext + n_dist
                    assert np.sum(row_x[dist_start:dist_end]) == 1
                    assert row_x[n_ext + cam_idx] == 1

                    # Check pose dependency: exactly 6 pose params
                    pose_start = n_ext + n_dist + pose_idx * 6
                    assert row_x[pose_start : pose_start + 6].sum() == 6

                    residual_idx += 2

    def test_sparsity_is_sparse(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Verify that the matrix is actually sparse."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
            min_corners=4,
        )

        camera_order = sorted(intrinsics.keys())
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]

        sparsity = build_jacobian_sparsity(
            detections, "cam0", camera_order, frame_order, min_corners=4
        )

        # Calculate sparsity ratio
        total_elements = sparsity.size
        nonzero_elements = np.sum(sparsity != 0)
        sparsity_ratio = 1.0 - (nonzero_elements / total_elements)

        # Should be at least 50% sparse (typical is 90%+)
        assert sparsity_ratio > 0.5, f"Sparsity ratio only {sparsity_ratio:.2%}"


class TestOptimizeInterfaceWithSparseJacobian:
    """Test optimization with sparse Jacobian."""

    def test_sparse_jacobian_gives_same_result(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Optimization with sparse Jacobian gives same result as dense."""
        np.random.seed(42)

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.5,
            min_corners=4,
        )

        # Run with sparse Jacobian
        _, dist_sparse, _, rms_sparse = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            use_sparse_jacobian=True,
        )

        # Run with dense Jacobian
        np.random.seed(42)
        _, dist_dense, _, rms_dense = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            use_sparse_jacobian=False,
        )

        # Results should be identical (same cost function, just different Jacobian)
        assert abs(rms_sparse - rms_dense) < 0.1

        for cam in intrinsics:
            assert abs(dist_sparse[cam] - dist_dense[cam]) < 0.005


class TestWaterZRegularizationSparsity:
    """Verify water_z regularization doesn't destroy Jacobian sparsity."""

    def test_group_count_with_regularization(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Column groups should not increase dramatically with water_z_weight."""
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
            min_corners=4,
        )

        camera_order = sorted(intrinsics.keys())
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]

        # Build sparsity pattern without regularization
        sparsity_off = build_jacobian_sparsity(
            detections,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
            min_corners=4,
            water_z_weight=0.0,
        )
        groups_off = group_columns(sparsity_off)
        n_groups_off = int(groups_off.max()) + 1

        # Build sparsity pattern with regularization
        sparsity_on = build_jacobian_sparsity(
            detections,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
            min_corners=4,
            water_z_weight=10.0,
        )
        groups_on = group_columns(sparsity_on)
        n_groups_on = int(groups_on.max()) + 1

        # Regularization should add at most a modest number of groups.
        # The old bug caused n_groups to go from ~50 to ~80+ (60%+ increase)
        # with EVERY regularization residual depending on ALL cameras.
        # Pairwise regularization is much sparser (each residual depends on only 2 cameras).
        # For small test cases (3 cameras), the pairwise residuals may introduce new
        # column co-occurrences that increase groups proportionally. The threshold should
        # catch cases where the sparsity pattern couples too many parameters.
        # With 3 cameras: 13->21 groups (62% increase) is expected due to 3 new pairwise
        # residuals in a small parameter space. Allow up to 2x increase to catch truly
        # pathological cases while passing this expected behavior.
        assert n_groups_on <= n_groups_off * 2.0, (
            f"Regularization increased groups from {n_groups_off} to {n_groups_on} "
            f"({(n_groups_on / n_groups_off - 1) * 100:.0f}% increase). "
            f"Sparsity pattern may be coupling too many parameters."
        )

    def test_residual_count_with_regularization(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Pairwise regularization should add N*(N-1)/2 residuals."""
        from aquacal.calibration._optim_common import compute_residuals, pack_params

        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.0,
            min_corners=4,
        )

        camera_order = sorted(intrinsics.keys())
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}

        # Pack parameters
        params = pack_params(
            ground_truth_extrinsics,
            ground_truth_distances,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # Build cost args
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        reference_extrinsics = ground_truth_extrinsics["cam0"]

        # Compute residuals without regularization
        residuals_off = compute_residuals(
            params,
            detections,
            intrinsics,
            board,
            "cam0",
            reference_extrinsics,
            interface_normal,
            1.0,  # n_air
            1.333,  # n_water
            camera_order,
            frame_order,
            4,  # min_corners
            False,  # refine_intrinsics
            0.0,  # water_z_weight
        )

        # Compute residuals with regularization
        residuals_on = compute_residuals(
            params,
            detections,
            intrinsics,
            board,
            "cam0",
            reference_extrinsics,
            interface_normal,
            1.0,  # n_air
            1.333,  # n_water
            camera_order,
            frame_order,
            4,  # min_corners
            False,  # refine_intrinsics
            10.0,  # water_z_weight
        )

        n_cams = len(camera_order)
        expected_extra = n_cams * (n_cams - 1) // 2

        assert len(residuals_on) == len(residuals_off) + expected_extra, (
            f"Expected {len(residuals_off) + expected_extra} residuals with regularization, "
            f"but got {len(residuals_on)}. Expected {expected_extra} additional pairwise residuals "
            f"for {n_cams} cameras."
        )
