"""Tests for interface estimation (Stage 3 optimization)."""

import sys

import numpy as np
import pytest

from aquacal.calibration._optim_common import (
    build_bounds,
    build_jacobian_sparsity,
    pack_params,
    unpack_params,
)
from aquacal.calibration.interface_estimation import (
    _compute_initial_board_poses,
    _multi_frame_pnp_init,
    optimize_interface,
    register_auxiliary_camera,
)
from aquacal.config.schema import (
    BoardConfig,
    BoardPose,
    CameraExtrinsics,
    CameraIntrinsics,
    DetectionResult,
    InsufficientDataError,
)
from aquacal.core.board import BoardGeometry
from aquacal.utils.transforms import matrix_to_rvec, rvec_to_matrix

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
        "cam0": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
        "cam1": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
        "cam2": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
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
    """Ground truth interface distances.

    All cameras have C_z=0 (identity R, t with no Z component), so with a
    single water surface plane, all interface distances must be identical.
    """
    return {"cam0": 0.15, "cam1": 0.15, "cam2": 0.15}


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

        # With all cameras at C_z=0, water_z equals the interface distance
        water_z = 0.15

        packed = pack_params(
            ground_truth_extrinsics,
            water_z,
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
        water_z = 0.15

        packed = pack_params(
            ground_truth_extrinsics,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # 2 non-reference cameras * 6 params + 1 water_z + N frames * 6 params
        expected = 6 * 2 + 1 + 6 * n_frames
        assert len(packed) == expected

    def test_reference_camera_extrinsics_preserved(
        self, ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Reference camera extrinsics come from input, not packed params."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [0]
        board_poses_dict = {
            synthetic_board_poses[0].frame_idx: synthetic_board_poses[0]
        }

        # Create modified extrinsics for reference camera
        modified_ref = CameraExtrinsics(
            R=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64),
            t=np.array([1, 2, 3], dtype=np.float64),
        )

        water_z = 0.15

        packed = pack_params(
            ground_truth_extrinsics,
            water_z,
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

    def test_unpack_with_nonzero_Cz(self, synthetic_board_poses):
        """unpack_params returns water_z for all cameras regardless of their C_z."""
        # Create cameras with varying C_z (including negative values)
        extrinsics_nonzero_cz = {
            "cam0": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.0, -0.05], dtype=np.float64),  # C_z = 0.05
            ),
            "cam1": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.1, 0.0, 0.03], dtype=np.float64),  # C_z = -0.03
            ),
            "cam2": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.1, -0.02], dtype=np.float64),  # C_z = 0.02
            ),
        }

        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses[:3]]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses[:3]}

        water_z = 0.15

        packed = pack_params(
            extrinsics_nonzero_cz,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )

        ext_out, dist_out, _, _ = unpack_params(
            packed,
            reference_camera="cam0",
            reference_extrinsics=extrinsics_nonzero_cz["cam0"],
            camera_order=camera_order,
            frame_order=frame_order,
        )

        # All cameras should have water_z = water_z, regardless of C_z
        for cam in camera_order:
            assert abs(dist_out[cam] - water_z) < 1e-10, (
                f"Camera {cam} with C_z={extrinsics_nonzero_cz[cam].C[2]:.4f} "
                f"should have water_z={water_z}, got {dist_out[cam]}"
            )


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
            initial_water_zs=initial_distances,
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
            assert abs(dist_opt[cam] - ground_truth_distances[cam]) < 0.03, (
                f"Distance for {cam} off by {abs(dist_opt[cam] - ground_truth_distances[cam])}"
            )

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

    def test_default_water_zs(
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

        # Don't provide initial_water_zs
        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            initial_water_zs=None,  # Should default to 0.15
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

    def test_nonzero_cz_cameras_no_scale_bias(
        self,
        board,
        intrinsics,
        synthetic_board_poses,
    ):
        """Full optimization with non-zero C_z cameras produces correct reconstruction.

        This test verifies that the fix for B.6 (C_z double-counting) is correct.
        With cameras at different Z heights, the water surface Z should still be
        recovered correctly and reconstruction should not have systematic scale bias.
        """
        np.random.seed(43)

        # Create ground truth with cameras at varying Z heights
        ground_truth_extrinsics_nonzero = {
            "cam0": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.0, -0.05], dtype=np.float64),  # C_z = 0.05
            ),
            "cam1": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.1, 0.0, 0.03], dtype=np.float64),  # C_z = -0.03
            ),
            "cam2": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.1, -0.02], dtype=np.float64),  # C_z = 0.02
            ),
        }

        # Ground truth water surface Z (should be same for all cameras)
        ground_truth_water_z = 0.15
        ground_truth_distances_nonzero = {
            "cam0": ground_truth_water_z,
            "cam1": ground_truth_water_z,
            "cam2": ground_truth_water_z,
        }

        # Generate synthetic detections
        detections = generate_synthetic_detections(
            intrinsics,
            ground_truth_extrinsics_nonzero,
            ground_truth_distances_nonzero,
            board,
            synthetic_board_poses,
            noise_std=0.5,
            min_corners=4,
        )

        # Perturb initial guess
        initial_extrinsics = {}
        for cam, ext in ground_truth_extrinsics_nonzero.items():
            if cam == "cam0":
                initial_extrinsics[cam] = ext
            else:
                noise = np.random.randn(3) * 0.01
                initial_extrinsics[cam] = CameraExtrinsics(
                    R=ext.R.copy(),
                    t=ext.t + noise,
                )

        initial_water_z_perturbed = ground_truth_water_z + 0.02
        initial_water_zs = {cam: initial_water_z_perturbed for cam in intrinsics.keys()}

        # Run optimization
        opt_extrinsics, opt_distances, opt_board_poses, rms_error = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=initial_extrinsics,
            initial_water_zs=initial_water_zs,
            board=board,
            reference_camera="cam0",
            verbose=0,
        )

        # After the fix, all water_zs should be identical (= water_z)
        water_zs = list(opt_distances.values())
        recovered_water_z = water_zs[0]

        # Verify water_z is recovered
        assert abs(recovered_water_z - ground_truth_water_z) < 0.01, (
            f"Water Z not recovered: expected {ground_truth_water_z}, "
            f"got {recovered_water_z}"
        )

        # Verify all cameras have same water_z (= water_z)
        assert np.std(water_zs) < 1e-6, (
            f"Interface distances should be identical (all equal to water_z), "
            f"got std={np.std(water_zs)}"
        )

        # Verify reprojection error is low
        assert rms_error < 2.0, f"RMS error too high: {rms_error}"

        # Verify extrinsics are recovered reasonably
        for cam_name in ["cam0", "cam1", "cam2"]:
            gt_ext = ground_truth_extrinsics_nonzero[cam_name]
            opt_ext = opt_extrinsics[cam_name]

            # Check C_z is close to ground truth (most important for this test)
            gt_C_z = gt_ext.C[2]
            opt_C_z = opt_ext.C[2]
            assert abs(opt_C_z - gt_C_z) < 0.02, (
                f"Camera {cam_name} C_z not recovered: "
                f"expected {gt_C_z:.4f}, got {opt_C_z:.4f}"
            )


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

        # Expected params: extrinsics(6*(N-1)) + water_z(1) + board_poses(6*M)
        n_cams = len(camera_order)
        n_frames = len(frame_order)
        n_params = 6 * (n_cams - 1) + 1 + 6 * n_frames

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
        # water_z(12) = 1 water surface Z
        # frame0_pose(13-18), frame1_pose(19-24), frame2_pose(25-30) = 18 pose
        # Total = 31 params

        n_cams = len(camera_order)
        _n_frames = len(frame_order)
        n_ext = 6 * (n_cams - 1)  # 12
        water_z_col = n_ext  # 12

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

                    # Check water_z dependency: single column, always 1
                    assert row_x[water_z_col] == 1

                    # Check pose dependency: exactly 6 pose params
                    pose_start = water_z_col + 1 + pose_idx * 6
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


class TestTiltEstimation:
    """Tests for normal_fixed=False (reference camera tilt estimation)."""

    def test_pack_unpack_roundtrip_with_tilt(
        self, ground_truth_extrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Pack and unpack with normal_fixed=False roundtrips correctly."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses[:3]]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses[:3]}

        # Give reference camera a non-identity rotation (small tilt)
        tilt_rvec = np.array([0.05, -0.03, 0.0])
        R_tilt = rvec_to_matrix(tilt_rvec)
        tilted_extrinsics = dict(ground_truth_extrinsics)
        tilted_extrinsics["cam0"] = CameraExtrinsics(
            R=R_tilt, t=np.zeros(3, dtype=np.float64)
        )

        water_z = 0.15

        packed = pack_params(
            tilted_extrinsics,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
            normal_fixed=False,
        )

        # Should have 2 extra params compared to normal_fixed=True
        packed_fixed = pack_params(
            tilted_extrinsics,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
            normal_fixed=True,
        )
        assert len(packed) == len(packed_fixed) + 2

        # First 2 params should be the tilt rx, ry
        np.testing.assert_allclose(packed[0], tilt_rvec[0], atol=1e-10)
        np.testing.assert_allclose(packed[1], tilt_rvec[1], atol=1e-10)

        # Roundtrip unpack
        ext_out, dist_out, poses_out, _ = unpack_params(
            packed,
            reference_camera="cam0",
            reference_extrinsics=ground_truth_extrinsics["cam0"],  # ignored
            camera_order=camera_order,
            frame_order=frame_order,
            normal_fixed=False,
        )

        # Reference camera should have the tilt rotation (rz forced to 0)
        ref_rvec_out = matrix_to_rvec(ext_out["cam0"].R)
        np.testing.assert_allclose(ref_rvec_out[0], tilt_rvec[0], atol=1e-6)
        np.testing.assert_allclose(ref_rvec_out[1], tilt_rvec[1], atol=1e-6)

        # Reference camera t should be zeros
        np.testing.assert_allclose(ext_out["cam0"].t, np.zeros(3), atol=1e-10)

        # Non-reference cameras should roundtrip exactly
        for cam in ["cam1", "cam2"]:
            np.testing.assert_allclose(
                ext_out[cam].R, tilted_extrinsics[cam].R, atol=1e-10
            )
            np.testing.assert_allclose(
                ext_out[cam].t, tilted_extrinsics[cam].t, atol=1e-10
            )

        # Distances should roundtrip
        for cam in camera_order:
            assert abs(dist_out[cam] - ground_truth_distances[cam]) < 1e-10

    def test_sparsity_pattern_with_tilt(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Sparsity with normal_fixed=False has 2 extra columns, correctly marked."""
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

        sparsity_fixed = build_jacobian_sparsity(
            detections,
            "cam0",
            camera_order,
            frame_order,
            min_corners,
            normal_fixed=True,
        )
        sparsity_tilt = build_jacobian_sparsity(
            detections,
            "cam0",
            camera_order,
            frame_order,
            min_corners,
            normal_fixed=False,
        )

        # Same number of residual rows, 2 extra columns
        assert sparsity_tilt.shape[0] == sparsity_fixed.shape[0]
        assert sparsity_tilt.shape[1] == sparsity_fixed.shape[1] + 2

        # Check that reference camera residuals depend on tilt params
        # and non-reference residuals don't
        n_tilt = 2
        n_ext = 6 * (len(camera_order) - 1)
        residual_idx = 0
        for frame_idx in frame_order:
            if frame_idx not in detections.frames:
                continue
            frame_det = detections.frames[frame_idx]
            for cam_name in camera_order:
                if cam_name not in frame_det.detections:
                    continue
                det = frame_det.detections[cam_name]
                if det.num_corners < min_corners:
                    continue
                for _ in range(det.num_corners):
                    row = sparsity_tilt[residual_idx]
                    if cam_name == "cam0":
                        # Reference camera: tilt cols should be 1
                        assert row[0] == 1 and row[1] == 1
                        # No extrinsic params for reference
                        assert np.sum(row[n_tilt : n_tilt + n_ext]) == 0
                    else:
                        # Non-reference: tilt cols should be 0
                        assert row[0] == 0 and row[1] == 0
                    residual_idx += 2

    def test_bounds_with_tilt(self, synthetic_board_poses):
        """Bounds with normal_fixed=False have 2 extra elements with tilt bounds."""
        camera_order = ["cam0", "cam1", "cam2"]
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]

        lower_fixed, upper_fixed = build_bounds(
            camera_order,
            frame_order,
            "cam0",
            normal_fixed=True,
        )
        lower_tilt, upper_tilt = build_bounds(
            camera_order,
            frame_order,
            "cam0",
            normal_fixed=False,
        )

        # 2 extra elements
        assert len(lower_tilt) == len(lower_fixed) + 2
        assert len(upper_tilt) == len(upper_fixed) + 2

        # First 2 bounds are tilt bounds [-0.2, 0.2]
        np.testing.assert_allclose(lower_tilt[0:2], -0.2)
        np.testing.assert_allclose(upper_tilt[0:2], 0.2)

        # Remaining bounds should match (shifted by 2)
        np.testing.assert_allclose(lower_tilt[2:], lower_fixed)
        np.testing.assert_allclose(upper_tilt[2:], upper_fixed)

    def test_optimize_interface_with_tilt(
        self,
        board,
        intrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """Optimizer recovers a known small tilt when normal_fixed=False."""
        np.random.seed(42)

        # Create ground truth with a tilted reference camera (~3 degrees)
        tilt_rvec = np.array([0.03, -0.02, 0.0])  # ~3 deg combined tilt
        R_tilt = rvec_to_matrix(tilt_rvec)
        gt_extrinsics = {
            "cam0": CameraExtrinsics(
                R=R_tilt,
                t=np.zeros(3, dtype=np.float64),
            ),
            "cam1": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.1, 0.0, 0.0], dtype=np.float64),
            ),
            "cam2": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.1, 0.0], dtype=np.float64),
            ),
        }

        detections = generate_synthetic_detections(
            intrinsics,
            gt_extrinsics,
            ground_truth_distances,
            board,
            synthetic_board_poses,
            noise_std=0.3,
            min_corners=4,
        )

        # Initialize without tilt (as Stage 2 would)
        initial_extrinsics = {
            "cam0": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.zeros(3, dtype=np.float64),
            ),
            "cam1": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.1, 0.0, 0.0], dtype=np.float64)
                + np.random.normal(0, 0.01, 3),
            ),
            "cam2": CameraExtrinsics(
                R=np.eye(3, dtype=np.float64),
                t=np.array([0.0, 0.1, 0.0], dtype=np.float64)
                + np.random.normal(0, 0.01, 3),
            ),
        }

        ext_opt, dist_opt, poses_opt, rms = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=initial_extrinsics,
            board=board,
            reference_camera="cam0",
            initial_water_zs=ground_truth_distances,
            normal_fixed=False,
        )

        # Should converge
        assert rms < 2.0, f"RMS too high: {rms}"

        # Recovered tilt should be close to ground truth
        recovered_rvec = matrix_to_rvec(ext_opt["cam0"].R)
        np.testing.assert_allclose(recovered_rvec[:2], tilt_rvec[:2], atol=0.02)

        # Distances should be reasonable
        for cam in intrinsics:
            assert abs(dist_opt[cam] - ground_truth_distances[cam]) < 0.05

    def test_normal_fixed_true_unchanged(
        self,
        board,
        intrinsics,
        ground_truth_extrinsics,
        ground_truth_distances,
        synthetic_board_poses,
    ):
        """normal_fixed=True produces identical results to omitting the parameter."""
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

        camera_order = sorted(intrinsics.keys())
        frame_order = [bp.frame_idx for bp in synthetic_board_poses]
        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}

        water_z = 0.15

        # Pack params: explicit normal_fixed=True should give same as default
        packed_default = pack_params(
            ground_truth_extrinsics,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
        )
        packed_explicit = pack_params(
            ground_truth_extrinsics,
            water_z,
            board_poses_dict,
            reference_camera="cam0",
            camera_order=camera_order,
            frame_order=frame_order,
            normal_fixed=True,
        )
        np.testing.assert_array_equal(packed_default, packed_explicit)

        # Sparsity: same shape and values
        sparsity_default = build_jacobian_sparsity(
            detections,
            "cam0",
            camera_order,
            frame_order,
            4,
        )
        sparsity_explicit = build_jacobian_sparsity(
            detections,
            "cam0",
            camera_order,
            frame_order,
            4,
            normal_fixed=True,
        )
        np.testing.assert_array_equal(sparsity_default, sparsity_explicit)

        # Bounds: same length and values
        lo_default, hi_default = build_bounds(
            camera_order,
            frame_order,
            "cam0",
        )
        lo_explicit, hi_explicit = build_bounds(
            camera_order,
            frame_order,
            "cam0",
            normal_fixed=True,
        )
        np.testing.assert_array_equal(lo_default, lo_explicit)
        np.testing.assert_array_equal(hi_default, hi_explicit)

        # Optimize: same result
        np.random.seed(42)
        _, dist_default, _, rms_default = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
        )
        np.random.seed(42)
        _, dist_explicit, _, rms_explicit = optimize_interface(
            detections=detections,
            intrinsics=intrinsics,
            initial_extrinsics=ground_truth_extrinsics,
            board=board,
            reference_camera="cam0",
            normal_fixed=True,
        )
        assert abs(rms_default - rms_explicit) < 1e-10
        for cam in camera_order:
            assert abs(dist_default[cam] - dist_explicit[cam]) < 1e-10


class TestRegisterAuxiliaryCamera:
    """Tests for auxiliary camera registration with multi-frame PnP initialization."""

    def test_multi_frame_init_robust_to_outlier(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Multi-frame PnP initialization filters out outlier frames."""
        np.random.seed(42)

        # Create a near-overhead auxiliary camera (small tilt)
        aux_extrinsics = CameraExtrinsics(
            R=rvec_to_matrix(np.array([0.05, 0.03, 0.0])),  # ~3 deg tilt
            t=np.array([0.2, 0.0, 0.0], dtype=np.float64),  # C_z = 0
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        # Generate detections for this aux camera
        # Use more frames to test multi-frame averaging (10 frames)
        multi_frame_poses = []
        for i in range(10):
            x_offset = 0.05 * (i % 3 - 1)
            y_offset = 0.02 * (i % 2)
            multi_frame_poses.append(
                BoardPose(
                    frame_idx=i,
                    rvec=np.array(
                        [0.1 * (i % 3), 0.1 * (i % 2), 0.0], dtype=np.float64
                    ),
                    tvec=np.array([x_offset, y_offset, 0.4], dtype=np.float64),
                )
            )

        # Generate clean detections for most frames
        # Use min_corners=6 since cv2.solvePnP ITERATIVE requires at least 6 points
        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            multi_frame_poses,
            noise_std=0.3,
            min_corners=6,
        )

        # Manually corrupt one frame with large noise to simulate PnP failure
        # We'll add extra noise to one frame's corners
        frame_to_corrupt = 3
        if frame_to_corrupt in detections.frames:
            corrupted_corners = (
                detections.frames[frame_to_corrupt].detections[aux_name].corners_2d
                + np.random.randn(
                    detections.frames[frame_to_corrupt]
                    .detections[aux_name]
                    .num_corners,
                    2,
                )
                * 10.0
            )
            detections.frames[frame_to_corrupt].detections[
                aux_name
            ].corners_2d = corrupted_corners

        # Create board_poses dict from list
        board_poses_dict = {bp.frame_idx: bp for bp in multi_frame_poses}

        # Run register_auxiliary_camera
        water_z = ground_truth_distances["cam0"]
        opt_extrinsics, opt_iface_dist, rms_error = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            verbose=0,
        )

        # Should achieve reasonable RMS despite the outlier
        # The outlier frame still participates in optimization after init, so RMS may be higher
        assert rms_error < 5.0, f"RMS error too high: {rms_error}"

        # C_z should be close to ground truth (0.0)
        recovered_C_z = opt_extrinsics.C[2]
        assert abs(recovered_C_z) < 0.1, (
            f"C_z should be near 0, got {recovered_C_z:.4f}. "
            "Multi-frame PnP should filter outlier frames."
        )

        # Interface distance should equal water_z
        assert abs(opt_iface_dist - water_z) < 1e-6

    def test_single_frame_fallback(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Falls back gracefully when only one frame is available."""
        np.random.seed(42)

        # Create auxiliary camera
        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        # Generate detections for only one frame
        single_frame_pose = [synthetic_board_poses[0]]

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            single_frame_pose,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {single_frame_pose[0].frame_idx: single_frame_pose[0]}
        water_z = ground_truth_distances["cam0"]

        # Should not raise, even with single frame
        opt_extrinsics, opt_iface_dist, rms_error = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            verbose=0,
        )

        # Should produce reasonable result
        assert isinstance(opt_extrinsics, CameraExtrinsics)
        assert isinstance(rms_error, float)
        assert rms_error < 5.0

    def test_produces_correct_C_z(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Recovered C_z is within tolerance of ground truth."""
        np.random.seed(42)

        # Create auxiliary camera at known position
        ground_truth_C_z = (
            0.02  # 2cm above reference (smaller offset for better visibility)
        )
        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array(
                [0.15, 0.0, -ground_truth_C_z], dtype=np.float64
            ),  # Closer in X too
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        # Use existing synthetic board poses which are known to work
        # Just use the first 3 frames from fixture
        multi_frame_poses = synthetic_board_poses

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            multi_frame_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in multi_frame_poses}
        water_z = ground_truth_distances["cam0"]

        opt_extrinsics, opt_iface_dist, rms_error = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            verbose=0,
        )

        # Check C_z is close to ground truth
        recovered_C_z = opt_extrinsics.C[2]
        assert abs(recovered_C_z - ground_truth_C_z) < 0.05, (
            f"C_z should be near {ground_truth_C_z:.4f}, got {recovered_C_z:.4f}"
        )

        # Check interface distance equals water_z
        assert abs(opt_iface_dist - water_z) < 1e-6

        # Check RMS is reasonable
        assert rms_error < 2.0

    def test_raises_for_no_frames(self, board, intrinsics):
        """Raises InsufficientDataError when no usable frames."""
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        # Empty detections
        detections = DetectionResult(
            frames={},
            camera_names=[aux_name],
            total_frames=0,
        )

        board_poses_dict = {}
        water_z = 0.15

        with pytest.raises(InsufficientDataError, match="No usable frames"):
            register_auxiliary_camera(
                camera_name=aux_name,
                intrinsics=aux_intrinsics,
                detections=detections,
                board_poses=board_poses_dict,
                board=board,
                water_z=water_z,
            )

    def test_multi_frame_pnp_init_averages_multiple_frames(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """_multi_frame_pnp_init averages multiple frames correctly."""
        np.random.seed(42)

        # Create auxiliary camera
        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        # Generate detections with 5 frames
        multi_frame_poses = synthetic_board_poses[:5]

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            multi_frame_poses,
            noise_std=0.5,
            min_corners=6,
        )

        # Collect obs_frames in the same format as register_auxiliary_camera
        obs_frames = []
        for frame_idx, frame_det in detections.frames.items():
            if aux_name in frame_det.detections:
                det = frame_det.detections[aux_name]
                obs_frames.append((frame_idx, det.corner_ids, det.corners_2d))

        board_poses_dict = {bp.frame_idx: bp for bp in multi_frame_poses}
        water_z = ground_truth_distances["cam0"]
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Call the helper directly
        result = _multi_frame_pnp_init(
            obs_frames=obs_frames,
            board_poses=board_poses_dict,
            intrinsics=aux_intrinsics,
            board=board,
            water_z=water_z,
            interface_normal=interface_normal,
            n_air=1.0,
            n_water=1.333,
        )

        assert result is not None
        rvec, tvec = result

        # Result should be reasonable
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)

        # C_z should be near ground truth (0 in this case)
        R = rvec_to_matrix(rvec)
        C = -R.T @ tvec
        C_z = C[2]
        assert abs(C_z) < 0.1, f"C_z should be near 0, got {C_z:.4f}"

    def test_register_auxiliary_refine_intrinsics_improves_rms(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Refining intrinsics improves RMS when initial intrinsics are perturbed."""
        np.random.seed(42)

        # Create auxiliary camera
        ground_truth_C_z = 0.02
        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, -ground_truth_C_z], dtype=np.float64),
        )

        # Use perturbed intrinsics (fx off by 5%)
        true_intrinsics = intrinsics["cam0"]
        perturbed_K = true_intrinsics.K.copy()
        perturbed_K[0, 0] *= 1.05  # fx 5% too high
        perturbed_intrinsics = CameraIntrinsics(
            K=perturbed_K,
            dist_coeffs=true_intrinsics.dist_coeffs,
            image_size=true_intrinsics.image_size,
            is_fisheye=true_intrinsics.is_fisheye,
        )

        aux_name = "aux_cam"
        multi_frame_poses = synthetic_board_poses

        # Generate detections with true intrinsics but optimize with perturbed
        detections = generate_synthetic_detections(
            {aux_name: true_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            multi_frame_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in multi_frame_poses}
        water_z = ground_truth_distances["cam0"]

        # Run without intrinsic refinement (should have higher RMS)
        _, _, rms_no_refine = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=perturbed_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=False,
            verbose=0,
        )

        # Run with intrinsic refinement (should improve RMS and fx)
        _, _, rms_with_refine, refined_intr = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=perturbed_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=True,
            verbose=0,
        )

        # RMS should improve
        assert rms_with_refine < rms_no_refine, (
            f"Refinement should improve RMS: {rms_with_refine:.2f} vs {rms_no_refine:.2f}"
        )

        # Refined fx should be closer to true value
        refined_fx = refined_intr.K[0, 0]
        true_fx = true_intrinsics.K[0, 0]
        perturbed_fx = perturbed_intrinsics.K[0, 0]

        error_before = abs(perturbed_fx - true_fx)
        error_after = abs(refined_fx - true_fx)

        assert error_after < error_before, (
            f"Refinement should move fx closer to true value. "
            f"Before: {perturbed_fx:.1f}, After: {refined_fx:.1f}, True: {true_fx:.1f}"
        )

    def test_register_auxiliary_refine_intrinsics_returns_4_tuple(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Returns 4-tuple with CameraIntrinsics when refine_intrinsics=True."""
        np.random.seed(42)

        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            synthetic_board_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}
        water_z = ground_truth_distances["cam0"]

        result = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=True,
            verbose=0,
        )

        # Should return 4-tuple
        assert len(result) == 4
        ext, dist, rms, refined_intr = result

        # Check types
        assert isinstance(ext, CameraExtrinsics)
        assert isinstance(dist, float)
        assert isinstance(rms, float)
        assert isinstance(refined_intr, CameraIntrinsics)

    def test_register_auxiliary_refine_intrinsics_preserves_distortion(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Refined intrinsics preserve distortion coefficients and is_fisheye flag."""
        np.random.seed(42)

        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )

        # Create intrinsics with non-zero distortion and fisheye flag set
        # Fisheye cameras need exactly 4 distortion coefficients
        K = intrinsics["cam0"].K.copy()
        dist = np.array([0.1, -0.05, 0.001, 0.002], dtype=np.float64)
        aux_intrinsics = CameraIntrinsics(
            K=K,
            dist_coeffs=dist,
            image_size=(640, 480),
            is_fisheye=True,
        )
        aux_name = "aux_cam"

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            synthetic_board_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}
        water_z = ground_truth_distances["cam0"]

        _, _, _, refined_intr = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=True,
            verbose=0,
        )

        # Distortion coefficients should be unchanged
        np.testing.assert_array_equal(
            refined_intr.dist_coeffs,
            aux_intrinsics.dist_coeffs,
            err_msg="Distortion coefficients should not be modified",
        )

        # is_fisheye flag should be preserved
        assert refined_intr.is_fisheye == aux_intrinsics.is_fisheye

        # image_size should be preserved
        assert refined_intr.image_size == aux_intrinsics.image_size

    def test_register_auxiliary_no_refine_returns_3_tuple(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Returns 3-tuple when refine_intrinsics=False (backward compatibility)."""
        np.random.seed(42)

        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )
        aux_intrinsics = intrinsics["cam0"]
        aux_name = "aux_cam"

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            synthetic_board_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}
        water_z = ground_truth_distances["cam0"]

        result = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=False,
            verbose=0,
        )

        # Should return 3-tuple
        assert len(result) == 3
        ext, dist, rms = result

        # Check types
        assert isinstance(ext, CameraExtrinsics)
        assert isinstance(dist, float)
        assert isinstance(rms, float)

    def test_register_auxiliary_refine_intrinsics_bounds(
        self, board, intrinsics, ground_truth_distances, synthetic_board_poses
    ):
        """Intrinsic bounds are enforced: fx/fy in [0.5x, 2.0x], cx/cy in [0, w/h]."""
        np.random.seed(42)

        aux_extrinsics = CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.15, 0.0, 0.0], dtype=np.float64),
        )

        # Create intrinsics with extreme initial fx (to test bounds)
        K = intrinsics["cam0"].K.copy()
        initial_fx = 100.0  # Very low fx
        K[0, 0] = initial_fx
        aux_intrinsics = CameraIntrinsics(
            K=K,
            dist_coeffs=np.zeros(5, dtype=np.float64),
            image_size=(640, 480),
            is_fisheye=False,
        )
        aux_name = "aux_cam"

        detections = generate_synthetic_detections(
            {aux_name: aux_intrinsics},
            {aux_name: aux_extrinsics},
            {aux_name: ground_truth_distances["cam0"]},
            board,
            synthetic_board_poses,
            noise_std=0.3,
            min_corners=6,
        )

        board_poses_dict = {bp.frame_idx: bp for bp in synthetic_board_poses}
        water_z = ground_truth_distances["cam0"]

        _, _, _, refined_intr = register_auxiliary_camera(
            camera_name=aux_name,
            intrinsics=aux_intrinsics,
            detections=detections,
            board_poses=board_poses_dict,
            board=board,
            water_z=water_z,
            refine_intrinsics=True,
            verbose=0,
        )

        # Check fx is within bounds [0.5*initial_fx, 2.0*initial_fx]
        refined_fx = refined_intr.K[0, 0]
        assert 0.5 * initial_fx <= refined_fx <= 2.0 * initial_fx, (
            f"fx should be in [0.5*{initial_fx}, 2.0*{initial_fx}], got {refined_fx}"
        )

        # Check cx, cy are within image bounds
        refined_cx = refined_intr.K[0, 2]
        refined_cy = refined_intr.K[1, 2]
        w, h = aux_intrinsics.image_size

        assert 0.0 <= refined_cx <= float(w), (
            f"cx should be in [0, {w}], got {refined_cx}"
        )
        assert 0.0 <= refined_cy <= float(h), (
            f"cy should be in [0, {h}], got {refined_cy}"
        )
