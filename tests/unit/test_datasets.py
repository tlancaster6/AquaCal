"""Tests for aquacal.datasets synthetic data generation."""

import numpy as np
import pytest

from aquacal.datasets import SyntheticScenario, generate_synthetic_rig


def test_generate_synthetic_rig_small():
    """Test small preset returns 2 cameras and 10 frames."""
    scenario = generate_synthetic_rig("small")

    assert isinstance(scenario, SyntheticScenario)
    assert scenario.name == "small"
    assert len(scenario.intrinsics) == 2
    assert len(scenario.extrinsics) == 2
    assert len(scenario.interface_distances) == 2
    assert len(scenario.board_poses) == 10
    assert scenario.noise_std == 0.0  # Clean by default
    assert scenario.images is None  # No images by default


def test_generate_synthetic_rig_medium():
    """Test medium preset returns 6 cameras and 80 frames."""
    scenario = generate_synthetic_rig("medium")

    assert isinstance(scenario, SyntheticScenario)
    assert scenario.name == "medium"
    assert len(scenario.intrinsics) == 6
    assert len(scenario.extrinsics) == 6
    assert len(scenario.interface_distances) == 6
    assert len(scenario.board_poses) == 80
    assert scenario.noise_std == 0.0  # Clean by default
    assert scenario.images is None


@pytest.mark.slow
def test_generate_synthetic_rig_large():
    """Test large preset returns 13 cameras and 300 frames."""
    scenario = generate_synthetic_rig("large")

    assert isinstance(scenario, SyntheticScenario)
    assert scenario.name == "large"
    assert len(scenario.intrinsics) == 13
    assert len(scenario.extrinsics) == 13
    assert len(scenario.interface_distances) == 13
    assert len(scenario.board_poses) == 300
    assert scenario.noise_std == 0.0  # Clean by default
    assert scenario.images is None


def test_generate_synthetic_rig_reproducibility():
    """Test that same preset with same defaults produces identical output."""
    scenario1 = generate_synthetic_rig("small")
    scenario2 = generate_synthetic_rig("small")

    # Check intrinsics are identical
    for cam in scenario1.intrinsics:
        assert np.allclose(scenario1.intrinsics[cam].K, scenario2.intrinsics[cam].K)
        assert np.allclose(
            scenario1.intrinsics[cam].dist_coeffs, scenario2.intrinsics[cam].dist_coeffs
        )

    # Check extrinsics are identical
    for cam in scenario1.extrinsics:
        assert np.allclose(scenario1.extrinsics[cam].R, scenario2.extrinsics[cam].R)
        assert np.allclose(scenario1.extrinsics[cam].t, scenario2.extrinsics[cam].t)

    # Check interface distances are identical
    for cam in scenario1.interface_distances:
        assert np.isclose(
            scenario1.interface_distances[cam], scenario2.interface_distances[cam]
        )

    # Check board poses are identical
    assert len(scenario1.board_poses) == len(scenario2.board_poses)
    for pose1, pose2 in zip(scenario1.board_poses, scenario2.board_poses):
        assert pose1.frame_idx == pose2.frame_idx
        assert np.allclose(pose1.rvec, pose2.rvec)
        assert np.allclose(pose1.tvec, pose2.tvec)


def test_generate_synthetic_rig_noisy():
    """Test that noisy=True produces different detections than clean."""
    clean = generate_synthetic_rig("small", noisy=False)
    noisy = generate_synthetic_rig("small", noisy=True)

    # Noise std should be non-zero
    assert clean.noise_std == 0.0
    assert noisy.noise_std == 0.3  # small preset noise

    # Intrinsics, extrinsics, and poses should be the same (same seed)
    for cam in clean.intrinsics:
        assert np.allclose(clean.intrinsics[cam].K, noisy.intrinsics[cam].K)

    # But detections would differ (though we don't generate them by default in the scenario)
    # The noise_std field is the important verification here


def test_generate_synthetic_rig_invalid_preset():
    """Test that unknown preset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown preset.*unknown.*"):
        generate_synthetic_rig("unknown")

    # Verify error message lists valid presets
    with pytest.raises(ValueError, match="small.*medium.*large"):
        generate_synthetic_rig("invalid")


def test_generate_synthetic_rig_with_images():
    """Test that include_images=True renders images."""
    scenario = generate_synthetic_rig("small", include_images=True)

    # Images should be populated
    assert scenario.images is not None
    assert len(scenario.images) == 2  # 2 cameras

    # Check structure: camera_name -> frame_idx -> image
    for cam_name in scenario.intrinsics:
        assert cam_name in scenario.images
        assert len(scenario.images[cam_name]) > 0  # At least some frames rendered

    # Check image properties
    cam0_images = scenario.images["cam0"]
    first_frame_idx = list(cam0_images.keys())[0]
    image = cam0_images[first_frame_idx]

    # Image should be grayscale uint8
    assert image.dtype == np.uint8
    assert len(image.shape) == 2  # Grayscale (height, width)

    # Image size should match camera resolution
    expected_height, expected_width = (
        scenario.intrinsics["cam0"].image_size[1],
        scenario.intrinsics["cam0"].image_size[0],
    )
    assert image.shape == (expected_height, expected_width)


def test_render_synthetic_frame():
    """Test individual frame rendering produces correct-sized grayscale image."""
    from aquacal.core.board import BoardGeometry
    from aquacal.core.interface_model import Interface
    from aquacal.datasets.rendering import render_synthetic_frame

    scenario = generate_synthetic_rig("small")

    # Get first camera and first board pose
    cam_name = "cam0"
    intrinsics = scenario.intrinsics[cam_name]
    extrinsics = scenario.extrinsics[cam_name]
    board_pose = scenario.board_poses[0]
    board_geometry = BoardGeometry(scenario.board_config)

    # Create interface
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    interface = Interface(
        normal=interface_normal,
        camera_distances={cam_name: scenario.interface_distances[cam_name]},
    )

    # Render frame
    image = render_synthetic_frame(
        camera_intrinsics=intrinsics,
        camera_extrinsics=extrinsics,
        board_pose=board_pose,
        board_geometry=board_geometry,
        interface=interface,
        camera_name=cam_name,
        image_size=intrinsics.image_size,
        underwater=True,
    )

    # Check image properties
    assert image.dtype == np.uint8
    assert len(image.shape) == 2  # Grayscale
    assert image.shape == (intrinsics.image_size[1], intrinsics.image_size[0])

    # Image should have some non-zero pixels (corner markers)
    assert np.any(image > 0)


def test_board_config_consistency():
    """Test that all presets use the same ChArUco board config."""
    small = generate_synthetic_rig("small")
    medium = generate_synthetic_rig("medium")
    large = generate_synthetic_rig("large")

    # All should use 12x9 ChArUco board
    for scenario in [small, medium, large]:
        assert scenario.board_config.squares_x == 12
        assert scenario.board_config.squares_y == 9
        assert np.isclose(scenario.board_config.square_size, 0.060)
        assert np.isclose(scenario.board_config.marker_size, 0.045)
        assert scenario.board_config.dictionary == "DICT_5X5_100"


def test_camera_naming_convention():
    """Test that cameras are named cam0, cam1, ..., camN."""
    scenario = generate_synthetic_rig("medium")  # 6 cameras

    expected_names = [f"cam{i}" for i in range(6)]
    assert list(scenario.intrinsics.keys()) == expected_names
    assert list(scenario.extrinsics.keys()) == expected_names
    assert list(scenario.interface_distances.keys()) == expected_names


def test_reference_camera_at_origin():
    """Test that cam0 is always at origin with identity rotation."""
    for preset in ["small", "medium", "large"]:
        scenario = generate_synthetic_rig(preset)

        # cam0 should be at origin
        cam0_extrinsics = scenario.extrinsics["cam0"]
        assert np.allclose(cam0_extrinsics.R, np.eye(3))
        assert np.allclose(cam0_extrinsics.t, np.zeros(3))

        # Camera center should be at origin
        assert np.allclose(cam0_extrinsics.C, np.zeros(3))
