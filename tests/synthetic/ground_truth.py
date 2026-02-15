"""Synthetic ground truth generation for full pipeline testing.

This module now re-exports functions from the public aquacal.datasets API
and provides test-specific scenario presets via create_scenario().
"""

from __future__ import annotations

from aquacal.config.schema import BoardConfig
from aquacal.datasets.synthetic import (
    SyntheticScenario,
    generate_board_trajectory,
    generate_camera_array,
    generate_real_rig_array,
    generate_real_rig_trajectory,
)

# All generation functions are now imported from aquacal.datasets.synthetic


def create_scenario(name: str, seed: int = 42) -> SyntheticScenario:
    """
    Create a predefined test scenario.

    Available scenarios:
    - "ideal": 4 cameras, 20 frames, 0 noise - verify math correctness
    - "minimal": 2 cameras, 10 frames, 0.3px noise - edge case
    - "realistic": 13 cameras matching actual hardware, 100 frames, 0.5px noise

    Args:
        name: Scenario name
        seed: Random seed for reproducibility

    Returns:
        Complete SyntheticScenario with all ground truth

    Raises:
        ValueError: If scenario name not recognized
    """
    # Common board config (matches real hardware)
    default_board = BoardConfig(
        squares_x=12,
        squares_y=9,
        square_size=0.060,
        marker_size=0.045,
        dictionary="DICT_5X5_100",
    )

    if name == "ideal":
        intrinsics, extrinsics, distances = generate_camera_array(
            n_cameras=4,
            layout="grid",
            spacing=0.1,
            height_above_water=0.15,
            height_variation=0.0,  # No variation for ideal case
            seed=seed,
        )
        # Extract camera positions for trajectory generation
        camera_positions = {cam: ext.C for cam, ext in extrinsics.items()}
        board_poses = generate_board_trajectory(
            n_frames=20,
            camera_positions=camera_positions,
            interface_distances=distances,
            depth_range=(0.25, 0.45),
            xy_extent=0.08,
            seed=seed,
        )
        return SyntheticScenario(
            name="ideal",
            board_config=default_board,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            interface_distances=distances,
            board_poses=board_poses,
            noise_std=0.0,
            description="Ideal conditions: 4 cameras, 20 frames, 0 noise",
        )

    elif name == "minimal":
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
            interface_distances=distances,
            depth_range=(0.25, 0.40),
            xy_extent=0.06,
            seed=seed,
        )
        return SyntheticScenario(
            name="minimal",
            board_config=default_board,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            interface_distances=distances,
            board_poses=board_poses,
            noise_std=0.3,
            description="Minimal scenario: 2 cameras, 10 frames, 0.3px noise",
        )

    elif name == "realistic":
        intrinsics, extrinsics, distances = generate_real_rig_array(
            height_above_water=0.75,
            height_variation=0.002,
            seed=seed,
        )

        board_poses = generate_real_rig_trajectory(
            n_frames=30,
            depth_range=(0.9, 1.5),
            seed=seed,
        )

        return SyntheticScenario(
            name="realistic",
            board_config=default_board,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            interface_distances=distances,
            board_poses=board_poses,
            noise_std=0.5,
            description="13-camera rig matching real hardware: center + 2 concentric rings",
        )

    raise ValueError(f"Unknown scenario: {name}")


# compute_calibration_errors is now imported from aquacal.datasets.synthetic
