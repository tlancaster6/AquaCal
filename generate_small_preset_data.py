"""Generate static small preset data files for in-package distribution.

This script generates the small preset data once and saves it as JSON files
that ship with the package. Run this when the synthetic data generation changes.
"""

import json
from pathlib import Path

from aquacal.datasets.synthetic import generate_synthetic_rig


def serialize_scenario(scenario):
    """Serialize SyntheticScenario to JSON-serializable dict."""
    data = {
        "name": scenario.name,
        "description": scenario.description,
        "noise_std": scenario.noise_std,
        "board_config": {
            "squares_x": scenario.board_config.squares_x,
            "squares_y": scenario.board_config.squares_y,
            "square_size": scenario.board_config.square_size,
            "marker_size": scenario.board_config.marker_size,
            "dictionary": scenario.board_config.dictionary,
        },
        "intrinsics": {
            cam_name: {
                "K": intr.K.tolist(),
                "dist_coeffs": intr.dist_coeffs.tolist(),
                "image_size": list(intr.image_size),
            }
            for cam_name, intr in scenario.intrinsics.items()
        },
        "extrinsics": {
            cam_name: {
                "R": extr.R.tolist(),
                "t": extr.t.tolist(),
            }
            for cam_name, extr in scenario.extrinsics.items()
        },
        "interface_distances": {
            cam_name: float(dist)
            for cam_name, dist in scenario.interface_distances.items()
        },
        "board_poses": [
            {
                "frame_idx": pose.frame_idx,
                "rvec": pose.rvec.tolist(),
                "tvec": pose.tvec.tolist(),
            }
            for pose in scenario.board_poses
        ],
    }
    return data


def serialize_detections(scenario):
    """Serialize DetectionResult from scenario to JSON-serializable dict."""
    from aquacal.core.board import BoardGeometry
    from aquacal.datasets.synthetic import generate_synthetic_detections

    # Generate detections
    board = BoardGeometry(scenario.board_config)
    detections = generate_synthetic_detections(
        intrinsics=scenario.intrinsics,
        extrinsics=scenario.extrinsics,
        interface_distances=scenario.interface_distances,
        board=board,
        board_poses=scenario.board_poses,
        noise_std=scenario.noise_std,
        min_corners=8,
        seed=42,  # Same seed as small preset
    )

    data = {
        "camera_names": detections.camera_names,
        "total_frames": detections.total_frames,
        "frames": {
            str(frame_idx): {
                "frame_idx": fd.frame_idx,
                "detections": {
                    cam_name: {
                        "corner_ids": det.corner_ids.tolist(),
                        "corners_2d": det.corners_2d.tolist(),
                    }
                    for cam_name, det in fd.detections.items()
                },
            }
            for frame_idx, fd in detections.frames.items()
        },
    }
    return data


def main():
    """Generate and save small preset data files."""
    print("Generating small preset...")
    scenario = generate_synthetic_rig("small")

    # Serialize ground truth
    ground_truth_data = serialize_scenario(scenario)

    # Serialize detections
    detections_data = serialize_detections(scenario)

    # Save to data/small/
    data_dir = Path("src/aquacal/datasets/data/small")
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Saving ground_truth.json...")
    with open(data_dir / "ground_truth.json", "w") as f:
        json.dump(ground_truth_data, f, indent=2)

    print("Saving detections.json...")
    with open(data_dir / "detections.json", "w") as f:
        json.dump(detections_data, f, indent=2)

    print(f"✓ Small preset data saved to {data_dir}")
    print(f"  - Cameras: {len(scenario.intrinsics)}")
    print(f"  - Frames: {len(scenario.board_poses)}")
    print(
        f"  - Ground truth size: {(data_dir / 'ground_truth.json').stat().st_size / 1024:.1f} KB"
    )
    print(
        f"  - Detections size: {(data_dir / 'detections.json').stat().st_size / 1024:.1f} KB"
    )


if __name__ == "__main__":
    main()
