"""Coordinate frame visualization diagram.

Generates a 3D perspective view showing world coordinate frame, camera positions,
water surface, and underwater region.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate(output_dir: Path):
    """Generate coordinate frame diagram showing world frame conventions.

    Args:
        output_dir: Directory to save the PNG diagram
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw coordinate axes from world frame origin
    axis_length = 0.4
    ax.quiver(
        0,
        0,
        0,
        axis_length,
        0,
        0,
        color="r",
        arrow_length_ratio=0.15,
        linewidth=2.5,
        label="+X (right)",
    )
    ax.quiver(
        0,
        0,
        0,
        0,
        axis_length,
        0,
        color="g",
        arrow_length_ratio=0.15,
        linewidth=2.5,
        label="+Y (forward)",
    )
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        axis_length,
        color="b",
        arrow_length_ratio=0.15,
        linewidth=2.5,
        label="+Z (down)",
    )

    # Axis labels
    ax.text(axis_length + 0.05, 0, 0, "+X", color="r", fontsize=12, weight="bold")
    ax.text(0, axis_length + 0.05, 0, "+Y", color="g", fontsize=12, weight="bold")
    ax.text(0, 0, axis_length + 0.05, "+Z", color="b", fontsize=12, weight="bold")

    # Mark origin (reference camera)
    ax.scatter([0], [0], [0], color="black", s=100, marker="^", label="Camera (Z≈0)")
    ax.text(0, 0, -0.08, "Origin\n(Cam0)", ha="center", va="top", fontsize=9)

    # Draw additional camera positions
    camera_positions = [
        [0.3, 0.0, 0.01],
        [-0.2, 0.25, -0.01],
        [0.1, -0.3, 0.02],
    ]
    for i, pos in enumerate(camera_positions, start=1):
        ax.scatter(pos[0], pos[1], pos[2], color="black", s=60, marker="^", alpha=0.6)

    # Water surface plane
    water_z = 0.5
    x_plane = np.linspace(-0.5, 0.5, 10)
    y_plane = np.linspace(-0.5, 0.5, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.full_like(X_plane, water_z)

    ax.plot_surface(
        X_plane,
        Y_plane,
        Z_plane,
        alpha=0.2,
        color="cyan",
        edgecolor="blue",
        linewidth=0.5,
    )
    ax.text(
        0.4,
        0.4,
        water_z,
        "Water surface\n(Z = water_z)",
        color="blue",
        fontsize=10,
        weight="bold",
        ha="center",
    )

    # Underwater target region (show a few points)
    target_positions = [
        [0.15, 0.1, 0.8],
        [-0.1, 0.2, 0.9],
        [0.05, -0.15, 1.0],
    ]
    target_x = [p[0] for p in target_positions]
    target_y = [p[1] for p in target_positions]
    target_z = [p[2] for p in target_positions]
    ax.scatter(
        target_x,
        target_y,
        target_z,
        color="darkblue",
        s=50,
        marker="o",
        alpha=0.5,
        label="Targets (Z>water_z)",
    )

    # Draw a sample board (rectangle in underwater region)
    board_z = 0.85
    board_corners = np.array(
        [
            [-0.15, -0.1, board_z],
            [0.15, -0.1, board_z],
            [0.15, 0.1, board_z],
            [-0.15, 0.1, board_z],
            [-0.15, -0.1, board_z],  # Close the loop
        ]
    )
    ax.plot(
        board_corners[:, 0],
        board_corners[:, 1],
        board_corners[:, 2],
        "k-",
        linewidth=2,
        alpha=0.4,
    )
    ax.text(0, -0.12, board_z, "Calibration\nboard", ha="center", va="top", fontsize=9)

    # Annotate key Z levels
    ax.text(
        -0.55,
        0,
        0.05,
        "Z ≈ 0\n(cameras)",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
    )
    ax.text(
        -0.55,
        0,
        water_z,
        f"Z = {water_z:.1f}m\n(surface)",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.4),
    )
    ax.text(
        -0.55,
        0,
        board_z,
        f"Z > {water_z:.1f}m\n(underwater)",
        fontsize=9,
        ha="right",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.4),
    )

    # Configure 3D axes
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.2, 1.2)
    ax.set_xlabel("X (meters)", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (meters)", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (meters, down)", fontsize=10, labelpad=8)
    ax.set_title(
        "AquaCal World Coordinate Frame\n(Z-down convention)", fontsize=13, pad=15
    )

    # Adjust viewing angle
    ax.view_init(elev=20, azim=135)

    # Legend
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Save diagram
    output_path = output_dir / "coordinate_frames.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved {output_path}")
