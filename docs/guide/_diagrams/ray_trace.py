"""Ray trace diagram generation using actual AquaCal functions.

Generates a 2D cross-section showing refractive ray path from camera to underwater target.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project to path to import aquacal
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


def generate(output_dir: Path):
    """Generate ray trace diagram showing Snell's law refraction.

    Args:
        output_dir: Directory to save the PNG diagram
    """
    # Setup: camera in air, water surface, underwater target
    water_z = 0.5  # Water surface at Z = 0.5m
    C = np.array([0.0, 0.0, 0.0])  # Camera at origin
    Q = np.array([0.3, 0.0, 1.2])  # Underwater target (offset in X and Z)

    # Compute interface crossing point using actual Snell's law
    # This is a simplified 2D version for visualization
    h_c = water_z - C[2]  # Camera to interface gap
    h_q = Q[2] - water_z  # Interface to target gap
    r_q = abs(Q[0] - C[0])  # Horizontal offset

    # Solve for interface crossing point using Newton-Raphson approach
    # (same logic as in _refractive_project_newton)
    n_air = 1.0
    n_water = 1.333

    # Initial guess: pinhole projection
    r_p = r_q * h_c / (h_c + h_q)

    # Newton iteration (a few steps to get accurate result)
    for _ in range(5):
        # Snell equation: n_air * sin(theta_air) - n_water * sin(theta_water) = 0
        sin_air = r_p / np.sqrt(r_p**2 + h_c**2)
        sin_water = (r_q - r_p) / np.sqrt((r_q - r_p) ** 2 + h_q**2)
        f = n_air * sin_air - n_water * sin_water

        # Derivative
        f_prime = (
            n_air * h_c**2 / (r_p**2 + h_c**2) ** 1.5
            + n_water * h_q**2 / ((r_q - r_p) ** 2 + h_q**2) ** 1.5
        )

        # Update
        r_p = r_p - f / f_prime

    # Interface point
    P = np.array([C[0] + r_p, 0.0, water_z])

    # Compute angles for annotation
    theta_air = np.arctan2(r_p, h_c) * 180 / np.pi
    theta_water = np.arctan2(r_q - r_p, h_q) * 180 / np.pi

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw water surface
    ax.axhline(water_z, color="blue", linewidth=2, linestyle="--", alpha=0.5)
    ax.text(
        0.95,
        water_z + 0.05,
        f"Water surface (Z = {water_z:.1f}m)",
        ha="right",
        va="bottom",
        fontsize=10,
        color="blue",
    )

    # Draw air and water regions
    ax.fill_between(
        [-0.1, 1.0], 0, water_z, alpha=0.05, color="cyan", label="Air (n=1.0)"
    )
    ax.fill_between(
        [-0.1, 1.0], water_z, 1.5, alpha=0.1, color="blue", label="Water (n=1.333)"
    )

    # Draw incident ray (camera to interface)
    ax.plot(
        [C[0], P[0]],
        [C[2], P[2]],
        "r-",
        linewidth=2,
        label="Incident ray (air)",
        marker="o",
        markersize=4,
    )

    # Draw refracted ray (interface to target)
    ax.plot(
        [P[0], Q[0]],
        [P[2], Q[2]],
        "g-",
        linewidth=2,
        label="Refracted ray (water)",
        marker="o",
        markersize=4,
    )

    # Draw interface normal at P
    normal_length = 0.2
    ax.arrow(
        P[0],
        P[2],
        0,
        -normal_length,
        head_width=0.03,
        head_length=0.04,
        fc="black",
        ec="black",
        linewidth=1.5,
    )
    ax.text(
        P[0] + 0.05,
        P[2] - normal_length / 2,
        "n [0,0,-1]",
        fontsize=9,
        va="center",
    )

    # Mark points
    ax.plot(C[0], C[2], "ko", markersize=8, label="Camera C")
    ax.text(C[0] - 0.05, C[2] - 0.08, "C (camera)", ha="right", fontsize=10)

    ax.plot(P[0], P[2], "mo", markersize=8, label="Interface point P")
    ax.text(P[0], P[2] + 0.08, "P (interface)", ha="center", va="bottom", fontsize=10)

    ax.plot(Q[0], Q[2], "bo", markersize=8, label="Target Q")
    ax.text(Q[0] + 0.05, Q[2], "Q (target)", ha="left", fontsize=10)

    # Add angle annotations
    angle_offset = 0.15
    ax.annotate(
        f"θᵢ = {theta_air:.1f}°",
        xy=(P[0], P[2]),
        xytext=(P[0] - angle_offset, P[2] - angle_offset),
        fontsize=9,
        color="red",
    )
    ax.annotate(
        f"θᵣ = {theta_water:.1f}°",
        xy=(P[0], P[2]),
        xytext=(P[0] + angle_offset, P[2] + angle_offset),
        fontsize=9,
        color="green",
    )

    # Add Snell's law equation
    ax.text(
        0.5,
        0.15,
        r"$n_{\mathrm{air}} \sin\theta_i = n_{\mathrm{water}} \sin\theta_r$",
        fontsize=12,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3),
    )

    # Configure axes
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(0, 1.5)
    ax.invert_yaxis()  # Z increases downward
    ax.set_xlabel("X (horizontal, meters)", fontsize=11)
    ax.set_ylabel("Z (depth, meters)", fontsize=11)
    ax.set_title("Refractive Ray Path: Camera → Water → Target", fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_aspect("equal")

    # Save diagram
    output_path = output_dir / "ray_trace.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved {output_path}")
