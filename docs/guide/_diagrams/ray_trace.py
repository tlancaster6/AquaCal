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

# Add shared scripts dir to import palette
scripts_dir = project_root / "docs" / "_static" / "scripts"
sys.path.insert(0, str(scripts_dir))

from palette import (  # noqa: E402
    AIR_FILL,
    BOARD_COLOR,
    CAMERA_COLOR,
    INTERFACE_POINT,
    LABEL_COLOR,
    RAY_AIR,
    RAY_WATER,
    WATER_FILL,
    WATER_SURFACE,
)

from aquacal.core.refractive_geometry import snells_law_3d  # noqa: E402


def generate(output_dir: Path):
    """Generate ray trace diagram showing Snell's law refraction.

    Args:
        output_dir: Directory to save the PNG diagram
    """
    # Setup: camera in air, water surface, underwater target
    water_z = 0.5  # Water surface at Z = 0.5m
    C = np.array([0.0, 0.0, 0.0])  # Camera at origin
    Q = np.array([0.3, 0.0, 1.2])  # Underwater target (offset in X and Z)

    # Compute interface crossing point using actual Snell's law from AquaCal
    h_c = water_z - C[2]  # Camera to interface gap
    h_q = Q[2] - water_z  # Interface to target gap
    r_q = abs(Q[0] - C[0])  # Horizontal offset

    # Refractive indices
    n_air = 1.0
    n_water = 1.333
    normal = np.array([0.0, 0.0, -1.0])  # Interface normal (points up, water->air)

    # Solve for interface crossing point using Newton-Raphson approach
    # with Snell's law from AquaCal library
    r_p = r_q * h_c / (h_c + h_q)  # Initial guess: pinhole projection

    # Newton iteration (a few steps to get accurate result)
    for _ in range(5):
        # Compute incident ray from C to interface point P
        P_guess = np.array([C[0] + r_p, 0.0, water_z])
        d_inc = P_guess - C
        d_inc = d_inc / np.linalg.norm(d_inc)

        # Apply Snell's law using AquaCal function
        d_refracted = snells_law_3d(d_inc, normal, n_air / n_water)

        if d_refracted is None:
            break  # Shouldn't happen for air->water at normal angles

        # Check where the refracted ray from P intersects Q's depth
        # The refracted ray should reach Q: P + t * d_refracted = Q
        # Solve for horizontal offset at Q's depth
        # Z component: P[2] + t * d_refracted[2] = Q[2]
        if abs(d_refracted[2]) < 1e-10:
            break  # Ray is horizontal, shouldn't happen

        t_to_Q = (Q[2] - P_guess[2]) / d_refracted[2]
        x_at_Q = P_guess[0] + t_to_Q * d_refracted[0]

        # Residual: difference between where ray goes and where target is
        residual = x_at_Q - Q[0]

        # Update r_p based on residual (simple step)
        r_p = r_p - residual * 0.5  # Damped update for stability

        if abs(residual) < 1e-9:
            break

    # Final interface point
    P = np.array([C[0] + r_p, 0.0, water_z])

    # Compute angles for annotation using actual ray directions
    # Incident ray direction
    d_inc_final = P - C
    d_inc_final = d_inc_final / np.linalg.norm(d_inc_final)

    # Refracted ray direction (using AquaCal function)
    d_refracted_final = snells_law_3d(d_inc_final, normal, n_air / n_water)

    # Angles from normal (vertical Z-axis in this 2D case)
    # theta = angle from vertical = arctan(horizontal / vertical)
    theta_air = np.arctan2(r_p, h_c) * 180 / np.pi
    theta_water = (
        np.arctan2(abs(d_refracted_final[0]), abs(d_refracted_final[2])) * 180 / np.pi
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw water surface
    ax.axhline(water_z, color=WATER_SURFACE, linewidth=2, linestyle="--", alpha=0.8)
    ax.text(
        0.95,
        water_z + 0.05,
        f"Water surface (Z = {water_z:.1f}m)",
        ha="right",
        va="bottom",
        fontsize=10,
        color=WATER_SURFACE,
    )

    # Draw air and water regions
    ax.fill_between(
        [-0.1, 1.0], 0, water_z, alpha=0.12, color=AIR_FILL, label="Air (n=1.0)"
    )
    ax.fill_between(
        [-0.1, 1.0],
        water_z,
        1.5,
        alpha=0.15,
        color=WATER_FILL,
        label="Water (n=1.333)",
    )

    # Draw incident ray (camera to interface)
    ax.plot(
        [C[0], P[0]],
        [C[2], P[2]],
        color=RAY_AIR,
        linewidth=2,
        label="Incident ray (air)",
        marker="o",
        markersize=4,
    )

    # Draw refracted ray (interface to target)
    ax.plot(
        [P[0], Q[0]],
        [P[2], Q[2]],
        color=RAY_WATER,
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
        fc=LABEL_COLOR,
        ec=LABEL_COLOR,
        linewidth=1.5,
    )
    ax.text(
        P[0] + 0.05,
        P[2] - normal_length / 2,
        "n [0,0,-1]",
        fontsize=9,
        va="center",
        color=LABEL_COLOR,
    )

    # Mark points
    ax.plot(C[0], C[2], "o", color=CAMERA_COLOR, markersize=8, label="Camera C")
    ax.text(C[0] - 0.05, C[2] - 0.08, "C (camera)", ha="right", fontsize=10)

    ax.plot(
        P[0], P[2], "o", color=INTERFACE_POINT, markersize=8, label="Interface point P"
    )
    ax.text(P[0], P[2] + 0.08, "P (interface)", ha="center", va="bottom", fontsize=10)

    ax.plot(Q[0], Q[2], "o", color=BOARD_COLOR, markersize=8, label="Target Q")
    ax.text(Q[0] + 0.05, Q[2], "Q (target)", ha="left", fontsize=10)

    # Add angle annotations
    angle_offset = 0.15
    ax.annotate(
        f"θᵢ = {theta_air:.1f}°",
        xy=(P[0], P[2]),
        xytext=(P[0] - angle_offset, P[2] - angle_offset),
        fontsize=9,
        color=RAY_AIR,
    )
    ax.annotate(
        f"θᵣ = {theta_water:.1f}°",
        xy=(P[0], P[2]),
        xytext=(P[0] + angle_offset, P[2] + angle_offset),
        fontsize=9,
        color=RAY_WATER,
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
