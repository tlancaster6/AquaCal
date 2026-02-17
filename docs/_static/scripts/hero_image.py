"""Hero image generation script for AquaCal documentation.

Generates a 2D cross-section hero image showing three cameras above a water
surface with rays bending at the water interface (Snell's law refraction) and
a calibration board below. Targets ~1200x500 px at 150 DPI.
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Resolve paths so this script can be run from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

# Add project src so we can import aquacal
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Add this scripts dir so we can import palette
sys.path.insert(0, str(_SCRIPT_DIR))

from palette import (  # noqa: E402
    AIR_FILL,
    BOARD_COLOR,
    CAMERA_COLOR,
    INTERFACE_POINT,
    RAY_AIR,
    RAY_WATER,
    WATER_FILL,
    WATER_SURFACE,
)

from aquacal.core.refractive_geometry import snells_law_3d  # noqa: E402

# ---------------------------------------------------------------------------
# Scene geometry parameters
# ---------------------------------------------------------------------------

N_AIR = 1.0
N_WATER = 1.333
N_RATIO = N_AIR / N_WATER  # air -> water

NORMAL = np.array([0.0, 0.0, -1.0])  # interface normal points up (water->air)

WATER_Z = 0.5  # water surface depth (m)

# Three camera positions: evenly spread horizontally, just above water surface
CAMERAS = [
    np.array([-0.55, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.02]),
    np.array([0.55, 0.0, -0.01]),
]

# Board: four corners slightly tilted, below the water surface
BOARD_Z_CENTER = 1.15
BOARD_HALF_W = 0.35
BOARD_TILT = 0.06  # Z offset across board width to give a slight tilt
BOARD_LEFT = np.array([-BOARD_HALF_W, 0.0, BOARD_Z_CENTER + BOARD_TILT])
BOARD_RIGHT = np.array([BOARD_HALF_W, 0.0, BOARD_Z_CENTER - BOARD_TILT])

# Representative target point on the board for each camera's ray endpoint
TARGET_X_OFFSETS = [-0.20, 0.05, 0.22]


def _solve_interface_point(camera: np.ndarray, target_x: float) -> np.ndarray:
    """Find where a camera ray hits the water interface using Newton-Raphson.

    Args:
        camera: Camera position [x, y, z] in world frame.
        target_x: X-coordinate of the underwater target.

    Returns:
        Interface crossing point [x, 0, water_z].
    """
    target = np.array([target_x, 0.0, BOARD_Z_CENTER])

    h_c = WATER_Z - camera[2]
    h_q = target[2] - WATER_Z

    # Initial guess: pinhole projection
    r_p = (target_x - camera[0]) * h_c / (h_c + h_q)

    for _ in range(10):
        p_guess = np.array([camera[0] + r_p, 0.0, WATER_Z])
        d_inc = p_guess - camera
        d_inc = d_inc / np.linalg.norm(d_inc)

        d_refr = snells_law_3d(d_inc, NORMAL, N_RATIO)
        if d_refr is None:
            break

        if abs(d_refr[2]) < 1e-10:
            break

        t_to_q = (target[2] - p_guess[2]) / d_refr[2]
        x_at_q = p_guess[0] + t_to_q * d_refr[0]
        residual = x_at_q - target_x

        r_p -= residual * 0.5
        if abs(residual) < 1e-9:
            break

    return np.array([camera[0] + r_p, 0.0, WATER_Z])


def generate(output_path: Path) -> None:
    """Generate the hero image and save it as a PNG.

    Args:
        output_path: Full path (including filename) where the PNG is saved.
    """
    # Figure sized for ~1200 x 500 px at 150 DPI
    fig, ax = plt.subplots(figsize=(8, 3.33), dpi=150)
    fig.patch.set_facecolor("white")

    # Scene X and Z extents
    X_MIN, X_MAX = -0.85, 0.85
    Z_MIN, Z_MAX = -0.12, 1.35

    # ------------------------------------------------------------------
    # Background fills: air (above) and water (below)
    # ------------------------------------------------------------------
    ax.fill_between(
        [X_MIN, X_MAX],
        Z_MIN,
        WATER_Z,
        color=AIR_FILL,
        alpha=1.0,
        zorder=0,
    )
    ax.fill_between(
        [X_MIN, X_MAX],
        WATER_Z,
        Z_MAX,
        color=WATER_FILL,
        alpha=0.18,
        zorder=0,
    )

    # ------------------------------------------------------------------
    # Water surface line
    # ------------------------------------------------------------------
    ax.axhline(
        WATER_Z,
        color=WATER_SURFACE,
        linewidth=2.5,
        zorder=2,
    )
    ax.text(
        X_MAX - 0.02,
        WATER_Z - 0.04,
        "Water surface",
        ha="right",
        va="bottom",
        fontsize=8,
        color=WATER_SURFACE,
        weight="bold",
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Calibration board (tilted line)
    # ------------------------------------------------------------------
    ax.plot(
        [BOARD_LEFT[0], BOARD_RIGHT[0]],
        [BOARD_LEFT[2], BOARD_RIGHT[2]],
        color=BOARD_COLOR,
        linewidth=4,
        solid_capstyle="round",
        zorder=3,
    )
    ax.text(
        0.0,
        BOARD_Z_CENTER + 0.07,
        "Board",
        ha="center",
        va="top",
        fontsize=8,
        color=BOARD_COLOR,
        weight="bold",
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Rays and interface points for each camera
    # ------------------------------------------------------------------
    for cam_idx, (cam, tgt_x) in enumerate(zip(CAMERAS, TARGET_X_OFFSETS)):
        p_iface = _solve_interface_point(cam, tgt_x)
        target = np.array([tgt_x, 0.0, BOARD_Z_CENTER])

        # Ray in air
        ax.plot(
            [cam[0], p_iface[0]],
            [cam[2], p_iface[2]],
            color=RAY_AIR,
            linewidth=1.6,
            alpha=0.85,
            zorder=4,
        )

        # Ray in water
        ax.plot(
            [p_iface[0], target[0]],
            [p_iface[2], target[2]],
            color=RAY_WATER,
            linewidth=1.6,
            alpha=0.85,
            zorder=4,
        )

        # Interface crossing point
        ax.plot(
            p_iface[0],
            p_iface[2],
            "o",
            color=INTERFACE_POINT,
            markersize=5,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # Camera icons (downward-pointing triangles)
    # ------------------------------------------------------------------
    cam_label_done = False
    for cam in CAMERAS:
        # Draw a simple trapezoid camera body
        body_w = 0.045
        body_h = 0.055
        lens_w = 0.022
        # Trapezoid corners (wider at top, narrower toward lens at bottom)
        trap_x = [
            cam[0] - body_w,
            cam[0] + body_w,
            cam[0] + lens_w,
            cam[0] - lens_w,
            cam[0] - body_w,
        ]
        trap_z = [
            cam[2] - body_h,
            cam[2] - body_h,
            cam[2],
            cam[2],
            cam[2] - body_h,
        ]
        ax.fill(trap_x, trap_z, color=CAMERA_COLOR, zorder=6)

        # Label only once to avoid clutter
        if not cam_label_done:
            ax.text(
                CAMERAS[0][0] - body_w - 0.03,
                CAMERAS[0][2] - body_h / 2,
                "Camera",
                ha="right",
                va="center",
                fontsize=8,
                color=CAMERA_COLOR,
                weight="bold",
                zorder=7,
            )
            cam_label_done = True

    # ------------------------------------------------------------------
    # Axes configuration
    # ------------------------------------------------------------------
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.invert_yaxis()  # Z increases downward
    ax.set_aspect("equal")
    ax.axis("off")  # Clean hero image — no axes ticks or labels

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved hero image: {output_path}")


if __name__ == "__main__":
    _output = _PROJECT_ROOT / "docs" / "_static" / "hero_ray_trace.png"
    generate(_output)
