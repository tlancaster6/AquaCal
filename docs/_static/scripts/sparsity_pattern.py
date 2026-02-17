"""Jacobian sparsity pattern diagram for AquaCal documentation.

Generates a block-sparse Jacobian illustration for a small example:
3 cameras, 3 frames, with varied observations per camera-frame pair.

The pattern shows that each residual (camera-frame pair) only touches:
  - That camera's 6 extrinsic parameters
  - The single global water_z parameter
  - That frame's 6 board pose parameters

This block-diagonal structure (plus a dense water_z column) enables
efficient sparse finite differencing.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Ensure palette.py is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))
from palette import (  # noqa: E402
    BOARD_COLOR,
    CAMERA_COLOR,
    GRID_COLOR,
    LABEL_COLOR,
    WATER_SURFACE,
)

# ---------------------------------------------------------------------------
# Problem setup: 3 cameras, 3 frames
# cam0 is the reference camera (no extrinsic parameters optimized)
# cam1 and cam2 each have 6 extrinsic parameters
# water_z is a single global parameter
# Each frame has 6 board pose parameters
# ---------------------------------------------------------------------------

N_CAMS = 3  # cam0 (reference), cam1, cam2
N_FRAMES = 3  # F0, F1, F2
N_CORNERS = 4  # corners detected per observation (simplified)

# Which camera-frame pairs have observations (not all cameras see all frames)
OBSERVATIONS = [
    (0, 0),
    (0, 1),  # cam0 sees F0, F1
    (1, 1),
    (1, 2),  # cam1 sees F1, F2
    (2, 0),
    (2, 2),  # cam2 sees F0, F2
]

# Parameter layout:
#   extrinsics: 6 per non-reference camera = 6 * (N_CAMS - 1) = 12 columns
#   water_z: 1 column
#   board_poses: 6 per frame = 6 * N_FRAMES = 18 columns
N_EXT_PARAMS = 6 * (N_CAMS - 1)  # 12
N_WATER_PARAMS = 1
N_BOARD_PARAMS = 6 * N_FRAMES  # 18
N_PARAMS = N_EXT_PARAMS + N_WATER_PARAMS + N_BOARD_PARAMS  # 31

# Residuals: 2 per corner per observation
N_RESIDUALS = len(OBSERVATIONS) * N_CORNERS * 2  # 6 obs * 4 corners * 2 = 48

# Build sparsity pattern matrix
pattern = np.zeros((N_RESIDUALS, N_PARAMS), dtype=float)

# Column ranges
ext_start = 0  # extrinsics block starts at col 0
water_col = N_EXT_PARAMS  # water_z column
board_start = N_EXT_PARAMS + N_WATER_PARAMS  # board poses block

residual_row = 0
for cam, frame in OBSERVATIONS:
    n_rows = N_CORNERS * 2  # residuals for this observation

    # Extrinsic columns for this camera (cam0 is reference: no params)
    if cam > 0:
        cam_ext_start = ext_start + (cam - 1) * 6
        pattern[
            residual_row : residual_row + n_rows, cam_ext_start : cam_ext_start + 6
        ] = 1.0

    # water_z column (always present for every observation)
    pattern[residual_row : residual_row + n_rows, water_col] = 1.0

    # Board pose columns for this frame
    frame_board_start = board_start + frame * 6
    pattern[
        residual_row : residual_row + n_rows, frame_board_start : frame_board_start + 6
    ] = 1.0

    residual_row += n_rows


def generate(output_dir: Path) -> None:
    """Generate and save the sparsity pattern diagram.

    Args:
        output_dir: Directory where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Build a colored version: different colors per parameter block
    colored = np.zeros((N_RESIDUALS, N_PARAMS, 4))  # RGBA

    # Colors for each block (RGBA)
    ext_rgba = (
        *[
            int(c, 16) / 255
            for c in [CAMERA_COLOR[1:3], CAMERA_COLOR[3:5], CAMERA_COLOR[5:7]]
        ],
        0.85,
    )
    water_rgba = (
        *[
            int(c, 16) / 255
            for c in [WATER_SURFACE[1:3], WATER_SURFACE[3:5], WATER_SURFACE[5:7]]
        ],
        0.85,
    )
    board_rgba = (
        *[
            int(c, 16) / 255
            for c in [BOARD_COLOR[1:3], BOARD_COLOR[3:5], BOARD_COLOR[5:7]]
        ],
        0.85,
    )
    empty_rgba = (0.96, 0.96, 0.96, 1.0)

    # Fill background (empty cells)
    colored[:, :] = empty_rgba

    # Color non-zero entries per block
    ext_mask = pattern[:, :N_EXT_PARAMS].astype(bool)
    water_mask = pattern[:, water_col].astype(bool)
    board_mask = pattern[:, board_start:].astype(bool)

    for row in range(N_RESIDUALS):
        for col in range(N_EXT_PARAMS):
            if ext_mask[row, col]:
                colored[row, col] = ext_rgba
        if water_mask[row]:
            colored[row, water_col] = water_rgba
        for col in range(N_BOARD_PARAMS):
            if board_mask[row, col]:
                colored[row, board_start + col] = board_rgba

    ax.imshow(colored, aspect="auto", interpolation="nearest")

    # Grid lines
    for x in range(N_PARAMS + 1):
        lw = 1.5 if x in (N_EXT_PARAMS, N_EXT_PARAMS + 1) else 0.4
        ax.axvline(x - 0.5, color=GRID_COLOR, linewidth=lw)
    for y in range(N_RESIDUALS + 1):
        ax.axhline(y - 0.5, color=GRID_COLOR, linewidth=0.4)

    # Column group separator: thick line between extrinsics, water_z, boards
    ax.axvline(N_EXT_PARAMS - 0.5, color=LABEL_COLOR, linewidth=2.0)
    ax.axvline(N_EXT_PARAMS + 0.5, color=LABEL_COLOR, linewidth=2.0)
    ax.axvline(board_start - 0.5, color=LABEL_COLOR, linewidth=2.0)

    # Row group separators between observations
    cum_rows = 0
    for i, _ in enumerate(OBSERVATIONS[:-1]):
        cum_rows += N_CORNERS * 2
        ax.axhline(cum_rows - 0.5, color=LABEL_COLOR, linewidth=1.2)

    # X-axis: column group labels
    ax.set_xticks(
        [N_EXT_PARAMS / 2 - 0.5, water_col, board_start + N_BOARD_PARAMS / 2 - 0.5]
    )
    ax.set_xticklabels(
        [
            f"Extrinsics\n(6 × {N_CAMS - 1} non-ref cams = {N_EXT_PARAMS} cols)",
            "water_z\n(1 col)",
            f"Board Poses\n(6 × {N_FRAMES} frames = {N_BOARD_PARAMS} cols)",
        ],
        fontsize=9,
        color=LABEL_COLOR,
    )
    ax.tick_params(axis="x", which="both", length=0)

    # Y-axis: observation group labels
    obs_labels = []
    for cam, frame in OBSERVATIONS:
        obs_labels.append(f"cam{cam} / F{frame}")
    obs_centers = [
        i * N_CORNERS * 2 + N_CORNERS - 0.5 for i in range(len(OBSERVATIONS))
    ]
    ax.set_yticks(obs_centers)
    ax.set_yticklabels(obs_labels, fontsize=8, color=LABEL_COLOR)
    ax.tick_params(axis="y", which="both", length=0)

    ax.set_xlabel("Parameters", fontsize=10, color=LABEL_COLOR, labelpad=8)
    ax.set_ylabel(
        "Residuals (camera × frame observations)",
        fontsize=10,
        color=LABEL_COLOR,
        labelpad=8,
    )
    ax.set_title(
        "Block-Sparse Jacobian Structure\n"
        f"({N_CAMS} cameras, {N_FRAMES} frames, {len(OBSERVATIONS)} observations)",
        fontsize=11,
        color=LABEL_COLOR,
        pad=10,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(
            facecolor=CAMERA_COLOR, alpha=0.85, label="Camera extrinsics (6 per camera)"
        ),
        mpatches.Patch(
            facecolor=WATER_SURFACE, alpha=0.85, label="water_z (global, 1 param)"
        ),
        mpatches.Patch(
            facecolor=BOARD_COLOR, alpha=0.85, label="Board pose (6 per frame)"
        ),
        mpatches.Patch(
            facecolor="#F5F5F5", edgecolor=GRID_COLOR, label="Zero (no dependency)"
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
        edgecolor=GRID_COLOR,
    )

    # Density annotation
    nonzero = int(pattern.sum())
    total = N_RESIDUALS * N_PARAMS
    density = nonzero / total * 100
    ax.text(
        0.01,
        -0.12,
        f"Sparsity: {100 - density:.0f}% zeros  ({nonzero}/{total} entries non-zero)",
        transform=ax.transAxes,
        fontsize=8,
        color=LABEL_COLOR,
        alpha=0.7,
    )

    plt.tight_layout()
    output_path = Path(output_dir) / "sparsity_pattern.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    output_dir = Path(__file__).parent.parent / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate(output_dir)
