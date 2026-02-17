"""BFS pose graph diagram for AquaCal documentation.

Generates a bipartite graph showing camera-board connectivity used during
Stage 2 (Extrinsic Initialization). Highlights the BFS traversal path
from the reference camera (cam0) to all other cameras.

Node types:
    - Camera nodes (circles, CAMERA_COLOR): cam0, cam1, cam2, cam3
    - Frame/board nodes (squares via scatter, BOARD_COLOR): F1, F2, F3

Edges connect a camera to each frame it observes. BFS traversal edges
are highlighted to show how extrinsics are chained from the reference.
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
    RAY_AIR,
    WATER_SURFACE,
)

# ---------------------------------------------------------------------------
# Graph definition
# Cameras: cam0 (reference), cam1, cam2, cam3
# Frames: F1, F2, F3
# Edges (camera -> frame observations):
#   cam0-F1, cam0-F2       (cam0 is reference; anchors F1 and F2)
#   cam1-F2, cam1-F3       (cam1 linked to cam0 via F2)
#   cam2-F1, cam2-F3       (cam2 linked to cam0 via F1, or cam1 via F3)
#   cam3-F3                (cam3 linked to cam1 or cam2 via F3)
# ---------------------------------------------------------------------------

CAMERA_NODES = ["cam0", "cam1", "cam2", "cam3"]
FRAME_NODES = ["F1", "F2", "F3"]
EDGES = [
    ("cam0", "F1"),
    ("cam0", "F2"),
    ("cam1", "F2"),
    ("cam1", "F3"),
    ("cam2", "F1"),
    ("cam2", "F3"),
    ("cam3", "F3"),
]

# BFS traversal path from cam0 (reference):
# cam0 -> F2 -> cam1 -> F3 -> cam2, cam3
# cam0 -> F1 -> cam2 (already reached via F3, but also via F1)
BFS_EDGES = {
    ("cam0", "F2"),  # cam0 observes F2
    ("cam1", "F2"),  # cam1 discovered via F2
    ("cam1", "F3"),  # cam1 observes F3
    ("cam2", "F3"),  # cam2 discovered via F3
    ("cam3", "F3"),  # cam3 discovered via F3
    ("cam0", "F1"),  # cam0 observes F1
    ("cam2", "F1"),  # cam2 also confirmed via F1
}

# Node positions: cameras on left, frames on right in a bipartite layout
CAM_X = 0.15
FRAME_X = 0.85
CAM_SPACING = 1.0 / (len(CAMERA_NODES) + 1)
FRAME_SPACING = 1.0 / (len(FRAME_NODES) + 1)

NODE_POS = {}
for i, cam in enumerate(CAMERA_NODES):
    NODE_POS[cam] = (CAM_X, 1.0 - (i + 1) * CAM_SPACING)
for i, frame in enumerate(FRAME_NODES):
    NODE_POS[frame] = (FRAME_X, 1.0 - (i + 1) * FRAME_SPACING)


def _draw_camera_node(ax, pos, label, is_reference=False, node_size=0.055):
    """Draw a camera node as a circle with label."""
    x, y = pos
    color = CAMERA_COLOR
    edge_color = RAY_AIR if is_reference else CAMERA_COLOR
    lw = 3.5 if is_reference else 1.5
    circle = plt.Circle(
        (x, y), node_size, color=color, ec=edge_color, linewidth=lw, zorder=3
    )
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold" if is_reference else "normal",
        color="white",
        zorder=4,
    )


def _draw_frame_node(ax, pos, label, node_size=0.055):
    """Draw a frame/board node as a rounded rectangle."""
    x, y = pos
    rect = mpatches.FancyBboxPatch(
        (x - node_size, y - node_size * 0.7),
        node_size * 2,
        node_size * 1.4,
        boxstyle="round,pad=0.01",
        facecolor=BOARD_COLOR,
        edgecolor="#CC8800",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=LABEL_COLOR,
        zorder=4,
    )


def _draw_edge(ax, src, dst, is_bfs=False, node_size=0.055):
    """Draw an edge between two nodes, clipped to node boundaries."""
    x0, y0 = NODE_POS[src]
    x1, y1 = NODE_POS[dst]

    # Compute direction and clip start/end to node edge
    dx = x1 - x0
    dy = y1 - y0
    dist = np.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist

    x_start = x0 + ux * node_size
    y_start = y0 + uy * node_size
    x_end = x1 - ux * node_size
    y_end = y1 - uy * node_size

    color = WATER_SURFACE if is_bfs else GRID_COLOR
    lw = 2.2 if is_bfs else 0.9
    alpha = 1.0 if is_bfs else 0.5
    zorder = 2 if is_bfs else 1

    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        arrowprops=dict(
            arrowstyle="->" if is_bfs else "-",
            color=color,
            lw=lw,
        ),
        alpha=alpha,
        zorder=zorder,
    )


def generate(output_dir: Path) -> None:
    """Generate and save the BFS pose graph diagram.

    Args:
        output_dir: Directory where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    NODE_SIZE = 0.055

    # Draw all edges first (behind nodes)
    for src, dst in EDGES:
        is_bfs = (src, dst) in BFS_EDGES or (dst, src) in BFS_EDGES
        _draw_edge(ax, src, dst, is_bfs=is_bfs, node_size=NODE_SIZE)

    # Draw camera nodes
    for cam in CAMERA_NODES:
        _draw_camera_node(
            ax, NODE_POS[cam], cam, is_reference=(cam == "cam0"), node_size=NODE_SIZE
        )

    # Draw frame nodes
    for frame in FRAME_NODES:
        _draw_frame_node(ax, NODE_POS[frame], frame, node_size=NODE_SIZE)

    # Column headers
    ax.text(
        CAM_X,
        0.97,
        "Cameras",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=LABEL_COLOR,
    )
    ax.text(
        FRAME_X,
        0.97,
        "Board Frames",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=LABEL_COLOR,
    )

    # Title
    ax.set_title(
        "Stage 2: BFS Pose Graph\n(cameras linked through shared board observations)",
        fontsize=11,
        color=LABEL_COLOR,
        pad=6,
    )

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=CAMERA_COLOR, label="Camera node"),
        mpatches.Patch(
            facecolor=BOARD_COLOR, edgecolor="#CC8800", label="Board frame node"
        ),
        mpatches.Patch(facecolor=WATER_SURFACE, label="BFS traversal edge (directed)"),
        mpatches.Patch(
            facecolor=GRID_COLOR, alpha=0.6, label="Observation edge (undirected)"
        ),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        fontsize=8,
        framealpha=0.9,
        edgecolor=GRID_COLOR,
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
    )

    # Reference camera annotation
    ref_x, ref_y = NODE_POS["cam0"]
    ax.annotate(
        "reference\n(R=I, t=0)",
        xy=(ref_x - NODE_SIZE, ref_y),
        xytext=(ref_x - 0.18, ref_y),
        ha="right",
        va="center",
        fontsize=7.5,
        color=RAY_AIR,
        arrowprops=dict(arrowstyle="->", color=RAY_AIR, lw=1.2),
    )

    plt.tight_layout()
    output_path = Path(output_dir) / "bfs_pose_graph.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    output_dir = Path(__file__).parent.parent / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate(output_dir)
