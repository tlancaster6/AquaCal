"""Master diagram generation script.

This script imports and runs all diagram generators for the documentation site.
Called during Sphinx build via conf.py setup hook.
"""

import sys
from pathlib import Path

# Ensure matplotlib uses headless backend
import matplotlib

matplotlib.use("Agg")

# Add docs/_static/scripts/ to path so shared scripts can be imported
_scripts_dir = Path(__file__).parent.parent.parent / "_static" / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


def generate_all_diagrams():
    """Generate all documentation diagrams."""
    # Import diagram generators (from docs/guide/_diagrams/)
    # Import new diagram generators (from docs/_static/scripts/)
    from bfs_pose_graph import generate as generate_bfs_pose_graph
    from coordinate_frames import generate as generate_coordinate_frames
    from hero_image import generate as generate_hero_image
    from ray_trace import generate as generate_ray_trace
    from sparsity_pattern import generate as generate_sparsity_pattern

    # Ensure output directories exist
    diagrams_dir = Path(__file__).parent.parent.parent / "_static" / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    static_dir = Path(__file__).parent.parent.parent / "_static"
    static_dir.mkdir(parents=True, exist_ok=True)

    # Generate guide/_diagrams/ outputs (saved to _static/diagrams/)
    print("Generating ray trace diagram...")
    generate_ray_trace(diagrams_dir)

    print("Generating coordinate frames diagram...")
    generate_coordinate_frames(diagrams_dir)

    # Generate _static/scripts/ outputs
    print("Generating sparsity pattern diagram...")
    generate_sparsity_pattern(diagrams_dir)

    print("Generating BFS pose graph diagram...")
    generate_bfs_pose_graph(diagrams_dir)

    print("Generating hero image...")
    generate_hero_image(static_dir)

    print(f"All diagrams generated in {diagrams_dir}")


if __name__ == "__main__":
    generate_all_diagrams()
