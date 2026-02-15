"""Master diagram generation script.

This script imports and runs all diagram generators for the documentation site.
Called during Sphinx build via conf.py setup hook.
"""

from pathlib import Path

# Ensure matplotlib uses headless backend
import matplotlib

matplotlib.use("Agg")


def generate_all_diagrams():
    """Generate all documentation diagrams."""
    # Import diagram generators
    from coordinate_frames import generate as generate_coordinate_frames
    from ray_trace import generate as generate_ray_trace

    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent.parent / "_static" / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all diagrams
    print("Generating ray trace diagram...")
    generate_ray_trace(output_dir)

    print("Generating coordinate frames diagram...")
    generate_coordinate_frames(output_dir)

    print(f"All diagrams generated in {output_dir}")


if __name__ == "__main__":
    generate_all_diagrams()
