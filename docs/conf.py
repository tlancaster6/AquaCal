"""Sphinx configuration for AquaCal documentation."""

import subprocess
import sys
from pathlib import Path

# Add project source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "AquaCal"
copyright = "2024, Tucker Lancaster"
author = "Tucker Lancaster"
release = "1.0.3"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
]

# Templates and static files
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# nbsphinx configuration
nbsphinx_execute = "never"  # Use committed outputs, don't re-execute
nbsphinx_allow_errors = False
nbsphinx_requirejs_path = ""  # Avoid RequireJS conflicts

# HTML output
html_theme = "furo"
html_title = "AquaCal"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"


def setup(app):
    """Set up build-time hooks."""

    def run_diagram_generation(app, config):
        """Generate diagrams during build."""
        diagrams_script = (
            Path(__file__).parent / "guide" / "_diagrams" / "generate_all.py"
        )
        if diagrams_script.exists():
            # Set matplotlib to headless mode for diagram generation
            import matplotlib

            matplotlib.use("Agg")

            try:
                subprocess.run(
                    [sys.executable, str(diagrams_script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Diagram generation failed: {e.stderr}")

    app.connect("config-inited", run_diagram_generation)
