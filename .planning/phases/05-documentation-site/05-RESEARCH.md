# Phase 5: Documentation Site - Research

**Researched:** 2026-02-14
**Domain:** Sphinx documentation, Read the Docs hosting, technical writing for scientific Python
**Confidence:** HIGH

## Summary

Phase 5 requires building a comprehensive documentation site with auto-generated API reference, three theory pages (refractive geometry, coordinate conventions, optimizer pipeline), a conceptual overview page, and citation infrastructure. The standard stack is Sphinx with the Furo theme, napoleon extension for Google-style docstrings, and Read the Docs for hosting. MyST Markdown is the recommended markup format for manual pages due to its familiarity and ease of contribution, though this is at Claude's discretion. Matplotlib diagrams should be generated at build time using actual codebase functions (importing from aquacal modules) to ensure accuracy. The existing minimal Sphinx setup provides a foundation; tasks will focus on content creation, autodoc configuration, diagram generation scripts, and Read the Docs deployment configuration.

**Primary recommendation:** Use MyST Markdown for theory/guide pages (easier to write/review), keep RST for API reference sections if needed for autodoc compatibility, generate diagrams with matplotlib at build time via custom Python scripts that import actual AquaCal functions, and deploy to Read the Docs with `.readthedocs.yaml` configuration.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Theory page depth & style:**
- Practical guide level — enough math to understand what the code does, focus on intuition over full derivations
- Key equations rendered with MathJax, supported by diagrams
- Diagrams generated with matplotlib at build time, using actual codebase functions where possible (e.g., import projection functions to draw accurate ray traces)
- Simple flow/architecture diagrams use ASCII/text in source
- Each theory page is self-contained — users can jump to any topic without reading others first, with brief recaps where needed
- Add a conceptual overview page: "What is refractive calibration and why do you need it?" with links into the three detail pages
- Include inline "gotcha" callouts (Sphinx admonitions) for common mistakes (e.g., wrong coordinate convention, interface_distance misunderstanding)

**Site structure & navigation:**
- Organized by audience need: Overview | User Guide (theory pages) | API Reference | Contributing
- Rich landing page with feature highlights, a code snippet showing basic usage, badges, and section links
- Reserve a "Tutorials" placeholder section in the sidebar with a "Coming soon" note for Phase 6 notebook integration

**API reference approach:**
- Complete docstrings for all public API functions/classes; internal/private items can be sparse
- Cross-link API docs to relevant theory pages (e.g., `refractive_project()` links to refractive geometry explanation)
- Key functions include brief (3-5 line) usage examples in docstrings

**Tone & audience:**
- Tiered content: overview and quickstart accessible to broader scientific Python users; theory pages assume some calibration background
- Professional but approachable tone — clear, direct prose like scikit-learn docs
- Briefly explain OpenCV conventions when referenced (don't assume prior knowledge)
- Style reference: scikit-learn documentation

**Sphinx theme:**
- Use Furo theme

### Claude's Discretion

- Markup format (RST vs MyST Markdown) — pick what works best for the content
- API reference layout (grouped by functionality vs one page per module) — pick what fits the codebase
- Exact sidebar ordering and page titles

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Sphinx | ≥6.2 (latest stable as of 2026) | Documentation generator | De facto standard for Python projects; powerful autodoc, cross-referencing, and extensibility |
| sphinx.ext.autodoc | Built-in | Auto-generate API docs from docstrings | Core Sphinx extension, handles Python introspection |
| sphinx.ext.napoleon | Built-in | Parse Google/NumPy style docstrings | AquaCal uses Google style; napoleon converts to RST internally |
| sphinx.ext.viewcode | Built-in | Add links to source code | Standard practice for API docs, helps users find implementation |
| sphinx.ext.intersphinx | Built-in | Cross-link to external docs (NumPy, SciPy, OpenCV) | Allows linking to standard library types in docstrings |
| Furo | Latest (actively maintained) | Clean, responsive Sphinx theme | User-specified; modern, customizable, excellent UX |
| Read the Docs | Platform | Documentation hosting | Industry standard for open-source Python; free, automatic builds on commit |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| myst-parser | Latest | Parse MyST Markdown in Sphinx | If using Markdown for manual pages (recommended for ease of contribution) |
| sphinx.ext.mathjax | Built-in | Render LaTeX math equations | Theory pages need Snell's law equations, vector math |
| matplotlib | Already in dependencies | Generate diagrams at build time | User-specified requirement; import actual functions to draw accurate ray diagrams |
| sphinx-copybutton | Optional | Add copy button to code blocks | Quality-of-life for code examples; widely used |
| sphinx-design | Optional | Cards, tabs, grids for rich layouts | For feature highlights on landing page; modern look |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MyST Markdown | Pure RST | RST has richer directive support but steeper learning curve; harder for casual contributors. MyST combines Markdown familiarity with Sphinx power. |
| Furo theme | sphinx_rtd_theme (Read the Docs default) | RTD theme is widely recognized but less customizable and modern. Furo has better mobile support, cleaner design, and active development. User chose Furo. |
| Manual diagrams | Pre-rendered PNG/SVG | Build-time generation ensures diagrams stay in sync with code; changes to projection logic automatically update diagrams. |

**Installation:**

Existing `docs/conf.py` already has `sphinx.ext.autodoc` and `sphinx.ext.napoleon`. Need to add:

```bash
pip install myst-parser sphinx-copybutton sphinx-design furo
```

And update `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "sphinx>=6.2",
    "furo",
    "myst-parser",
    "sphinx-copybutton",
    "sphinx-design",
]
```

---

## Architecture Patterns

### Recommended Project Structure

Based on scikit-learn and standard Sphinx layouts:

```
docs/
├── conf.py                  # Sphinx configuration
├── index.rst or index.md    # Landing page (MyST or RST)
├── overview.md              # "What is refractive calibration?" (new)
├── guide/                   # User guide (theory pages)
│   ├── index.md             # Guide landing page with links
│   ├── refractive_geometry.md   # Theory page 1 (THEO-01)
│   ├── coordinates.md           # Theory page 2 (THEO-02)
│   ├── optimizer.md             # Theory page 3 (THEO-03)
│   └── _diagrams/               # Build-time diagram generation scripts
│       ├── generate_all.py      # Master script to generate all diagrams
│       ├── ray_trace.py         # Import aquacal.core.refractive_geometry
│       ├── coordinate_frames.py # Draw world/camera/board frames
│       └── pipeline_flow.py     # ASCII or simple matplotlib pipeline diagram
├── api/                     # API reference
│   ├── index.rst            # API landing page (autodoc summary tables)
│   ├── core.rst             # aquacal.core subpackage
│   ├── calibration.rst      # aquacal.calibration subpackage
│   ├── config.rst           # aquacal.config (schema, types)
│   ├── io.rst               # aquacal.io
│   ├── validation.rst       # aquacal.validation
│   └── triangulation.rst    # aquacal.triangulation
├── tutorials/               # Placeholder for Phase 6
│   └── index.md             # "Coming soon" stub
├── contributing.md          # Link to CONTRIBUTING.md or include directly
├── changelog.md             # Link to CHANGELOG.md
├── _static/                 # Static assets (logos, CSS overrides)
│   └── diagrams/            # Generated diagrams saved here at build time
└── _templates/              # Custom Jinja templates if needed
```

**Key decisions:**
- Theory pages in `guide/` subdirectory for clarity
- API reference organized by subpackage (mirrors codebase structure)
- Diagram generation scripts in `docs/guide/_diagrams/` — run at Sphinx build time
- MyST Markdown (`.md`) for manual pages, RST (`.rst`) for API reference where autodoc is heavy

### Pattern 1: Build-Time Diagram Generation

**What:** Execute Python scripts during Sphinx build to generate matplotlib diagrams that import actual AquaCal functions.

**When to use:** For diagrams that visualize code behavior (ray tracing, projection geometry).

**Example:**

`docs/guide/_diagrams/ray_trace.py`:
```python
"""Generate refractive ray tracing diagram for theory page."""

import matplotlib.pyplot as plt
import numpy as np

# Import actual AquaCal functions to ensure accuracy
from aquacal.core.refractive_geometry import snells_law_3d, trace_ray_air_to_water
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface

def generate_ray_diagram(output_path):
    """Draw camera -> interface -> target ray with actual Snell's law."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Camera at origin, interface at Z=0.5, target at Z=1.2
    # Use actual snells_law_3d to compute refracted ray
    # ...draw camera, rays, interface line, annotations...

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    generate_ray_diagram("../../_static/diagrams/ray_trace.png")
```

`docs/conf.py`:
```python
import subprocess
import sys
from pathlib import Path

# Run diagram generation scripts before build
def generate_diagrams(app, config):
    diagrams_dir = Path(__file__).parent / "guide/_diagrams"
    for script in diagrams_dir.glob("*.py"):
        if script.name != "generate_all.py":
            subprocess.run([sys.executable, str(script)], check=True)

def setup(app):
    app.connect('config-inited', generate_diagrams)
```

Reference in theory page:
```markdown
## Refractive Ray Tracing

![Snell's law ray path](../_static/diagrams/ray_trace.png)
```

**Why this works:** Diagrams always match code. If `snells_law_3d` logic changes, next doc build regenerates accurate diagrams. Reviewers can verify diagram code imports the right functions.

### Pattern 2: MyST Markdown for Manual Pages

**What:** Use MyST Markdown for theory pages, guides, and narrative documentation.

**When to use:** All non-API pages (overview, theory, contributing).

**Example:**

`docs/guide/refractive_geometry.md`:
```markdown
# Refractive Geometry

## Snell's Law in 3D

The core of AquaCal's projection model is Snell's law applied in three dimensions:

$$
n_1 \sin \theta_1 = n_2 \sin \theta_2
$$

where $n_1 = 1.0$ (air), $n_2 = 1.333$ (water), and $\theta$ is the angle
from the surface normal.

:::{admonition} Common Pitfall
:class: warning

The interface normal `[0, 0, -1]` points **up** (from water toward air),
not down. This is opposite to the Z-axis direction.
:::

See {func}`aquacal.core.refractive_geometry.snells_law_3d` for implementation.
```

**Why MyST:** Easier to write and review than RST. Admonitions for gotchas (`:::{admonition}`). Cross-references work (`{func}`). Math via `$$` (MathJax). Source: [MyST vs RST guide](https://www.pyopensci.org/python-package-guide/documentation/hosting-tools/myst-markdown-rst-doc-syntax.html), [Read the Docs migration guide](https://docs.readthedocs.com/platform/stable/guides/migrate-rest-myst.html).

### Pattern 3: Autodoc with Napoleon for API Reference

**What:** Use `.. automodule::` directives to auto-generate API docs from Google-style docstrings.

**When to use:** All API reference pages.

**Example:**

`docs/api/core.rst`:
```rst
Core Geometry (`aquacal.core`)
================================

.. automodule:: aquacal.core.refractive_geometry
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aquacal.core.camera
   :members:
   :undoc-members:

.. automodule:: aquacal.core.interface_model
   :members:
```

AquaCal already uses Google-style docstrings (verified in `refractive_geometry.py`, `schema.py`). Napoleon extension converts these to RST for Sphinx processing. Source: [Napoleon documentation](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).

### Pattern 4: Intersphinx for External Links

**What:** Link to external library docs (NumPy, SciPy, OpenCV) in type hints and docstrings.

**Configuration:**

`docs/conf.py`:
```python
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
}
```

**Usage:** Type hints like `NDArray[np.float64]` automatically link to NumPy docs. Source: [Intersphinx extension](https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html).

### Pattern 5: Furo Theme Customization

**What:** Configure Furo theme for clean, modern look with light/dark mode.

**Configuration:**

`docs/conf.py`:
```python
html_theme = "furo"

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0066cc",  # AquaCal brand color
        "color-brand-content": "#0066cc",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

html_title = "AquaCal"
html_short_title = "AquaCal"
# html_logo = "_static/logo.png"  # Optional
```

Furo features:
- Responsive design (mobile-friendly)
- Light/dark mode toggle (automatic)
- Customizable CSS variables
- Clean sidebar navigation

Source: [Furo customization docs](https://pradyunsg.me/furo/customisation/), [Furo GitHub](https://github.com/pradyunsg/furo).

### Anti-Patterns to Avoid

- **Hand-writing API docs:** Don't duplicate docstrings in RST files. Use autodoc to pull from source.
- **Static diagrams:** Pre-rendered PNGs fall out of sync with code. Generate at build time.
- **Mixing MyST and RST in same page:** Choose one format per file. MyST for guides, RST for API reference.
- **Deep nesting in sidebar:** Keep navigation shallow (2-3 levels max). Users should find pages in ≤2 clicks.
- **Overly formal tone:** Aim for scikit-learn style: clear, helpful, approachable. Avoid academic paper tone.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Diagram generation at build time | Custom Sphinx extension | `subprocess.run()` in `conf.py` `setup()` hook | Sphinx build hooks are simple and well-documented. Custom extensions add complexity. |
| Math rendering | Convert LaTeX to images manually | `sphinx.ext.mathjax` (built-in) | MathJax renders LaTeX client-side; no build overhead, crisp on all displays. |
| Code example syntax highlighting | Manual HTML/CSS | Sphinx's built-in Pygments integration | Automatic language detection, 100+ languages, customizable themes. |
| Cross-references between pages | Hardcoded links | Sphinx roles (`:func:`, `:class:`, `:ref:`) | Automatic validation, broken link warnings, hover tooltips. |
| Responsive theme | Custom CSS grid | Furo (or sphinx_rtd_theme) | Mobile support, accessibility, dark mode are complex. Use tested themes. |

**Key insight:** Sphinx has 15+ years of ecosystem development. Almost every documentation need has a standard solution. Custom code adds maintenance burden and breaks tooling expectations.

---

## Common Pitfalls

### Pitfall 1: Autodoc Fails to Import Module

**What goes wrong:** `autodoc` can't find `aquacal` package during build; API reference shows "Module not found" errors.

**Why it happens:** Sphinx runs in `docs/` directory; Python path doesn't include `src/aquacal` by default.

**How to avoid:** Install package in editable mode or add to `sys.path` in `conf.py`:

```python
import sys
from pathlib import Path

# Add src/ to Python path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
```

**Warning signs:** Build output shows `WARNING: autodoc: failed to import module 'aquacal.core.camera'`.

**Source:** [Simon Willison's TIL on autodoc setup](https://til.simonwillison.net/sphinx/sphinx-autodoc), [Sphinx autodoc docs](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html).

### Pitfall 2: MyST Markdown Not Rendering Directives

**What goes wrong:** MyST admonitions, code blocks with options, or cross-references don't render; appear as raw text.

**Why it happens:** `myst_parser` extension not installed or not added to `extensions` list in `conf.py`.

**How to avoid:** Ensure `myst_parser` is in `extensions`:

```python
extensions = [
    "sphinx.ext.autodoc",
    "myst_parser",  # Must be present
    # ...
]
```

**Warning signs:** Admonitions like `:::{note}` render as literal colons and braces.

**Source:** [MyST Parser documentation](https://myst-parser.readthedocs.io/), [Sphinx Markdown usage](https://www.sphinx-doc.org/en/master/usage/markdown.html).

### Pitfall 3: Read the Docs Build Fails (Missing Dependencies)

**What goes wrong:** Local builds succeed, but Read the Docs builds fail with `ModuleNotFoundError`.

**Why it happens:** Read the Docs builds in a clean environment. If `aquacal` imports `numpy`, `scipy`, etc., those must be installed during RTD build.

**How to avoid:** Create `.readthedocs.yaml` and specify installation method:

```yaml
version: 2
build:
  os: ubuntu-24.04
  tools:
    python: "3.10"
sphinx:
  configuration: docs/conf.py
python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

This installs `aquacal[docs]` (includes Sphinx, Furo, etc.) before building docs. Autodoc can then import the package.

**Warning signs:** RTD build log shows `ImportError: No module named 'numpy'` or similar.

**Source:** [RTD configuration reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html), [RTD Sphinx deployment guide](https://docs.readthedocs.com/platform/stable/intro/sphinx.html), [RTD example config](https://github.com/readthedocs-examples/example-sphinx-basic/blob/main/.readthedocs.yaml).

### Pitfall 4: Diagram Generation Scripts Fail in RTD Environment

**What goes wrong:** Diagrams generate locally but RTD build fails with matplotlib errors (missing display, font issues).

**Why it happens:** RTD runs in headless environment; matplotlib defaults to GUI backend.

**How to avoid:** Force non-interactive backend in diagram scripts:

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

Or set `MPLBACKEND=Agg` environment variable in `.readthedocs.yaml`:

```yaml
build:
  os: ubuntu-24.04
  tools:
    python: "3.10"
  commands:
    - export MPLBACKEND=Agg
    - cd docs && python guide/_diagrams/generate_all.py
```

**Warning signs:** RTD build log shows `_tkinter.TclError: no display name and no $DISPLAY environment variable`.

**Source:** [Matplotlib documentation tips](https://matplotlib.org/stable/devel/document.html), common RTD/CI pattern.

### Pitfall 5: Cross-References to Code Break

**What goes wrong:** Links like `:func:`aquacal.core.refractive_geometry.snells_law_3d`` don't resolve; Sphinx warns "undefined label."

**Why it happens:** Autodoc hasn't generated target for that symbol (module not imported, or symbol not public).

**How to avoid:**
1. Ensure module is in autodoc (check `api/*.rst` includes `.. automodule:: aquacal.core.refractive_geometry`)
2. Verify symbol is in `__all__` or is a top-level class/function
3. Use full path: `:func:`aquacal.core.refractive_geometry.snells_law_3d`` not `:func:`snells_law_3d``

**Warning signs:** Build output shows `WARNING: undefined label: aquacal.core.refractive_geometry.snells_law_3d`.

---

## Code Examples

Verified patterns from official sources:

### Landing Page with Badges and Feature Highlights

`docs/index.md` (MyST Markdown):
```markdown
# AquaCal Documentation

![Build](https://img.shields.io/github/actions/workflow/status/tlancaster6/AquaCal/test.yml?branch=main&label=build)
![Coverage](https://img.shields.io/codecov/c/github/tlancaster6/AquaCal)
![PyPI](https://img.shields.io/pypi/v/aquacal)

Refractive multi-camera calibration library for underwater arrays.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} What is AquaCal?
:link: overview
:link-type: doc

Learn about refractive calibration and when you need it
:::

:::{grid-item-card} User Guide
:link: guide/index
:link-type: doc

Theory pages covering geometry, coordinates, and optimization
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Complete API documentation with examples
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Step-by-step guides (coming soon)
:::
::::

## Quick Start

\`\`\`python
from aquacal import run_calibration, load_calibration

# Run calibration
run_calibration("config.yaml")

# Load results
result = load_calibration("output/calibration.json")
pixel = result.project("cam0", [0.1, 0.05, 0.4])
\`\`\`
```

Uses `sphinx-design` for grid cards. Source: [Sphinx Design docs](https://sphinx-design.readthedocs.io/).

### Theory Page with Math and Admonitions

`docs/guide/refractive_geometry.md`:
```markdown
# Refractive Geometry

## Snell's Law

The refracted ray direction is computed using the 3D vector form of Snell's law:

$$
\mathbf{t} = \eta \mathbf{d} + (\cos \theta_t - \eta \cos \theta_i) \mathbf{n}
$$

where $\eta = n_1 / n_2$ is the refractive index ratio.

:::{admonition} Common Pitfall
:class: warning

The interface normal `[0, 0, -1]` points **from water toward air** (upward),
opposite to the +Z axis direction. Don't flip it manually.
:::

Implementation: {func}`aquacal.core.refractive_geometry.snells_law_3d`

## Ray Tracing

![Refractive ray path](../_static/diagrams/ray_trace.png)

The ray travels from camera $C$ through interface point $P$ to target $Q$.
Finding $P$ requires solving a 1D root-finding problem.
```

Math via `$$` (MathJax). Admonitions with `:::{admonition}`. Cross-ref with `{func}`. Source: [MyST syntax guide](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html).

### Autodoc with Custom Grouping

`docs/api/core.rst`:
```rst
Core Geometry
=============

Refractive Projection
---------------------

.. autofunction:: aquacal.core.refractive_geometry.refractive_project

.. autofunction:: aquacal.core.refractive_geometry.refractive_project_batch

Snell's Law
-----------

.. autofunction:: aquacal.core.refractive_geometry.snells_law_3d

.. autofunction:: aquacal.core.refractive_geometry.trace_ray_air_to_water

Camera Models
-------------

.. autoclass:: aquacal.core.camera.Camera
   :members:
   :undoc-members:

.. autoclass:: aquacal.core.camera.FisheyeCamera
   :members:
   :show-inheritance:
```

Manual grouping (better than flat `automodule`) for large modules. Source: Autodoc directive reference.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| sphinx_rtd_theme | Furo, Book, PyData themes | 2020-2023 | Modern themes have better mobile support, dark mode, customization |
| Pure RST | MyST Markdown | 2019+ | Easier contribution; Markdown familiarity lowers barrier |
| Manual examples | Sphinx-Gallery | 2016+ | Jupyter notebook integration, auto-generated galleries (Phase 6) |
| MathJax v2 | MathJax v4 | Sphinx 9.0 (2024) | Faster rendering, better LaTeX support |
| requirements.txt for RTD | .readthedocs.yaml | 2018+ | Explicit build config, versioned, validates before build |

**Deprecated/outdated:**
- `sphinx_rtd_theme` still works but less actively developed than Furo/PyData
- RST-only projects: MyST now mature enough for production use
- Hardcoded Python path in `conf.py`: Better to install package in RTD config

---

## Open Questions

1. **Should API reference be single-page or multi-page?**
   - What we know: AquaCal has 6 subpackages (`core`, `calibration`, `config`, `io`, `validation`, `triangulation`). Single-page would be 1000+ lines.
   - What's unclear: User preference for navigability vs. search-ability.
   - Recommendation: Multi-page (one per subpackage) with a landing page that has summary tables. Easier navigation, clearer sidebar.

2. **Include examples in docstrings or separate examples page?**
   - What we know: User decision says "key functions include brief (3-5 line) usage examples in docstrings."
   - What's unclear: Should there also be a separate "Examples" page with longer workflows?
   - Recommendation: Follow user decision (examples in docstrings). Defer longer examples to Phase 6 tutorials.

3. **ASCII diagrams vs. matplotlib for simple flows?**
   - What we know: User said "simple flow/architecture diagrams use ASCII/text in source."
   - What's unclear: Which diagrams qualify as "simple"?
   - Recommendation: Pipeline flow (4 stages linear) = ASCII. Ray geometry (3D spatial) = matplotlib. Coordinate frames (3D axes) = matplotlib.

---

## Sources

### Primary (HIGH confidence)

- [Sphinx Official Documentation](https://www.sphinx-doc.org/) - Core tool documentation
- [Furo Theme Documentation](https://pradyunsg.me/furo/) - User-chosen theme
- [MyST Parser Documentation](https://myst-parser.readthedocs.io/) - Markdown for Sphinx
- [Read the Docs Configuration Reference](https://docs.readthedocs.com/platform/stable/config-file/v2.html) - `.readthedocs.yaml` spec
- [Napoleon Extension Docs](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) - Google/NumPy docstring support
- [Autodoc Extension Docs](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) - API generation
- [MathJax Extension Docs](https://www.sphinx-doc.org/en/master/usage/extensions/math.html) - Math rendering
- [CITATION.cff Format Specification](https://citation-file-format.github.io/) - Citation file standard
- [GitHub CITATION.cff Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) - GitHub integration

### Secondary (MEDIUM confidence)

- [MyST vs RST Comparison (PyOpenSci Guide)](https://www.pyopensci.org/python-package-guide/documentation/hosting-tools/myst-markdown-rst-doc-syntax.html) - Markup format tradeoffs
- [Read the Docs Migration Guide (MyST)](https://docs.readthedocs.com/platform/stable/guides/migrate-rest-myst.html) - MyST adoption guide
- [Sphinx-Gallery Configuration](https://sphinx-gallery.github.io/stable/configuration.html) - Gallery setup (Phase 6 reference)
- [Scikit-learn Sphinx Configuration](https://github.com/scikit-learn/scikit-learn/blob/main/doc/conf.py) - Style reference example
- [Matplotlib Build-Time Plots](https://matplotlib.org/stable/devel/document.html) - Diagram generation patterns
- [Simon Willison's Autodoc TIL](https://til.simonwillison.net/sphinx/sphinx-autodoc) - Common setup issues
- [Read the Docs Sphinx Example](https://github.com/readthedocs-examples/example-sphinx-basic) - Minimal working config

### Tertiary (LOW confidence)

- [Sphinx Discussion on Markdown vs RST](https://discuss.python.org/t/sphinx-rest-vs-markdown/2470) - Community opinions (dated 2020, but directional)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Sphinx, Furo, napoleon, autodoc are industry standard; verified in official docs
- Architecture: HIGH - Patterns verified from official examples and major projects (scikit-learn, matplotlib)
- Pitfalls: MEDIUM - Based on common issues in Sphinx/RTD forums and personal experience; not exhaustively verified
- Diagram generation: MEDIUM - matplotlib build-time generation is documented but less common than static images; subprocess approach is straightforward but not official Sphinx pattern

**Research date:** 2026-02-14
**Valid until:** ~60 days (Sphinx ecosystem is stable; major changes rare)
