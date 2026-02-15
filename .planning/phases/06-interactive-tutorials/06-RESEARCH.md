# Phase 6: Image Input Support, Interactive Tutorials & README Overhaul - Research

**Researched:** 2026-02-15
**Domain:** Jupyter notebook tutorials, image loading abstraction, documentation presentation
**Confidence:** HIGH

## Summary

Phase 6 delivers three parallel capabilities: (1) abstracted frame loading via FrameSet interface supporting both video files and image directories, (2) Jupyter notebook tutorials demonstrating end-to-end workflows integrated into Sphinx docs via nbsphinx, and (3) a concise README with visual assets linking to detailed documentation. The codebase already has strong foundations: VideoSet class for video loading, comprehensive datasets API (synthetic + example data), Sphinx + Furo documentation site, and matplotlib-based diagram generation. The research confirms that the standard Python ecosystem provides robust tools for all three capabilities without requiring custom solutions.

**Primary recommendation:** Use nbsphinx 0.9.8+ for notebook integration (already standard in scientific Python docs), create abstract FrameSet protocol with ImageSet/VideoSet implementations using pathlib + natural sorting, and generate hero visual from existing ray trace diagram code for README impact.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Tutorial content & depth:**
- Audience: researchers who know Python and OpenCV basics but are new to AquaCal
- Each notebook is self-contained — no required reading order, some setup repetition is acceptable
- Failure modes covered via inline warning callouts where relevant, not separate sections
- Data source: runtime-toggleable — cell near top lets user switch between synthetic, in-package preset, or Zenodo download
- Rich visualization emphasis: 3D camera rig plots, ray traces, error heatmaps, before/after comparisons
- Diagnostics notebook covers reprojection error analysis, parameter convergence, AND 3D rig visualization
- Synthetic validation notebook focuses on refractive vs non-refractive comparison (set n_air=n_water=1.0) — interactive extension of tests/synthetic/experiments.py, showing users first-hand why refractive calibration matters

**Image input behavior:**
- Auto-detect input type: directory = images, file = video — no config flag needed
- Supported image formats: JPEG and PNG only
- Image ordering: alphabetical/natural sort by filename
- No subsampling for image directories — assumed pre-curated; subsampling only applies to video

**README structure & visuals:**
- Hero visual: ray trace visualization (Snell's law refraction diagram — immediately communicates "refractive")
- Quick start: 3-step minimal (pip install, generate config, run calibrate)
- Short feature list: 4-5 bullets covering key capabilities
- Citation: brief one-liner in README with link to docs for full BibTeX and details
- Bulk content (CLI reference, config reference, methodology, output details) moves to docs with links

**Notebook integration:**
- Pre-rendered outputs committed with notebooks — fast doc builds, curated outputs, no build-time deps
- Notebooks live in docs/tutorials/ — picked up naturally by nbsphinx
- "Tutorials" as top-level section in docs sidebar alongside Theory, API Reference
- Google Colab badge on each notebook for interactive cloud use

### Claude's Discretion

- Narrative verbosity per notebook — Claude picks appropriate balance of explanation vs code
- Exact matplotlib styling and plot layouts
- Cell structure and markdown formatting within notebooks
- How the data source toggle is implemented (config dict, enum, etc.)

### Deferred Ideas (OUT OF SCOPE)

- Dedicated docs page proving necessity of refractive model (when it matters vs when approximation is good enough) using experiment output plots — new documentation capability beyond Phase 6 scope
- TIFF and other scientific image format support — keep it simple with JPEG+PNG for now

</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nbsphinx | 0.9.8+ | Jupyter notebook integration with Sphinx | Standard in scientific Python docs (NumPy, SciPy, pandas); auto-executes notebooks with outputs, supports MyST markdown |
| pathlib | stdlib | Object-oriented filesystem paths | Python 3.10+ standard for file operations; replaces os.path in modern code |
| natsort | 8.4.0+ | Natural sorting of filenames | Standard for filename sorting with embedded numbers (image_001.jpg < image_010.jpg); robust cross-platform support |
| matplotlib | (existing) | Static + interactive plots in notebooks | Already in dependencies; ipympl backend enables widget-based interactivity in notebooks |
| opencv-python | >=4.6 (existing) | Image loading (cv2.imread) | Already used throughout codebase; handles JPEG/PNG natively |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ipywidgets | latest | Interactive notebook widgets | Optional enhancement for data source toggles; not required for basic functionality |
| sphinx-copybutton | (existing) | Copy button for code blocks | Already in docs dependencies; improves notebook code cell UX in rendered HTML |
| sphinx-design | (existing) | Grid layouts, cards, dropdowns | Already in docs dependencies; useful for organizing notebook sections |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| nbsphinx | MyST-NB | MyST-NB requires markdown-based notebooks (.md); nbsphinx works with standard .ipynb which is more familiar to users |
| natsort | Manual sorting with regex | natsort handles edge cases (version strings, unicode) and is well-tested; manual sorting is error-prone |
| pathlib.suffix check | imghdr module | imghdr was deprecated in Python 3.11 and removed in 3.13; extension check is simpler and sufficient |

**Installation:**

nbsphinx and natsort are the only new dependencies:

```bash
# Add to pyproject.toml [project.optional-dependencies] docs section:
nbsphinx>=0.9.8
natsort>=8.4.0
```

## Architecture Patterns

### Recommended Project Structure

```
src/aquacal/io/
├── video.py            # Existing VideoSet class
├── images.py           # NEW: ImageSet class (mirrors VideoSet API)
├── frameset.py         # NEW: FrameSet protocol (abstract interface)
└── detection.py        # MODIFY: accept FrameSet instead of VideoSet

docs/
├── tutorials/
│   ├── index.md                     # Tutorial landing page
│   ├── 01_full_pipeline.ipynb       # End-to-end calibration
│   ├── 02_diagnostics.ipynb         # Visualization & diagnostics
│   └── 03_synthetic_validation.ipynb # Refractive comparison
├── _static/
│   └── hero_ray_trace.png           # README hero visual
└── conf.py                          # MODIFY: add nbsphinx extension

.github/workflows/
└── docs.yml                         # MODIFY: install nbsphinx dep
```

### Pattern 1: FrameSet Protocol (Duck Typing)

**What:** Abstract interface defining common API for VideoSet and ImageSet

**When to use:** When detection pipeline needs to work with either videos or image directories without isinstance checks

**Example:**

```python
# src/aquacal/io/frameset.py
from typing import Protocol, Iterator
from numpy.typing import NDArray
import numpy as np

class FrameSet(Protocol):
    """Protocol for frame sources (videos or image directories).

    Implementations must provide:
    - camera_names: sorted list of camera identifiers
    - frame_count: total number of synchronized frames
    - iterate_frames(start, stop, step): iterator over frames
    - Context manager protocol (__enter__, __exit__)
    """

    @property
    def camera_names(self) -> list[str]:
        """List of camera names (sorted for deterministic ordering)."""
        ...

    @property
    def frame_count(self) -> int:
        """Total number of synchronized frames."""
        ...

    def iterate_frames(
        self,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, dict[str, NDArray[np.uint8] | None]]]:
        """Iterate over synchronized frames."""
        ...

    def __enter__(self) -> "FrameSet":
        """Context manager entry."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        ...
```

**Why:** VideoSet already implements this API; ImageSet can mirror it without inheritance. Detection pipeline works with any FrameSet-compatible object via duck typing.

### Pattern 2: ImageSet Natural Sorting

**What:** Load images from directory with natural filename ordering

**When to use:** When config path points to directory instead of video file

**Example:**

```python
# src/aquacal/io/images.py
from pathlib import Path
from natsort import natsorted

class ImageSet:
    """Frame source from image directory (JPEG/PNG).

    Images are sorted naturally by filename to determine frame order.
    All cameras must have the same number of images.
    """

    def __init__(self, image_dirs: dict[str, str]) -> None:
        """Initialize ImageSet from camera directories.

        Args:
            image_dirs: Dict mapping camera_name to directory path

        Raises:
            ValueError: If directories have different image counts
            FileNotFoundError: If directory doesn't exist
        """
        self._image_paths: dict[str, list[Path]] = {}

        for cam_name, dir_path in image_dirs.items():
            dir_p = Path(dir_path)
            if not dir_p.is_dir():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            # Collect JPEG/PNG files with natural sort
            images = []
            for pattern in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG"]:
                images.extend(dir_p.glob(pattern))

            # Natural sort by filename
            images = natsorted(images, key=lambda p: p.name)

            if not images:
                raise ValueError(f"No JPEG/PNG images found in {dir_path}")

            self._image_paths[cam_name] = images

        # Validate synchronized frame count
        counts = [len(paths) for paths in self._image_paths.values()]
        if len(set(counts)) > 1:
            raise ValueError(
                f"Image directories have different counts: {dict(zip(self._image_paths.keys(), counts))}"
            )

        self._frame_count = min(counts) if counts else 0
```

**Source:** [natsort documentation](https://natsort.readthedocs.io/), [pathlib documentation](https://docs.python.org/3/library/pathlib.html)

### Pattern 3: Auto-Detection in Config Loading

**What:** Detect video file vs image directory without explicit config flag

**When to use:** When loading paths from config YAML; transparently support both input types

**Example:**

```python
# src/aquacal/config/__init__.py (modify existing load_config)
from pathlib import Path
from aquacal.io.video import VideoSet
from aquacal.io.images import ImageSet
from aquacal.io.frameset import FrameSet

def _create_frame_source(paths: dict[str, str]) -> FrameSet:
    """Create VideoSet or ImageSet based on path types.

    Auto-detects: directory = images, file = video.
    All cameras must use same input type.
    """
    path_objs = {cam: Path(p) for cam, p in paths.items()}

    # Check first camera's path type
    first_path = next(iter(path_objs.values()))
    if first_path.is_dir():
        input_type = "directory"
    elif first_path.is_file():
        input_type = "file"
    else:
        raise FileNotFoundError(f"Path not found: {first_path}")

    # Validate all cameras use same type
    for cam, p in path_objs.items():
        is_dir = p.is_dir()
        is_file = p.is_file()
        if input_type == "directory" and not is_dir:
            raise ValueError(f"Mixed input types: {cam} is not a directory")
        if input_type == "file" and not is_file:
            raise ValueError(f"Mixed input types: {cam} is not a file")

    if input_type == "directory":
        return ImageSet(paths)
    else:
        return VideoSet(paths)
```

### Pattern 4: Jupyter Notebook Data Source Toggle

**What:** Runtime-configurable data source (synthetic/preset/Zenodo) via notebook cell

**When to use:** At top of each tutorial notebook for user flexibility

**Example:**

```python
# docs/tutorials/01_full_pipeline.ipynb - Cell 1
"""
# Full Pipeline Tutorial

Choose your data source (edit DATA_SOURCE below):
- "synthetic": Generate on-the-fly (fast, no download)
- "preset": Small included dataset (2 cameras, 10 frames)
- "zenodo": Real rig dataset (13 cameras, requires download)
"""

DATA_SOURCE = "preset"  # EDIT THIS LINE

# Load data based on selection
from aquacal.datasets import load_example, generate_synthetic_rig

if DATA_SOURCE == "synthetic":
    scenario = generate_synthetic_rig("small")
    dataset = None  # Will generate detections inline
elif DATA_SOURCE == "preset":
    dataset = load_example("small")
    scenario = dataset.ground_truth
elif DATA_SOURCE == "zenodo":
    dataset = load_example("real-rig")  # Auto-downloads if needed
    scenario = dataset.ground_truth
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

print(f"Using {DATA_SOURCE} data source")
```

**Why:** Simple string variable is more accessible than ipywidgets for notebook users; works in Google Colab without extra dependencies.

### Pattern 5: nbsphinx Configuration

**What:** Configure Sphinx to render notebooks with pre-executed outputs

**When to use:** In docs/conf.py for Phase 6 notebook integration

**Example:**

```python
# docs/conf.py additions
extensions = [
    # ... existing extensions ...
    "nbsphinx",  # Add this
]

# nbsphinx configuration
nbsphinx_execute = "never"  # Use committed outputs, don't re-execute
nbsphinx_allow_errors = False  # Fail build on notebook errors
nbsphinx_requirejs_path = ""  # Avoid RequireJS conflicts

# Notebook kernel timeout (if execution enabled in future)
nbsphinx_timeout = 600  # 10 minutes

# Custom CSS for notebook cells (optional)
html_static_path = ["_static"]
html_css_files = ["custom_notebook.css"]  # Optional styling
```

**Source:** [nbsphinx documentation](https://nbsphinx.readthedocs.io/), [Read the Docs Jupyter guide](https://docs.readthedocs.com/platform/latest/guides/jupyter.html)

### Pattern 6: Google Colab Badge in Notebooks

**What:** Add "Open in Colab" badge to notebook markdown header

**When to use:** First cell of each tutorial notebook

**Example:**

```markdown
# Full Pipeline Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlancaster6/AquaCal/blob/main/docs/tutorials/01_full_pipeline.ipynb)

This tutorial demonstrates the complete AquaCal calibration workflow...
```

**Badge URL format:**
```
https://colab.research.google.com/github/{username}/{repo}/blob/{branch}/{path_to_notebook}
```

**Source:** [Colab Badge Action](https://github.com/marketplace/actions/colab-badge-action), [Open in Colab](https://openincolab.com/)

### Anti-Patterns to Avoid

- **Running notebooks during Sphinx build:** Slow builds, non-deterministic outputs, dependency hell. Commit pre-executed notebooks instead.
- **ipywidgets for data source toggle:** Adds dependency, breaks in Colab without extra setup, harder for users to understand. Use simple string variable.
- **Manual filename sorting:** Will break on `img_10.jpg` < `img_2.jpg`. Use natsort library.
- **Type checking with isinstance:** Tight coupling to concrete classes. Use duck typing with FrameSet protocol.
- **Multiple image format libraries:** opencv-python already loaded; no need for PIL/Pillow for JPEG/PNG reading.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Jupyter notebook rendering in Sphinx | Custom notebook parser | nbsphinx 0.9.8+ | Handles kernel outputs, cell metadata, image embedding, LaTeX math; used by NumPy/SciPy/pandas |
| Natural filename sorting | Regex-based number extraction | natsort library | Edge cases: version strings (v1.10 vs v1.2), unicode, mixed alphanumeric; cross-platform tested |
| Image file validation | Check magic bytes / use imghdr | pathlib.suffix check + OpenCV imread | imghdr deprecated/removed; OpenCV imread returns None for invalid images; simple and sufficient |
| Notebook execution during docs build | Sphinx-based kernel runner | Pre-execute notebooks locally, commit outputs | Build speed (seconds vs minutes), reproducibility, no compute dependencies in CI |
| Interactive plot updates in notebooks | Custom matplotlib wrapper | fig.canvas.draw_idle() + reusing axes | Standard matplotlib pattern; avoid creating new figures per update |

**Key insight:** Scientific Python ecosystem already solved notebook documentation (nbsphinx) and filename sorting (natsort). Custom solutions introduce bugs and maintenance burden for solved problems.

## Common Pitfalls

### Pitfall 1: Mixed Natural and Lexicographic Sorting

**What goes wrong:** Image sequences named `img_1.jpg`, `img_10.jpg`, `img_2.jpg` load in wrong order (1, 10, 2 instead of 1, 2, 10) using standard `sorted()`.

**Why it happens:** Python's default sort is lexicographic (string comparison); "10" < "2" as strings.

**How to avoid:** Use natsort library's `natsorted()` function for any filename-based ordering. Already handles embedded numbers correctly.

**Warning signs:** Frame indices jump (frame 10 appears after frame 1, before frame 2); calibration fails due to non-sequential board movements.

**Code example:**

```python
# WRONG - lexicographic sort
image_files = sorted(Path("data").glob("*.jpg"))
# Result: [img_1.jpg, img_10.jpg, img_2.jpg, ...]

# CORRECT - natural sort
from natsort import natsorted
image_files = natsorted(Path("data").glob("*.jpg"))
# Result: [img_1.jpg, img_2.jpg, img_10.jpg, ...]
```

### Pitfall 2: Notebook Execution During Sphinx Build

**What goes wrong:** Documentation builds become slow (minutes instead of seconds), fail non-deterministically, or require GPU/large datasets in CI environment.

**Why it happens:** nbsphinx can auto-execute notebooks during build if outputs are missing; this runs calibration pipelines, downloads datasets, etc.

**How to avoid:** Set `nbsphinx_execute = "never"` in docs/conf.py and commit notebooks with pre-executed outputs. Re-execute locally when updating.

**Warning signs:** Sphinx build takes >5 minutes, fails with "dataset not found" or memory errors, works locally but fails in CI.

**Workflow:**

```bash
# Local notebook development:
jupyter notebook docs/tutorials/01_full_pipeline.ipynb
# Run all cells, curate outputs, save

# Commit executed notebook:
git add docs/tutorials/01_full_pipeline.ipynb
git commit -m "docs: add full pipeline tutorial with outputs"

# Sphinx build (fast, uses committed outputs):
cd docs && make html  # Completes in seconds
```

### Pitfall 3: Assuming Directory = Video Parent Directory

**What goes wrong:** User has `videos/cam0/video.mp4` structure and passes directory `videos/cam0/` to ImageSet, which expects images directly in that directory.

**Why it happens:** Ambiguity between "directory of videos" vs "directory of images"; VideoSet takes file paths, ImageSet takes directory paths.

**How to avoid:** Auto-detect based on first path's type: if directory, create ImageSet; if file, create VideoSet. Validate all cameras use same type.

**Warning signs:** FileNotFoundError on valid paths, or "no images found" when directory contains video file.

**Detection logic:**

```python
first_path = Path(list(paths.values())[0])
if first_path.is_dir():
    # All paths must be directories containing images
    return ImageSet(paths)
elif first_path.is_file():
    # All paths must be video files
    return VideoSet(paths)
else:
    raise FileNotFoundError(f"Path does not exist: {first_path}")
```

### Pitfall 4: Matplotlib Figure Accumulation in Notebooks

**What goes wrong:** Notebook creates 50+ figures over repeated cells; memory bloat, slow rendering, confusing output.

**Why it happens:** Each `plt.figure()` creates a new figure; figures stay in memory until explicitly closed or kernel restarted.

**How to avoid:** Use `plt.close()` after displaying plots, or reuse figure/axes objects. In tutorials, close figures after each section.

**Warning signs:** Notebook file size >10MB after execution, slow rendering in JupyterLab, many duplicate plots in output.

**Best practice:**

```python
# Plot in a cell
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
plt.close(fig)  # Clean up immediately

# OR: Reuse figure for updates
fig, ax = plt.subplots()
for i in range(10):
    ax.clear()
    ax.plot(data[i])
    fig.canvas.draw_idle()
```

### Pitfall 5: Google Colab Badge Linking to Wrong Branch

**What goes wrong:** Colab badge opens old version of notebook from `main` branch; user edits aren't reflected.

**Why it happens:** Badge URL hard-codes branch name; defaults to `main` but notebooks might be in development branch.

**How to avoid:** Verify badge URL points to `main` branch before merging PR. Update badge when notebook is finalized.

**Warning signs:** Users report tutorial doesn't match README description; Colab shows different code than docs site.

**Correct badge:**

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tlancaster6/AquaCal/blob/main/docs/tutorials/01_full_pipeline.ipynb)
                                                                                                                    ^^^^
                                                                                            Ensure this matches your default branch
```

### Pitfall 6: README Hero Image Not Mobile-Responsive

**What goes wrong:** Large diagram looks great on desktop but is unreadable on mobile GitHub app; text too small, details lost.

**Why it happens:** Hero images are often >1200px wide; GitHub README viewer doesn't auto-scale images well.

**How to avoid:** Export hero visual at 800-1000px width (balances detail vs mobile readability); test on mobile view before committing.

**Warning signs:** Image width >1500px; text in diagram <12pt font size; GitHub mobile app requires horizontal scrolling.

**Export guidelines:**

```python
# When generating hero_ray_trace.png from docs/guide/_diagrams/ray_trace.py
fig = plt.figure(figsize=(10, 6), dpi=100)  # 1000x600px - good for README
# ... plotting code ...
plt.savefig("docs/_static/hero_ray_trace.png", dpi=100, bbox_inches="tight")
```

## Code Examples

Verified patterns from existing codebase:

### Existing VideoSet API (Reference for ImageSet)

```python
# From src/aquacal/io/video.py (existing)
class VideoSet:
    def __init__(self, video_paths: dict[str, str]) -> None:
        """Initialize with dict mapping camera_name to video file path."""
        ...

    @property
    def camera_names(self) -> list[str]:
        """Sorted list of camera names."""
        return sorted(self.video_paths.keys())

    @property
    def frame_count(self) -> int:
        """Minimum frame count across all videos."""
        ...

    def iterate_frames(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[tuple[int, dict[str, NDArray[np.uint8] | None]]]:
        """Yield (frame_idx, {camera: image}) tuples."""
        ...

    def __enter__(self) -> VideoSet:
        """Context manager: opens video captures."""
        ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager: releases video captures."""
        ...
```

**ImageSet must mirror this API** for drop-in compatibility with detection pipeline.

### Existing Dataset Loading (Reference for Notebook Data Toggle)

```python
# From src/aquacal/datasets/__init__.py (existing)
from aquacal.datasets import load_example, generate_synthetic_rig

# Synthetic data (no download)
scenario = generate_synthetic_rig("small")  # 2 cameras, 10 frames
scenario = generate_synthetic_rig("medium")  # 6 cameras, 80 frames
scenario = generate_synthetic_rig("large")  # 13 cameras, 300 frames

# Example datasets (download if needed)
dataset = load_example("small")  # Included in package
dataset = load_example("real-rig")  # Zenodo download (cached)

# Access structure
dataset.detections  # DetectionResult
dataset.ground_truth  # SyntheticScenario (if synthetic)
dataset.metadata  # Dict with description
```

**Notebooks should use this API** for data source toggle; already supports synthetic/preset/download modes.

### Existing Matplotlib Diagram Generation (Reference for README Hero)

```python
# From docs/guide/_diagrams/ray_trace.py (existing)
"""Generate ray trace diagram showing Snell's law refraction."""

import matplotlib.pyplot as plt
from aquacal.core.refractive_geometry import trace_refracted_ray
# ... imports ...

def generate_ray_trace_diagram(output_path: Path) -> None:
    """Generate refractive ray trace visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot camera, interface, underwater target
    # ... plotting code using actual library functions ...

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    generate_ray_trace_diagram(Path("docs/_static/hero_ray_trace.png"))
```

**README can reuse this diagram** by copying output to root or linking directly to docs/_static/ asset.

### Existing Detection Pipeline (Modify to Accept FrameSet)

```python
# From src/aquacal/io/detection.py (existing - will be modified)
def detect_all_frames(
    video_paths: dict[str, str] | VideoSet,  # CHANGE TO: FrameSet
    board: BoardGeometry,
    intrinsics: dict[str, tuple[NDArray, NDArray]] | None = None,
    min_corners: int = 4,
    frame_step: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DetectionResult:
    """Detect ChArUco corners in all frames.

    Now accepts any FrameSet (VideoSet or ImageSet).
    """
    # Current code creates VideoSet if dict passed
    # CHANGE TO: auto-detect and create VideoSet OR ImageSet
    if isinstance(video_paths, dict):
        frame_set = _create_frame_source(video_paths)  # Auto-detect
        owns_frame_set = True
    else:
        frame_set = video_paths  # Already a FrameSet
        owns_frame_set = False

    # Rest of function works unchanged - uses FrameSet protocol methods
    for frame_idx, frame_dict in frame_set.iterate_frames(step=frame_step):
        # ... detection logic ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual notebook HTML conversion | nbsphinx auto-rendering | nbsphinx 0.3.0 (2017) | Standard in SciPy docs; replaced custom Sphinx extensions |
| Lexicographic filename sorting | Natural sorting (natsort) | natsort 3.0 (2013), mature 8.x (2023) | Intuitive image sequence ordering; avoids img_10 < img_2 bugs |
| imghdr for image type detection | pathlib.suffix + try/except imread | Python 3.11 (2022 deprecation), removed 3.13 (2024) | Simpler code; OpenCV already validates on imread |
| %matplotlib inline (static) | %matplotlib widget (interactive) | ipympl 0.9+ (2023) | Enables zooming/panning in notebook; better for diagnostics |
| os.path file operations | pathlib object-oriented API | Python 3.4 (2014), standard in 3.10+ code | More readable; cross-platform; chaining operations |

**Deprecated/outdated:**
- **imghdr module:** Removed in Python 3.13 after deprecation in 3.11. Use pathlib.suffix + OpenCV imread instead.
- **%matplotlib notebook:** Deprecated backend; use `%matplotlib widget` (ipympl) for interactive plots in JupyterLab/Notebook 7+.
- **os.path for file operations:** Use pathlib.Path for modern Python 3.10+ code (project minimum version).

## Open Questions

1. **Should ImageSet support non-uniform frame counts across cameras?**
   - What we know: VideoSet uses minimum frame count across videos (partial support)
   - What's unclear: Should ImageSet error on mismatch or use minimum like VideoSet?
   - Recommendation: **Error on mismatch** (stricter) — image directories are assumed pre-curated, unlike videos where recording may stop early. Fail fast with clear message if camera directories have different image counts.

2. **Should README hero visual be generated during Sphinx build or committed?**
   - What we know: Existing diagrams in docs/guide/_diagrams/ are generated during Sphinx build (via setup() hook in conf.py)
   - What's unclear: README is in repository root; Sphinx doesn't touch it
   - Recommendation: **Generate and commit hero visual separately** — Add task to create docs/_static/hero_ray_trace.png, then copy to root README_hero.png and commit both. Keeps README self-contained without build dependencies.

3. **Should notebooks include full pipeline runtime (minutes) or abbreviated examples?**
   - What we know: Real rig calibration (13 cameras, 100+ frames) takes ~5 minutes; Colab may timeout or lack compute
   - What's unclear: Should tutorial notebooks run full pipeline or use smaller synthetic datasets?
   - Recommendation: **Use small synthetic/preset by default, show command for full run** — Notebook demonstrates workflow with fast data (30 seconds); include commented cell with full rig command for users with compute resources.

## Sources

### Primary (HIGH confidence)

- [nbsphinx 0.9.8 documentation](https://nbsphinx.readthedocs.io/) - Jupyter notebook integration with Sphinx
- [natsort 8.4.0 documentation](https://natsort.readthedocs.io/) - Natural sorting in Python
- [pathlib official documentation](https://docs.python.org/3/library/pathlib.html) - Object-oriented filesystem paths
- Existing codebase: src/aquacal/io/video.py (VideoSet API reference), docs/conf.py (Sphinx configuration), docs/guide/_diagrams/ (matplotlib diagram generation)

### Secondary (MEDIUM confidence)

- [Read the Docs Jupyter guide](https://docs.readthedocs.com/platform/latest/guides/jupyter.html) - nbsphinx integration patterns
- [GitHub Colab Badge Action](https://github.com/marketplace/actions/colab-badge-action) - Badge implementation
- [Open in Colab](https://openincolab.com/) - Badge URL format
- [awesome-readme](https://github.com/matiassingers/awesome-readme) - README visual best practices
- [Matplotlib interactive figures](https://matplotlib.org/stable/users/explain/figure/interactive.html) - ipympl backend usage

### Tertiary (LOW confidence)

- Web searches for general best practices (matplotlib in notebooks, README hero images) - informative but not authoritative; cross-referenced with official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - nbsphinx and natsort are industry-standard libraries with extensive documentation; pathlib is Python stdlib
- Architecture: HIGH - VideoSet API already exists and is tested; ImageSet mirrors it; detection.py already accepts VideoSet or dict
- Pitfalls: HIGH - natural sorting issues are well-documented; notebook execution pitfalls are common in scientific Python projects
- Open questions: MEDIUM - Recommendations based on similar projects (SciPy docs, pandas tutorials) but not project-specific testing

**Research date:** 2026-02-15
**Valid until:** ~60 days (stable domain; nbsphinx and natsort are mature libraries with infrequent breaking changes)

---

*Sources marked with URLs are authoritative official documentation. Existing codebase references point to project files that already implement similar patterns.*
