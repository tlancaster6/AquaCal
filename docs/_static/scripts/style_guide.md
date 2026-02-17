# AquaCal Visual Style Guide

Blue/aqua theme matching the AquaCal water-optics domain. Blues and aquas
represent the aquatic environment; warm tones distinguish cameras and ray paths.

This palette is portable — copy `palette.py` to any related project to adopt
the same visual identity.

## Color Palette

| Name              | Hex       | Usage                                          |
|-------------------|-----------|------------------------------------------------|
| `WATER_SURFACE`   | `#4DD0E1` | Water surface plane line / interface band      |
| `WATER_FILL`      | `#00BCD4` | Underwater region fill (use alpha ≈ 0.15)      |
| `AIR_FILL`        | `#E1F5FE` | Air region fill above water (use alpha ≈ 0.10) |
| `CAMERA_COLOR`    | `#37474F` | Camera body icons and position markers         |
| `BOARD_COLOR`     | `#FFB300` | Calibration board outlines and markers         |
| `RAY_AIR`         | `#FF7043` | Ray segment in air (incident ray)              |
| `RAY_WATER`       | `#00897B` | Ray segment in water (refracted ray)           |
| `INTERFACE_POINT` | `#E91E63` | Refraction crossing point at interface         |
| `AXIS_X`          | `#F44336` | +X coordinate axis                             |
| `AXIS_Y`          | `#4CAF50` | +Y coordinate axis                             |
| `AXIS_Z`          | `#1565C0` | +Z coordinate axis                             |
| `LABEL_COLOR`     | `#263238` | Text labels and annotations                    |
| `GRID_COLOR`      | `#B0BEC5` | Grid lines and minor decorations               |

## Typography

Matplotlib defaults are used throughout (DejaVu Sans). Suggested sizes:

- Diagram title: 13 pt, no bold
- Axis labels: 11 pt
- Annotations / sub-labels: 9–10 pt

## Diagram Conventions

- **Coordinate system**: Z increases downward (air at top, water at bottom).
  Y-axis is inverted in 2D matplotlib plots via `ax.invert_yaxis()`.
- **Aspect ratio**: Hero image targets ~1200×500 px (2.4:1 landscape).
  Technical diagrams use equal aspect or near-square as appropriate.
- **DPI**: 150 for all saved PNGs.
- **Line widths**: 2–2.5 pt for primary lines; 1–1.5 pt for secondary.

## Rationale

AquaCal calibrates cameras viewing underwater through a flat water surface.
The blue/aqua palette immediately communicates "water optics." Warm red/orange
for air-side rays and teal for water-side rays visually encodes the refraction
boundary — the central concept of the library.

## Portability

Copy `palette.py` into any related project's scripts directory and import:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "docs" / "_static" / "scripts"))
from palette import WATER_SURFACE, RAY_AIR, RAY_WATER  # etc.
```
