"""Shared color palette for AquaCal documentation visuals.

Defines a blue/aqua theme matching the AquaCal brand — blues and aquas for
water optics elements, warm tones for cameras and ray paths. Import these
constants in all diagram generation scripts to ensure visual consistency.

This palette is intentionally portable: any project in the AquaCal family
(or related underwater optics work) can adopt it by copying this file.
"""

# ---------------------------------------------------------------------------
# Water and environment
# ---------------------------------------------------------------------------

#: Light aqua/cyan for the water surface plane and interface line.
WATER_SURFACE = "#4DD0E1"

#: Semi-transparent aqua for the underwater region fill. Use with alpha=0.15.
WATER_FILL = "#00BCD4"

#: Very light sky blue for the air region above the water surface. Use with alpha=0.10.
AIR_FILL = "#E1F5FE"

# ---------------------------------------------------------------------------
# Cameras and geometry
# ---------------------------------------------------------------------------

#: Charcoal/dark blue-gray for camera body icons and markers.
CAMERA_COLOR = "#37474F"

#: Warm amber/gold for calibration board outlines and markers.
BOARD_COLOR = "#FFB300"

# ---------------------------------------------------------------------------
# Ray paths
# ---------------------------------------------------------------------------

#: Warm red/orange for ray segments travelling through air.
RAY_AIR = "#FF7043"

#: Deep teal/blue for ray segments travelling through water.
RAY_WATER = "#00897B"

#: Bright magenta/pink for the refraction point at the water interface.
INTERFACE_POINT = "#E91E63"

# ---------------------------------------------------------------------------
# Coordinate axes (standard RGB convention)
# ---------------------------------------------------------------------------

#: Red for the +X axis.
AXIS_X = "#F44336"

#: Green for the +Y axis.
AXIS_Y = "#4CAF50"

#: Blue for the +Z axis.
AXIS_Z = "#1565C0"

# ---------------------------------------------------------------------------
# Typography and chrome
# ---------------------------------------------------------------------------

#: Dark blue-gray for text labels and annotations.
LABEL_COLOR = "#263238"

#: Light gray for grid lines and minor decorations.
GRID_COLOR = "#B0BEC5"
