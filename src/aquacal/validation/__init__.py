"""Validation and diagnostics modules."""

from aquacal.validation.comparison import (
    ComparisonResult,
    compare_calibrations,
    write_comparison_report,
)
from aquacal.validation.diagnostics import (
    plot_error_distribution,
    plot_per_camera_error,
)

__all__ = [
    # diagnostics
    "plot_per_camera_error",
    "plot_error_distribution",
    # comparison
    "compare_calibrations",
    "ComparisonResult",
    "write_comparison_report",
]
