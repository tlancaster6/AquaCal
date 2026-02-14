"""Validation and diagnostics modules."""

from aquacal.validation.comparison import (
    ComparisonResult,
    compare_calibrations,
    write_comparison_report,
)

# Public API - will be populated as modules are implemented
__all__ = [
    # From reprojection.py (Task 5.1):
    # "compute_reprojection_errors",
    # From reconstruction.py (Task 5.2):
    # "compute_3d_distance_errors",
    # From diagnostics.py (Task 5.3):
    # "generate_diagnostic_report",
    # From comparison.py (Task 6.1, 6.2):
    "compare_calibrations",
    "ComparisonResult",
    "write_comparison_report",
]

# Imports will be added as modules are implemented:
# from aquacal.validation.reprojection import compute_reprojection_errors
# from aquacal.validation.reconstruction import compute_3d_distance_errors
# from aquacal.validation.diagnostics import generate_diagnostic_report
