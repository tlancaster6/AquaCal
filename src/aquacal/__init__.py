"""AquaCal: Refractive multi-camera calibration library."""

__version__ = "0.1.0"

# Load/save calibration results
# Run calibration
from aquacal.calibration.pipeline import load_config, run_calibration

# Core types
from aquacal.config.schema import (
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)
from aquacal.io.serialization import load_calibration, save_calibration

__all__ = [
    "__version__",
    # Load/save
    "load_calibration",
    "save_calibration",
    # Core types
    "CalibrationResult",
    "CameraCalibration",
    "CameraIntrinsics",
    "CameraExtrinsics",
    # Run calibration
    "run_calibration",
    "load_config",
]
