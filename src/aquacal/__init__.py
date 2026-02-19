"""AquaCal: Refractive multi-camera calibration library."""


def _check_torch():
    """Verify PyTorch is installed before importing aquakit."""
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise ImportError(
            "AquaCal requires PyTorch. Install it first: `pip install torch`. "
            "See https://pytorch.org/get-started/ for GPU variants."
        ) from None


_check_torch()

from importlib.metadata import version as _get_version  # noqa: E402

__version__ = _get_version("aquacal")

# Load/save calibration results
# Run calibration
from aquacal.calibration.pipeline import (  # noqa: E402
    calibrate_from_detections,
    load_config,
    run_calibration,
)

# Core types
from aquacal.config.schema import (  # noqa: E402
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)
from aquacal.io.serialization import load_calibration, save_calibration  # noqa: E402

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
    "calibrate_from_detections",
    "run_calibration",
    "load_config",
]
