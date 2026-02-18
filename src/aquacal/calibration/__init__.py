"""Calibration pipeline modules."""

from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    build_pose_graph,
    estimate_board_pose,
    estimate_extrinsics,
    refractive_solve_pnp,
)
from aquacal.calibration.interface_estimation import (
    optimize_interface,
)
from aquacal.calibration.intrinsics import (
    calibrate_intrinsics_all,
    calibrate_intrinsics_single,
)
from aquacal.calibration.pipeline import (
    calibrate_from_detections,
    load_config,
    run_calibration,
    run_calibration_from_config,
    split_detections,
)
from aquacal.calibration.refinement import (
    joint_refinement,
)

__all__ = [
    # intrinsics
    "calibrate_intrinsics_single",
    "calibrate_intrinsics_all",
    # extrinsics
    "Observation",
    "PoseGraph",
    "estimate_board_pose",
    "refractive_solve_pnp",
    "build_pose_graph",
    "estimate_extrinsics",
    # interface_estimation
    "optimize_interface",
    # refinement
    "joint_refinement",
    # pipeline
    "calibrate_from_detections",
    "load_config",
    "split_detections",
    "run_calibration",
    "run_calibration_from_config",
]
