"""Calibration pipeline modules."""

from aquacal.calibration.intrinsics import (
    calibrate_intrinsics_single,
    calibrate_intrinsics_all,
)
from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    estimate_board_pose,
    refractive_solve_pnp,
    build_pose_graph,
    estimate_extrinsics,
)
from aquacal.calibration.interface_estimation import (
    optimize_interface,
)
from aquacal.calibration.refinement import (
    joint_refinement,
)
from aquacal.calibration.pipeline import (
    load_config,
    split_detections,
    run_calibration,
    run_calibration_from_config,
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
    "load_config",
    "split_detections",
    "run_calibration",
    "run_calibration_from_config",
]
