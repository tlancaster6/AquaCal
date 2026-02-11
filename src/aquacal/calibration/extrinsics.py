"""Stage 2: Extrinsic initialization via pose graph."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from aquacal.config.schema import (
    CameraExtrinsics,
    CameraIntrinsics,
    ConnectivityError,
    DetectionResult,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project_fast
from aquacal.utils.transforms import compose_poses, invert_pose, rvec_to_matrix


@dataclass
class Observation:
    """A single camera's observation of the board in one frame."""

    camera: str
    frame_idx: int
    corner_ids: NDArray[np.int32]  # shape (N,)
    corners_2d: NDArray[np.float64]  # shape (N, 2)


@dataclass
class PoseGraph:
    """
    Graph connecting cameras through shared board observations.

    Cameras connect indirectly: if cameras A and B both see the board in
    frame F, they are linked through that frame's board pose node.

    Attributes:
        camera_names: List of all camera names in the graph
        frame_indices: List of frame indices with 2+ camera observations
        observations: All camera-board observations
        adjacency: Dict mapping each node to its neighbors for connectivity analysis.
                   Node names: camera names (str) and frame indices prefixed with "f" (e.g., "f42")
    """

    camera_names: list[str]
    frame_indices: list[int]
    observations: list[Observation]
    adjacency: dict[str, set[str]] = field(default_factory=dict)


def estimate_board_pose(
    intrinsics: CameraIntrinsics,
    corners_2d: NDArray[np.float64],
    corner_ids: NDArray[np.int32],
    board: BoardGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """
    Estimate board pose relative to camera using PnP.

    Args:
        intrinsics: Camera intrinsic parameters
        corners_2d: Detected corner positions, shape (N, 2)
        corner_ids: Corner IDs corresponding to corners_2d, shape (N,)
        board: Board geometry for 3D corner positions

    Returns:
        Tuple of (rvec, tvec) representing board pose in camera frame,
        or None if PnP fails (e.g., too few points).
        - rvec: Rodrigues rotation vector, shape (3,)
        - tvec: Translation vector, shape (3,)

    Example:
        >>> result = estimate_board_pose(intrinsics, corners, ids, board)
        >>> if result is not None:
        ...     rvec, tvec = result
    """
    if len(corner_ids) < 4:
        return None

    # Get 3D points in board frame
    object_points = board.get_corner_array(corner_ids).astype(np.float32)
    image_points = corners_2d.astype(np.float32)

    # Run PnP
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.K.astype(np.float64),
        intrinsics.dist_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    return rvec.flatten().astype(np.float64), tvec.flatten().astype(np.float64)


def refractive_solve_pnp(
    intrinsics: CameraIntrinsics,
    corners_2d: NDArray[np.float64],
    corner_ids: NDArray[np.int32],
    board: BoardGeometry,
    interface_distance: float,
    interface_normal: NDArray[np.float64] | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """
    Estimate board pose with refractive correction using LM refinement.

    Uses standard solvePnP as initial guess, applies rough depth correction,
    then refines by minimizing refractive reprojection error.

    The key trick: use an identity-extrinsics camera so camera frame = world
    frame. Board corners transformed by the candidate pose become "world points"
    that get projected through the refractive interface.

    Args:
        intrinsics: Camera intrinsic parameters
        corners_2d: Detected corner positions, shape (N, 2)
        corner_ids: Corner IDs corresponding to corners_2d, shape (N,)
        board: Board geometry for 3D corner positions
        interface_distance: Distance from camera to water surface in meters
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)

    Returns:
        Tuple of (rvec, tvec) representing board pose in camera frame,
        or None if PnP fails.
    """
    # Get initial guess from standard PnP
    result = estimate_board_pose(intrinsics, corners_2d, corner_ids, board)
    if result is None:
        return None

    rvec_init, tvec_init = result

    # Apply rough depth correction for refraction
    tvec_init[2] *= n_water

    if interface_normal is None:
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Set up identity-extrinsics camera (camera frame = world frame)
    identity_ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
    camera = Camera("_pnp", intrinsics, identity_ext)
    interface = Interface(
        normal=interface_normal,
        base_height=0.0,
        camera_offsets={"_pnp": interface_distance},
        n_air=n_air,
        n_water=n_water,
    )

    # Get 3D object points in board frame
    object_points = board.get_corner_array(corner_ids).astype(np.float64)
    image_points = corners_2d.astype(np.float64)
    n_pts = len(object_points)

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        rvec = x[:3]
        tvec = x[3:]
        R = rvec_to_matrix(rvec)
        # Board corners in camera frame (= world frame for identity camera)
        pts_cam = (R @ object_points.T).T + tvec
        resid = np.empty(n_pts * 2, dtype=np.float64)
        for i, pt in enumerate(pts_cam):
            projected = refractive_project_fast(camera, interface, pt)
            if projected is None:
                resid[2 * i] = 100.0
                resid[2 * i + 1] = 100.0
            else:
                resid[2 * i] = projected[0] - image_points[i, 0]
                resid[2 * i + 1] = projected[1] - image_points[i, 1]
        return resid

    x0 = np.concatenate([rvec_init, tvec_init])

    result_opt = least_squares(residuals, x0, method="lm", max_nfev=200)

    rvec_out = result_opt.x[:3].astype(np.float64)
    tvec_out = result_opt.x[3:].astype(np.float64)
    return rvec_out, tvec_out


def _find_connected_components(
    adjacency: dict[str, set[str]],
    camera_names: list[str],
) -> list[set[str]]:
    """Find connected components in the pose graph using BFS."""
    visited: set[str] = set()
    components: list[set[str]] = []

    for start in camera_names:
        if start in visited:
            continue

        # BFS from this camera
        component: set[str] = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)

            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        components.append(component)

    return components


def build_pose_graph(
    detections: DetectionResult,
    min_cameras: int = 2,
) -> PoseGraph:
    """
    Build pose graph from detection results.

    Creates a bipartite graph where cameras and frames are nodes,
    connected by observation edges. Only includes frames where
    at least min_cameras see the board.

    Args:
        detections: Detection results from detect_all_frames
        min_cameras: Minimum cameras per frame to include (default 2)

    Returns:
        PoseGraph with adjacency structure for connectivity analysis

    Raises:
        ConnectivityError: If the graph is not connected (some cameras
            cannot be linked to others through shared observations).
            Error message includes details about disconnected components.

    Example:
        >>> detections = detect_all_frames(videos, board)
        >>> pose_graph = build_pose_graph(detections, min_cameras=2)
        >>> print(f"Graph has {len(pose_graph.frame_indices)} usable frames")
    """
    # Filter frames with enough cameras
    usable_frames = detections.get_frames_with_min_cameras(min_cameras)

    if not usable_frames:
        raise ConnectivityError(
            f"No frames with {min_cameras}+ cameras detecting the board"
        )

    # Collect observations and build adjacency
    observations: list[Observation] = []
    adjacency: dict[str, set[str]] = {}
    camera_set: set[str] = set()

    for frame_idx in usable_frames:
        frame_node = f"f{frame_idx}"
        if frame_node not in adjacency:
            adjacency[frame_node] = set()

        frame_det = detections.frames[frame_idx]
        for cam_name, det in frame_det.detections.items():
            camera_set.add(cam_name)

            # Add observation
            observations.append(
                Observation(
                    camera=cam_name,
                    frame_idx=frame_idx,
                    corner_ids=det.corner_ids,
                    corners_2d=det.corners_2d,
                )
            )

            # Build adjacency (bipartite: cameras <-> frames)
            if cam_name not in adjacency:
                adjacency[cam_name] = set()
            adjacency[cam_name].add(frame_node)
            adjacency[frame_node].add(cam_name)

    camera_names = sorted(camera_set)

    # Check connectivity
    components = _find_connected_components(adjacency, camera_names)
    if len(components) > 1:
        # Format error message with component details
        comp_strs = []
        for i, comp in enumerate(components):
            cameras_in_comp = [n for n in comp if not n.startswith("f")]
            comp_strs.append(f"Component {i+1}: {cameras_in_comp}")
        raise ConnectivityError(
            f"Pose graph is not connected. Found {len(components)} components:\n"
            + "\n".join(comp_strs)
        )

    return PoseGraph(
        camera_names=camera_names,
        frame_indices=sorted(usable_frames),
        observations=observations,
        adjacency=adjacency,
    )


def estimate_extrinsics(
    pose_graph: PoseGraph,
    intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str | None = None,
    interface_distances: dict[str, float] | None = None,
    interface_normal: NDArray[np.float64] | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
) -> dict[str, CameraExtrinsics]:
    """
    Estimate camera extrinsics by chaining poses through the graph.

    Uses BFS traversal from reference camera, computing each camera's
    pose relative to world frame (centered at reference camera).

    Process:
    1. Fix reference camera at world origin (R=I, t=0)
    2. BFS through pose graph:
       - When visiting a frame node from a known camera:
         compute board pose via PnP
       - When visiting a camera node from a known frame:
         compute camera pose via PnP + inversion

    Args:
        pose_graph: Pose graph from build_pose_graph
        intrinsics: Dict mapping camera names to intrinsics
        board: Board geometry
        reference_camera: Camera to place at world origin.
            If None, uses first camera name (sorted).
        interface_distances: Optional dict mapping camera names to interface
            distances in meters. When provided, uses refractive PnP for
            cameras with known distances.
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)

    Returns:
        Dict mapping camera names to CameraExtrinsics.
        Reference camera has R=I, t=0.

    Raises:
        ValueError: If reference_camera not in pose_graph
        ValueError: If intrinsics missing for any camera in graph

    Example:
        >>> extrinsics = estimate_extrinsics(pose_graph, intrinsics, board)
        >>> cam0_pose = extrinsics['cam0']
        >>> print(f"cam0 at world origin: {np.allclose(cam0_pose.t, 0)}")
    """
    # Validate reference camera
    if reference_camera is None:
        reference_camera = pose_graph.camera_names[0]
    elif reference_camera not in pose_graph.camera_names:
        raise ValueError(f"Reference camera '{reference_camera}' not in pose graph")

    # Validate intrinsics
    for cam in pose_graph.camera_names:
        if cam not in intrinsics:
            raise ValueError(f"Missing intrinsics for camera '{cam}'")

    # Index observations by (camera, frame) for quick lookup
    obs_index: dict[tuple[str, int], Observation] = {}
    for obs in pose_graph.observations:
        obs_index[(obs.camera, obs.frame_idx)] = obs

    # Initialize poses
    # camera_poses: R, t transforming world -> camera
    # board_poses: R, t transforming board -> world (for each frame)
    camera_poses: dict[str, tuple[NDArray, NDArray]] = {}
    board_poses: dict[int, tuple[NDArray, NDArray]] = {}

    # Reference camera at origin
    camera_poses[reference_camera] = (
        np.eye(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )

    # BFS to propagate poses
    visited_cameras: set[str] = {reference_camera}
    visited_frames: set[int] = set()
    queue = deque([reference_camera])

    while queue:
        node = queue.popleft()

        if isinstance(node, str) and not node.startswith("f"):
            # Camera node - propagate to connected frames
            cam_name = node
            cam_R, cam_t = camera_poses[cam_name]

            for neighbor in pose_graph.adjacency.get(cam_name, []):
                frame_idx = int(neighbor[1:])  # "f42" -> 42
                if frame_idx in visited_frames:
                    continue

                # Compute board pose in world frame via this camera
                obs_maybe = obs_index.get((cam_name, frame_idx))
                if obs_maybe is None:
                    continue
                obs = obs_maybe

                if interface_distances is not None and cam_name in interface_distances:
                    result = refractive_solve_pnp(
                        intrinsics[cam_name], obs.corners_2d, obs.corner_ids,
                        board, interface_distances[cam_name],
                        interface_normal, n_air, n_water,
                    )
                else:
                    result = estimate_board_pose(
                        intrinsics[cam_name], obs.corners_2d, obs.corner_ids, board
                    )
                if result is None:
                    continue

                rvec_bc, tvec_bc = result  # board in camera frame
                R_bc: NDArray[np.float64] = cv2.Rodrigues(rvec_bc)[0].astype(np.float64)

                # board_in_world = cam_in_world @ board_in_cam
                # cam_in_world = invert(world_in_cam) = invert(cam_R, cam_t)
                R_cw, t_cw = invert_pose(cam_R, cam_t)  # camera in world
                R_bw, t_bw = compose_poses(R_cw, t_cw, R_bc, tvec_bc)  # board in world

                board_poses[frame_idx] = (R_bw, t_bw)
                visited_frames.add(frame_idx)
                queue.append(f"f{frame_idx}")

        elif isinstance(node, str) and node.startswith("f"):
            # Frame node - propagate to connected cameras
            frame_idx = int(node[1:])
            if frame_idx not in board_poses:
                continue

            R_bw, t_bw = board_poses[frame_idx]

            for neighbor in pose_graph.adjacency.get(node, []):
                cam_name = neighbor
                if cam_name in visited_cameras:
                    continue

                # Compute camera pose from board pose
                obs_maybe = obs_index.get((cam_name, frame_idx))
                if obs_maybe is None:
                    continue
                obs = obs_maybe

                if interface_distances is not None and cam_name in interface_distances:
                    result = refractive_solve_pnp(
                        intrinsics[cam_name], obs.corners_2d, obs.corner_ids,
                        board, interface_distances[cam_name],
                        interface_normal, n_air, n_water,
                    )
                else:
                    result = estimate_board_pose(
                        intrinsics[cam_name], obs.corners_2d, obs.corner_ids, board
                    )
                if result is None:
                    continue

                rvec_bc, tvec_bc = result  # board in camera frame
                R_bc = cv2.Rodrigues(rvec_bc)[0].astype(np.float64)

                # world_in_cam = board_in_cam @ world_in_board
                # world_in_board = invert(board_in_world)
                R_wb, t_wb = invert_pose(R_bw, t_bw)
                R_wc, t_wc = compose_poses(R_bc, tvec_bc, R_wb, t_wb)

                camera_poses[cam_name] = (R_wc, t_wc)
                visited_cameras.add(cam_name)
                queue.append(cam_name)

    # Convert to CameraExtrinsics
    extrinsics_result: dict[str, CameraExtrinsics] = {}
    for cam_name in pose_graph.camera_names:
        if cam_name not in camera_poses:
            raise ValueError(f"Could not determine pose for camera '{cam_name}'")
        R, t = camera_poses[cam_name]
        extrinsics_result[cam_name] = CameraExtrinsics(R=R, t=t)

    return extrinsics_result
