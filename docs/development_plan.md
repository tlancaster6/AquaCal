# Development Plan: Refractive Multi-Camera Calibration Pipeline

## Project Overview

A Python library for calibrating a multi-camera array (up to 14 cameras) viewing an underwater volume through the air-water interface. The system uses ChArUco board calibration targets and explicitly models refraction at the water surface.

### Key Characteristics
- Cameras in air, viewing downward through water surface (approximately normal incidence)
- Hardware-synchronized cameras
- ChArUco board moved through underwater volume for calibration
- Separate in-air video for intrinsic calibration
- Interface modeled as single plane with per-camera height offsets

### Camera Array Geometry

The camera array has a specific physical arrangement that informs the calibration:

**Geometric properties**:
- All optical axes are approximately **parallel** (pointing straight down)
- All optical axes are approximately **perpendicular** to water surface
- Cameras are at approximately the **same height** above water (small variations exist)
- Each camera has a **unique roll angle** around its optical axis
- All point in the volume of interest are visible by 2+ cameras, but no point is visible to all cameras. 

**Implications for calibration**:
- Extrinsics between cameras: primarily differ in XY translation and roll (Z-rotation)
- Small pitch/yaw variations exist and must be estimated, but are minor corrections
- Interface distance variations between cameras are small but significant for accuracy
- Pose graph must chain through board observations since there is no input where all cameras see the board simultaneously

**Optional initialization hint**: If mechanical roll angles are approximately known (e.g., from CAD or mounting), they can be used to seed extrinsic initialization. The config can include:
```yaml
cameras:
  cam0: {approximate_roll_deg: 0}    # reference
  cam1: {approximate_roll_deg: 45}
  cam2: {approximate_roll_deg: 90}
  ...
```

### Downstream Use Cases
1. **Dense sand surface reconstruction**: High spatial density, low temporal resolution (~5 min intervals)
2. **Fish tracking**: High temporal resolution (every frame), sparse 3D output (keypoints or shells)

---

## Architecture Overview

```
multicam_refractive/
│
├── config/
│   └── schema.py                 # Configuration and output schema definitions
│
├── core/
│   ├── camera.py                 # Camera model (intrinsics, extrinsics, projection)
│   ├── interface_model.py        # Refractive interface model
│   ├── refractive_geometry.py    # Ray tracing, Snell's law, refractive projection
│   └── board.py                  # ChArUco board geometry and 3D point generation
│
├── io/
│   ├── video.py                  # Video loading, frame extraction, sync handling
│   ├── detection.py              # ChArUco detection wrapper
│   └── serialization.py          # Save/load calibration results
│
├── calibration/
│   ├── intrinsics.py             # Stage 1: Per-camera intrinsic calibration
│   ├── extrinsics.py             # Stage 2: Multi-camera extrinsic calibration
│   ├── interface_estimation.py   # Stage 3: Interface + board pose estimation
│   ├── refinement.py             # Stage 4: Optional joint refinement
│   └── pipeline.py               # Orchestrates the full calibration workflow
│
├── validation/
│   ├── reprojection.py           # Reprojection error computation
│   ├── reconstruction.py         # 3D reconstruction metrics (ChArUco distances)
│   └── diagnostics.py            # Per-camera, per-frame, spatial error analysis
│
├── triangulation/
│   └── triangulate.py            # Refractive triangulation (for validation & downstream)
│
└── utils/
    ├── transforms.py             # Rotation conversions, coordinate transforms
    └── visualization.py          # Plotting utilities for debugging
```

---

## Module Specifications

### 1. Configuration & Schema (`config/`)

#### `schema.py`

Defines the calibration output format and configuration structures.

**Calibration Output Schema**:
```
CalibrationResult:
  cameras: dict[str, CameraCalibration]
    - K: 3x3 intrinsic matrix
    - dist_coeffs: distortion coefficients (k1, k2, p1, p2, k3 or subset)
    - R: 3x3 rotation matrix (world → camera)
    - t: 3x1 translation vector
    - C: 3x1 camera center in world coordinates
    - interface_distance: float (meters from camera center to interface)
    
  interface: InterfaceParams
    - normal: 3x1 unit vector (shared across cameras)
    - n_air: float (refractive index, default 1.0)
    - n_water: float (refractive index, default 1.333)
    
  board: BoardParams
    - squares_x: int
    - squares_y: int
    - square_size: float (meters)
    - marker_size: float (meters)
    - dictionary: str (e.g., "DICT_4X4_50")
    
  diagnostics: DiagnosticsData
    - reprojection_error_rms: float
    - reprojection_error_per_camera: dict[str, float]
    - validation_3d_error_mean: float
    - validation_3d_error_std: float
    - per_corner_residuals: optional detailed data
    - per_frame_residuals: optional detailed data
    
  metadata: Metadata
    - calibration_date: str
    - software_version: str
    - config_hash: str (for reproducibility)
```

**Configuration Input Schema**:
```
CalibrationConfig:
  board:
    squares_x, squares_y, square_size, marker_size, dictionary
    
  cameras:
    names: list[str]
    
  paths:
    intrinsic_videos: dict[str, path] or path with pattern
    extrinsic_videos: dict[str, path] or path with pattern
    output_dir: path
    
  interface:
    n_air: float
    n_water: float
    normal_fixed: bool (if true, fix to [0,0,-1])
    
  optimization:
    robust_loss: str ("huber", "soft_l1", or "linear")
    loss_scale: float
    
  detection:
    min_corners: int (minimum corners per frame to use)
    min_cameras: int (minimum cameras seeing board to use frame)
    
  validation:
    holdout_fraction: float (fraction of frames for validation)
    save_detailed_residuals: bool
```

---

### 2. Core Geometry (`core/`)

#### `camera.py`

Camera model representation and basic operations.

**Responsibilities**:
- Store intrinsic parameters (K, distortion)
- Store extrinsic parameters (R, t) and derived quantities (C)
- Undistort points
- Back-project pixel to normalized ray (in camera frame)
- Project 3D point to pixel (without refraction — used for in-air calibration)

**Key Functions**:
- `Camera` class with intrinsic/extrinsic properties
- `pixel_to_ray(camera, pixel) → ray_direction` (normalized, camera frame)
- `world_to_pixel(camera, point_3d) → pixel` (standard projection, no refraction)
- `undistort_points(camera, points) → undistorted_points`

#### `interface.py`

Refractive interface model.

**Responsibilities**:
- Store interface parameters (normal, per-camera distances, refractive indices)
- Compute interface plane for a given camera
- Provide ray-plane intersection

**Key Functions**:
- `Interface` class
- `get_interface_point(interface, camera) → point_on_plane` (where camera's optical axis hits interface)
- `ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal) → intersection_point, t`

#### `refractive_geometry.py`

Core refractive ray tracing operations. **This is the shared module used by calibration and downstream tasks.**

**Responsibilities**:
- Snell's law in 3D
- Trace ray from camera through interface into water
- Refractive forward projection (3D point in water → pixel)
- Refractive back-projection (pixel → ray in water)

**Key Functions**:
- `snells_law_3d(incident_dir, normal, n1, n2) → refracted_dir`
- `trace_ray_air_to_water(camera, interface, pixel) → (intersection_point, ray_dir_in_water)`
- `refractive_project(camera, interface, point_3d) → pixel` (for reprojection error)
- `refractive_back_project(camera, interface, pixel) → (ray_origin_in_water, ray_dir_in_water)`

#### `board.py`

ChArUco board geometry.

**Responsibilities**:
- Generate 3D coordinates of all corners in board frame
- Transform board points by pose (R, t) to world frame
- Create OpenCV CharucoBoard object for detection

**Key Functions**:
- `BoardGeometry` class constructed from config
- `get_corner_positions(board) → dict[corner_id, point_3d]` (in board frame)
- `transform_corners(board, rvec, tvec) → dict[corner_id, point_3d]` (in world frame)
- `get_opencv_board(board) → cv2.aruco.CharucoBoard`

---

### 3. Input/Output (`io/`)

#### `video.py`

Video loading and frame extraction.

**Responsibilities**:
- Load video files
- Extract frames at specified interval or all frames
- Handle synchronized multi-camera video (frames indexed consistently)
- Provide iterator interface for memory efficiency

**Key Functions**:
- `VideoSet` class managing multiple synchronized videos
- `extract_frames(video_set, frame_indices) → dict[camera, dict[frame_idx, image]]`
- `get_frame_count(video_set) → int`
- `iterate_synchronized_frames(video_set) → generator of dict[camera, image]`

#### `detection.py`

ChArUco detection wrapper.

**Responsibilities**:
- Detect ChArUco corners in images
- Filter detections by quality/count
- Organize detections by frame and camera

**Key Functions**:
- `detect_charuco(image, board) → (corner_ids, corners_2d, num_detected)`
- `detect_all_frames(video_set, board, min_corners) → DetectionResult`
  - `DetectionResult.detections[frame_idx][camera_name] = {'ids': array, 'corners': array}`
  - `DetectionResult.frames_by_visibility` — frames sorted by how many cameras see the board

#### `serialization.py`

Save and load calibration results.

**Responsibilities**:
- Serialize CalibrationResult to disk (JSON + numpy arrays, or single pickle/HDF5)
- Load CalibrationResult
- Version compatibility handling

**Key Functions**:
- `save_calibration(result, path)`
- `load_calibration(path) → CalibrationResult`
- `export_for_downstream(result, path)` — minimal format optimized for triangulation speed

---

### 4. Calibration Stages (`calibration/`)

#### `intrinsics.py`

**Stage 1: Intrinsic Calibration**

**Input**: In-air calibration videos (one per camera), board config
**Output**: Per-camera intrinsic parameters (K, dist_coeffs)

**Process**:
1. For each camera:
   a. Extract frames from in-air video
   b. Detect ChArUco corners
   c. Select subset of frames with good coverage (corners across full image)
   d. Run OpenCV camera calibration
   e. Store intrinsics

**Key Functions**:
- `calibrate_intrinsics_single(video_path, board) → (K, dist_coeffs, reprojection_error)`
- `calibrate_intrinsics_all(config) → dict[camera_name, IntrinsicCalibration]`

**Frame Selection Strategy**:
- Aim for ~50-100 frames per camera
- Prioritize frames with corners spread across image (avoid clustering in center)
- Reject frames with too few corners or poor detection confidence

#### `extrinsics.py`

**Stage 2: Extrinsic Calibration (Initialization)**

**Input**: Underwater calibration videos, intrinsics from Stage 1, board config
**Output**: Initial estimate of camera extrinsics (R, t for each camera relative to reference camera)

**Critical Constraint — Non-Overlapping FOVs**:

The cameras do not have overlapping fields of view, meaning no single frame will ever show the board in all cameras simultaneously. Extrinsic calibration must work by **chaining observations through shared board poses**:

- Frame 1: Cameras A, B see board → establishes A↔B relative pose
- Frame 2: Cameras B, C see board → establishes B↔C relative pose  
- Combined: A↔C inferred transitively through B

This requires building a **pose graph** where:
- Nodes: cameras + board poses (one per frame)
- Edges: camera-to-board observations (PnP constraints)
- Cameras connect to each other *indirectly* through board nodes

**Connectivity Requirement**: The pose graph must be **connected**—there must exist a path of observations linking every camera to every other camera. If the graph has disconnected components, those camera groups cannot be calibrated relative to each other.

**Process**:
1. Detect ChArUco in all underwater frames across all cameras
2. Find frames where board is visible in 2+ cameras
3. Build pose graph:
   a. For each usable frame, create a board pose node
   b. For each camera seeing the board in that frame, create an edge (observation)
   c. Verify graph connectivity; report error if disconnected
4. Initialize poses:
   a. Fix reference camera (cam0) at world origin
   b. Use BFS/DFS from reference camera through pose graph to initialize all camera poses
   c. For each board pose, initialize via PnP from any observing camera
5. **Ignoring refraction**, optimize full pose graph:
   a. Cost function: sum of squared reprojection errors across all observations
   b. Parameters: all camera poses (except reference) + all board poses
   c. Run sparse bundle adjustment
6. Output camera extrinsics (will be refined in Stage 3)

**Key Functions**:
- `estimate_board_pose(camera_intrinsics, corners_2d, corner_ids, board) → (rvec, tvec)`
- `build_pose_graph(detections, intrinsics, board) → PoseGraph`
- `check_connectivity(pose_graph) → (is_connected, components)`
- `initialize_poses_from_graph(pose_graph, reference_camera) → (camera_poses, board_poses)`
- `optimize_extrinsics(pose_graph, initial_poses) → dict[camera_name, (R, t)]`

**PoseGraph Structure**:
```
PoseGraph:
  camera_nodes: list[str]  # camera names
  board_nodes: list[int]   # frame indices with usable detections
  observations: list[Observation]
    - camera: str
    - frame: int
    - corner_ids: array
    - corners_2d: array

  adjacency: dict  # for connectivity analysis
```

**Notes**:
- This ignores refraction, so extrinsics will be biased—that's expected
- Serves as initialization for Stage 3; doesn't need to be perfect
- Reference camera (cam0) is placed at world origin
- Data collection tip: ensure board path visits regions visible by camera pairs that bridge between groups

#### `interface.py`

**Stage 3: Interface and Board Pose Estimation**

**Input**: Intrinsics (fixed), initial extrinsics from Stage 2, underwater detections, board config
**Output**: Refined extrinsics, interface parameters (per-camera distances), board poses

**Process**:
1. Initialize parameters:
   - Camera extrinsics: from Stage 2
   - Interface normal: [0, 0, -1] (fixed or free based on config)
   - Interface distances: initial guess (e.g., measured or estimated from Stage 2 residual patterns)
   - Board poses: from PnP estimates in Stage 2

2. Define parameter vector:
   ```
   params = [
     cam1_rvec (3), cam1_tvec (3),   # cam0 is reference, fixed at origin
     cam2_rvec (3), cam2_tvec (3),
     ...
     cam_N_rvec (3), cam_N_tvec (3),
     interface_distance_cam1,         # cam0 distance is reference (or estimated)
     interface_distance_cam2,
     ...
     interface_distance_camN,
     (optional) interface_tilt_x, interface_tilt_y,
     board_pose_frame0_rvec (3), board_pose_frame0_tvec (3),
     board_pose_frame1_rvec (3), board_pose_frame1_tvec (3),
     ...
   ]
   ```

3. Define cost function:
   - For each frame, for each camera, for each detected corner:
     - Get corner 3D position in world frame (from board pose)
     - Refractive project to pixel
     - Compute residual vs detected pixel
   - Return all residuals (for least_squares)

4. Run optimization:
   - `scipy.optimize.least_squares` with Huber loss
   - Monitor convergence

5. Extract results

**Key Functions**:
- `pack_params(cameras, interface, board_poses) → param_vector`
- `unpack_params(param_vector, config) → (cameras, interface, board_poses)`
- `cost_function(params, detections, board, intrinsics, config) → residuals`
- `optimize_interface(initial_params, detections, board, intrinsics, config) → OptimizationResult`

**Parameter Bounds** (for Trust Region Reflective):
- Interface distances: [0.01, 2.0] meters (physical constraints)
- Interface tilt angles: [-0.1, 0.1] radians if free (~±6°)

#### `refinement.py`

**Stage 4: Joint Refinement (Optional)**

**Input**: Results from Stage 3, option to also refine intrinsics
**Output**: Final calibration with all parameters jointly optimized

**Process**:
1. If enabled, add intrinsic parameters to optimization
2. Run joint optimization over all parameters
3. Typically minor adjustments if Stage 3 converged well

**When to Use**:
- If Stage 3 residuals show systematic patterns
- If intrinsics might have shifted (unlikely given your setup)
- For maximum accuracy when calibration time isn't critical

**Key Functions**:
- `joint_refinement(stage3_result, detections, config, refine_intrinsics=False) → CalibrationResult`

#### `pipeline.py`

**Orchestration**

**Responsibilities**:
- Load configuration
- Run stages in sequence
- Handle intermediate outputs
- Coordinate validation
- Save final results

**Key Functions**:
- `run_calibration(config_path) → CalibrationResult`
- `run_calibration_from_config(config) → CalibrationResult`

**Workflow**:
```
1. Load config
2. Load videos
3. Detect ChArUco in all videos
4. Split detections into calibration and validation sets
5. Run Stage 1 (intrinsics) on in-air data
6. Run Stage 2 (extrinsics init) on underwater data
7. Run Stage 3 (interface + refinement) on underwater data
8. Optionally run Stage 4 (joint refinement)
9. Run validation on held-out data
10. Compile diagnostics
11. Save results
```

---

### 5. Validation (`validation/`)

#### `reprojection.py`

Compute reprojection errors.

**Key Functions**:
- `compute_reprojection_errors(calibration, detections) → ReprojectionErrors`
  - `.rms`: overall RMS error in pixels
  - `.per_camera`: dict[camera, rms]
  - `.per_frame`: dict[frame, rms]
  - `.per_observation`: array of individual errors
- `compute_reprojection_error_single(camera, interface, board_pose, detected_corners) → errors`

#### `reconstruction.py`

3D reconstruction metrics using known ChArUco geometry.

**Key Functions**:
- `triangulate_charuco_corners(calibration, detections, frame) → dict[corner_id, point_3d]`
- `compute_3d_distance_errors(calibration, detections, board) → DistanceErrors`
  - `.mean`: mean absolute error in meters
  - `.std`: standard deviation
  - `.per_corner_pair`: detailed breakdown
  - `.max_error`: worst case
- `compute_board_planarity_error(triangulated_corners) → float` (RMS distance from best-fit plane)

#### `diagnostics.py`

Detailed error analysis.

**Key Functions**:
- `spatial_error_map(calibration, detections) → dict[camera, error_image]`
  - Visualizes reprojection error magnitude across image
- `depth_stratified_error(calibration, detections) → DataFrame`
  - Error vs estimated depth below surface
- `generate_diagnostic_report(calibration, detections) → DiagnosticReport`
  - Summary statistics, plots, recommendations

---

### 6. Triangulation (`triangulation/`)

#### `triangulate.py`

Refractive triangulation for downstream use.

**Key Functions**:
- `triangulate_point(calibration, observations) → point_3d`
  - `observations`: dict[camera_name, pixel_2d]
  - Uses refractive back-projection + least-squares ray intersection
  
- `triangulate_points_batch(calibration, observations_batch) → points_3d`
  - Vectorized for efficiency
  
- `ray_intersection_least_squares(rays) → point_3d`
  - `rays`: list of (origin, direction) tuples
  - Minimizes sum of squared distances to all rays

**Notes**:
- This module is intentionally minimal for now
- Dense reconstruction and tracking will build on this foundation
- For dense use, consider adding LUT-based acceleration later

---

### 7. Utilities (`utils/`)

#### `transforms.py`

Rotation and coordinate transform helpers.

**Key Functions**:
- `rvec_to_rotation_matrix(rvec) → R`
- `rotation_matrix_to_rvec(R) → rvec`
- `compose_poses(R1, t1, R2, t2) → (R_combined, t_combined)`
- `invert_pose(R, t) → (R_inv, t_inv)`
- `camera_center_from_pose(R, t) → C`

#### `visualization.py`

Plotting for debugging and diagnostics.

**Key Functions**:
- `plot_detections(image, corners, ids)` — show detected corners on image
- `plot_reprojection(image, detected, projected)` — compare detected vs reprojected
- `plot_camera_arrangement(calibration)` — 3D plot of camera positions and orientations
- `plot_residual_histogram(errors)`
- `plot_spatial_error_map(camera, errors)`

---

## Development Phases

> **Note**: The phases below are for conceptual organization of the implementation roadmap.
> For canonical task IDs and status tracking, see `TASKS.md` which is the authoritative source.

### Phase 1: Foundation
**Goal**: Core geometry and data structures working

**Tasks**:
1. Define configuration and output schemas (`config/schema.py`)
2. Implement camera model (`core/camera.py`)
3. Implement refractive geometry (`core/refractive_geometry.py`)
4. Implement interface model (`core/interface_model.py`)
5. Implement board geometry (`core/board.py`)
6. Write unit tests for ray tracing and projection

**Validation Milestone**: 
- Create synthetic test case: known camera, known interface, known 3D point
- Verify forward projection and back-projection are consistent
- Verify Snell's law implementation against analytical examples

### Phase 2: Data Pipeline
**Goal**: Load videos, detect ChArUco, organize data

**Tasks**:
1. Implement video loading (`io/video.py`)
2. Implement ChArUco detection wrapper (`io/detection.py`)
3. Implement serialization (`io/serialization.py`)
4. Write integration tests with sample video

**Validation Milestone**:
- Load multi-camera video set
- Detect ChArUco across all cameras and frames
- Verify detection consistency (same corners detected in overlapping views)

### Phase 3: Intrinsic Calibration
**Goal**: Stage 1 working end-to-end

**Tasks**:
1. Implement intrinsic calibration (`calibration/intrinsics.py`)
2. Add frame selection logic
3. Test on in-air calibration video

**Validation Milestone**:
- Calibrate all cameras from in-air video
- Verify reprojection error < 0.5 pixels
- Compare results to OpenCV standalone calibration (should match)

### Phase 4: Extrinsic Initialization
**Goal**: Stage 2 working, produces reasonable initial extrinsics

**Tasks**:
1. Implement pose graph construction (`calibration/extrinsics.py`)
2. Implement extrinsic optimization (ignoring refraction)
3. Test on underwater video

**Validation Milestone**:
- Estimate relative camera poses from underwater data
- Visualize camera arrangement (should roughly match physical setup)
- Reprojection error will be elevated due to ignored refraction — that's expected

### Phase 5: Refractive Calibration
**Goal**: Stage 3 working, core deliverable complete

**Tasks**:
1. Implement parameter packing/unpacking
2. Implement refractive cost function
3. Implement optimization loop (`calibration/interface_estimation.py`)
4. Tune convergence criteria and loss function

**Validation Milestone**:
- Run full Stage 3 optimization
- Reprojection error drops significantly vs Stage 2
- Interface distances are physically plausible
- 3D ChArUco corner distances match known geometry within ~1-2mm

### Phase 6: Validation & Diagnostics
**Goal**: Comprehensive validation and debugging tools

**Tasks**:
1. Implement reprojection error computation (`validation/reprojection.py`)
2. Implement 3D reconstruction metrics (`validation/reconstruction.py`)
3. Implement diagnostic tools (`validation/diagnostics.py`)
4. Implement visualization utilities (`utils/visualization.py`)

**Validation Milestone**:
- Generate full diagnostic report
- Identify any systematic error patterns
- Confirm validation set performance matches calibration set (no overfitting)

### Phase 7: Pipeline Integration
**Goal**: End-to-end pipeline, ready for use

**Tasks**:
1. Implement pipeline orchestration (`calibration/pipeline.py`)
2. Add configuration file loading
3. Add command-line interface
4. Write documentation
5. Optional: implement Stage 4 joint refinement

**Validation Milestone**:
- Run calibration from config file with single command
- Output is complete, loadable, and usable by downstream code

### Phase 8: Triangulation Module
**Goal**: Downstream-ready triangulation

**Tasks**:
1. Implement refractive triangulation (`triangulation/triangulate.py`)
2. Add batch processing for efficiency
3. Verify against ChArUco known geometry

**Validation Milestone**:
- Triangulate arbitrary points from multi-camera observations
- Accuracy consistent with calibration validation metrics

---

## Testing Strategy

### Unit Tests
- Ray tracing: Snell's law correctness, edge cases (normal incidence, grazing angles)
- Projection: Round-trip consistency (project then back-project)
- Transforms: Rotation conversions, pose composition/inversion
- Board geometry: Corner positions match expected values

### Integration Tests
- Detection pipeline: Load video → detect → organize
- Intrinsic calibration: Full Stage 1 on test data
- Extrinsic calibration: Full Stage 2 on test data
- Refractive calibration: Full Stage 3 on test data

### Synthetic Tests
- Generate synthetic observations from known ground truth
- Run calibration, verify recovered parameters match ground truth
- Test sensitivity to noise levels

### Real Data Tests
- Run on actual calibration videos
- Compare interface distances to physical measurements
- Validate 3D reconstruction against known ChArUco geometry

---

## Dependencies

**Required**:
- numpy
- scipy
- opencv-python (cv2)
- pyyaml or toml (for config files)

**Optional**:
- matplotlib (visualization)
- pandas (diagnostics tables)
- jax (if automatic differentiation needed for speed)
- h5py (if HDF5 output desired)

---

## Open Questions / Future Considerations

1. **LUT acceleration for dense reconstruction**: Not needed for calibration, but downstream dense reconstruction may benefit from precomputed ray lookup tables. The calibration output format supports generating these.

2. **Partial re-calibration**: The staged design supports re-running only Stage 3 (interface estimation) with fixed intrinsics/extrinsics if only water level changed. Consider adding explicit support in pipeline.

3. **Multi-board or large volume calibration**: Current design assumes single board visible in multiple cameras. For very large volumes, may need to extend to board-hopping or wand-based calibration.

4. **Radiometric calibration**: Not addressed here. If downstream tasks need color consistency across cameras (e.g., for texture reconstruction), that's a separate calibration step.

5. **Online/streaming triangulation**: Current triangulation module is batch-oriented. Real-time fish tracking may need optimized streaming implementation.
