# AquaCal Geometry Reference

This document is the definitive reference for the coordinate systems, transform
conventions, and refractive geometry used throughout AquaCal. Every claim has
been verified analytically and/or tested against the code.

---

## 1. Physical Setup

An array of cameras is mounted in air, pointing downward at a calibration
target submerged in water. Light from the underwater target travels upward
through water, refracts at the flat air-water interface, continues through air,
and enters a camera.

```
Camera array (in air, near Z ≈ 0)
    ○  ○  ○  ○
    │  │  │  │       air  (n = 1.000)
~~~~│~~│~~│~~│~~~~   water surface at Z = water_z
    │  │  │  │       water (n = 1.333)
    ┌──────────┐
    │  target  │     calibration board, underwater
    └──────────┘
        ▼ +Z
```

---

## 2. Coordinate Frames

### 2.1 World Frame

| Axis | Direction | Notes |
|------|-----------|-------|
| +X   | Horizontal (right from above) | |
| +Y   | Horizontal (forward from above) | |
| +Z   | **Down**, into the water | Depth increases with Z |

- **Origin**: Reference camera (cam0) optical center
- **Units**: Meters
- Cameras sit near Z ≈ 0; the water surface is at Z = water_z > 0;
  underwater targets have Z > water_z

### 2.2 Camera Frame (OpenCV Convention)

| Axis | Direction | Notes |
|------|-----------|-------|
| +X   | Right in the image | |
| +Y   | Down in the image | |
| +Z   | **Forward**, into the scene (optical axis) | |

- **Origin**: Camera optical center
- **Units**: Meters
- For a camera pointing straight down, camera +Z aligns with world +Z

### 2.3 Board Frame

| Axis | Direction | Notes |
|------|-----------|-------|
| +X   | Right (along column direction) | |
| +Y   | Down (along row direction) | |
| +Z   | Into the board (away from viewer) | |

- **Origin**: Top-left interior corner (corner_id = 0)
- **Units**: Meters
- All corners lie in the Z = 0 plane: `corner[id] = [col * square_size, row * square_size, 0]`
- Matches OpenCV 4.6+ CharucoBoard convention
- **Source**: `core/board.py:BoardGeometry._compute_corner_positions` (line 41)

### 2.4 Pixel Coordinates

| Component | Direction | Notes |
|-----------|-----------|-------|
| u         | Right (column index) | Corresponds to camera X |
| v         | Down (row index)     | Corresponds to camera Y |

- **Origin**: Top-left corner of the image, (u, v) = (0, 0)
- **Units**: Pixels
- **Convention**: `pixel[0] = u = column`, `pixel[1] = v = row`
- Principal point (cx, cy) maps to the ray [0, 0, 1] in camera frame

---

## 3. Transforms

### 3.1 World-to-Camera Transform

The extrinsics (R, t) transform a point from world frame to camera frame:

```
p_cam = R @ p_world + t
```

- `R`: 3x3 rotation matrix (world → camera)
- `t`: 3x1 translation vector
- **Source**: `core/camera.py:Camera.world_to_camera` (line 81)

### 3.2 Camera Center in World Frame

```
C = -R^T @ t
```

The camera center C is the point in world coordinates that maps to the
origin of the camera frame. Equivalently, it is where the optical center sits
in the world.

- Reference camera: R = I, t = 0, so C = [0, 0, 0]
- Other cameras: C_z ≈ 0 (small variations from coplanarity)
- **Source**: `config/schema.py:CameraExtrinsics.C` (line 78)

### 3.3 Board-to-World Transform

A board pose (rvec, tvec) transforms board-frame points to world-frame points:

```
p_world = R_bw @ p_board + t_bw
```

where `R_bw = cv2.Rodrigues(rvec)` and `t_bw = tvec`.

- tvec gives the world-frame position of the board's origin (corner 0)
- For a face-up board underwater: tvec[2] > water_z
- **Source**: `core/board.py:BoardGeometry.transform_corners` (line 107)

### 3.4 Pose Composition

`compose_poses(T1, T2)` computes the matrix product T1 @ T2:

```python
R_out = R1 @ R2
t_out = R1 @ t2 + t1
```

**Semantics**: T2 is applied first, T1 is applied second.

If T2 maps frame A → frame B and T1 maps frame B → frame C, then
`compose_poses(T1, T2)` maps frame A → frame C.

**Usage pattern** (from `extrinsics.py`):
```python
R_cw, t_cw = invert_pose(R_wc, t_wc)       # cam→world (invert world→cam)
R_bw, t_bw = compose_poses(R_cw, t_cw,      # outer: cam→world
                            R_bc, tvec_bc)    # inner: board→cam
# Result: board→world
```

- **Source**: `utils/transforms.py:compose_poses` (line 54)

> **Note**: The docstring of `compose_poses` describes the argument order
> incorrectly ("T1: A→B, T2: B→C"). The code correctly computes T1 @ T2
> (inner applied first), and all call sites use it correctly. The docstring
> should read: "T1 maps B→C, T2 maps A→B, result maps A→C."

### 3.5 Pose Inversion

```python
R_inv = R^T
t_inv = -R^T @ t
```

Converts a world→camera transform into a camera→world transform (or vice versa).

- **Source**: `utils/transforms.py:invert_pose` (line 83)

### 3.6 Rodrigues Vectors

Compact 3-element rotation representation:
- Axis of rotation: `rvec / ||rvec||`
- Angle of rotation: `||rvec||` (radians)
- Identity rotation: `rvec = [0, 0, 0]`
- Conversion: `R = cv2.Rodrigues(rvec)[0]` (note the `[0]` to extract the matrix)

---

## 4. The Refractive Interface

### 4.1 Interface Plane

The water surface is modeled as an infinite horizontal plane at a fixed
Z-coordinate in the world frame:

```
Z = 0           ← Reference camera (origin)
│
│  air  (n_air = 1.000)
│
Z = water_z     ← Water surface plane
│
│  water (n_water = 1.333)
│
Z > water_z     ← Underwater targets
▼ +Z
```

### 4.2 Interface Normal

```
normal = [0, 0, -1]
```

Points **from water toward air** (opposite to +Z). This is the outward normal
of the water surface when viewed from the water side.

The normal direction is used by `snells_law_3d`, which handles orientation
internally: it flips the normal to point into the destination medium before
applying the vector form of Snell's law.

- **Source**: `core/interface_model.py:Interface` (line 9), default in
  `calibration/interface_estimation.py` (line 188)

### 4.3 The `interface_distance` Parameter

Despite its name, **`interface_distance` is the Z-coordinate of the water
surface in world frame**, not a per-camera physical distance.

After optimization, every camera receives the same value:

```
interface_distance = water_z    (for all cameras)
```

The physical gap from camera i to the water surface is computed internally
by the projection functions:

```
h_c_i = interface_distance - C_z_i
```

For the reference camera (C_z = 0): `h_c = water_z`
For other cameras: `h_c = water_z - C_z_i`

- **Source**: `calibration/_optim_common.py:unpack_params` (line 163-165),
  `core/refractive_geometry.py:_refractive_project_newton` (line 351)
- **History**: Originally stored per-camera distances. Reparameterized to a
  single `water_z` to break the height/distance degeneracy (see KB entry).

### 4.4 The `water_z` Optimization Parameter

In the optimizer, a single scalar `water_z` replaces N per-camera interface
distances. This eliminates the degeneracy between camera Z position and
interface distance: moving a camera vertically (changing C_z) automatically
changes its physical gap to the water.

```
Parameter vector layout:
[tilt(0 or 2) | extrinsics(6*(N-1)) | water_z(1) | board_poses(6*F) | intrinsics(4*N or 0)]
```

- Bounded: `0.01 ≤ water_z ≤ 2.0` meters
- Initial value: `initial_interface_distances[reference_camera]`
  (since C_z_ref = 0, the reference camera's "distance" equals water_z)
- **Source**: `calibration/_optim_common.py:pack_params` (line 26),
  `calibration/interface_estimation.py:optimize_interface` (line 221)

---

## 5. Snell's Law (3D Vector Form)

### 5.1 Implementation

Given incident direction **d**, surface normal **n**, and refractive index
ratio η = n₁/n₂:

```
cos θᵢ = |d · n|
sin²θₜ = η² (1 - cos²θᵢ)

If sin²θₜ > 1: total internal reflection (return None)

cos θₜ = √(1 - sin²θₜ)
t = η·d + (cos θₜ - η·cos θᵢ)·n̂
```

where n̂ is the normal oriented to point into the destination medium.

- **Source**: `core/refractive_geometry.py:snells_law_3d` (line 16)

### 5.2 Air → Water (Camera → Target Direction)

| Quantity | Value |
|----------|-------|
| Incident ray | Points downward (+Z), from camera toward water |
| Surface normal | [0, 0, -1] (points up) |
| n_ratio | n_air / n_water ≈ 0.750 |
| Effect | Ray bends **toward** normal (becomes more vertical) |
| TIR possible? | No (entering denser medium) |

**Analytical check**: At 30° incidence, the refracted angle is:
θₜ = arcsin(0.750 × sin 30°) = arcsin(0.375) ≈ 22.0°. Verified in code.

The refracted ray's horizontal components are scaled by η ≈ 0.75 (reduced),
and the Z component remains positive (still pointing down). The ray becomes
steeper.

### 5.3 Water → Air (Target → Camera Direction)

| Quantity | Value |
|----------|-------|
| Incident ray | Points upward (-Z), from underwater toward surface |
| Surface normal | [0, 0, -1] (points up, same as ray direction) |
| n_ratio | n_water / n_air ≈ 1.333 |
| Effect | Ray bends **away from** normal (becomes more horizontal) |
| TIR possible? | Yes, for θᵢ > arcsin(n_air/n_water) ≈ 48.6° |

The refracted ray's horizontal components are scaled by η ≈ 1.333 (amplified),
and the Z component is -cos θₜ (still pointing up). The ray becomes shallower.

### 5.4 Reversibility

Snell's law is path-reversible: a ray refracted from air to water at angle θᵢ
in air and θₜ in water can be exactly reversed. Tracing the same path from
water to air at angle θₜ recovers the original air angle θᵢ. Verified to
machine precision in tests.

---

## 6. Forward Projection (3D Point → Pixel)

Given an underwater 3D point Q and camera C, find the pixel where Q appears.

### 6.1 The Geometry

The problem reduces to finding the point P on the interface plane where
Snell's law is satisfied:

```
         C (camera, Z ≈ 0)
          \  ← air ray
           \  θ_air
────────────P──────────── Z = water_z
           /  θ_water
          /  ← water ray
         Q (underwater point, Z > water_z)
```

### 6.2 The 1D Snell Equation

By rotational symmetry about the vertical axis through C, the problem
reduces to finding a single scalar r_p (the horizontal distance from C to P):

```
f(r_p) = n_air · sin θ_air - n_water · sin θ_water = 0

where:
    sin θ_air   = r_p / √(r_p² + h_c²)
    sin θ_water = (r_q - r_p) / √((r_q - r_p)² + h_q²)
    h_c = water_z - C_z     (camera-to-interface gap)
    h_q = Q_z - water_z     (interface-to-point gap)
    r_q = ||Q_xy - C_xy||   (horizontal offset)
```

**Properties of f**:
- f(0) = -n_water · r_q / √(r_q² + h_q²) < 0
- f(r_q) = n_air · r_q / √(r_q² + h_c²) > 0
- f'(r_p) = n_air·h_c²/(r_p²+h_c²)^(3/2) + n_water·h_q²/((r_q-r_p)²+h_q²)^(3/2) > 0

Since f is strictly increasing and crosses zero exactly once on (0, r_q),
there is a **unique solution**, and Newton's method converges reliably.

### 6.3 Newton-Raphson Solution

```
Initial guess: r_p = r_q · h_c / (h_c + h_q)   (pinhole/straight-line)
Update:        r_p ← r_p - f(r_p) / f'(r_p)
Convergence:   typically 2-4 iterations to < 1e-9 m
```

After convergence, the interface point is:

```
P = (C_x + r_p · dir_x,  C_y + r_p · dir_y,  water_z)
```

where (dir_x, dir_y) is the unit XY direction from C toward Q.

The pixel is obtained by projecting P through the standard pinhole model:

```
pixel = camera.project(P)
```

This works because the camera observes the air-segment of the ray, which
travels from C through P. Projecting any point on this segment (including P
itself) gives the correct pixel.

- **Source**: `core/refractive_geometry.py:_refractive_project_newton` (line 319)
- **Brent fallback**: `_refractive_project_brent` (line 147) handles non-flat
  interfaces using 1D root-finding with bracketing. Both methods agree to
  < 1e-6 pixels on all test cases.

### 6.4 Batch Projection

`refractive_project_batch` vectorizes the Newton-Raphson solver across N
points, processing all points simultaneously with NumPy broadcasting.

- **Source**: `core/refractive_geometry.py:_refractive_project_newton_batch` (line 416)

---

## 7. Back-Projection (Pixel → Ray in Water)

Given a pixel, compute the refracted ray in water:

1. **Pixel → camera ray**: Undistort the pixel and form a unit direction
   vector in camera frame: `d_cam = [x_norm, y_norm, 1] / ||...||`
2. **Camera ray → world ray**: Rotate to world frame: `d_world = R^T @ d_cam`
3. **Intersect with interface**: Find where the world ray hits the plane
   Z = water_z. The ray origin is C (camera center in world).
4. **Refract**: Apply Snell's law (air→water) to get the refracted direction.
5. **Return**: (intersection_point, refracted_direction) — this defines the
   ray in water.

The intersection point serves as the ray origin in water, and the refracted
direction gives the ray's path through water toward the target.

- **Source**: `core/refractive_geometry.py:trace_ray_air_to_water` (line 72)

---

## 8. Refractive PnP

Estimating a board's pose relative to a camera when the board is underwater.

### 8.1 The Identity-Camera Trick

`refractive_solve_pnp` creates a virtual camera with R = I, t = 0 (camera
frame equals world frame). The board pose (rvec, tvec) then directly gives
the board corners' positions in this frame, and the refractive projection
from Section 6 handles the interface.

This avoids the complexity of jointly solving for the board pose and
accounting for the camera's own extrinsics during PnP.

### 8.2 Pipeline

1. Run standard `cv2.solvePnP` for an initial guess (ignoring refraction)
2. Apply rough depth correction: `tvec[2] *= n_water` (accounts for the
   apparent depth compression)
3. Refine with Levenberg-Marquardt, minimizing refractive reprojection error
4. The result is the board pose in camera frame

To get the board pose in world frame, compose with the camera's world pose:

```python
R_cw, t_cw = invert_pose(R_wc, t_wc)           # cam→world
R_bw, t_bw = compose_poses(R_cw, t_cw,          # outer: cam→world
                            R_bc, tvec_bc)        # inner: board→cam
```

- **Source**: `calibration/extrinsics.py:refractive_solve_pnp` (line 108)

---

## 9. Calibration Pipeline Geometry

### Stage 1: Intrinsics (In-Air)

Standard OpenCV camera calibration with a checkerboard in air. No refraction
involved. Produces K and distortion coefficients per camera.

### Stage 2: Extrinsics Initialization

BFS traversal through a pose graph. Reference camera is fixed at the world
origin (R=I, t=0). Other cameras are located by chaining board-to-camera
and camera-to-world transforms through shared board observations.

### Stage 3: Joint Refractive Optimization

Jointly optimizes:
- Non-reference camera extrinsics: 6 params each (rvec + tvec)
- Global water_z: 1 param
- Board poses: 6 params each (rvec + tvec)

Total parameters: 6(N-1) + 1 + 6F, where N = cameras, F = frames.

The cost function computes refractive reprojection error for every observed
corner across all cameras and frames.

### Stage 4: Intrinsic Refinement (Optional)

Same as Stage 3 but also optimizes per-camera (fx, fy, cx, cy). Adds 4N
parameters. Distortion coefficients are held fixed.

---

## 10. Key Invariants and Sanity Checks

These should always hold for a valid calibration:

| Invariant | Expected |
|-----------|----------|
| Reference camera C | [0, 0, 0] |
| Reference camera R, t | I, [0, 0, 0] |
| Other camera C_z | Near 0 (cameras are roughly coplanar) |
| water_z | Positive (surface is below cameras) |
| interface_distance | Same as water_z for all cameras |
| Board corner Z (world) | > water_z (board is underwater) |
| h_c = water_z - C_z | Positive for all cameras |
| Interface normal | [0, 0, -1] (points up) |
| Camera rays | Have positive Z component (point downward) |
| Refracted rays (water) | Have positive Z component (still downward) |
| Refracted rays (air→water) | More vertical than incident rays |

---

## 11. Issues Found During This Audit

### 12.1 `compose_poses` Docstring (Minor)

**File**: `utils/transforms.py`, line 54

The docstring states: "If T1 transforms from frame A to frame B, and T2
transforms from frame B to frame C, then T_combined transforms from frame
A to frame C."

This is incorrect. The function computes T1 @ T2 (T2 applied first), so the
correct statement is: "If T1 transforms B to C and T2 transforms A to B,
then compose_poses(T1, T2) transforms A to C."

All call sites use the function correctly (outer transform first, inner
second). Only the docstring is wrong.

### 12.2 `Interface.camera_distances` Naming (Cosmetic)

**File**: `core/interface_model.py`, line 14

The attribute name `camera_distances` and its docstring ("Per-camera interface
distances (distance from camera to water surface)") suggest per-camera
physical distances. In reality, every camera receives the same value
(water_z), and the physical gap is computed internally. The name is a
historical artifact from before the water_z reparameterization.

Renaming is not urgent since the KB entry and this document clarify the
semantics, but a future cleanup could rename to `camera_interface_z` or
similar.
