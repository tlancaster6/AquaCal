# Glossary

Quick reference for domain-specific terms used in AquaCal.

---

**Auxiliary camera**
: Camera excluded from joint optimization (Stages 2-4) and registered post-hoc against fixed board poses and water_z. Used for cameras with poor intrinsic calibration or different viewing geometry that would degrade the primary solution.

**Board pose**
: Rigid transformation (rotation + translation) defining the position and orientation of the calibration board in world coordinates for a single frame. Optimized in Stage 3.

**Bundle adjustment**
: Simultaneous optimization of multiple camera poses and 3D point positions (or board poses) to minimize reprojection error. AquaCal uses refractive bundle adjustment in Stage 3.

**Camera frame**
: Coordinate system with origin at camera optical center. OpenCV convention: +X right, +Y down, +Z forward (optical axis).

**ChArUco board**
: Calibration target combining a checkerboard pattern with ArUco markers. Provides robust corner detection and unique corner identification. "ChArUco" = Checkerboard + ArUco.

**Extrinsics**
: Camera pose parameters defining the transformation from world coordinates to camera coordinates. Consists of rotation matrix **R** (3×3) and translation vector **t** (3×1).

**Interface**
: The water surface (air-water boundary) where light refraction occurs. Modeled as a plane with normal vector and refractive indices on both sides.

**Interface normal**
: Unit vector pointing from water toward air, typically [0, 0, -1] in world frame. Defines the orientation of the water surface.

**Intrinsics**
: Camera-specific parameters independent of pose: intrinsic matrix **K** (focal lengths and principal point) and distortion coefficients. Calibrated in Stage 1.

**Pose graph**
: Graph structure where nodes represent camera-frame observations and edges connect observations of the same board pose. Used in Stage 2 for extrinsic initialization via BFS traversal.

**Reference camera**
: The first camera in your configuration, which defines the world coordinate origin (R = I, t = 0). All other camera poses are expressed relative to this camera.

**Refractive index**
: Ratio of light speed in vacuum to light speed in a medium. AquaCal uses n_air (1.0) and n_water (typically 1.333 for fresh water at 20°C).

**Refractive projection**
: Process of projecting a 3D world point to 2D pixel coordinates while accounting for light bending at the water surface. Uses Snell's law and ray tracing.

**Reprojection error**
: Distance (in pixels) between an observed 2D corner and its predicted location from 3D position via camera projection. Minimized during calibration.

**Rodrigues vector**
: Compact 3-parameter rotation representation: axis direction encodes rotation axis, magnitude encodes rotation angle in radians. Equivalent to axis-angle representation.

**Snell's law**
: Physical law governing light refraction at an interface: n₁ sin(θ₁) = n₂ sin(θ₂), where n is refractive index and θ is angle from surface normal.

**Validation set**
: Subset of calibration frames held out during optimization and used only for quality assessment. Prevents overfitting and provides unbiased error estimates.

**water_z**
: Z-coordinate of the water surface in world frame, in meters. Shared across all cameras after Stage 3 optimization. Replaces the older term "interface_distance" (which was misleading).

**World frame**
: Global coordinate system with origin at the reference camera's optical center. AquaCal convention: +X right, +Y forward, +Z down (into water). Units: meters.

---

## See Also

- [Coordinate Conventions](coordinates.md) — Detailed explanation of coordinate frames and transforms
- [Refractive Geometry](refractive_geometry.md) — How refraction and ray tracing work
- [Optimizer Pipeline](optimizer.md) — The four calibration stages
