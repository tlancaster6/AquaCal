# Coordinate Systems Quick Reference

## World Frame

```
     +X (horizontal)
      │
      │
      └───────── +Y (horizontal)
     ╱
    ╱
   +Z (DOWN, into water)
```

- **Origin**: Reference camera (cam0) optical center
- **+X**: Horizontal (right when viewed from above)
- **+Y**: Horizontal (forward when viewed from above)
- **+Z**: Down, into the water
- **Units**: Meters

## Camera Frame (OpenCV Convention)

```
         +Z (forward, into scene)
        ╱
       ╱
      ╱
     └───────── +X (right in image)
     │
     │
     +Y (down in image)
```

- **Origin**: Camera optical center (projection center)
- **+X**: Right in the image
- **+Y**: Down in the image
- **+Z**: Forward, into the scene (optical axis direction)
- **Units**: Meters

**For cameras pointing straight down**: Camera +Z aligns with world +Z. Reference camera has R = I, t = 0.

## Pixel Coordinates

```
    (0,0) ────────────► +u (column)
      │
      │
      │
      ▼
     +v (row)
```

- **(0, 0)**: Top-left corner of image
- **u**: Column index (horizontal, corresponds to X)
- **v**: Row index (vertical, corresponds to Y)
- **Convention**: `pixel[0] = u = column`, `pixel[1] = v = row`
- **Units**: Pixels

## Interface (Water Surface)

```
    Z = 0          ← Reference camera (origin)
      │
      │  (air)
      │
    Z = d          ← Interface plane (water surface), d = interface_distance
      │
      │  (water)
      │
    Z = d + depth  ← Underwater point
      ▼
     +Z
```

- **Normal vector**: `[0, 0, -1]` — points UP, from water toward air (opposite to +Z)
- **Interface height**: Z-coordinate of water surface (positive value, below camera)
- **Interface distance**: Same as interface height when origin is at camera
- **In air**: Z < interface_distance (camera is here, near Z = 0)
- **In water**: Z > interface_distance (targets are here)
- **Deeper**: Larger positive Z values

## Transforms

### World to Camera

```python
p_camera = R @ p_world + t
```

- `R`: 3×3 rotation matrix (world → camera)
- `t`: 3×1 translation vector
- `p_world`: Point in world coordinates
- `p_camera`: Point in camera coordinates

### Reference Camera

For cam0 (at origin, pointing down):
```python
R = np.eye(3)  # Identity
t = np.zeros(3)  # Zero
```

### Other Cameras

For other cameras in the planar array:
- `R` is primarily a rotation about Z (roll difference)
- `t` is primarily XY translation (horizontal offset)
- Small corrections for pitch/yaw misalignment

### Camera Center

```python
C = -R.T @ t
```

- `C`: Camera center (optical center) in world coordinates

### Pose Composition

If `T1` transforms A → B and `T2` transforms B → C, then A → C is:

```python
R_combined = R2 @ R1
t_combined = R2 @ t1 + t2
```

### Pose Inversion

```python
R_inv = R.T
t_inv = -R.T @ t
```

## Rotation Representations

| Representation | Size | Notes |
|----------------|------|-------|
| Rotation matrix | 3×3 | Direct use in transforms |
| Rodrigues vector | 3 | Compact, used in optimization |
| Quaternion | 4 | No gimbal lock, needs normalization |
| Euler angles | 3 | Avoid (gimbal lock issues) |

### Rodrigues Convention

```python
import cv2
R, _ = cv2.Rodrigues(rvec)  # rvec (3,) → R (3,3)
rvec, _ = cv2.Rodrigues(R)  # R (3,3) → rvec (3,)
```

- Axis: `rvec / ||rvec||`
- Angle: `||rvec||` (radians)
- Identity rotation: `rvec = [0, 0, 0]`

## Refractive Index Values

| Medium | n |
|--------|---|
| Air | 1.000 |
| Fresh water (20°C) | 1.333 |
| Sea water (typical) | 1.339–1.341 |

## Snell's Law Direction

Light traveling from camera into water:
1. Ray starts at camera (Z ≈ 0)
2. Travels in +Z direction (down toward water)
3. Hits interface at Z = interface_distance
4. Refracts toward normal (bends toward vertical)
5. Continues in +Z direction (down into water)

Since entering denser medium (air → water), ray bends toward the normal, becoming more vertical.

## Quick Sanity Checks

- Reference camera center is at Z = 0
- Other camera centers have Z ≈ 0 (small variations)
- Interface is at Z = interface_distance > 0
- Target points have Z > interface_distance (below surface)
- All cameras looking down: rays have positive Z component
- Interface normal is `[0, 0, -1]` (pointing up, opposite to +Z)
- After refraction, rays still have positive Z component (still going down)

## Example Values

Typical setup:
```python
cam0_center = [0, 0, 0]              # Origin
cam1_center = [0.15, 0, 0.002]       # 15cm right, 2mm lower
interface_distance = 0.12            # Water surface 12cm below cam0
interface_normal = [0, 0, -1]        # Points up
target_point = [0.05, 0.03, 0.35]    # 35cm below cam0, i.e., 23cm underwater
```