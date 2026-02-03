# Coordinate Systems Quick Reference

## World Frame

```
        +Z (up, out of water)
         │
         │
         │
         └───────── +Y
        ╱
       ╱
      +X
```

- **Origin**: Reference camera (cam0) center, or center of interface plane
- **+X**: Horizontal (e.g., East or rightward)
- **+Y**: Horizontal (e.g., North or forward)  
- **+Z**: Up, pointing out of the water into air
- **Units**: Meters

## Camera Frame

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
    AIR (n = 1.0)
    ─────────────────────  ← Interface plane (Z = interface_height)
    WATER (n = 1.333)
```

- **Normal vector**: `[0, 0, 1]` — points UP, from water toward air
- **Interface height**: Z-coordinate of the water surface in world frame
- **Interface distance**: Vertical distance from a camera center to the interface
- **Above water**: Z > interface_height (cameras are here)
- **Below water**: Z < interface_height (calibration targets are here)

## Transforms

### World to Camera

```python
p_camera = R @ p_world + t
```

- `R`: 3×3 rotation matrix (world → camera)
- `t`: 3×1 translation vector
- `p_world`: Point in world coordinates
- `p_camera`: Point in camera coordinates

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

## Quick Sanity Checks

- Camera center Z should be **positive** (above water)
- Target points Z should be **negative** (below water)
- Interface normal Z should be **positive** (pointing up)
- Rays from camera should have **negative** Z component (pointing down into water)
- After refraction, rays should still have **negative** Z component (pointing down)
