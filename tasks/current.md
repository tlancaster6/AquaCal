# Task: 1.3 Implement board geometry

## Objective

Implement `core/board.py` with the `BoardGeometry` class for ChArUco board 3D geometry and corner position calculations.

## Context Files

Read these files before starting (in order):

1. `CLAUDE.md` — Project conventions (especially coordinate frames)
2. `docs/agent_implementation_spec.md` (lines 366-471) — BoardGeometry class specification
3. `docs/COORDINATES.md` — Board frame definition
4. `src/aquacal/config/schema.py` — BoardConfig dataclass (dependency)

## Dependencies

This task depends on:
- **Task 1.1** (`config/schema.py`) — Must be complete for `BoardConfig` import

## Modify

Files to create:

- `src/aquacal/core/board.py`
- `tests/unit/test_board.py`

Files to update:

- `src/aquacal/core/__init__.py` — Uncomment BoardGeometry export

## Do Not Modify

Everything not listed above. In particular:

- `config/schema.py` — already implemented
- `utils/transforms.py` — board.py should use cv2.Rodrigues directly, not import transforms
- Other core modules (camera.py, interface_model.py, etc.)

## Acceptance Criteria

- [ ] `board.py` implements `BoardGeometry` class with:
  - `__init__(self, config: BoardConfig)`
  - `corner_positions` property returning `dict[int, Vec3]`
  - `num_corners` property returning `int`
  - `get_opencv_board()` returning `cv2.aruco.CharucoBoard`
  - `transform_corners(rvec, tvec)` returning `dict[int, Vec3]`
  - `get_corner_array(corner_ids)` returning `NDArray[np.float64]` shape (N, 3)
- [ ] Board frame convention: origin at top-left corner, X right, Y down, Z into board
- [ ] Corner IDs match OpenCV CharucoBoard corner ordering
- [ ] `num_corners` equals `(squares_x - 1) * (squares_y - 1)`
- [ ] Corner positions are in meters (using `square_size` from config)
- [ ] Tests pass: `pytest tests/unit/test_board.py -v`
- [ ] Type check passes: `mypy src/aquacal/core/board.py --ignore-missing-imports`
- [ ] `core/__init__.py` exports `BoardGeometry`

## Notes

### Board frame convention

The board frame follows OpenCV CharucoBoard convention (OpenCV 4.6+):
- **Origin**: Top-left interior corner (corner ID 0)
- **X-axis**: Points right (toward increasing column)
- **Y-axis**: Points down (toward increasing row)
- **Z-axis**: Points into the board (away from viewer, completing right-handed system)

```
    Corner layout (8x6 board has 7x5=35 interior corners):

    0----1----2----3----4----5----6    → X
    |    |    |    |    |    |    |
    7----8----9---10---11---12---13
    |    |    |    |    |    |    |
   14---15---16---17---18---19---20
    ...
    ↓
    Y
```

### Corner position calculation

For an 8x6 board (squares_x=8, squares_y=6):
- Interior corners: (8-1) × (6-1) = 35 corners
- Corner ID `i` is at row `i // (squares_x - 1)`, column `i % (squares_x - 1)`
- Position: `[col * square_size, row * square_size, 0.0]`

```python
def _compute_corner_positions(self) -> dict[int, Vec3]:
    positions = {}
    cols = self.config.squares_x - 1
    rows = self.config.squares_y - 1
    for corner_id in range(cols * rows):
        col = corner_id % cols
        row = corner_id // cols
        positions[corner_id] = np.array([
            col * self.config.square_size,
            row * self.config.square_size,
            0.0
        ], dtype=np.float64)
    return positions
```

### OpenCV CharucoBoard creation

Use OpenCV 4.6+ API:

```python
def get_opencv_board(self) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, self.config.dictionary)
    )
    board = cv2.aruco.CharucoBoard(
        (self.config.squares_x, self.config.squares_y),
        self.config.square_size,
        self.config.marker_size,
        dictionary
    )
    return board
```

Note: The dictionary string like `"DICT_4X4_50"` maps to `cv2.aruco.DICT_4X4_50`.

### Transform corners

Apply rigid transform (rvec, tvec) to move corners from board frame to world frame:

```python
def transform_corners(self, rvec: Vec3, tvec: Vec3) -> dict[int, Vec3]:
    R, _ = cv2.Rodrigues(rvec)
    return {
        corner_id: R @ pos + tvec
        for corner_id, pos in self.corner_positions.items()
    }
```

### Test cases

```python
import numpy as np
import pytest
from aquacal.config.schema import BoardConfig
from aquacal.core.board import BoardGeometry

@pytest.fixture
def board_config():
    return BoardConfig(
        squares_x=8,
        squares_y=6,
        square_size=0.03,  # 3cm
        marker_size=0.022,
        dictionary="DICT_4X4_50"
    )

class TestBoardGeometry:
    def test_num_corners(self, board_config):
        """8x6 board has 7x5=35 interior corners."""
        board = BoardGeometry(board_config)
        assert board.num_corners == 35

    def test_corner_0_at_origin(self, board_config):
        """Corner 0 should be at origin."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[0],
            np.array([0, 0, 0])
        )

    def test_corner_1_position(self, board_config):
        """Corner 1 should be one square_size to the right."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[1],
            np.array([0.03, 0, 0])
        )

    def test_corner_7_position(self, board_config):
        """Corner 7 (first of second row) should be one square_size down."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[7],
            np.array([0, 0.03, 0])
        )

    def test_transform_identity(self, board_config):
        """Identity transform should not change positions."""
        board = BoardGeometry(board_config)
        transformed = board.transform_corners(np.zeros(3), np.zeros(3))
        for corner_id, pos in board.corner_positions.items():
            np.testing.assert_allclose(transformed[corner_id], pos)

    def test_transform_translation(self, board_config):
        """Pure translation should shift all corners."""
        board = BoardGeometry(board_config)
        tvec = np.array([1.0, 2.0, 3.0])
        transformed = board.transform_corners(np.zeros(3), tvec)
        for corner_id, pos in board.corner_positions.items():
            np.testing.assert_allclose(transformed[corner_id], pos + tvec)

    def test_get_corner_array_shape(self, board_config):
        """get_corner_array should return (N, 3) array."""
        board = BoardGeometry(board_config)
        ids = np.array([0, 1, 7, 8])
        pts = board.get_corner_array(ids)
        assert pts.shape == (4, 3)

    def test_get_corner_array_values(self, board_config):
        """get_corner_array values should match corner_positions."""
        board = BoardGeometry(board_config)
        ids = np.array([0, 1])
        pts = board.get_corner_array(ids)
        np.testing.assert_allclose(pts[0], board.corner_positions[0])
        np.testing.assert_allclose(pts[1], board.corner_positions[1])

    def test_opencv_board_creation(self, board_config):
        """Should create valid OpenCV CharucoBoard."""
        board = BoardGeometry(board_config)
        cv_board = board.get_opencv_board()
        assert cv_board is not None
        # Verify it has the expected number of corners
        assert len(cv_board.getChessboardCorners()) == 35
```

### Import structure

After implementation, update `core/__init__.py`:

```python
"""Core geometry modules."""

from aquacal.core.board import BoardGeometry

__all__ = [
    "BoardGeometry",
    # Future: Camera, Interface, refractive functions
]
```