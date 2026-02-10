# Task: Add Legacy ChArUco Board Pattern Support

## Objective

Add a `legacy_pattern` option to `BoardConfig` so users can specify that a board uses the legacy ChArUco layout (marker in top-left corner instead of a solid square). This affects detection only — corner positions are identical between legacy and new patterns.

## Background

OpenCV 4.6+ defaults to a new ChArUco pattern where the top-left cell is a solid square. Older printed boards have a marker in the top-left cell (legacy pattern). If the wrong pattern is used, detection will fail or return incorrect corner IDs. The OpenCV API provides `CharucoBoard.setLegacyPattern(True)` to switch.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/config/schema.py` (lines 26-41) — `BoardConfig` dataclass
2. `src/aquacal/core/board.py` (lines 87-103) — `get_opencv_board()` creates the CharucoBoard
3. `src/aquacal/calibration/pipeline.py` (lines 69-89) — `load_config()` parses `board` and `intrinsic_board` sections
4. `src/aquacal/config/example_config.yaml` — Example config
5. `tests/unit/test_board.py` — Existing board tests

## Modify

- `src/aquacal/config/schema.py`
- `src/aquacal/core/board.py`
- `src/aquacal/calibration/pipeline.py`
- `src/aquacal/config/example_config.yaml`
- `tests/unit/test_board.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/io/detection.py` — Uses `get_opencv_board()` which will handle legacy automatically
- `src/aquacal/calibration/intrinsics.py` — Same, uses board from BoardGeometry
- `src/aquacal/cli.py` — Generated config can omit `legacy_pattern` since default is False

## Design

### Schema Change (`schema.py`)

Add `legacy_pattern` to `BoardConfig`:

```python
@dataclass
class BoardConfig:
    squares_x: int
    squares_y: int
    square_size: float
    marker_size: float
    dictionary: str
    legacy_pattern: bool = False  # True for boards with marker in top-left cell
```

Update the docstring to mention the new field.

### Board Change (`board.py`)

In `get_opencv_board()`, call `setLegacyPattern` when the flag is set:

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
    if self.config.legacy_pattern:
        board.setLegacyPattern(True)
    return board
```

**Do NOT change `_compute_corner_positions()`** — corner positions are identical between legacy and new patterns (verified empirically).

### Config Parsing Change (`pipeline.py`)

Parse `legacy_pattern` from both board sections in `load_config()`:

```python
# In the main board section (~line 71):
board = BoardConfig(
    squares_x=board_data["squares_x"],
    squares_y=board_data["squares_y"],
    square_size=board_data["square_size"],
    marker_size=board_data["marker_size"],
    dictionary=board_data.get("dictionary", "DICT_4X4_50"),
    legacy_pattern=board_data.get("legacy_pattern", False),
)

# In the intrinsic_board section (~line 83):
intrinsic_board = BoardConfig(
    squares_x=intrinsic_data["squares_x"],
    squares_y=intrinsic_data["squares_y"],
    square_size=intrinsic_data["square_size"],
    marker_size=intrinsic_data["marker_size"],
    dictionary=intrinsic_data.get("dictionary", "DICT_4X4_50"),
    legacy_pattern=intrinsic_data.get("legacy_pattern", False),
)
```

### Example Config Change (`example_config.yaml`)

Add `legacy_pattern` as a commented-out option to both board sections:

```yaml
board:
  squares_x: 8
  squares_y: 6
  square_size: 0.030
  marker_size: 0.022
  dictionary: "DICT_4X4_50"
  # legacy_pattern: true  # Uncomment if board has marker in top-left cell
```

Same for the commented-out `intrinsic_board` section.

## Acceptance Criteria

- [ ] `BoardConfig` has `legacy_pattern: bool = False`
- [ ] `get_opencv_board()` calls `setLegacyPattern(True)` when flag is set
- [ ] `load_config()` parses `legacy_pattern` from both `board` and `intrinsic_board` sections
- [ ] Default is `False` (backward compatible)
- [ ] `example_config.yaml` shows `legacy_pattern` as commented-out option
- [ ] New test: `get_opencv_board()` with `legacy_pattern=True` returns a board where `getLegacyPattern()` is True
- [ ] New test: `get_opencv_board()` with default config returns a board where `getLegacyPattern()` is False
- [ ] New test: `load_config()` with `legacy_pattern: true` parses correctly
- [ ] New test: `load_config()` without `legacy_pattern` defaults to False
- [ ] Existing tests pass: `pytest tests/unit/test_board.py tests/unit/test_pipeline.py -v`
- [ ] Do NOT run the synthetic test suite

## Notes

1. **Corner positions are NOT affected**: Verified empirically — `getChessboardCorners()` returns identical positions regardless of legacy flag. Only marker placement differs.

2. **This is detection-critical**: If a physical board uses the legacy pattern but config says `false` (default), ChArUco detection will either fail entirely or return wrong corner IDs. This is a silent correctness issue, not a crash.

3. **Each board can differ**: The extrinsic board and intrinsic board may use different patterns. The user's scenario: one board is legacy, the other is not. Both `board` and `intrinsic_board` sections need independent `legacy_pattern` fields.

4. **CLI `init` command**: No changes needed. Generated configs omit `legacy_pattern` which defaults to `False`. Users who need it can add it manually.

## Model Recommendation

**Haiku** — Trivial plumbing: one field, one method call, two parse sites, example config update. No logic complexity.