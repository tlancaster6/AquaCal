# CLAUDE.md

## Project

Refractive multi-camera calibration library for an array of cameras in air viewing an underwater volume through the water surface.

## Workflow

1. Read this file (you're doing that now)
2. Read `tasks/current.md` for your specific task
3. Read files listed in "Context Files" section of the task
4. Implement changes to files listed in "Modify" section
5. Run tests specified in "Acceptance Criteria"
6. Update `CHANGELOG.md` (see Exit Protocol below)
7. If task is incomplete, write `tasks/handoff.md`

## Exit Protocol

Before ending your session:

1. Run the tests specified in Acceptance Criteria
2. Append an entry to `CHANGELOG.md`:
   ```
   ## YYYY-MM-DD
   ### [module/file.py]
   - Summary of changes (2-3 lines)
   - Additional changes if multiple files
   ```
3. If task is **INCOMPLETE**:
   - Create or update `tasks/handoff.md` with what's done, what's not, and specific next steps
4. If task is **COMPLETE**:
   - Delete `tasks/handoff.md` if it exists
5. Report final status to user

## Conventions

### Coordinates
- **World frame**: Z-up (out of water), X-Y horizontal, origin at reference camera or interface
- **Camera frame**: Z-forward (into scene), X-right, Y-down
- **Interface normal**: [0, 0, 1] points up (from water toward air)
- **Pixels**: (u, v) = (column, row), origin at top-left

### Code Style
- Formatter: Black
- Docstrings: Google style
- Type hints: Use `numpy.typing.NDArray` with shapes in docstrings
- Imports: Standard library, then third-party, then local (blank line between groups)

### Error Handling
- **Geometric failures** (TIR, ray misses plane, etc.): Return `None`, do not raise
- **Invalid input** (wrong type, bad shape): Raise `ValueError`
- **Missing files**: Raise `FileNotFoundError`
- **Calibration failures**: Raise `CalibrationError` or subclass

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

## Key Reference Files

| File | Purpose |
|------|---------|
| `docs/agent_implementation_spec.md` | Detailed function signatures and type definitions |
| `docs/development_plan.md` | Architecture overview and module responsibilities |
| `docs/COORDINATES.md` | Coordinate frame quick reference |
| `DEPENDENCIES.yaml` | Module dependency graph |
| `TASKS.md` | Overall task status (do not modify) |

## Common Commands

```bash
# Test single module
pytest tests/unit/test_<name>.py -v

# Test all unit tests
pytest tests/unit/ -v

# Type check a module
mypy <path> --ignore-missing-imports

# Format code
black <path>

# Check what depends on a module
grep -A 5 "^  <module>:" DEPENDENCIES.yaml
```

## Do Not

- Modify files not listed in your task's "Modify" section
- Add new dependencies without noting in CHANGELOG
- Skip running tests before exit
- Modify `TASKS.md` (orchestrator maintains this)
- Delete or overwrite `CHANGELOG.md` (append only)
