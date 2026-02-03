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
- **World frame**: Z-down (into water), X-Y horizontal, origin at reference camera optical center
- **Camera frame**: Z-forward (into scene), X-right, Y-down (OpenCV convention)
- **Interface normal**: [0, 0, -1] points up (from water toward air, opposite to +Z)
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

## Testing Conventions

- **Unit test files**: `tests/unit/test_<module_name>.py` (e.g., `test_transforms.py` for `src/aquacal/utils/transforms.py`)
- **Test classes**: `TestClassName` (e.g., `TestSnellsLaw`)
- **Test functions**: `test_<description>` (e.g., `test_normal_incidence`)
- **Fixtures**: Place shared fixtures in `tests/conftest.py`

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

---

## Task Writer Mode

If the orchestrator says **"task writer mode"** or asks you to write/create a task file:

### Your Role
You help the orchestrator write task files for coding agents. You do NOT write implementation code.

### What To Read
1. This file
2. `TASKS.md` — to see overall progress and task IDs
3. `DEPENDENCIES.yaml` — to determine what context files are needed
4. `docs/agent_implementation_spec.md` — for function signatures and types
5. `docs/development_plan.md` — for module responsibilities

### What To Output
A complete `tasks/current.md` file following this template:

```markdown
# Task: [ID from TASKS.md] [Short name]

## Objective

[One sentence describing what this task accomplishes]

## Context Files

Read these files before starting (in order):

1. `path/to/file.py` — [why this file is needed]
2. `path/to/other.py` (lines X-Y) — [why this section is needed]

## Modify

Files to create or edit:

- `path/to/target.py`
- `tests/unit/test_target.py`

## Do Not Modify

Everything not listed above. In particular:
- [list any files that might seem related but should not be touched]

## Acceptance Criteria

- [ ] [Specific, verifiable criterion]
- [ ] [Another criterion]
- [ ] Tests pass: `pytest tests/unit/test_<name>.py -v`
- [ ] No modifications to files outside "Modify" list

## Notes

[Design decisions, gotchas, references to specific sections of the spec]
```

### Guidelines

1. **Context Files must be minimal** — only what's needed for THIS task
2. **Include line numbers** when only part of a file is relevant
3. **Acceptance Criteria must be verifiable** — runnable commands or inspectable outcomes
4. **Reference the spec** — point to specific sections of `agent_implementation_spec.md`
5. **One module per task** — don't combine unrelated work
6. **Check dependencies** — ensure all dependencies are implemented before assigning a task

### Do Not (In Task Writer Mode)

- Write implementation code
- Modify any source files
- Create overly broad tasks
- Assign tasks whose dependencies aren't complete (check `DEPENDENCIES.yaml` status)