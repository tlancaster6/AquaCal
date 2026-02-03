# Task: 0.3 Add requirements.txt

## Objective

Create `requirements.txt` for users who prefer `pip install -r requirements.txt` over installing the package.

## Context Files

Read these files before starting (in order):

1. `CLAUDE.md` — Project conventions and workflow
2. `pyproject.toml` — Authoritative source for dependencies (created in task 0.1)

## Modify

Files to create:

- `requirements.txt`

## Do Not Modify

Everything not listed above. In particular:

- `pyproject.toml` — this is the authoritative source; requirements.txt mirrors it
- Any source code files

## Acceptance Criteria

- [ ] `requirements.txt` exists at project root
- [ ] Contains all required dependencies from `pyproject.toml`: `numpy`, `scipy`, `opencv-python`, `pyyaml`
- [ ] Contains development dependencies: `pytest`, `pytest-cov`, `mypy`, `black`
- [ ] Contains visualization dependencies: `matplotlib`, `pandas`
- [ ] Has a header comment explaining relationship to pyproject.toml
- [ ] No version pins (matches pyproject.toml philosophy), except `opencv-python>=4.6` if specified there
- [ ] Installs successfully: `pip install -r requirements.txt`

## Notes

### Format

```txt
# Requirements for AquaCal
#
# This file mirrors dependencies from pyproject.toml for pip-only users.
# For development, prefer: pip install -e ".[dev,viz]"
#
# Authoritative source: pyproject.toml

# Core dependencies
numpy
scipy
opencv-python>=4.6
pyyaml

# Development
pytest
pytest-cov
mypy
black

# Visualization
matplotlib
pandas
```

### Why both pyproject.toml and requirements.txt?

- `pyproject.toml` is the modern standard for package metadata and dependencies
- `requirements.txt` is familiar to many users and works with `pip install -r`
- Some CI/CD systems and Docker workflows expect requirements.txt
- The comment header clarifies that pyproject.toml is authoritative

### Keep in sync

When dependencies change, update pyproject.toml first, then mirror to requirements.txt. The comment header reminds future maintainers of this relationship.
