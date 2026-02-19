# Phase 13: Setup - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Add AquaKit as a hard dependency of AquaCal and handle the PyTorch prerequisite gracefully. After this phase, `pip install aquacal` also installs aquakit, and missing torch produces a clear error at import time. CI is updated to install CPU-only torch.

</domain>

<decisions>
## Implementation Decisions

### Dependency strategy
- `aquakit>=1.0,<2` as a hard dependency in pyproject.toml (not optional)
- PyTorch is NOT declared in AquaCal's dependencies — user installs it themselves
- AquaKit v1.0.0 is the current version on PyPI

### Import-time behavior
- Fail fast: `import aquacal` raises `ImportError` immediately if torch is not installed
- Check lives in `aquacal/__init__.py` — top-level, before any aquakit imports
- Check for torch presence only (no minimum version enforcement)
- Error message: minimal + link — "AquaCal requires PyTorch. Install it first: `pip install torch`. See https://pytorch.org/get-started/ for GPU variants."

### Version compatibility
- CI-tested range strategy: pyproject.toml allows `>=1.0,<2`, CI tests against latest aquakit
- If CI breaks on a new aquakit release, update AquaCal accordingly
- Add torch + aquakit to CI matrix in this phase (not deferred)
- CI uses CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### PyTorch documentation
- Document in all entry points: README install section, Sphinx getting-started page, import-time error message
- Torch requirement is top of install section — first thing users see: "Install PyTorch first, then pip install aquacal"
- CHANGELOG entry noting the new torch/aquakit dependency (not a full migration guide)

### Claude's Discretion
- Exact wording of the import error message (minimal + link style decided)
- CHANGELOG entry formatting
- Whether to add a `_check_torch()` helper or inline the check

</decisions>

<specifics>
## Specific Ideas

- Follow the same pattern as AquaKit's own REWIRING.md prerequisite section for the install instructions
- The error message should be actionable — not just "missing torch" but "here's how to fix it"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-setup*
*Context gathered: 2026-02-19*
