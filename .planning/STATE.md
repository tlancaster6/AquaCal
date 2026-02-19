# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** v1.5 AquaKit Integration — Phase 14: Geometry Rewiring

## Current Position

Phase: 14 of 17 (Geometry Rewiring)
Plan: 1 of TBD in current phase
Status: In progress
Last activity: 2026-02-19 — Phase 14 Plan 01 complete (bridge module)

Progress: [█████████░░░░░░░░░░░] 13/17 phases complete (v1.2 + v1.4 shipped, Phase 13 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 32
- v1.2: 20 plans, ~1.85 hours
- v1.4: 10 plans

**Recent Trend:**
- Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Recent decisions affecting current work:
- [v1.5 start]: NumPy internals retained; torch conversion happens only at AquaKit call boundaries
- [v1.5 start]: Delete-after-tests strategy — rewire first, test equivalence, then delete originals
- [v1.5 start]: AquaKit bug fixes performed as needed during rewiring
- [13-01]: aquakit added as hard dep (>=1.0,<2); _check_torch() guard in __init__.py fails fast with clear ImportError
- [13-01]: CI uses CPU-only torch index URL to minimize download size; noqa: E402 suppresses ruff for post-guard imports
- [13-02]: PyTorch prerequisite is the FIRST item in the install section (not a footnote)
- [13-02]: CHANGELOG uses Unreleased section since v1.5 not yet released
- [14-01]: _make_interface_params() factory prevents aquakit.types.InterfaceParams name collision with aquacal's own InterfaceParams
- [14-01]: _bridge_refractive_project uses two-step: AquaKit finds interface point, camera.project() maps to pixel
- [14-01]: Bridge module owns all torch imports; callers import bridge functions, never torch directly

### Pending Todos

- Design better hero image for README (deferred from Phase 11 — user wants to rethink concept)
- Reduce memory and CPU load during calibration

### Blockers/Concerns

- PyTorch is not an aquakit pip dependency — users must install it manually; SETUP-02 must document this clearly
- `refractive_project` API change is non-trivial (two-step process; returns interface point not pixel) — Phase 14 needs careful call-site audit

## Session Continuity

Last session: 2026-02-19 (phase 14 plan 01 executed)
Stopped at: Completed 14-geometry-rewiring/14-01-PLAN.md
Resume file: .planning/phases/14-geometry-rewiring/14-01-SUMMARY.md
