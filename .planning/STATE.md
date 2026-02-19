# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.
**Current focus:** v1.5 AquaKit Integration — Phase 13: Setup

## Current Position

Phase: 13 of 17 (Setup)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-02-19 — Roadmap created for v1.5 AquaKit Integration

Progress: [████████░░░░░░░░░░░░] 12/17 phases complete (v1.2 + v1.4 shipped)

## Performance Metrics

**Velocity:**
- Total plans completed: 30
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

### Pending Todos

- Design better hero image for README (deferred from Phase 11 — user wants to rethink concept)
- Reduce memory and CPU load during calibration

### Blockers/Concerns

- PyTorch is not an aquakit pip dependency — users must install it manually; SETUP-02 must document this clearly
- `refractive_project` API change is non-trivial (two-step process; returns interface point not pixel) — Phase 14 needs careful call-site audit

## Session Continuity

Last session: 2026-02-19 (roadmap created)
Stopped at: Roadmap written, ready to plan Phase 13
Resume file: None
