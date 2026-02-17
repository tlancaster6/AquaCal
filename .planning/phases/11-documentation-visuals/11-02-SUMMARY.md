---
phase: 11-documentation-visuals
plan: 02
subsystem: docs
tags: [mermaid, sphinx, sparsity, bfs, visualization, diagrams]

# Dependency graph
requires:
  - phase: 11-documentation-visuals
    plan: 01
    provides: Shared palette module for consistent diagram colors
provides:
  - Mermaid pipeline flowchart replacing ASCII diagram in optimizer.md
  - Sparsity pattern visualization for Jacobian structure
  - BFS pose graph visualization for extrinsic initialization
  - sphinxcontrib-mermaid configured for Sphinx builds
affects: [documentation site rendering]

# Tech tracking
tech-stack:
  added: [sphinxcontrib-mermaid, pypandoc_binary]
  patterns:
    - "Mermaid diagrams with classDef styling for operation vs data distinction"
    - "matplotlib diagram scripts in docs/_static/scripts/ with generate(output_dir) signature"

key-files:
  created:
    - docs/_static/scripts/sparsity_pattern.py
    - docs/_static/scripts/bfs_pose_graph.py
    - docs/_static/diagrams/sparsity_pattern.png
    - docs/_static/diagrams/bfs_pose_graph.png
  modified:
    - pyproject.toml
    - docs/conf.py
    - docs/guide/optimizer.md
    - docs/guide/_diagrams/generate_all.py

key-decisions:
  - "Mermaid pipeline: horizontal LR flow, no intermediate data labels on arrows, color-coded (teal stages, aqua data nodes)"
  - "Sparsity pattern: small 3-camera 3-frame example showing block-diagonal structure with dense water_z column"
  - "BFS pose graph: networkx layout with 4 cameras and 3 frames, BFS traversal edges highlighted"
  - "Hero image deferred — user wants to rethink concept before redesigning"

patterns-established:
  - "New diagram scripts use generate(output_dir) function signature for scriptability"
  - "Mermaid classDef for consistent styling across pipeline diagrams"

# Metrics
duration: 8min
completed: 2026-02-17
---

# Phase 11 Plan 02: Documentation Visual Aids Summary

**Mermaid pipeline flowchart, sparsity pattern diagram, BFS pose graph, and user verification**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-17
- **Completed:** 2026-02-17
- **Tasks:** 2 (1 auto + 1 checkpoint)
- **Files modified:** 8

## Accomplishments

- Replaced ASCII pipeline diagram in optimizer.md with a colored Mermaid flowchart (teal stages, aqua data nodes)
- Created sparsity pattern visualization showing block-sparse Jacobian structure for 3-camera, 3-frame example
- Created BFS pose graph diagram showing camera-board connectivity with highlighted traversal path
- Configured sphinxcontrib-mermaid extension in Sphinx and pyproject.toml
- Updated generate_all.py to include new diagram scripts
- User verified and approved all Phase 11 visuals

## Task Commits

1. **Task 1: Add Mermaid support, create new diagrams, update optimizer.md** - `5d78e07` (feat)
2. **Mermaid simplification: drop arrow labels, fix input label** - `0175506` (fix)
3. **Mermaid color styling: teal stages, aqua data nodes** - `75516d4` (feat)
4. **Task 2: User verification** - approved

## Files Created/Modified

- `docs/_static/scripts/sparsity_pattern.py` - Block-sparse Jacobian visualization script
- `docs/_static/scripts/bfs_pose_graph.py` - BFS pose graph visualization with networkx
- `docs/_static/diagrams/sparsity_pattern.png` - Jacobian sparsity pattern for optimizer docs
- `docs/_static/diagrams/bfs_pose_graph.png` - Camera-board connectivity graph for optimizer docs
- `pyproject.toml` - Added sphinxcontrib-mermaid to docs extras
- `docs/conf.py` - Added sphinxcontrib.mermaid to extensions list
- `docs/guide/optimizer.md` - Mermaid pipeline, sparsity pattern image ref, BFS graph image ref
- `docs/guide/_diagrams/generate_all.py` - Added new diagram script calls

## Decisions Made

- Dropped intermediate data labels from Mermaid arrows for cleaner appearance
- Changed Stage 2 input from "Intrinsics" to "Board params" for accuracy
- Color-coded Mermaid: deep teal (#00897B) for stages, light aqua (#E0F7FA) for data nodes
- Hero image redesign deferred — added as pending todo

## Deviations from Plan

- Hero image reverted to original (user didn't like the generated version, wants to rethink concept)
- Mermaid intermediate data labels removed (user preference override of original plan)

## Issues Encountered

- pandoc missing for nbsphinx — installed via conda and pypandoc_binary

## Self-Check: PASSED

All new PNGs exist, Mermaid renders in built docs, user approved all visuals.
