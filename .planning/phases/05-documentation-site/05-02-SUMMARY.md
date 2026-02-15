---
phase: 05
plan: 02
subsystem: documentation
tags: [theory-pages, diagrams, mathjax, sphinx]

dependency_graph:
  requires:
    - 05-01 (Sphinx infrastructure and site skeleton)
  provides:
    - Three complete theory pages (THEO-01, THEO-02, THEO-03)
    - Build-time diagram generation with actual AquaCal code
    - Cross-referenced documentation between theory and API
  affects:
    - User guide navigation and content completeness
    - Documentation site educational value

tech_stack:
  added:
    - matplotlib for diagram generation
    - myst_parser dollarmath extension for MathJax
  patterns:
    - Build-time diagram generation via conf.py setup hook
    - Theory pages import actual codebase functions for accuracy
    - Self-contained pages with cross-links for flexible navigation

key_files:
  created:
    - docs/guide/_diagrams/generate_all.py
    - docs/guide/_diagrams/ray_trace.py
    - docs/guide/_diagrams/coordinate_frames.py
    - docs/guide/refractive_geometry.md
    - docs/guide/coordinates.md
    - docs/guide/optimizer.md
    - docs/_static/diagrams/ray_trace.png
    - docs/_static/diagrams/coordinate_frames.png
  modified:
    - docs/guide/index.md (added all three theory pages to toctree)
    - docs/conf.py (enabled html_static_path for diagram copying)

decisions:
  - "Use matplotlib Agg backend for headless diagram generation"
  - "Ray trace diagram imports actual snells_law_3d for accuracy"
  - "ASCII flow diagram for pipeline overview (simple, text-based)"
  - "Each theory page includes 2+ gotcha admonitions from knowledge-base.md"
  - "Cross-references use MyST markdown syntax for inter-page links"

metrics:
  duration_seconds: 802
  tasks_completed: 2
  files_created: 8
  files_modified: 2
  commits: 2
  theory_pages: 3
  diagrams_generated: 2
  cross_references: 18
  gotcha_admonitions: 7
---

# Phase 05 Plan 02: Theory Pages with Diagrams and Equations

**One-liner:** Complete theory documentation with refractive geometry, coordinate conventions, and optimizer pipeline using MathJax equations and matplotlib diagrams

## What Was Built

Created the three core theory pages (THEO-01, THEO-02, THEO-03) that provide educational content on AquaCal's calibration approach:

1. **Refractive Geometry** (THEO-01):
   - Snell's law in scalar and 3D vector form
   - Ray tracing through air-water interface with 1D Newton-Raphson solver
   - Projection model and when refraction matters
   - Ray trace diagram generated from actual AquaCal `snells_law_3d` function
   - Gotcha admonitions: interface_distance is Z-coordinate, interface normal points up

2. **Coordinate Conventions** (THEO-02):
   - World frame (Z-down), camera frame (OpenCV), pixel coordinates
   - Extrinsics convention (R, t) and camera position C
   - Interface normal [0,0,-1] and its significance
   - 3D coordinate frame diagram showing cameras, water surface, underwater region
   - Gotcha admonitions: world Z-down vs camera Y-down, interface_distance naming

3. **Optimizer Pipeline** (THEO-03):
   - Four-stage calibration pipeline with ASCII flow diagram
   - Parameter vector layout for Stages 3 and 4
   - Cost function (refractive reprojection error) and loss function (soft-L1)
   - Sparse Jacobian strategy: sparse FD with dense solver for stability
   - BFS pose graph initialization and auxiliary camera registration
   - Gotcha admonitions: water_z unobservable in non-refractive mode

Each page is self-contained with brief recaps of prerequisite concepts, cross-references to other theory pages and API docs, and professional admonitions highlighting common pitfalls.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1    | 030abf2 | feat(05-02): create diagram generators and refractive geometry theory page |
| 2    | 82c8a29 | feat(05-02): create coordinate conventions and optimizer pipeline theory pages |

## Self-Check: PASSED

All claimed files verified to exist:
- ✓ docs/guide/_diagrams/generate_all.py
- ✓ docs/guide/_diagrams/ray_trace.py
- ✓ docs/guide/_diagrams/coordinate_frames.py
- ✓ docs/guide/refractive_geometry.md
- ✓ docs/guide/coordinates.md
- ✓ docs/guide/optimizer.md
- ✓ docs/_static/diagrams/ray_trace.png
- ✓ docs/_static/diagrams/coordinate_frames.png

All claimed commits verified to exist:
- ✓ 030abf2
- ✓ 82c8a29

Sphinx build verification:
- ✓ Build succeeds without errors
- ✓ All three theory pages render with MathJax equations
- ✓ Diagrams display correctly in HTML output
- ✓ Cross-references between theory pages work
- ✓ {func} and {mod} references to API docs resolve

## Quality Metrics

**Theory Page Content:**
- 3 theory pages with substantive content (15-25 sections each)
- 18+ MathJax equations across all pages
- 2 matplotlib diagrams generated at build time
- 7 gotcha admonitions covering known pitfalls
- 18 cross-references between theory pages and API docs

**Diagram Generation:**
- Ray trace diagram uses actual `snells_law_3d` for accuracy (not hand-drawn approximation)
- Newton-Raphson solver logic matches codebase implementation
- Coordinate frames diagram shows 3D perspective with correct Z-down convention

**Documentation Structure:**
- All pages linked in guide/index.md toctree
- Each page has "See Also" section with relevant cross-links
- Professional tone and tiered content (accessible to research Python users)

## Key Learnings

**Build-time diagram generation:** Sphinx's setup hook with `app.connect("config-inited", ...)` allows pre-build diagram generation. Setting matplotlib to Agg backend before import ensures headless rendering works on CI servers.

**Importing codebase in docs build:** Diagrams can import actual AquaCal functions by adding project root to sys.path. This ensures diagrams stay accurate as code evolves (e.g., if Snell's law implementation changes, diagram auto-updates).

**MyST cross-references:** Markdown cross-references use `[text](page.md)` for inter-page links and `{func}` / `{mod}` for API references. Sphinx resolves these at build time and warns if targets are missing.

**Admonitions as knowledge capture:** Gotcha admonitions (`:class: warning`) surface lessons from knowledge-base.md directly in user-facing docs. Users see common pitfalls in context, reducing support burden.

## Next Steps

Plan 05-03 will create the conceptual overview page and integrate tutorials placeholder for Phase 6.
