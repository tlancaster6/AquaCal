# Phase 11: Documentation Visuals - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Improve visual aids in documentation where they enhance understanding of the refractive multi-camera calibration system. Replace the ASCII pipeline diagram, create a new hero image, add targeted new diagrams, and update existing visuals to a consistent style. No new documentation pages or text content.

</domain>

<decisions>
## Implementation Decisions

### Hero Image
- 2D cross-section layout showing 3 cameras above water surface, rays bending at interface, calibration board below
- Polished matplotlib style — professional but technical
- Minimal labels only (Camera, Water surface, Board) — no equations or angle annotations
- Must convey "multi-camera array" + "refractive calibration" at a glance
- Replaces current `hero_ray_trace.png`

### Pipeline Diagram
- Replace ASCII diagram in `docs/guide/optimizer.md` with a **Mermaid** flowchart
- Horizontal left-to-right flow: Stage 1 → Stage 2 → Stage 3 → Stage 4
- Full detail: include abbreviated inputs/outputs per stage (matching current ASCII content)
- Mermaid chosen for maintainability (text in markdown, renders on GitHub)
- Requires `sphinxcontrib-mermaid` extension (add if not already present)

### New Diagrams
- **Sparsity pattern** (optimizer.md, Sparse Jacobian section): Small illustrative example (3 cameras, 3 frames) showing block-sparse Jacobian structure. Matplotlib PNG.
- **BFS pose graph** (optimizer.md, Stage 2): Small graph showing cameras connected through shared board observations. Claude's discretion on format (Mermaid or matplotlib/networkx).

### Color Scheme & Style Guide
- Blue/aqua theme matching "AquaCal" brand: blues and aquas for water, warm tones for cameras/rays
- Define a shared color palette used across ALL diagrams
- Save palette decisions as a style guide file (e.g., `docs/_static/scripts/style_guide.md`) so it can be ported to related projects
- Update existing diagrams (coordinate_frames.png, ray_trace.png) to match the new palette

### Visual Generation
- Generation scripts live in `docs/_static/scripts/` for regeneration
- All matplotlib diagrams are PNG (consistent with existing)
- Mermaid diagrams are inline in markdown (no separate files)

### Claude's Discretion
- Exact color hex values for the palette (within blue/aqua theme)
- BFS pose graph format (Mermaid vs matplotlib)
- Spacing, typography, and composition details
- How to structure the generation scripts (one per diagram or combined)

</decisions>

<specifics>
## Specific Ideas

- Hero image should communicate both "this is a camera array" and "light bends at the water surface" — the two key concepts
- Style guide should be portable — usable as a reference for styling visuals in related projects
- Existing diagrams already work well conceptually; they just need color consistency with the new palette

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-documentation-visuals*
*Context gathered: 2026-02-17*
