# Phase 5: Documentation Site - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive documentation site with auto-generated API reference and user guide, hosted on Read the Docs. Covers three theory pages (refractive geometry, coordinate conventions, optimizer pipeline), a conceptual overview page, API reference from docstrings, and citation infrastructure. Interactive tutorials are Phase 6.

</domain>

<decisions>
## Implementation Decisions

### Theory page depth & style
- Practical guide level — enough math to understand what the code does, focus on intuition over full derivations
- Key equations rendered with MathJax, supported by diagrams
- Diagrams generated with matplotlib at build time, using actual codebase functions where possible (e.g., import projection functions to draw accurate ray traces)
- Simple flow/architecture diagrams use ASCII/text in source
- Each theory page is self-contained — users can jump to any topic without reading others first, with brief recaps where needed
- Add a conceptual overview page: "What is refractive calibration and why do you need it?" with links into the three detail pages
- Include inline "gotcha" callouts (Sphinx admonitions) for common mistakes (e.g., wrong coordinate convention, interface_distance misunderstanding)

### Site structure & navigation
- Organized by audience need: Overview | User Guide (theory pages) | API Reference | Contributing
- Rich landing page with feature highlights, a code snippet showing basic usage, badges, and section links
- Reserve a "Tutorials" placeholder section in the sidebar with a "Coming soon" note for Phase 6 notebook integration

### API reference approach
- Complete docstrings for all public API functions/classes; internal/private items can be sparse
- Cross-link API docs to relevant theory pages (e.g., `refractive_project()` links to refractive geometry explanation)
- Key functions include brief (3-5 line) usage examples in docstrings

### Tone & audience
- Tiered content: overview and quickstart accessible to broader scientific Python users; theory pages assume some calibration background
- Professional but approachable tone — clear, direct prose like scikit-learn docs
- Briefly explain OpenCV conventions when referenced (don't assume prior knowledge)
- Style reference: scikit-learn documentation

### Claude's Discretion
- Markup format (RST vs MyST Markdown) — pick what works best for the content
- API reference layout (grouped by functionality vs one page per module) — pick what fits the codebase
- Exact sidebar ordering and page titles
- Sphinx theme choice

</decisions>

<specifics>
## Specific Ideas

- Matplotlib diagrams should import actual AquaCal functions to generate accurate visualizations (e.g., ray tracing through water surface using real projection code)
- scikit-learn docs as the style benchmark — clean layout, tiered content, good API docs with examples
- Gotcha callouts should cover known pitfalls from `.planning/knowledge-base.md`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-documentation-site*
*Context gathered: 2026-02-14*
