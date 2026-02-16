# Phase 10: Documentation Audit - Context

**Gathered:** 2026-02-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Audit all docstrings and Sphinx documentation for quality, consistency, accuracy, and formatting. Resolve all 6 accumulated documentation todos. Produce an audit report for user review before applying fixes. Does not include new tutorials, visual aids, or hero images (those are Phases 11-12).

</domain>

<decisions>
## Implementation Decisions

### Audit scope & depth
- Audit public API + key internal modules (optimization, refractive geometry, pipeline) — skip trivial internal helpers
- Produce a findings report first — user reviews and approves which fixes to apply before changes are made
- Full style unification: standardize Google-style docstring format, consistent tense, consistent detail level across all audited docstrings
- Sphinx source files audited; Claude decides whether a build check is also needed

### Pending doc todos (all 6 included)
1. **CLI usage guide** — Reference-style (command syntax, flags, options, brief descriptions). Not a workflow walkthrough.
2. **Camera model documentation** — Document standard 5-param, rational 8-param, fisheye 4-param models + auto-simplification logic. Add to existing docs page, not a new standalone page.
3. **Troubleshooting section** — Create or expand a troubleshooting/tips section for usage tips:
   - Reference camera choice affects extrinsic calibration quality
   - Bad RMS / high round-trip errors → lower frame_step, optionally set max_calibration_frames
4. **Clarify allowed camera combinations** — Document which combinations of auxiliary_cameras, fisheye_cameras, and rational_model_cameras are valid
5. **Spelling check** — Check for misspellings of "auxiliary" (commonly "auxillary") across all docs
6. **Rename interface_distance → water_z** — Change config parameter name from `interface_distance` to `water_z`. Add inline comment in generated config clarifying that water_z is the approximate camera-rig to water-surface distance. Update all docs/code references.

### Terminology & conventions
- Rename `interface_distance` to `water_z` everywhere: config, code, docs. Config gets an inline comment explaining the parameter.
- Create a glossary section in the Sphinx docs (water_z, extrinsics, board pose, etc.) that other pages can link to
- Claude identifies and proposes standardizations for inconsistently used terms (e.g., "calibration target" vs "board" vs "ChArUco board")
- Units convention: state "all values in meters" once in a prominent location, only annotate exceptions

### Documentation gaps
- Claude identifies missing documentation sections by comparing existing docs against what a new user would need
- Audience: CV researchers are primary (assume calibration/OpenCV knowledge), but include a Background/Concepts section for general engineers
- README included in audit: check accuracy, completeness, alignment with current features and badges
- API reference: meaningful coverage — every public class/function/method documented except trivially obvious items

### Claude's Discretion
- Whether to run a Sphinx build check in addition to source file audit
- Exact glossary term list (Claude identifies which terms need defining)
- Where to place the Background/Concepts section for newcomers
- How to structure the audit findings report
- Identifying which existing docs pages best host the camera model and CLI content

</decisions>

<specifics>
## Specific Ideas

- The `water_z` config parameter should have an inline comment: approximate camera-rig to water-surface distance
- Glossary should cover the domain-specific terms that trip up newcomers (refractive geometry, interface normal, board pose, etc.)
- Troubleshooting section should feel practical — "if you see X, try Y"

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 10-documentation-audit*
*Context gathered: 2026-02-15*
