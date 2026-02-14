# Phase 3: Public Release - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

AquaCal v1.0.0 is live on PyPI with community files, README badges, and Zenodo DOI. Users can `pip install aquacal` and cite the package. This phase covers the release itself, community infrastructure, and discoverability — not new library features, documentation site, or tutorials.

</domain>

<decisions>
## Implementation Decisions

### License
- MIT license
- Copyright: "Copyright (c) 2026-present AquaCal Contributors"
- Include a non-binding citation request in README (alongside standard MIT LICENSE file)
- CITATION.cff created in this phase (not deferred to Phase 5)

### README presentation
- Academic/technical tone for project description (assumes domain knowledge)
- Quick-start shows both CLI workflow (`aquacal init` + `aquacal calibrate`) and Python API snippet
- Badge order: Build | Coverage | PyPI | Python | License | DOI (standard scientific — quality signals first)
- Include a 3-5 bullet feature list highlighting key capabilities (refractive modeling, multi-camera, sparse Jacobian, CLI + API)

### Release process
- TestPyPI dry-run before real v1.0.0 publish (catch metadata/packaging issues)
- Trigger via python-semantic-release (auto-detects conventional commits, creates tag + release) — already configured in Phase 2
- Zenodo DOI via GitHub-Zenodo webhook integration (auto-archives every release, not manual)
- Full pre-release validation checklist: TestPyPI install test, all tests pass, README renders correctly, CHANGELOG reviewed, badges working

### Community files
- CONTRIBUTING.md: minimal practical depth — dev setup, how to run tests, PR guidelines. Expand later as needed.
- CODE_OF_CONDUCT.md: Contributor Covenant (industry standard)
- GitHub issue templates: bug report + feature request (basic)
- GitHub PR template (basic)

### Claude's Discretion
- Exact README wording and feature bullet phrasing
- Badge shield service choice (shields.io vs alternatives)
- CITATION.cff metadata structure details
- Issue/PR template field specifics
- CHANGELOG formatting for v1.0.0

</decisions>

<specifics>
## Specific Ideas

- Citation request should be non-binding and placed in README near "How to Cite" section, not in the LICENSE file itself
- Zenodo integration should be webhook-based so future releases auto-mint DOIs without manual steps
- Semantic release already configured in Phase 2 — leverage that, don't create a parallel release mechanism

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-public-release*
*Context gathered: 2026-02-14*
