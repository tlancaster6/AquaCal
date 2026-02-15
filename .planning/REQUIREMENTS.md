# Requirements: AquaCal

**Defined:** 2026-02-15
**Core Value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.

## v1.4 Requirements

Requirements for QA & Polish milestone. Each maps to roadmap phases.

### CLI QA

- [ ] **QA-01**: User runs `calibrate` workflow end-to-end with real rig data and confirms output correctness
- [ ] **QA-02**: User runs `init` workflow to generate config from real data directories and confirms correct config generation
- [ ] **QA-03**: User runs `compare` workflow to compare multiple calibration runs and confirms correct comparison output
- [ ] **QA-04**: Bugs and friction points discovered during QA are triaged (quick fixes applied, larger items captured as todos)

### Documentation Audit

- [ ] **DOCS-01**: Agent audits docstrings across codebase for inconsistencies, redundancy, and factual errors
- [ ] **DOCS-02**: Agent audits Sphinx documentation for inconsistencies, redundancy, and factual errors
- [ ] **DOCS-03**: User reviews documentation and confirms formatting issues are identified and fixed

### Documentation Visuals

- [ ] **VIS-01**: Existing ASCII diagrams in documentation replaced with better visualizations
- [ ] **VIS-02**: Images and visualizations added or improved in documentation where they aid understanding
- [ ] **VIS-03**: Visual abstract / hero image created for README and docs

### Tutorial Verification

- [ ] **TUT-01**: User confirms all Jupyter tutorials run correctly end-to-end
- [ ] **TUT-02**: Static cell outputs in docs are verified as correctly embedded

### Infrastructure

- [ ] **INFRA-01**: Quick check whether pending todos are already resolved (Read the Docs, DOI badge, RELEASE_TOKEN)
- [ ] **INFRA-02**: Read the Docs deployment configured and working (if not already done)
- [ ] **INFRA-03**: Zenodo DOI badge live in README (if not already done)
- [ ] **INFRA-04**: RELEASE_TOKEN PAT configured for semantic-release (if not already done)

## Future Requirements

None deferred — this is a polish milestone.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New calibration features | This milestone is QA & polish only |
| Performance optimization | Not the focus of this milestone |
| New camera model support | Feature work deferred to future milestone |
| Automated integration tests | Focus is human-in-the-loop QA, not test automation |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 7 | Pending |
| QA-01 | Phase 8 | Pending |
| QA-02 | Phase 8 | Pending |
| QA-03 | Phase 8 | Pending |
| QA-04 | Phase 9 | Pending |
| DOCS-01 | Phase 10 | Pending |
| DOCS-02 | Phase 10 | Pending |
| DOCS-03 | Phase 10 | Pending |
| VIS-01 | Phase 11 | Pending |
| VIS-02 | Phase 11 | Pending |
| VIS-03 | Phase 11 | Pending |
| TUT-01 | Phase 12 | Pending |
| TUT-02 | Phase 12 | Pending |
| INFRA-02 | Phase 13 | Pending |
| INFRA-03 | Phase 13 | Pending |
| INFRA-04 | Phase 13 | Pending |

**Coverage:**
- v1.4 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0 (100% coverage)

---
*Requirements defined: 2026-02-15*
*Last updated: 2026-02-15 after roadmap creation*
