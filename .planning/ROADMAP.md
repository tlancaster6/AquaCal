# Roadmap: AquaCal

## Milestones

- ✅ **v1.2 MVP** — Phases 1-6 (shipped 2026-02-15)
- 🚧 **v1.4 QA & Polish** — Phases 7-12 (in progress)

## Phases

<details>
<summary>✅ v1.2 MVP (Phases 1-6) — SHIPPED 2026-02-15</summary>

- [x] Phase 1: Foundation and Cleanup (3/3 plans) — completed 2026-02-14
- [x] Phase 2: CI/CD Automation (3/3 plans) — completed 2026-02-14
- [x] Phase 3: Public Release (3/3 plans) — completed 2026-02-14
- [x] Phase 4: Example Data (3/3 plans) — completed 2026-02-14
- [x] Phase 5: Documentation Site (4/4 plans) — completed 2026-02-14
- [x] Phase 6: Interactive Tutorials (4/4 plans) — completed 2026-02-15

See `.planning/milestones/v1.2-ROADMAP.md` for full details.

</details>

### 🚧 v1.4 QA & Polish (In Progress)

**Milestone Goal:** Human-in-the-loop QA of CLI workflows, documentation audit and polish, visual enhancements, and shipping pending infrastructure todos.

#### Phase 7: Infrastructure Check ✓ (completed 2026-02-15)
**Goal**: Quickly validate whether pending infrastructure todos are already resolved
**Depends on**: Phase 6
**Requirements**: INFRA-01
**Success Criteria** (what must be TRUE):
  1. User knows current status of Read the Docs deployment
  2. User knows current status of Zenodo DOI badge
  3. User knows current status of RELEASE_TOKEN configuration
  4. User has clear action list of what infrastructure work remains
**Plans**: 1 plan

Plans:
- [x] 07-01-PLAN.md — Audit infrastructure status (RTD, DOI, RELEASE_TOKEN)

#### Phase 8: CLI QA Execution
**Goal**: User validates all CLI workflows work correctly with real rig data
**Depends on**: Phase 7
**Requirements**: QA-01, QA-02, QA-03
**Success Criteria** (what must be TRUE):
  1. User successfully runs `aquacal calibrate` end-to-end with real data and confirms output correctness
  2. User successfully runs `aquacal init` with real data directories and verifies generated config
  3. User successfully runs `aquacal compare` on multiple calibration runs and validates comparison output
  4. User documents any bugs, friction points, or unexpected behaviors discovered during QA
**Plans**: 1 plan

Plans:
- [ ] 08-01-PLAN.md — QA all CLI workflows (init, calibrate, compare) with real rig data

#### Phase 9: Bug Triage
**Goal**: Issues discovered during CLI QA are triaged and resolved or captured
**Depends on**: Phase 8
**Requirements**: QA-04
**Success Criteria** (what must be TRUE):
  1. All bugs discovered in Phase 8 are categorized (quick fix vs. future work)
  2. Quick fixes are applied and verified
  3. Larger issues are captured as todos or GitHub issues
  4. User confirms CLI workflows are in releasable state
**Plans**: TBD

Plans:
- [ ] 09-01: TBD

#### Phase 10: Documentation Audit
**Goal**: Docstrings and Sphinx documentation are audited for quality and consistency
**Depends on**: Phase 9
**Requirements**: DOCS-01, DOCS-02, DOCS-03
**Success Criteria** (what must be TRUE):
  1. Agent has audited all docstrings for inconsistencies, redundancy, and factual errors
  2. Agent has audited Sphinx documentation for inconsistencies, redundancy, and factual errors
  3. User has reviewed audit findings and confirmed fixes are applied
  4. Documentation has consistent terminology, accurate technical content, and no formatting issues
**Plans**: 3 plans

Plans:
- [ ] 10-01-PLAN.md — Audit docstrings and Sphinx docs, produce findings report
- [ ] 10-02-PLAN.md — Rename interface_distance to water_z across codebase
- [ ] 10-03-PLAN.md — User review + create new doc sections + apply audit fixes

#### Phase 11: Documentation Visuals
**Goal**: Documentation has improved visual aids where they enhance understanding
**Depends on**: Phase 10
**Requirements**: VIS-01, VIS-02, VIS-03
**Success Criteria** (what must be TRUE):
  1. ASCII diagrams in documentation are replaced with better visualizations
  2. New images or visualizations are added where they aid understanding
  3. Visual abstract or hero image exists for README and documentation
  4. User confirms visuals improve documentation clarity
**Plans**: TBD

Plans:
- [ ] 11-01: TBD

#### Phase 12: Tutorial Verification
**Goal**: All Jupyter tutorials run correctly with proper embedded outputs
**Depends on**: Phase 11
**Requirements**: TUT-01, TUT-02
**Success Criteria** (what must be TRUE):
  1. User has run all Jupyter tutorials end-to-end and confirmed they execute without errors
  2. Static cell outputs in documentation are verified as correctly embedded
  3. Tutorial content is accurate and matches current API
  4. Tutorials provide clear learning path for new users
**Plans**: TBD

Plans:
- [ ] 12-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 7 → 8 → 9 → 10 → 11 → 12

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation and Cleanup | v1.2 | 3/3 | Complete | 2026-02-14 |
| 2. CI/CD Automation | v1.2 | 3/3 | Complete | 2026-02-14 |
| 3. Public Release | v1.2 | 3/3 | Complete | 2026-02-14 |
| 4. Example Data | v1.2 | 3/3 | Complete | 2026-02-14 |
| 5. Documentation Site | v1.2 | 4/4 | Complete | 2026-02-14 |
| 6. Interactive Tutorials | v1.2 | 4/4 | Complete | 2026-02-15 |
| 7. Infrastructure Check | v1.4 | 1/1 | Complete | 2026-02-15 |
| 8. CLI QA Execution | v1.4 | 0/TBD | Not started | - |
| 9. Bug Triage | v1.4 | 0/TBD | Not started | - |
| 10. Documentation Audit | v1.4 | Complete    | 2026-02-16 | - |
| 11. Documentation Visuals | v1.4 | 0/TBD | Not started | - |
| 12. Tutorial Verification | v1.4 | 0/TBD | Not started | - |
