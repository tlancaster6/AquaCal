# Phase 1: Foundation and Cleanup - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Make AquaCal a clean, pip-installable package with proper metadata, no legacy artifacts, and validated cross-platform installation. Package infrastructure only -- no new features, no documentation site, no CI/CD.

</domain>

<decisions>
## Implementation Decisions

### Repository cleanup
- Migrate useful content from `dev/` into `.planning/` (architecture docs, geometry docs, knowledge base), then delete entire `dev/` folder
- Specifically: `dev/DESIGN.md` and `dev/GEOMETRY.md` go to `.planning/architecture.md` and `.planning/geometry.md`
- `dev/KNOWLEDGE_BASE.md` content should be migrated to `.planning/` as well
- `dev/CHANGELOG.md`, task files, handoffs -- discard (GSD workflow replaces these)
- `results/` folder: leave as-is (user is actively using it, will delete later). Do NOT add to .gitignore
- Clean up `.claude/` directory: remove confusing workflows, unneeded agent specs, bad config settings
- Clean up `CLAUDE.md`: bring up to GSD standard (Claude's discretion on specific changes)

### Claude's Discretion
- Package metadata completeness (PyPI URLs, classifiers, description) -- use scientific Python conventions
- Dependency pinning strategy -- choose what's standard for research libraries
- Versioning and CHANGELOG format -- follow Keep a Changelog as specified in success criteria
- Specific `.claude/` cleanup decisions (which workflows/agents/configs to remove)
- CLAUDE.md restructuring approach

</decisions>

<specifics>
## Specific Ideas

- Dev docs should migrate to `.planning/` not `docs/` -- `.planning/` is the GSD workflow home for project reference material
- The `.claude/` cleanup should remove pre-GSD agent infrastructure that's now superseded by GSD workflows

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation-and-cleanup*
*Context gathered: 2026-02-14*
