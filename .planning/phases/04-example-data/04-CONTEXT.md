# Phase 4: Example Data - Context

**Gathered:** 2026-02-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Provide researchers with synthetic and real calibration datasets for testing, learning, and validating AquaCal. This includes a synthetic data generation API (refactored from existing test helpers), a real-world calibration dataset from a production rig, and a unified loading interface. Creating tutorials or documentation that use these datasets belongs in later phases.

</domain>

<decisions>
## Implementation Decisions

### Synthetic data API
- Function name: `generate_synthetic_rig()` (not `generate_sample_data()`)
- Lives in `aquacal.datasets` module (single module for both synthetic and real data)
- Three presets via string names: `'small'`, `'medium'`, `'large'` — presets only, no custom configuration
- Each preset has a fixed random seed for reproducibility (same preset always generates identical data)
- Presets include a default board configuration (e.g. 7x5 ChArUco, 30mm squares) — zero config required
- ChArUco boards only (no standard checkerboard support)
- Refactor existing `generate_synthetic_detections()` test helper into the public API — avoid duplication
- Default: returns detections only (no images). Optional flag to also generate synthetic images
- Noise model: clean by default, optional noise preset adds Gaussian pixel jitter on detected corners (no missed corners)

### Synthetic image rendering
- OpenCV drawing functions (project corners via refractive model, draw at projected positions) — no new deps
- Minimal rendering: black background with ChArUco pattern — correct refractive geometry, not photorealistic
- Must support both above-water boards (intrinsic calibration) and below-water boards (extrinsic calibration)
- Full frame at realistic camera resolution (e.g. 1920x1080), not cropped

### Real dataset scope
- Full production rig: 9+ cameras, both intrinsic (in-air) and extrinsic (underwater) image sets
- Images extracted from video frames to reduce data size (not raw video)
- Frame count per camera TBD — needs experimentation to find sweet spot between calibration quality and dataset size
- Include a reference calibration result from a production-quality run (low framestep) so users can compare
- Reference result demonstrates that more data = better results

### Dataset loading UX
- `aquacal.datasets.load_example()` — auto-downloads on first call if not cached
- Returns a loaded data object (structured, with images/intrinsics/config parsed) — not just a path
- Cache location: working directory (`./aquacal_data/`) with `.gitignore` inside — transparent, avoids hidden dirs collecting dust
- tqdm-style progress bar for downloads (important for 100MB+ files)

### Hosting & size strategy
- Real dataset hosted on Zenodo (DOI-backed, academic credibility)
- Synthetic medium and large presets also hosted on Zenodo as separate zip files
- One zip file per dataset: synthetic-medium.zip, synthetic-large.zip, real-rig.zip
- In-package: small synthetic preset data ships with pip install (zero-download quick start) + download metadata/URLs for other datasets
- Package includes dataset manifest so load_example() knows where to download from

### Claude's Discretion
- Return type of `generate_synthetic_rig()` — structured object with detections + ground truth bundle
- Above-water board projection method (pinhole vs refractive with no refraction effect)
- Exact preset parameters (camera count, frame count, noise levels per tier)
- Data object structure returned by `load_example()`
- Download retry/error handling behavior
- Exact directory structure within `./aquacal_data/`

</decisions>

<specifics>
## Specific Ideas

- Reference calibration result uses a much lower framestep (more frames) than the example dataset — demonstrates that results improve with more data
- Small synthetic preset ships in-package so users can verify their installation works without any download
- User emphasized: don't hide large datasets in `~/.aquacal` — working directory is more transparent for one-time demo usage
- Synthetic images must respect refractive geometry correctly, even if rendering is minimal

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-example-data*
*Context gathered: 2026-02-14*
