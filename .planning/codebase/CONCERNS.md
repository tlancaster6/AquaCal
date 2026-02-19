# Codebase Concerns

**Analysis Date:** 2026-02-14

## Tech Debt

**Comparison metric `reproj_3d_pct` becomes meaningless for non-refractive calibration:**
- Issue: In `src/aquacal/validation/comparison.py` (lines 107-111), the percent error metric divides 3D reconstruction mean error by `water_z` as a heuristic denominator. When calibrating with `n_air == n_water == 1.0` (non-refractive baseline), the water_z parameter becomes unobservable and drift arbitrarily during optimization, making the percent error nonsensical.
- Files: `src/aquacal/validation/comparison.py` (line 111), `dev/KNOWLEDGE_BASE.md` (water_z unobservable section)
- Impact: Comparison reports mixing refractive and non-refractive calibrations show invalid `reproj_3d_pct` values in metric tables; users cannot trust this metric for non-refractive runs.
- Fix approach: Task 6.8 (incomplete in TASKS.md) - remove `reproj_3d_pct` column from comparison metric output. Substitute with raw mean error (mm) only, which is independent of refractive index.
- Status: Open task (6.8)

**Depth binning uses absolute Z coordinates, breaks with non-refractive calibration:**
- Issue: Tasks 6.6 and 6.7 (depth-stratified error analysis) bin measurements by absolute Z coordinate in world frame. Non-refractive calibration produces biased camera Z positions (due to unobservable water_z), which shifts all triangulated points and invalidates inter-run depth comparisons when Z ranges differ.
- Files: `src/aquacal/validation/reconstruction.py` (bin_by_depth, line ~180), `src/aquacal/validation/comparison.py` (plot_depth_error_comparison)
- Impact: Depth-binned comparison plots show different Z ranges for refractive vs non-refractive runs, making depth generalization analysis meaningless.
- Fix approach: Task 6.9 (incomplete in TASKS.md) - switch to per-run Z normalization. Each run's Z range computed from its own triangulated data (with outlier trimming), divided into N equal slices. Depth axis becomes relative (shallowest to deepest) instead of absolute, enabling fair cross-model comparison.
- Status: Open task (6.9)

**Missing import in synthetic experiments:**
- Issue: `tests/synthetic/experiments.py` imports only `calibrate_synthetic` and `compute_per_camera_errors` from `experiment_helpers` (lines 21-24) but uses `evaluate_reconstruction()` at lines 475 and 981 without importing it.
- Files: `tests/synthetic/experiments.py` (line 21), `tests/synthetic/experiment_helpers.py` (line 163)
- Impact: Running `run_experiment_2()` or `run_experiment_3()` crashes with `NameError: name 'evaluate_reconstruction' is not defined`. Task B.2 documents this.
- Fix approach: Add `evaluate_reconstruction` to the import statement in `experiments.py` lines 21-24.
- Status: Open bug (B.2)

## Known Bugs

**B.2 - Missing `evaluate_reconstruction` import in experiments.py**
- Symptoms: `NameError: name 'evaluate_reconstruction' is not defined` at `experiments.py:475` and `experiments.py:981`
- Files: `tests/synthetic/experiments.py`, `tests/synthetic/experiment_helpers.py`
- Trigger: `python tests/synthetic/compare_refractive.py`
- Workaround: Manually add `evaluate_reconstruction,` to line 23 in `tests/synthetic/experiments.py`
- Status: Documented in `dev/tasks/b2_report.md`

## Performance Bottlenecks

**Sparse Jacobian computation falls back to dense for large rigs, risking OOM:**
- Problem: In `src/aquacal/calibration/_optim_common.py:make_sparse_jacobian_func()` (line 503), a 500M-element threshold determines whether to return dense (for QR solver) or sparse (for LSMR) Jacobian. For 13-camera, 629-frame rig with intrinsic refinement (~685 params × ~45,000 residuals), dense matrix would need ~13.5 GiB memory, exceeding typical workstation RAM.
- Files: `src/aquacal/calibration/_optim_common.py` (line 503, default threshold 500M), `dev/KNOWLEDGE_BASE.md` (Sparse Jacobian entry)
- Cause: Dense Jacobian enables exact (QR-based) trust-region solver, which is more stable than LSMR for ill-conditioned problems, but LSMR is required for large problems to avoid OOM.
- Improvement path: Tune `dense_threshold` parameter based on available memory. Document recommended settings in CLI schema (see `src/aquacal/config/schema.py` line 196). Add runtime memory check and warning when falling back to sparse/LSMR for stability-critical stages.

**No frame selection heuristic; full calibration trajectory may be unnecessary:**
- Problem: Pipeline processes ALL detected frames in Stage 3 (joint refractive optimization), even if many are redundant (e.g., identical board pose repeated). This inflates optimization cost and wall-clock time for long videos.
- Files: `src/aquacal/calibration/pipeline.py` (line ~450), `src/aquacal/config/schema.py` (max_calibration_frames, line 244)
- Cause: No frame selection logic beyond optional uniform subsampling (`max_calibration_frames`). Intelligent selection (maximize geometric diversity) would reduce frames needed.
- Improvement path: Implement greedy frame selection based on board pose entropy (e.g., select frames that maximize XY spread, depth range, and orientation variation). Document as Task F.2 (Future).

## Fragile Areas

**Refractive projection geometry assumes horizontal interface normal:**
- Files: `src/aquacal/core/refractive_geometry.py` (lines 335, 431), `src/aquacal/core/interface_model.py` (lines 13, 70)
- Why fragile: Functions `_refractive_project_newton()` and `_refractive_project_brent()` expect interface normal to be horizontal (pointing straight up or down). Code comments say "assumes horizontal normal" but don't enforce it. If user accidentally passes a tilted interface (Stage 5 would support this) into these functions, results silently diverge from physical reality.
- Safe modification: Add explicit assertion that `|interface.normal[0]| < 1e-6 and |interface.normal[1]| < 1e-6` at function entry. Or better: refactor to accept tilted interfaces via rotated coordinate transforms (future work).
- Test coverage: No regression tests for non-horizontal normals (there shouldn't be any in current code paths, but future tilted interface support could regress this).

**Interface distance Z-coordinate confusion (misnaming):**
- Files: `src/aquacal/config/schema.py` (schema naming), `src/aquacal/calibration/_optim_common.py` (unpack_params, line 165), `src/aquacal/core/refractive_geometry.py` (projection code)
- Why fragile: The name `interface_distance` suggests a camera-to-water gap (per-camera scalar), but the code treats it as a Z-coordinate (absolute world frame position, same for all cameras). Downstream projection functions compute the gap internally as `h_c = interface_distance - C_z`. Bug B.6 (now fixed) resulted from double-counting C_z when deriving distance from global water_z.
- Safe modification: If renaming is not possible (API stability), document in docstrings that `interface_distance` is ALWAYS a Z-coordinate, never a gap. Add a KB entry (already exists) that must be consulted by anyone touching the interface or extrinsics.
- Test coverage: Unit tests pass, but docstring clarity could be improved. See `dev/KNOWLEDGE_BASE.md` entry "interface_distance is a Z-coordinate, not a physical gap".

**Board pose initialization via BFS assumes well-connected camera network:**
- Files: `src/aquacal/calibration/extrinsics.py` (build_pose_graph, estimate_extrinsics)
- Why fragile: Stage 2 (extrinsic initialization) builds a pose graph where cameras are nodes and edges exist if two cameras see the board in the same frame. It runs BFS from reference camera to initialize other cameras. If the camera network is poorly connected (e.g., no frame where all 13 cameras see the board), BFS may fail to reach all cameras, leaving some uninitialized.
- Safe modification: Add validation after build_pose_graph to check that all cameras are reachable from reference. Emit warning or error if any camera has zero observations in Stage 2. Document minimum connectivity requirement (e.g., "at least one frame where all cameras see the board" or "connected network across all frames").
- Test coverage: Synthetic tests use full-coverage ground truth; real rig tests not visible. Risk for edge-case real rigs with poor overlap.

**Intrinsic calibration per-camera uses OpenCV ChArUco detection without adaptive parameters:**
- Files: `src/aquacal/calibration/intrinsics.py` (calibrate_intrinsics_all)
- Why fragile: Stage 1 uses OpenCV's default ChArUco detection for each camera's in-air video independently. If a camera has poor lighting, motion blur, or bad board pose in its in-air sequence, detection can fail silently (few corners found) and lead to unstable K matrix estimate.
- Safe modification: Add detection quality metrics (min corners per frame, reprojection error of detected corners back to board model) and emit warnings if below threshold. Consider multi-camera consistency check: if one camera's focal length differs wildly from others with same lens, flag as suspect.
- Test coverage: `validate_intrinsics()` in `intrinsics.py` does some checks but doesn't validate inter-camera consistency.

**Water Z parameter becomes unobservable and arbitrary in non-refractive mode:**
- Files: `src/aquacal/calibration/interface_estimation.py` (optimize_interface), `src/aquacal/core/refractive_geometry.py` (projection), `dev/KNOWLEDGE_BASE.md` (water_z unobservable entry)
- Why fragile: When `n_air == n_water == 1.0` (no refraction), the water surface Z becomes unobservable to the optimizer — the projected pixel is exactly independent of water_z. Stage 3 cost function has a flat valley; numerical noise and boundary penalties cause water_z to drift during optimization, making the final value arbitrary.
- Safe modification: Document that water_z is meaningless in non-refractive mode (already in KB). Reject comparison metrics (like `reproj_3d_pct`) that depend on it. For users comparing refractive vs non-refractive, recommend using relative metrics (intrinsic error, extrinsic error) instead.
- Test coverage: `test_interface_estimation.py` has a test for non-refractive (water_z_nonrefractive) but doesn't assert that water_z stays bounded or meaningful.

**CLI config generation uses placeholder board dimensions:**
- Files: `src/aquacal/cli.py` (cmd_init, lines 468-471)
- Why fragile: Generated config.yaml contains hardcoded example board dimensions (squares_x=8, squares_y=6, square_size=0.030m, marker_size=0.022m) with TODO comments. Users must manually edit all four values or calibration will fail on size mismatch. No validation that user actually measured correct values.
- Safe modification: Require user to provide board dimensions via CLI flags (`--squares-x`, `--squares-y`, `--square-size`, `--marker-size`) or prompt interactively. Don't generate a config with incorrect placeholders.
- Test coverage: CLI tests don't validate that generated config is complete/usable.

## Scaling Limits

**Camera network limited to ~14 cameras:**
- Current capacity: Tested on 13-camera rig; code architecture supports up to ~14
- Limit: Beyond 14 cameras, parameter vector exceeds 500M-element Jacobian threshold, forcing fallback to sparse/LSMR solver (less stable). Memory for sparse pattern also grows as O(n_cameras * n_frames).
- Scaling path: Implement block-diagonal or schur-complement structure for bundle adjustment (sparse QR or Ceres Solver). Current `build_jacobian_sparsity()` already computes block structure; a smarter solver could exploit it.

**Frame capacity limited by parameter count and solver stability:**
- Current capacity: 629 frames × 13 cameras with intrinsic refinement (~685 params) is near the dense Jacobian threshold.
- Limit: Adding more frames increases residual count (n_residuals ≈ n_cameras × n_frames × n_corners) and Jacobian memory. Beyond ~1000 frames with 14 cameras, fallback to LSMR becomes necessary, reducing convergence stability.
- Scaling path: Implement frame selection heuristic (Task F.2) to cap frames while maximizing geometric diversity.

**Intrinsic refinement adds 4 params per camera:**
- Current capacity: 13 cameras = 52 additional parameters; manageable.
- Limit: Attempting to refine more than ~20 cameras' intrinsics simultaneously risks overparameterization and numerical ill-conditioning.
- Scaling path: Stage 4 should validate that refined intrinsics are stable (e.g., fx, fy don't drift >10% from Stage 1 estimates). Flag as suspect if they do.

## Missing Critical Features

**No validation of in-air intrinsic calibration data quality before Stage 2:**
- Problem: Pipeline processes in-air videos without checking if detection was successful. If Stage 1 fails silently (few/no corners detected), Stage 2 runs with invalid intrinsics, producing garbage extrinsics.
- Blocks: Users cannot detect bad in-air calibration until later stages show implausible results.
- Recommendation: Add explicit check in `run_calibration()` (pipeline.py) after Stage 1. If any camera's stage 1 result has <50 valid frames detected or abnormally high reprojection error, emit fatal error with actionable message.

**No automatic detection of board occlusion or non-coplanar corners:**
- Problem: If the board is partially out of frame, rotated out-of-plane, or severely tilted, ChArUco detection can still find corners but their geometry is invalid. Current `detect_charuco()` only filters by min corner count and collinearity.
- Blocks: Garbage detections propagate to extrinsic estimation, producing wrong camera poses.
- Recommendation: Add board planarity check: triangulate detected corners and verify they lie (within tolerance) on a plane. Reject frames where RMS plane fit error exceeds threshold.

**No memory usage profiling or adaptive threshold tuning:**
- Problem: The `dense_threshold` parameter (500M) in `make_sparse_jacobian_func()` is hardcoded. Users with 16GB RAM might OOM on 13 cameras + 629 frames; users with 64GB could use dense safely up to 30 cameras.
- Blocks: Manual parameter tuning required for large rigs; no runtime guidance.
- Recommendation: Measure available RAM at startup and adjust threshold dynamically. Emit warning if falling back to sparse due to memory constraints.

## Test Coverage Gaps

**Untested: Large-rig performance (13+ cameras, 500+ frames)**
- What's not tested: Real wall-clock time and memory usage for realistic large rigs.
- Files: `tests/synthetic/` (ground_truth.py, experiments.py) - use 13 cameras but limited to 100-200 frames in tests.
- Risk: Stage 3/4 might take hours or crash on OOM in production; not discovered until user runs on real rig.
- Priority: Medium - synthetic tests should include a large-frame scenario marked `@pytest.mark.slow` with memory assertions.

**Untested: Extrinsic initialization failure modes**
- What's not tested: What happens when camera network is disconnected (no frame sees all cameras), or when board has poor illumination in some frames.
- Files: `tests/synthetic/test_full_pipeline.py` - generates ideal detections; doesn't test Stage 2 robustness.
- Risk: Users with poor calibration videos get cryptic BFS failure messages instead of actionable guidance.
- Priority: High - add synthetic tests with partial detection (cameras missing in some frames) and BFS path tracing.

**Untested: Non-refractive calibration metrics and depth analysis**
- What's not tested: Comparison module's depth binning and `reproj_3d_pct` when water_z is meaningless (n_air=n_water).
- Files: Tests don't cover mixing refractive and non-refractive runs in comparison.
- Risk: Users comparing models see meaningless percent errors; depth plots show arbitrary Z ranges.
- Priority: Medium - add test for non-refractive comparison with assertion that `reproj_3d_pct` is NaN or omitted.

**Untested: Fisheye camera model in full pipeline**
- What's not tested: Stage 1-4 with fisheye cameras; only unit tests for FisheyeCamera.project/back_project exist.
- Files: `tests/synthetic/` doesn't generate fisheye ground truth; `test_full_pipeline.py` uses pinhole only.
- Risk: Fisheye support advertised in DESIGN.md but not validated end-to-end.
- Priority: Low-Medium - add synthetic test with FisheyeCamera; mark `@pytest.mark.slow` since it requires separate ground truth generation.

**Untested: Interface tilt estimation (Stage 5 preview)**
- What's not tested: Any functionality for tilted interface planes.
- Files: `interface_model.py` and `refractive_geometry.py` hard-code horizontal normal; no tilt support yet.
- Risk: If tilt support is added in future, current projection functions may silently produce wrong results.
- Priority: Low - no current usage; becomes High when Stage 5 is implemented.

## Security Considerations

**YAML config parsing vulnerable to arbitrary code execution:**
- Risk: `pipeline.py:load_config()` uses `yaml.safe_load()` (safe), but if code ever switches to `yaml.load()`, user-provided config could execute arbitrary Python code.
- Files: `src/aquacal/calibration/pipeline.py` (line 69)
- Current mitigation: Uses `yaml.safe_load()` (safe). Config file is expected to be user-controlled, so risk is minimal if users trust their own config.
- Recommendations: Keep `yaml.safe_load()`. Add comment explaining why. Never use `yaml.load()` without explicit approval.

**Video file paths not validated; could read arbitrary files:**
- Risk: If config is generated from untrusted source, video paths could point to sensitive files. Attempting to open them as video will fail, but files are touched.
- Files: `src/aquacal/io/video.py` (VideoSet init)
- Current mitigation: `pathlib.Path` is used; no path traversal possible. OpenCV video open will fail for non-video files.
- Recommendations: Add path validation in config loading to reject absolute paths outside expected directory. Document security assumption (config is user-controlled, not attacker-controlled).

**Serialized JSON output includes all calibration parameters:**
- Risk: If output directory is world-readable, camera positions and intrinsics are exposed, potentially revealing facility layout or hardware specs.
- Files: `src/aquacal/io/serialization.py` (save_calibration)
- Current mitigation: None; relies on filesystem permissions.
- Recommendations: Document that output directories should not be world-readable. Consider optional encryption flag for sensitive deployments (future).

---

*Concerns audit: 2026-02-14*
