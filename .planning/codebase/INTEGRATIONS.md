# External Integrations

**Analysis Date:** 2026-02-14

## APIs & External Services

**Not Detected:**
- No third-party API integrations (no HTTP clients)
- No SaaS platform integrations (Stripe, Twilio, SendGrid, etc.)
- Library is self-contained: accepts video files, produces calibration files

## Data Storage

**Databases:**
- None in use
- All data is in-memory (NumPy arrays) or persisted to local files

**File Storage:**
- Local filesystem only
  - Input: MP4 video files (specified in config YAML under `paths.intrinsic_videos` and `paths.extrinsic_videos`)
  - Output: JSON calibration results → `src/aquacal/io/serialization.py`
  - Output: Optional PNG diagnostic plots (matplotlib)
  - Output: Optional CSV comparison reports (pandas)

**Caching:**
- None (no Redis, Memcached, etc.)
- In-memory numpy arrays cleared after each pipeline stage

## Authentication & Identity

**Auth Provider:**
- None required
- Library is headless; no user authentication, API keys, or credentials
- YAML config files contain only measurement parameters (no secrets)

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, etc.)
- Errors raised as Python exceptions (`CalibrationError` in `src/aquacal/config/schema.py`)

**Logs:**
- Standard output via Python's built-in logging or print statements
- Controlled by `-v / --verbose` CLI flag in `src/aquacal/cli.py`
- No log aggregation

## CI/CD & Deployment

**Hosting:**
- Not applicable - this is a library, not a service
- Can be installed locally via `pip install .` or `pip install .[dev]`
- No deployment target configured

**CI Pipeline:**
- Not detected in repository
- Tests exist (`tests/unit/`, `tests/synthetic/`) but no GitHub Actions or GitLab CI configured
- Developers run tests locally: `python -m pytest tests/`

## Environment Configuration

**Required env vars:**
- None
- All configuration via YAML file or CLI arguments

**Secrets location:**
- Not applicable - no API keys, credentials, or secrets in configuration
- YAML config files contain only:
  - File paths (input/output directories)
  - Board geometry (public measurement parameters)
  - Refractive indices (physical constants: n_air=1.0, n_water≈1.333)
  - Optimization tuning parameters (loss type, scaling, frame limits)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Flow

**Typical Workflow:**
1. User provides video files (MP4 format)
2. User creates YAML config file specifying:
   - Board geometry (ChArUco dimensions)
   - Camera names and video paths
   - Output directory
3. CLI: `aquacal calibrate config.yaml`
4. Pipeline stages:
   - **Stage 1** (Intrinsics): OpenCV in-air calibration, per-camera K matrices
   - **Stage 2** (Extrinsics): BFS pose graph initialization
   - **Stage 3** (Refractive optimization): Joint optimization of R, t, water_z via `scipy.optimize.least_squares()`
   - **Stage 4** (Optional intrinsic refinement): Refine focal length and principal point
5. Output: `calibration.json` saved to output directory
6. Optional diagnostics: PNG plots (matplotlib), CSV reports (pandas)

**File Format Details:**
- Input: MP4 video (any codec)
- Config: YAML (text)
- Output calibration: JSON
  - See `src/aquacal/io/serialization.py` for schema
  - Contains: camera intrinsics (K, dist_coeffs), extrinsics (R, t), water_z, diagnostics
- Output plots: PNG (matplotlib Agg backend)
- Output reports: CSV (pandas)

---

*Integration audit: 2026-02-14*
