---
created: 2026-02-18T22:40:46.073Z
title: Check version field in JSON output reads local version properly
area: api
files:
  - src/aquacal/cli.py
  - src/aquacal/io/serialization.py
---

## Problem

The version field in JSON output (calibration results or CLI output) may be using a hardcoded placeholder string instead of reading the actual installed package version at runtime. Need to verify that the version is dynamically resolved (e.g., via `importlib.metadata.version("aquacal")` or `aquacal.__version__`) and not a static string that could drift from the real version.

## Solution

1. Check `cli.py` and `serialization.py` for how `version` is set in JSON output
2. Verify it reads from `importlib.metadata` or package metadata at runtime
3. If hardcoded, replace with dynamic version lookup
