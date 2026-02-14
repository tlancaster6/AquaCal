# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Refractive multi-camera calibration pipeline with four stages:
  - Stage 1: In-air intrinsic calibration using OpenCV
  - Stage 2: Extrinsic initialization via BFS-based pose graph
  - Stage 3: Joint refractive bundle adjustment for extrinsics, water surface position, and board poses
  - Stage 4: Optional intrinsic refinement stage
- Snell's law projection modeling for accurate refractive ray tracing through flat water surface
- ChArUco board detection and pose estimation
- Sparse Jacobian optimization using custom finite-difference callable with column grouping
- CLI interface for running calibration from YAML configuration files
- Coordinate system conventions (Z-down world frame, OpenCV camera convention)
- Support for arbitrary camera positioning and orientation relative to water surface
