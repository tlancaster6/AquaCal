"""Command-line interface for AquaCal calibration pipeline."""

import argparse
import re
import sys
from pathlib import Path

from aquacal.calibration.pipeline import load_config, run_calibration
from aquacal.config.schema import CalibrationError
from aquacal.io.serialization import load_calibration
from aquacal.validation.comparison import compare_calibrations, write_comparison_report
from aquacal.validation.reconstruction import (
    bin_by_depth,
    load_spatial_measurements,
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for CLI.

    Returns:
        Configured ArgumentParser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="aquacal",
        description="Refractive multi-camera calibration for underwater imaging",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    # calibrate subcommand
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Run calibration pipeline",
        description="Run complete calibration pipeline from configuration file.",
    )
    cal_parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration YAML file",
    )
    cal_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    cal_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory from config",
    )
    cal_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running calibration",
    )
    cal_parser.set_defaults(func=cmd_calibrate)

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Generate configuration file from video directories",
        description="Scan intrinsic and extrinsic video directories and generate a configuration YAML file.",
    )
    init_parser.add_argument(
        "--intrinsic-dir",
        type=Path,
        required=True,
        help="Directory containing in-air calibration videos",
    )
    init_parser.add_argument(
        "--extrinsic-dir",
        type=Path,
        required=True,
        help="Directory containing underwater calibration videos",
    )
    init_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("config.yaml"),
        help="Output path for generated config file (default: config.yaml)",
    )
    init_parser.add_argument(
        "--pattern",
        type=str,
        default="(.+)",
        help="Regex with one capture group to extract camera name from filename stem (default: '(.+)')",
    )
    init_parser.set_defaults(func=cmd_init)

    # compare subcommand
    cmp_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple calibration runs",
        description="Load calibration results from N directories and compare quality metrics and parameters.",
    )
    cmp_parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="N directories containing calibration.json files (minimum 2)",
    )
    cmp_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("comparison_output"),
        help="Output directory for comparison results (default: comparison_output)",
    )
    cmp_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation",
    )
    cmp_parser.set_defaults(func=cmd_compare)

    return parser


def cmd_calibrate(args: argparse.Namespace) -> int:
    """
    Execute calibration command.

    Args:
        args: Parsed arguments with config_path, verbose, output_dir, dry_run

    Returns:
        Exit code: 0 for success, non-zero for errors
    """
    config_path = args.config_path

    # Check file exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Load and validate config
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 2

    # Override output directory if specified
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Dry run: just validate
    if args.dry_run:
        n_primary = len(config.camera_names)
        n_rational = len(config.rational_model_cameras)
        n_auxiliary = len(config.auxiliary_cameras)
        n_fisheye = len(config.fisheye_cameras)

        parts = [f"{n_primary} primary camera(s)"]
        if n_rational:
            parts.append(f"{n_rational} rational model")
        if n_auxiliary:
            parts.append(f"{n_auxiliary} auxiliary")
        if n_fisheye:
            parts.append(f"{n_fisheye} fisheye")

        print(
            f"Configuration valid. {', '.join(parts)} detected. "
            f"Output will be saved to {config.output_dir}. "
            f"Ready for calibration."
        )
        return 0

    # Run calibration
    try:
        _result = run_calibration(config_path, verbose=args.verbose)
        return 0
    except CalibrationError as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nCalibration interrupted.", file=sys.stderr)
        return 130


def cmd_init(args: argparse.Namespace) -> int:
    """
    Execute init command to generate config file from video directories.

    Args:
        args: Parsed arguments with intrinsic_dir, extrinsic_dir, output, pattern

    Returns:
        Exit code: 0 for success, non-zero for errors
    """
    intrinsic_dir = args.intrinsic_dir
    extrinsic_dir = args.extrinsic_dir
    output_path = args.output
    pattern_str = args.pattern

    # Check directories exist
    if not intrinsic_dir.exists():
        print(f"Error: Intrinsic directory not found: {intrinsic_dir}", file=sys.stderr)
        return 1
    if not intrinsic_dir.is_dir():
        print(
            f"Error: Intrinsic path is not a directory: {intrinsic_dir}",
            file=sys.stderr,
        )
        return 1
    if not extrinsic_dir.exists():
        print(f"Error: Extrinsic directory not found: {extrinsic_dir}", file=sys.stderr)
        return 1
    if not extrinsic_dir.is_dir():
        print(
            f"Error: Extrinsic path is not a directory: {extrinsic_dir}",
            file=sys.stderr,
        )
        return 1

    # Check output file doesn't already exist
    if output_path.exists():
        print(f"Error: Output file already exists: {output_path}", file=sys.stderr)
        return 1

    # Validate regex pattern
    try:
        regex = re.compile(pattern_str)
    except re.error as e:
        print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
        return 1

    n_groups = regex.groups
    if n_groups != 1:
        print(
            f"Error: Regex must have exactly one capture group, found {n_groups}",
            file=sys.stderr,
        )
        return 1

    # Scan directories for video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    def scan_directory(directory: Path) -> dict[str, Path]:
        """Scan directory for video files and extract camera names."""
        camera_files = {}
        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in video_extensions:
                continue

            # Apply regex to stem (filename without extension)
            stem = file_path.stem
            match = regex.match(stem)
            if match:
                camera_name = match.group(1)
                camera_files[camera_name] = file_path.resolve()

        return camera_files

    intrinsic_files = scan_directory(intrinsic_dir)
    extrinsic_files = scan_directory(extrinsic_dir)

    # Check for empty results
    if not intrinsic_files:
        print(
            f"Error: No video files found in intrinsic directory: {intrinsic_dir}",
            file=sys.stderr,
        )
        return 1
    if not extrinsic_files:
        print(
            f"Error: No video files found in extrinsic directory: {extrinsic_dir}",
            file=sys.stderr,
        )
        return 1

    # Compute intersection and check for mismatches
    intrinsic_cameras = set(intrinsic_files.keys())
    extrinsic_cameras = set(extrinsic_files.keys())
    intersection = intrinsic_cameras & extrinsic_cameras

    if not intersection:
        print(
            "Error: No common camera names found between directories", file=sys.stderr
        )
        print(f"  Intrinsic cameras: {sorted(intrinsic_cameras)}", file=sys.stderr)
        print(f"  Extrinsic cameras: {sorted(extrinsic_cameras)}", file=sys.stderr)
        return 1

    # Warn about mismatches
    if intrinsic_cameras != extrinsic_cameras:
        only_intrinsic = intrinsic_cameras - extrinsic_cameras
        only_extrinsic = extrinsic_cameras - intrinsic_cameras

        print("Warning: Camera mismatch between directories", file=sys.stderr)
        if only_intrinsic:
            print(f"  Only in intrinsic: {sorted(only_intrinsic)}", file=sys.stderr)
        if only_extrinsic:
            print(f"  Only in extrinsic: {sorted(only_extrinsic)}", file=sys.stderr)
        print(f"  Using intersection: {sorted(intersection)}", file=sys.stderr)

    # Sort camera names for consistent output
    camera_names = sorted(intersection)

    # Generate config YAML content
    config_content = _generate_config_yaml(
        camera_names, intrinsic_files, extrinsic_files, output_path
    )

    # Write to file
    try:
        output_path.write_text(config_content)
    except Exception as e:
        print(f"Error: Failed to write config file: {e}", file=sys.stderr)
        return 1

    # Print success message
    print(f"Config file created: {output_path}")
    print(f"  {len(camera_names)} camera(s): {', '.join(camera_names)}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """
    Execute compare command to compare multiple calibration runs.

    Args:
        args: Parsed arguments with directories, output_dir, no_plots

    Returns:
        Exit code: 0 for success, 1 for file/directory errors, 2 for comparison errors
    """
    directories = args.directories
    output_dir = args.output_dir
    save_plots = not args.no_plots

    # Validate minimum 2 directories
    if len(directories) < 2:
        print(
            f"Error: Need at least 2 directories to compare, got {len(directories)}",
            file=sys.stderr,
        )
        return 1

    # Check all directories exist
    for directory in directories:
        if not directory.exists():
            print(f"Error: Directory not found: {directory}", file=sys.stderr)
            return 1
        if not directory.is_dir():
            print(f"Error: Path is not a directory: {directory}", file=sys.stderr)
            return 1

    # Check all directories contain calibration.json
    for directory in directories:
        calib_file = directory / "calibration.json"
        if not calib_file.exists():
            print(f"Error: calibration.json not found in {directory}", file=sys.stderr)
            return 1

    # Derive labels from directory basenames
    raw_labels = [d.name for d in directories]

    # Disambiguate duplicates by appending _2, _3, etc.
    label_counts = {}
    labels = []
    for raw_label in raw_labels:
        if raw_label not in label_counts:
            label_counts[raw_label] = 0
            labels.append(raw_label)
        else:
            label_counts[raw_label] += 1
            labels.append(f"{raw_label}_{label_counts[raw_label] + 1}")

    # Now we need to go back and fix any labels that had duplicates
    # Count occurrences of each raw label
    raw_label_freq = {}
    for raw_label in raw_labels:
        raw_label_freq[raw_label] = raw_label_freq.get(raw_label, 0) + 1

    # Rebuild labels with proper disambiguation
    labels = []
    label_indices = {}
    for i, raw_label in enumerate(raw_labels):
        if raw_label_freq[raw_label] == 1:
            labels.append(raw_label)
        else:
            # This label appears multiple times, need to disambiguate
            if raw_label not in label_indices:
                label_indices[raw_label] = 1
            else:
                label_indices[raw_label] += 1
            labels.append(f"{raw_label}_{label_indices[raw_label]}")

    # Load calibration results
    results = []
    try:
        for directory in directories:
            calib_file = directory / "calibration.json"
            result = load_calibration(calib_file)
            results.append(result)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: Failed to load calibration: {e}", file=sys.stderr)
        return 1

    # Compare calibrations
    try:
        comparison = compare_calibrations(results, labels)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Load spatial measurements for depth analysis (if available)
    spatial_data = {}
    depth_data = {}
    for directory, label in zip(directories, labels):
        spatial_file = directory / "spatial_measurements.csv"
        if spatial_file.exists():
            try:
                spatial = load_spatial_measurements(spatial_file)
                spatial_data[label] = spatial
                depth_data[label] = bin_by_depth(spatial)
            except Exception:
                # Silently skip if spatial data can't be loaded
                pass

    # Write comparison report
    try:
        written_files = write_comparison_report(
            comparison,
            results,
            output_dir,
            save_plots=save_plots,
            depth_data=depth_data if depth_data else None,
            spatial_data=spatial_data if spatial_data else None,
        )
    except Exception as e:
        print(f"Error: Failed to write comparison report: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(f"Comparing {len(results)} calibration runs...")
    loaded_info = ", ".join(
        [
            f"{label} ({len(result.cameras)} cameras)"
            for label, result in zip(labels, results)
        ]
    )
    print(f"Loaded: {loaded_info}\n")

    print(f"Results written to {output_dir}/")
    for file_type in [
        "metrics_csv",
        "per_camera_csv",
        "parameter_diffs_csv",
        "rms_bar_chart",
        "position_overlay",
        "z_position_dumbbell",
        "depth_error_plot",
        "depth_binned_csv",
        "xy_error_heatmaps",
    ]:
        if file_type in written_files:
            file_path = written_files[file_type]
            print(f"  {file_path.name}")

    return 0


def _generate_config_yaml(
    camera_names: list[str],
    intrinsic_files: dict[str, Path],
    extrinsic_files: dict[str, Path],
    config_path: Path,
) -> str:
    """
    Generate YAML config content from camera information.

    Args:
        camera_names: Sorted list of camera names
        intrinsic_files: Mapping from camera names to intrinsic video paths
        extrinsic_files: Mapping from camera names to extrinsic video paths
        config_path: Path where the config file will be written (output_dir defaults to its parent)

    Returns:
        YAML config file content as string
    """
    lines = [
        "# AquaCal configuration file",
        "# Generated by 'aquacal init'",
        "",
        "board:",
        "  squares_x: 8           # TODO: measure your board",
        "  squares_y: 6           # TODO: measure your board",
        "  square_size: 0.030     # TODO: measure your board (meters)",
        "  marker_size: 0.022     # TODO: measure your board (meters)",
        '  dictionary: "DICT_4X4_50"  # ArUco dictionary name',
        "  legacy_pattern: false  # Set to true if board has marker in top-left cell (pre-OpenCV 4.6)",
        "",
        "# Optional: use a different board for in-air intrinsic calibration",
        "# If omitted, the board above is used for both intrinsic and extrinsic steps",
        "# intrinsic_board:",
        "#   squares_x: 12",
        "#   squares_y: 9",
        "#   square_size: 0.025",
        "#   marker_size: 0.018",
        '#   dictionary: "DICT_4X4_100"',
        "#   # legacy_pattern: false  # Set to true if board has marker in top-left cell",
        "",
        "# The first camera listed is the reference camera (world origin)",
        "cameras:",
    ]

    for cam in camera_names:
        lines.append(f"  - {cam}")

    lines.extend(
        [
            "",
            "# Optional: auxiliary cameras (registered post-hoc, excluded from joint optimization)",
            "# Move camera names from 'cameras' to here if they should not participate in Stage 3",
            "# auxiliary_cameras:",
            "#   - cam_name",
            "",
            "# Optional: cameras needing 8-coefficient rational distortion model",
            "# Use for wide-angle lenses where the standard 5-coefficient model is insufficient",
            "# rational_model_cameras:",
            "#   - cam_name",
            "",
            "# Optional: cameras using equidistant fisheye projection model",
            "# Must be a subset of auxiliary_cameras. Must not overlap with rational_model_cameras.",
            "# fisheye_cameras:",
            "#   - cam_name",
        ]
    )

    lines.extend(
        [
            "",
            "paths:",
            "  intrinsic_videos:",
        ]
    )

    for cam in camera_names:
        path_str = str(intrinsic_files[cam]).replace("\\", "/")
        lines.append(f'    {cam}: "{path_str}"')

    lines.extend(
        [
            "  extrinsic_videos:",
        ]
    )

    for cam in camera_names:
        path_str = str(extrinsic_files[cam]).replace("\\", "/")
        lines.append(f'    {cam}: "{path_str}"')

    lines.extend(
        [
            f'  output_dir: "{str(config_path.resolve().parent / "output").replace(chr(92), "/")}"',
            "",
            "interface:",
            "  n_air: 1.0             # Refractive index of air",
            "  n_water: 1.333         # Refractive index of water (1.333 for fresh water at 20C)",
            "  normal_fixed: false    # If true, assume reference camera is perpendicular to water surface",
            "",
            "  # Optional: approximate camera-to-water-surface distances (meters)",
            "  # Improves Stage 3 initialization",
            "  # initial_water_z:",
        ]
    )

    for cam in camera_names:
        lines.append(f"  #   {cam}: 0.20")

    lines.extend(
        [
            "",
            "optimization:",
            '  robust_loss: "huber"   # Options: "huber", "soft_l1", "linear"',
            "  loss_scale: 1.0        # Residual scale for robust loss (pixels)",
            "  # max_calibration_frames: 150  # Max frames for Stage 3/4 (null = no limit)",
            "  # refine_intrinsics: false  # Stage 4: refine focal lengths and principal points",
            "  # refine_auxiliary_intrinsics: false  # Stage 4b: refine auxiliary camera intrinsics",
            "",
            "detection:",
            "  min_corners: 8         # Minimum corners per frame to use detection",
            "  min_cameras: 2         # Minimum cameras seeing board to use frame",
            "  frame_step: 5          # Process every Nth frame (1 = all frames)",
            "",
            "validation:",
            "  holdout_fraction: 0.2  # Fraction of frames held out for validation",
            "  save_detailed_residuals: true  # Save per-corner residual data",
            "",
        ]
    )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
