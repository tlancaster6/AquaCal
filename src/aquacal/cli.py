"""Command-line interface for AquaCal calibration pipeline."""

import argparse
import sys
from pathlib import Path

from aquacal.calibration.pipeline import run_calibration, load_config
from aquacal.config.schema import CalibrationError


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
        print("Configuration valid.")
        print(f"  Cameras: {', '.join(config.camera_names)}")
        print(f"  Board: {config.board.squares_x}x{config.board.squares_y}")
        print(f"  Output: {config.output_dir}")
        return 0

    # Run calibration
    try:
        result = run_calibration(config_path)
        return 0
    except CalibrationError as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nCalibration interrupted.", file=sys.stderr)
        return 130


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
