"""Unit tests for CLI entry point."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from aquacal.cli import create_parser, cmd_calibrate, main


class TestCreateParser:
    def test_calibrate_subcommand_exists(self):
        parser = create_parser()
        # Parse valid calibrate command
        args = parser.parse_args(["calibrate", "config.yaml"])
        assert args.command == "calibrate"
        assert args.config_path == Path("config.yaml")

    def test_calibrate_options(self):
        parser = create_parser()
        args = parser.parse_args([
            "calibrate", "config.yaml",
            "-v", "-o", "/output", "--dry-run"
        ])
        assert args.verbose is True
        assert args.output_dir == Path("/output")
        assert args.dry_run is True


class TestCmdCalibrate:
    def test_missing_file(self, tmp_path, capsys):
        parser = create_parser()
        args = parser.parse_args(["calibrate", str(tmp_path / "missing.yaml")])

        exit_code = cmd_calibrate(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_dry_run_valid_config(self, tmp_path, capsys):
        # Create minimal valid config
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
board:
  squares_x: 8
  squares_y: 6
  square_size: 0.030
  marker_size: 0.022
  dictionary: DICT_4X4_50
cameras:
  - cam0
  - cam1
paths:
  intrinsic_videos:
    cam0: /data/i0.mp4
    cam1: /data/i1.mp4
  extrinsic_videos:
    cam0: /data/e0.mp4
    cam1: /data/e1.mp4
  output_dir: /output
""")
        parser = create_parser()
        args = parser.parse_args(["calibrate", str(config_file), "--dry-run"])

        exit_code = cmd_calibrate(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "valid" in captured.out.lower()


class TestMain:
    def test_no_args_exits_error(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @patch("aquacal.cli.run_calibration")
    def test_integration(self, mock_run, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
board:
  squares_x: 8
  squares_y: 6
  square_size: 0.030
  marker_size: 0.022
  dictionary: DICT_4X4_50
cameras:
  - cam0
paths:
  intrinsic_videos:
    cam0: /data/i0.mp4
  extrinsic_videos:
    cam0: /data/e0.mp4
  output_dir: /output
""")
        mock_run.return_value = MagicMock()

        exit_code = main(["calibrate", str(config_file)])

        assert exit_code == 0
        mock_run.assert_called_once()
