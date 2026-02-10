"""Unit tests for CLI entry point."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from aquacal.cli import create_parser, cmd_calibrate, cmd_init, main


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


class TestCmdInit:
    def test_basic_generation(self, tmp_path, capsys):
        # Create video directories
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        # Create video files
        (intrinsic_dir / "cam0.mp4").touch()
        (intrinsic_dir / "cam1.mp4").touch()
        (extrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam1.mp4").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 0
        assert output_file.exists()

        # Verify generated config structure
        with open(output_file) as f:
            config = yaml.safe_load(f)

        assert "board" in config
        assert "cameras" in config
        assert config["cameras"] == ["cam0", "cam1"]
        assert "paths" in config
        assert "cam0" in config["paths"]["intrinsic_videos"]
        assert "cam1" in config["paths"]["intrinsic_videos"]

        captured = capsys.readouterr()
        assert "2 camera(s)" in captured.out

    def test_custom_regex(self, tmp_path):
        # Create directories with pattern like "experiment_cam0_trial1.mp4"
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        (intrinsic_dir / "experiment_cam0_trial1.mp4").touch()
        (intrinsic_dir / "experiment_cam1_trial1.mp4").touch()
        (extrinsic_dir / "experiment_cam0_trial2.mp4").touch()
        (extrinsic_dir / "experiment_cam1_trial2.mp4").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
            "--pattern", r"experiment_(cam\d+)_trial",
        ])

        exit_code = cmd_init(args)

        assert exit_code == 0
        with open(output_file) as f:
            config = yaml.safe_load(f)
        assert config["cameras"] == ["cam0", "cam1"]

    def test_video_extensions_only(self, tmp_path, capsys):
        # Test that only .mp4, .avi, .mov, .mkv are scanned
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        # Valid extensions (case-insensitive)
        (intrinsic_dir / "cam0.mp4").touch()
        (intrinsic_dir / "cam1.MP4").touch()
        (intrinsic_dir / "cam2.avi").touch()
        (intrinsic_dir / "cam3.mov").touch()
        (intrinsic_dir / "cam4.mkv").touch()

        (extrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam1.avi").touch()
        (extrinsic_dir / "cam2.AVI").touch()
        (extrinsic_dir / "cam3.MOV").touch()
        (extrinsic_dir / "cam4.MKV").touch()

        # Invalid extensions (should be ignored)
        (intrinsic_dir / "cam5.txt").touch()
        (intrinsic_dir / "cam6.jpg").touch()
        (extrinsic_dir / "cam7.mp3").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 0
        with open(output_file) as f:
            config = yaml.safe_load(f)

        # Should only include cam0-cam4 (intersection)
        assert set(config["cameras"]) == {"cam0", "cam1", "cam2", "cam3", "cam4"}

    def test_camera_mismatch_warning(self, tmp_path, capsys):
        # Test partial overlap produces warning
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        # cam0 and cam1 in both, cam2 only in intrinsic, cam3 only in extrinsic
        (intrinsic_dir / "cam0.mp4").touch()
        (intrinsic_dir / "cam1.mp4").touch()
        (intrinsic_dir / "cam2.mp4").touch()

        (extrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam1.mp4").touch()
        (extrinsic_dir / "cam3.mp4").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert "cam2" in captured.err  # Only in intrinsic
        assert "cam3" in captured.err  # Only in extrinsic

        with open(output_file) as f:
            config = yaml.safe_load(f)
        assert set(config["cameras"]) == {"cam0", "cam1"}

    def test_empty_intersection_error(self, tmp_path, capsys):
        # Test no overlap produces error
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        (intrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam1.mp4").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1
        assert not output_file.exists()
        captured = capsys.readouterr()
        assert "No common camera names" in captured.err

    def test_output_exists_error(self, tmp_path, capsys):
        # Test existing output file produces error
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        (intrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam0.mp4").touch()

        output_file = tmp_path / "config.yaml"
        output_file.touch()  # Pre-create the file

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.err

    def test_missing_directory_error(self, tmp_path, capsys):
        # Test missing directories produce error
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "nonexistent"

        intrinsic_dir.mkdir()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err

    def test_invalid_regex_error(self, tmp_path, capsys):
        # Test regex with wrong number of capture groups
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        (intrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam0.mp4").touch()

        output_file = tmp_path / "config.yaml"

        # No capture groups
        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
            "--pattern", r"cam\d+",  # No capture group
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "exactly one capture group" in captured.err

        # Multiple capture groups
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
            "--pattern", r"(cam)(\d+)",  # Two capture groups
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1

    def test_no_videos_found_error(self, tmp_path, capsys):
        # Test empty directories produce error
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No video files found" in captured.err

    def test_generated_config_has_all_fields(self, tmp_path):
        # Verify generated config includes all CalibrationConfig fields
        intrinsic_dir = tmp_path / "intrinsic"
        extrinsic_dir = tmp_path / "extrinsic"
        intrinsic_dir.mkdir()
        extrinsic_dir.mkdir()

        (intrinsic_dir / "cam0.mp4").touch()
        (extrinsic_dir / "cam0.mp4").touch()

        output_file = tmp_path / "config.yaml"

        parser = create_parser()
        args = parser.parse_args([
            "init",
            "--intrinsic-dir", str(intrinsic_dir),
            "--extrinsic-dir", str(extrinsic_dir),
            "--output", str(output_file),
        ])

        exit_code = cmd_init(args)

        assert exit_code == 0

        with open(output_file) as f:
            config = yaml.safe_load(f)

        # Check all required sections
        assert "board" in config
        assert "cameras" in config
        assert "paths" in config
        assert "interface" in config
        assert "optimization" in config
        assert "detection" in config
        assert "validation" in config

        # Check defaults match CalibrationConfig
        assert config["interface"]["n_air"] == 1.0
        assert config["interface"]["n_water"] == 1.333
        assert config["interface"]["normal_fixed"] is True
        assert config["optimization"]["robust_loss"] == "huber"
        assert config["optimization"]["loss_scale"] == 1.0
        assert config["detection"]["min_corners"] == 8
        assert config["detection"]["min_cameras"] == 2
        assert config["validation"]["holdout_fraction"] == 0.2
        assert config["validation"]["save_detailed_residuals"] is True


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
