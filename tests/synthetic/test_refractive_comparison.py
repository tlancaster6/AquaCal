"""Pytest wrappers for refractive vs non-refractive comparison experiments."""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tests.synthetic.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
)


@pytest.mark.slow
class TestExperiment1:
    """Experiment 1: Parameter Fidelity tests."""

    @pytest.fixture(scope="class")
    def experiment_results(self, tmp_path_factory):
        """Run experiment 1 once for all tests in this class."""
        output_dir = tmp_path_factory.mktemp("exp1")
        try:
            results = run_experiment_1(output_dir, seed=42)
            return results, output_dir
        except Exception as e:
            pytest.fail(f"Experiment 1 calibration failed: {e}")

    def test_refractive_focal_length_error_small(self, experiment_results):
        """Refractive model should recover focal length accurately."""
        results, _ = experiment_results
        errors = results["errors_refractive"]

        for cam, errs in errors.items():
            focal_err = abs(errs["focal_length_error_pct"])
            assert focal_err < 0.5, (
                f"Refractive focal error for {cam} is {focal_err:.2f}% (threshold 0.5%)"
            )

    def test_nonrefractive_focal_length_error_large(self, experiment_results):
        """Non-refractive model should show focal length bias."""
        results, _ = experiment_results
        errors = results["errors_nonrefractive"]

        # At least half the cameras should show >1% error
        large_errors = sum(
            1 for errs in errors.values() if abs(errs["focal_length_error_pct"]) > 1.0
        )
        assert large_errors >= len(errors) / 2, (
            f"Only {large_errors}/{len(errors)} cameras show >1% focal error "
            f"for non-refractive model (expected >= {len(errors) / 2})"
        )

    def test_refractive_z_error_small(self, experiment_results):
        """Refractive model should recover camera Z positions accurately."""
        results, _ = experiment_results
        errors = results["errors_refractive"]

        for cam, errs in errors.items():
            z_err = abs(errs["z_position_error_mm"])
            assert z_err < 5.0, (
                f"Refractive Z error for {cam} is {z_err:.2f}mm (threshold 5mm)"
            )

    def test_nonrefractive_z_error_large(self, experiment_results):
        """Non-refractive model should show Z position bias."""
        results, _ = experiment_results
        errors = results["errors_nonrefractive"]

        # Mean Z error should exceed 10mm
        mean_z_err = np.mean([abs(e["z_position_error_mm"]) for e in errors.values()])
        assert mean_z_err > 10.0, (
            f"Non-refractive mean Z error is {mean_z_err:.2f}mm (threshold 10mm)"
        )

    def test_xy_error_similar(self, experiment_results):
        """XY position should be similar for both models (not affected by refraction)."""
        results, _ = experiment_results
        errors_refr = results["errors_refractive"]
        errors_nonrefr = results["errors_nonrefractive"]

        for cam in errors_refr:
            xy_err_refr = errors_refr[cam]["xy_position_error_mm"]
            xy_err_nonrefr = errors_nonrefr[cam]["xy_position_error_mm"]

            assert xy_err_refr < 15.0, (
                f"Refractive XY error for {cam} is {xy_err_refr:.2f}mm"
            )
            assert xy_err_nonrefr < 15.0, (
                f"Non-refractive XY error for {cam} is {xy_err_nonrefr:.2f}mm"
            )

    def test_plots_created(self, experiment_results):
        """All expected plots should be created."""
        _, output_dir = experiment_results

        expected_plots = [
            "exp1_focal_length_error.png",
            "exp1_camera_xy_positions.png",
            "exp1_camera_z_error.png",
            "exp1_distortion_error.png",
        ]

        for plot in expected_plots:
            assert (output_dir / plot).exists(), f"Plot {plot} was not created"


@pytest.mark.slow
class TestExperiment2:
    """Experiment 2: Depth Generalization tests."""

    @pytest.fixture(scope="class")
    def experiment_results(self, tmp_path_factory):
        """Run experiment 2 once for all tests in this class."""
        output_dir = tmp_path_factory.mktemp("exp2")
        try:
            results = run_experiment_2(output_dir, seed=42)
            return results, output_dir
        except Exception as e:
            pytest.fail(f"Experiment 2 calibration failed: {e}")

    def test_refractive_flat_across_depth(self, experiment_results):
        """Refractive model should have flat signed error across depths."""
        results, _ = experiment_results
        signed_means = [r["signed_mean_mm"] for r in results["results_refractive"]]

        error_range = max(signed_means) - min(signed_means)
        assert error_range < 0.5, (
            f"Refractive signed error range is {error_range:.2f}mm (threshold 0.5mm)"
        )

    def test_nonrefractive_depth_dependent(self, experiment_results):
        """Non-refractive model should show depth-dependent bias."""
        results, _ = experiment_results
        signed_means = [r["signed_mean_mm"] for r in results["results_nonrefractive"]]

        error_range = max(signed_means) - min(signed_means)
        assert error_range > 1.0, (
            f"Non-refractive signed error range is {error_range:.2f}mm (threshold 1mm)"
        )

    def test_refractive_rmse_stable(self, experiment_results):
        """Refractive model should have stable RMSE across depths."""
        results, _ = experiment_results
        _rmse_values = [r["rmse_mm"] for r in results["results_refractive"]]

        for depth_result in results["results_refractive"]:
            rmse = depth_result["rmse_mm"]
            depth = depth_result["depth"]
            assert rmse < 1.0, (
                f"Refractive RMSE at depth {depth}m is {rmse:.2f}mm (threshold 1mm)"
            )

    def test_scale_factor_refractive_near_one(self, experiment_results):
        """Refractive model should have scale factor near 1.0 at all depths."""
        results, _ = experiment_results

        for depth_result in results["results_refractive"]:
            scale = depth_result["scale"]
            depth = depth_result["depth"]
            assert abs(scale - 1.0) < 0.01, (
                f"Refractive scale factor at depth {depth}m is {scale:.4f} "
                f"(threshold 1.0 +/- 0.01)"
            )

    def test_plots_created(self, experiment_results):
        """All expected plots should be created."""
        _, output_dir = experiment_results

        expected_plots = [
            "exp2_signed_error_vs_depth.png",
            "exp2_rmse_vs_depth.png",
            "exp2_scale_factor_vs_depth.png",
            "exp2_xy_heatmaps.png",
        ]

        for plot in expected_plots:
            assert (output_dir / plot).exists(), f"Plot {plot} was not created"


@pytest.mark.slow
class TestExperiment3:
    """Experiment 3: Depth Scaling tests."""

    @pytest.fixture(scope="class")
    def experiment_results(self, tmp_path_factory):
        """Run experiment 3 once for all tests in this class."""
        output_dir = tmp_path_factory.mktemp("exp3")
        try:
            results = run_experiment_3(output_dir, seed=42)
            return results, output_dir
        except Exception as e:
            pytest.fail(f"Experiment 3 calibration failed: {e}")

    def test_refractive_rmse_stable_across_depth(self, experiment_results):
        """Refractive model should have stable RMSE across all depths."""
        results, _ = experiment_results

        for depth_result in results["results_refractive"]:
            rmse = depth_result["rmse_mm"]
            depth = depth_result["depth"]
            assert rmse < 1.0, (
                f"Refractive RMSE at depth {depth}m is {rmse:.2f}mm (threshold 1mm)"
            )

    def test_nonrefractive_rmse_grows(self, experiment_results):
        """Non-refractive model RMSE should grow with depth."""
        results, _ = experiment_results
        rmse_values = [r["rmse_mm"] for r in results["results_nonrefractive"]]

        rmse_shallowest = rmse_values[0]  # First depth
        rmse_deepest = rmse_values[-1]  # Last depth

        assert rmse_deepest > 2 * rmse_shallowest, (
            f"Non-refractive RMSE growth too small: "
            f"{rmse_shallowest:.2f}mm -> {rmse_deepest:.2f}mm "
            f"(expected >2x growth)"
        )

    def test_focal_error_grows_with_depth(self, experiment_results):
        """Non-refractive focal error should grow with calibration depth."""
        results, _ = experiment_results

        focal_at_shallow = results["results_nonrefractive"][0][
            "focal_err_pct"
        ]  # Z=0.85m
        focal_at_deep = results["results_nonrefractive"][-1]["focal_err_pct"]  # Z=2.5m

        assert focal_at_deep > focal_at_shallow, (
            f"Non-refractive focal error did not grow with depth: "
            f"{focal_at_shallow:.2f}% -> {focal_at_deep:.2f}%"
        )

    def test_plots_created(self, experiment_results):
        """All expected plots should be created."""
        _, output_dir = experiment_results

        expected_plots = [
            "exp3_rmse_vs_depth.png",
            "exp3_focal_error_vs_depth.png",
            "exp3_z_error_vs_depth.png",
            "exp3_xy_heatmaps.png",
        ]

        for plot in expected_plots:
            assert (output_dir / plot).exists(), f"Plot {plot} was not created"
