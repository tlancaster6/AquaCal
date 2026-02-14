"""Tests for optimization common utilities (_optim_common.py)."""

import numpy as np
import scipy.sparse
import pytest

from aquacal.calibration._optim_common import make_sparse_jacobian_func


def _toy_cost(params, A):
    """Simple quadratic cost: residuals = A @ params.

    For this linear function f(x) = A @ x, the Jacobian is exactly A.
    """
    return A @ params


class TestMakeSparseJacobianFunc:
    """Tests for make_sparse_jacobian_func with dense_threshold behavior."""

    def test_small_problem_returns_dense_array(self):
        """Small problem (below threshold) returns numpy.ndarray (dense)."""
        # Create a small sparsity pattern (10 residuals x 5 params = 50 elements)
        jac_sparsity = np.ones((10, 5), dtype=np.int8)

        # Create simple linear cost function with known Jacobian
        A = np.random.randn(10, 5)
        cost_func = _toy_cost
        cost_args = (A,)
        bounds = (
            -np.inf * np.ones(5),
            np.inf * np.ones(5),
        )

        # Use default dense_threshold (500M >> 50 elements)
        jac_func = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=500_000_000,
        )

        # Evaluate Jacobian at arbitrary point
        params = np.random.randn(5)
        J = jac_func(params, A)

        # Should return numpy.ndarray (dense)
        assert isinstance(J, np.ndarray)
        assert not scipy.sparse.issparse(J)
        assert J.shape == (10, 5)

    def test_large_problem_returns_sparse_matrix(self):
        """Large problem (exceeds threshold) returns sparse matrix."""
        # Create a small sparsity pattern but force sparse path with threshold=0
        jac_sparsity = np.ones((10, 5), dtype=np.int8)

        A = np.random.randn(10, 5)
        cost_func = _toy_cost
        cost_args = (A,)
        bounds = (
            -np.inf * np.ones(5),
            np.inf * np.ones(5),
        )

        # Set dense_threshold=0 to force sparse path
        jac_func = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=0,
        )

        # Evaluate Jacobian at arbitrary point
        params = np.random.randn(5)
        J = jac_func(params, A)

        # Should return sparse matrix
        assert scipy.sparse.issparse(J)
        assert J.shape == (10, 5)

    def test_threshold_boundary_behavior(self):
        """Threshold boundary: equal returns dense, exceeds returns sparse."""
        # Create sparsity pattern of known size (100 x 100 = 10,000 elements)
        jac_sparsity = np.ones((100, 100), dtype=np.int8)

        A = np.random.randn(100, 100)
        cost_func = _toy_cost
        cost_args = (A,)
        bounds = (
            -np.inf * np.ones(100),
            np.inf * np.ones(100),
        )
        params = np.random.randn(100)

        # Case 1: dense_threshold = 10,000 (equal) -> should return dense
        jac_func_dense = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=10_000,
        )
        J_dense = jac_func_dense(params, A)
        assert isinstance(J_dense, np.ndarray)
        assert not scipy.sparse.issparse(J_dense)

        # Case 2: dense_threshold = 9,999 (exceeds) -> should return sparse
        jac_func_sparse = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=9_999,
        )
        J_sparse = jac_func_sparse(params, A)
        assert scipy.sparse.issparse(J_sparse)

    def test_both_paths_produce_correct_jacobian(self):
        """Both dense and sparse paths produce correct Jacobian values."""
        # For linear cost f(x) = A @ x, the Jacobian is exactly A
        # Finite differences should recover this (up to numerical precision)

        # Create test problem
        jac_sparsity = np.ones((20, 10), dtype=np.int8)
        A = np.random.randn(20, 10)
        cost_func = _toy_cost
        cost_args = (A,)
        bounds = (
            -np.inf * np.ones(10),
            np.inf * np.ones(10),
        )
        params = np.random.randn(10)

        # Get dense Jacobian
        jac_func_dense = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=500_000_000,  # Force dense
        )
        J_dense = jac_func_dense(params, A)

        # Get sparse Jacobian
        jac_func_sparse = make_sparse_jacobian_func(
            cost_func,
            cost_args,
            jac_sparsity,
            bounds,
            dense_threshold=0,  # Force sparse
        )
        J_sparse = jac_func_sparse(params, A)

        # Convert sparse to dense for comparison
        J_sparse_dense = (
            J_sparse.toarray() if hasattr(J_sparse, "toarray") else J_sparse
        )

        # Both should match the true Jacobian (A) within FD tolerance
        # For a linear function, 2-point FD is exact up to floating-point precision
        np.testing.assert_allclose(J_dense, A, atol=1e-6)
        np.testing.assert_allclose(J_sparse_dense, A, atol=1e-6)

        # Both paths should produce identical results
        np.testing.assert_allclose(J_dense, J_sparse_dense, atol=1e-10)
