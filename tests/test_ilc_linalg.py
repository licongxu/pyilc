"""
Regression tests for constrained-ILC weight linear algebra.

There were no formal unit tests in the original repository. These tests
encode the mathematical requirements of McCarthy & Hill 2023 (Eqs. 29–30):

1. Response preservation: w · a_preserved = 1 for every pixel.
2. Response nulling:     w · a_deproj    = 0 for every deprojected column.
3. Backend parity: GPU (jax/cupy) and CPU (numpy/numba) weights agree to
   tight absolute/relative tolerances on the same random SPD covariances.
4. apply_weights_to_maps consistency across backends.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyilc.ilc_linalg import (
    apply_weights_to_maps,
    available_backends,
    compute_ilc_weights_from_cov,
    resolve_backend,
)


def _spd_cov(n_pix: int, n_freq: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_pix, n_freq, n_freq))
    return X @ np.transpose(X, (0, 2, 1)) + n_freq * np.eye(n_freq)


def _mixing(n_freq: int, n_comp: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_freq, n_comp))
    # Preserved SED column kept strictly positive / non-degenerate
    A[:, 0] = np.abs(A[:, 0]) + 0.5
    return A


def _assert_response(weights: np.ndarray, A: np.ndarray, tol: float = 1e-8):
    resp = weights @ A  # (P, C)
    assert np.max(np.abs(resp[:, 0] - 1.0)) < tol
    if A.shape[1] > 1:
        assert np.max(np.abs(resp[:, 1:])) < tol


@pytest.mark.parametrize("n_pix,n_freq,n_comp", [(32, 4, 1), (64, 6, 2), (48, 5, 3)])
def test_numpy_response_constraints(n_pix, n_freq, n_comp):
    cov = _spd_cov(n_pix, n_freq)
    A = _mixing(n_freq, n_comp)
    w, used = compute_ilc_weights_from_cov(cov, A, backend="numpy")
    assert used == "numpy"
    assert w.shape == (n_pix, n_freq)
    _assert_response(w, A, tol=1e-8)


def test_numba_matches_numpy():
    # Keep this test free of GPU imports: probing cupy/jax in-process before
    # numba parallel kernels can crash some OpenBLAS+CUDA combos.
    try:
        import numba  # noqa: F401
    except Exception:
        pytest.skip("numba not available")
    cov = _spd_cov(128, 6, seed=3)
    A = _mixing(6, 2, seed=4)
    w_np, _ = compute_ilc_weights_from_cov(cov, A, backend="numpy")
    w_nb, used = compute_ilc_weights_from_cov(cov, A, backend="numba")
    assert used == "numba"
    np.testing.assert_allclose(w_nb, w_np, rtol=1e-10, atol=1e-10)
    _assert_response(w_nb, A)


def test_jax_matches_numpy_and_response():
    backends = available_backends()
    if not backends["jax"]:
        pytest.skip("jax not available")
    cov = _spd_cov(256, 6, seed=5)
    A = _mixing(6, 2, seed=6)
    w_np, _ = compute_ilc_weights_from_cov(cov, A, backend="numpy")
    w_jx, used = compute_ilc_weights_from_cov(cov, A, backend="jax")
    assert used == "jax"
    # float64 GPU path should track the CPU reference tightly
    np.testing.assert_allclose(w_jx, w_np, rtol=1e-9, atol=1e-9)
    _assert_response(w_jx, A, tol=1e-8)


def test_cupy_matches_numpy_if_available():
    backends = available_backends()
    if not backends["cupy"]:
        pytest.skip("cupy/CUDA not usable in this environment")
    cov = _spd_cov(200, 5, seed=7)
    A = _mixing(5, 1, seed=8)
    w_np, _ = compute_ilc_weights_from_cov(cov, A, backend="numpy")
    w_cp, used = compute_ilc_weights_from_cov(cov, A, backend="cupy")
    assert used == "cupy"
    np.testing.assert_allclose(w_cp, w_np, rtol=1e-9, atol=1e-9)
    _assert_response(w_cp, A)


def test_auto_backend_resolves():
    chosen = resolve_backend("auto")
    assert chosen in {"jax", "numba", "numpy", "cupy"}
    cov = _spd_cov(64, 4)
    A = _mixing(4, 1)
    w, used = compute_ilc_weights_from_cov(cov, A, backend="auto")
    assert used == chosen
    _assert_response(w, A)


def test_apply_weights_numpy_vs_jax():
    rng = np.random.default_rng(9)
    n_pix, n_freq = 1000, 6
    weights = rng.standard_normal((n_pix, n_freq))
    maps_fp = rng.standard_normal((n_freq, n_pix))
    out_np, _ = apply_weights_to_maps(weights, maps_fp, backend="numpy")
    expected = np.sum(weights * maps_fp.T, axis=1)
    np.testing.assert_allclose(out_np, expected)
    backends = available_backends()
    if backends["jax"]:
        out_jx, used = apply_weights_to_maps(weights, maps_fp, backend="jax")
        assert used == "jax"
        np.testing.assert_allclose(out_jx, expected, rtol=1e-12, atol=1e-12)


def test_legacy_wavelets_helpers_still_importable():
    """Old numba helpers remain importable for external scripts.

    Run in a subprocess so a prior GPU (jax/cupy) import in this pytest
    process cannot trip OpenBLAS/numba crashes on some hosts.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import numpy as np
        from pyilc.wavelets import (
            my_numba_solver,
            my_numba_solver_parallelb,
            numba_det,
            numba_matmul,
        )
        rng = np.random.default_rng(0)
        X = rng.standard_normal((16, 3, 3))
        A = X @ np.transpose(X, (0, 2, 1)) + 3 * np.eye(3)
        b = np.ones((3, 2))
        out = my_numba_solver(A, b)
        assert out.shape == (16, 3, 2)
        bb = np.ones((16, 3))
        out2 = my_numba_solver_parallelb(A, bb)
        assert out2.shape == (16, 3)
        assert numba_det(A).shape == (16,)
        assert numba_matmul(A, np.eye(3)).shape == (16, 3, 3)
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**dict(**__import__("os").environ), "OPENBLAS_NUM_THREADS": "4"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
