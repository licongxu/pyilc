"""
Parity tests for Wavelets construction (CPU) and integration of the GPU
weight path through scale_info.weights_from_covmat_at_scale_j scaffolding.

Full-sky NILC with healpy SHTs remains CPU-bound (healpy has no CUDA path);
the accelerated kernel is the per-pixel constrained-ILC weight solve.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyilc.wavelets import Wavelets
from pyilc.ilc_linalg import available_backends, compute_ilc_weights_from_cov


def test_gaussian_needlet_partition_of_unity():
    wv = Wavelets(N_scales=6, ELLMAX=512, tol=1e-6, taper_width=0)
    fwhm = np.array([300.0, 120.0, 60.0, 30.0, 15.0])
    ell, filts = wv.GaussianNeedlets(FWHM_arcmin=fwhm)
    assert ell.shape[0] == 513
    assert filts.shape == (6, 513)
    assert np.max(np.abs(np.sum(filts**2, axis=0) - 1.0)) < 1e-6


def test_tophat_harmonic_partition_of_unity():
    wv = Wavelets(N_scales=5, ELLMAX=200, tol=1e-6, taper_width=0)
    bins = [0, 40, 80, 120, 160, 201]
    ell, filts = wv.TopHatHarmonic(bins)
    assert filts.shape[0] == 5
    assert np.max(np.abs(np.sum(filts**2, axis=0) - 1.0)) < 1e-6


def test_end_to_end_weight_pipeline_cpu_gpu():
    """Simulate the stack assembled inside weights_from_covmat_at_scale_j."""
    n_pix, n_freq, n_comp = 2048, 6, 2
    rng = np.random.default_rng(42)
    # Fake cov maps as produced by compute_covariance_at_scale_j: upper-triangle list
    cov_maps = []
    full = np.zeros((n_freq, n_freq, n_pix))
    for a in range(n_freq):
        for b in range(a, n_freq):
            m = rng.standard_normal(n_pix)
            full[a, b] = m
            full[b, a] = m
            cov_maps.append(m)
    # Make SPD by C = X X^T + n I using the random full as X proxy at each pixel
    X = rng.standard_normal((n_pix, n_freq, n_freq))
    cov_pff = X @ np.transpose(X, (0, 2, 1)) + n_freq * np.eye(n_freq)
    A = rng.standard_normal((n_freq, n_comp))
    A[:, 0] = np.abs(A[:, 0]) + 0.3

    w_cpu, _ = compute_ilc_weights_from_cov(cov_pff, A, backend="numpy")
    backends = available_backends()
    if backends["jax"]:
        w_gpu, used = compute_ilc_weights_from_cov(cov_pff, A, backend="jax")
        assert used == "jax"
        np.testing.assert_allclose(w_gpu, w_cpu, rtol=1e-9, atol=1e-9)
    else:
        pytest.skip("jax not available for GPU parity check")

    # Response constraints
    resp = w_cpu @ A
    assert np.max(np.abs(resp[:, 0] - 1.0)) < 1e-8
    assert np.max(np.abs(resp[:, 1:])) < 1e-8
