"""
Core constrained-ILC weight linear algebra for NILC/HILC.

Implements Eqs. (29)–(30) of McCarthy & Hill 2023 (arXiv:2307.01043) for a
stack of per-pixel frequency–frequency covariance matrices.

Backends
--------
- ``numpy``: pure NumPy (reference; fixed for NumPy 2.x batched solve API)
- ``numba``: original pixel-parallel CPU path (default legacy behavior)
- ``jax``:   GPU/CPU via JAX (preferred GPU path)
- ``cupy``:  GPU via CuPy (optional if CuPy + CUDA libs are available)
- ``auto``:  jax if a GPU is visible, else numba, else numpy

Public entry point: :func:`compute_ilc_weights_from_cov`.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

_BACKEND_CACHE: dict = {}


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def _jax_has_gpu() -> bool:
    try:
        import jax
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False


def _cupy_available() -> bool:
    try:
        import cupy as cp  # noqa: F401
        # Probe device count only — avoid allocating GPU memory here so that
        # discovery does not perturb subsequent CPU/numba runs in-process.
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def resolve_backend(backend: str = "auto") -> str:
    """Resolve a user backend request to a concrete backend name."""
    backend = (backend or "auto").lower().strip()
    if backend in _BACKEND_CACHE:
        return _BACKEND_CACHE[backend]

    if backend == "auto":
        if _jax_available() and _jax_has_gpu():
            chosen = "jax"
        else:
            try:
                from numba import njit  # noqa: F401
                chosen = "numba"
            except Exception:
                chosen = "numpy"
    elif backend == "jax":
        if not _jax_available():
            raise ImportError("backend='jax' requested but jax is not importable")
        chosen = "jax"
    elif backend == "cupy":
        if not _cupy_available():
            raise ImportError(
                "backend='cupy' requested but cupy/CUDA is not usable "
                "(check LD_LIBRARY_PATH for libcublas, etc.)"
            )
        chosen = "cupy"
    elif backend == "numba":
        from numba import njit  # noqa: F401
        chosen = "numba"
    elif backend == "numpy":
        chosen = "numpy"
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; choose from "
            "'auto', 'numpy', 'numba', 'jax', 'cupy'"
        )

    _BACKEND_CACHE[backend] = chosen
    return chosen


def available_backends() -> dict:
    """Return discovery info useful for tests and benchmarks."""
    info = {
        "numpy": True,
        "numba": False,
        "jax": False,
        "jax_gpu": False,
        "cupy": False,
        "auto": None,
    }
    try:
        from numba import njit  # noqa: F401
        info["numba"] = True
    except Exception:
        pass
    info["jax"] = _jax_available()
    info["jax_gpu"] = _jax_has_gpu() if info["jax"] else False
    info["cupy"] = _cupy_available()
    info["auto"] = resolve_backend("auto")
    return info


# ---------------------------------------------------------------------------
# NumPy reference implementation (matches wavelets.py non-numba branch,
# with NumPy 2.x-safe batched solves).
# ---------------------------------------------------------------------------

def _weights_numpy(cov: np.ndarray, A_mix: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    cov : (P, F, F) array
        Per-pixel covariance matrices (must be SPD / well-conditioned).
    A_mix : (F, C) array
        Mixing matrix: column 0 is the preserved component SED; remaining
        columns are deprojected component SEDs.

    Returns
    -------
    weights : (P, F) array
    """
    cov = np.asarray(cov, dtype=np.float64)
    A_mix = np.asarray(A_mix, dtype=np.float64)
    n_pix, n_freq, n_freq2 = cov.shape
    assert n_freq == n_freq2
    assert A_mix.shape[0] == n_freq
    n_comp = A_mix.shape[1]

    # solve C w = A  -> tmp1 (P, F, C); full transpose -> (C, F, P)
    rhs = np.broadcast_to(A_mix, (n_pix,) + A_mix.shape).copy()
    tmp1 = np.linalg.solve(cov, rhs)
    tmp1 = np.transpose(tmp1)  # (C, F, P)

    # Q_{αβ}(p) = A^T C^{-1} A  as einsum matching original code
    Qab_pix = np.einsum("ajp,bj->abp", tmp1, np.transpose(A_mix), optimize=True)

    tempvec = np.zeros((n_comp, n_pix), dtype=np.float64)
    if n_comp == 1:
        tempvec[0] = 1.0
    else:
        for a in range(n_comp):
            QSa = np.delete(np.delete(Qab_pix, a, 0), 0, 1)
            tempvec[a] = ((-1.0) ** float(a)) * np.linalg.det(
                np.transpose(QSa, (2, 0, 1))
            )

    tmp2 = np.einsum("ia,ap->ip", A_mix, tempvec, optimize=True)  # (F, P)
    # NumPy 2.x: 2-D rhs (P, F) is ambiguous vs (M, K); force (P, F, 1)
    tmp3 = np.linalg.solve(cov, np.transpose(tmp2)[..., None])[..., 0]  # (P, F)
    tmp3 = np.transpose(tmp3)  # (F, P)

    detQ = np.linalg.det(np.transpose(Qab_pix, (2, 0, 1)))  # (P,)
    weights = (1.0 / detQ)[:, None] * np.transpose(tmp3)  # (P, F)
    return weights


# ---------------------------------------------------------------------------
# Numba path (re-implements the original wavelets helpers in one place)
# ---------------------------------------------------------------------------

def _get_numba_kernels():
    from numba import njit, prange

    @njit(parallel=True)
    def jit_linsolve(A, b):
        m = A.shape[0]
        n = A.shape[-1]
        k = b.shape[-1]
        ret = np.empty((m, n, k))
        for i in prange(m):
            ret[i, :] = np.linalg.solve(A[i], b)
        return ret

    @njit(parallel=True)
    def jit_linsolve_parallelb(A, b):
        m = A.shape[0]
        k = b.shape[-1]
        ret = np.empty((m, k))
        for i in prange(m):
            ret[i, :] = np.linalg.solve(A[i], b[i])
        return ret

    @njit(parallel=True)
    def jit_det(A):
        m = A.shape[0]
        ret = np.empty((m))
        for i in prange(m):
            ret[i] = np.linalg.det(A[i])
        return ret

    @njit(parallel=True)
    def jit_matmul(A, b):
        m = A.shape[0]
        n = A.shape[1]
        ret = np.empty((m, n, n))
        for i in prange(m):
            ret[i] = np.dot(A[i], b)
        return ret

    return jit_linsolve, jit_linsolve_parallelb, jit_det, jit_matmul


_NUMBA_KERNELS = None


def _weights_numba(cov: np.ndarray, A_mix: np.ndarray) -> np.ndarray:
    global _NUMBA_KERNELS
    if _NUMBA_KERNELS is None:
        _NUMBA_KERNELS = _get_numba_kernels()
    jit_linsolve, jit_linsolve_parallelb, jit_det, jit_matmul = _NUMBA_KERNELS

    cov = np.asarray(cov, dtype=np.float64)
    A_mix = np.asarray(A_mix, dtype=np.float64)
    n_pix = cov.shape[0]
    n_comp = A_mix.shape[1]

    tmp1 = jit_linsolve(cov, A_mix)
    tmp1 = np.transpose(tmp1)
    Qab_pix = np.transpose(jit_matmul(np.transpose(tmp1, (2, 0, 1)), A_mix), (1, 2, 0))

    tempvec = np.zeros((n_comp, n_pix), dtype=np.float64)
    if n_comp == 1:
        tempvec[0] = 1.0
    else:
        for a in range(n_comp):
            QSa = np.delete(np.delete(Qab_pix, a, 0), 0, 1)
            tempvec[a] = ((-1.0) ** float(a)) * jit_det(np.transpose(QSa, (2, 0, 1)))

    tmp2 = np.einsum("ia,ap->ip", A_mix, tempvec)
    tmp3 = np.transpose(jit_linsolve_parallelb(cov, np.transpose(tmp2)))
    weights = (1.0 / jit_det(np.transpose(Qab_pix, (2, 0, 1))))[:, None] * np.transpose(
        tmp3
    )
    return weights


# ---------------------------------------------------------------------------
# JAX GPU/CPU path
# ---------------------------------------------------------------------------

def _weights_jax(cov: np.ndarray, A_mix: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    # Prefer float64 for numerical parity with CPU path
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception:
        pass

    cov_j = jnp.asarray(np.asarray(cov, dtype=np.float64))
    A_j = jnp.asarray(np.asarray(A_mix, dtype=np.float64))
    n_pix = cov_j.shape[0]
    n_comp = int(A_j.shape[1])

    rhs = jnp.broadcast_to(A_j, (n_pix,) + A_j.shape)
    tmp1 = jnp.linalg.solve(cov_j, rhs)
    tmp1 = jnp.transpose(tmp1)  # (C, F, P)
    Qab_pix = jnp.einsum("ajp,bj->abp", tmp1, jnp.transpose(A_j))

    if n_comp == 1:
        tempvec = jnp.ones((1, n_pix), dtype=cov_j.dtype)
    else:
        rows = []
        for a in range(n_comp):
            QSa = jnp.delete(jnp.delete(Qab_pix, a, axis=0), 0, axis=1)
            det = jnp.linalg.det(jnp.transpose(QSa, (2, 0, 1)))
            rows.append(((-1.0) ** float(a)) * det)
        tempvec = jnp.stack(rows, axis=0)

    tmp2 = jnp.einsum("ia,ap->ip", A_j, tempvec)
    tmp3 = jnp.linalg.solve(cov_j, jnp.transpose(tmp2)[..., None])[..., 0]
    tmp3 = jnp.transpose(tmp3)
    detQ = jnp.linalg.det(jnp.transpose(Qab_pix, (2, 0, 1)))
    weights = (1.0 / detQ)[:, None] * jnp.transpose(tmp3)
    # Block until ready and return host numpy
    weights = np.asarray(weights.block_until_ready())
    return weights


# ---------------------------------------------------------------------------
# CuPy path
# ---------------------------------------------------------------------------

def _weights_cupy(cov: np.ndarray, A_mix: np.ndarray) -> np.ndarray:
    import cupy as cp

    cov_g = cp.asarray(np.asarray(cov, dtype=np.float64))
    A_g = cp.asarray(np.asarray(A_mix, dtype=np.float64))
    n_pix = cov_g.shape[0]
    n_comp = int(A_g.shape[1])

    rhs = cp.broadcast_to(A_g, (n_pix,) + A_g.shape).copy()
    tmp1 = cp.linalg.solve(cov_g, rhs)
    tmp1 = cp.transpose(tmp1)
    Qab_pix = cp.einsum("ajp,bj->abp", tmp1, cp.transpose(A_g))

    if n_comp == 1:
        tempvec = cp.ones((1, n_pix), dtype=cov_g.dtype)
    else:
        rows = []
        for a in range(n_comp):
            QSa = cp.delete(cp.delete(Qab_pix, a, axis=0), 0, axis=1)
            det = cp.linalg.det(cp.transpose(QSa, (2, 0, 1)))
            rows.append(((-1.0) ** float(a)) * det)
        tempvec = cp.stack(rows, axis=0)

    tmp2 = cp.einsum("ia,ap->ip", A_g, tempvec)
    tmp3 = cp.linalg.solve(cov_g, cp.transpose(tmp2)[..., None])[..., 0]
    tmp3 = cp.transpose(tmp3)
    detQ = cp.linalg.det(cp.transpose(Qab_pix, (2, 0, 1)))
    weights = (1.0 / detQ)[:, None] * cp.transpose(tmp3)
    return cp.asnumpy(weights)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ilc_weights_from_cov(
    cov: np.ndarray,
    A_mix: np.ndarray,
    backend: str = "auto",
) -> Tuple[np.ndarray, str]:
    """
    Compute constrained ILC weights from per-pixel covariances.

    Parameters
    ----------
    cov : array, shape (N_pix, N_freq, N_freq)
    A_mix : array, shape (N_freq, N_comp)
    backend : {'auto', 'numpy', 'numba', 'jax', 'cupy'}

    Returns
    -------
    weights : ndarray, shape (N_pix, N_freq)
    backend_used : str
    """
    chosen = resolve_backend(backend)
    if chosen == "numpy":
        w = _weights_numpy(cov, A_mix)
    elif chosen == "numba":
        w = _weights_numba(cov, A_mix)
    elif chosen == "jax":
        w = _weights_jax(cov, A_mix)
    elif chosen == "cupy":
        w = _weights_cupy(cov, A_mix)
    else:
        raise RuntimeError(f"unhandled backend {chosen}")
    return np.asarray(w, dtype=np.float64), chosen


def apply_weights_to_maps(
    weights: np.ndarray,
    maps: np.ndarray,
    backend: str = "auto",
) -> Tuple[np.ndarray, str]:
    """
    ILC map = sum_i w_i(p) * m_i(p).

    Parameters
    ----------
    weights : (N_pix, N_freq)
    maps : (N_freq, N_pix) or (N_pix, N_freq)
        If shape[0] == N_freq and shape[1] == N_pix, treated as (F, P).
        If shape matches weights, treated as (P, F).
    """
    weights = np.asarray(weights, dtype=np.float64)
    maps = np.asarray(maps, dtype=np.float64)
    n_pix, n_freq = weights.shape

    if maps.shape == (n_freq, n_pix):
        maps_pf = maps.T
        layout = "fp"
    elif maps.shape == (n_pix, n_freq):
        maps_pf = maps
        layout = "pf"
    else:
        raise ValueError(
            f"maps shape {maps.shape} incompatible with weights {weights.shape}"
        )

    chosen = resolve_backend(backend)
    if chosen == "jax":
        import jax.numpy as jnp
        try:
            import jax
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass
        out = jnp.sum(jnp.asarray(weights) * jnp.asarray(maps_pf), axis=1)
        return np.asarray(out.block_until_ready()), chosen
    if chosen == "cupy":
        import cupy as cp
        out = cp.sum(cp.asarray(weights) * cp.asarray(maps_pf), axis=1)
        return cp.asnumpy(out), chosen
    # numpy / numba: plain numpy is already optimal for this elementwise sum
    return np.sum(weights * maps_pf, axis=1), "numpy"


def env_backend_override() -> Optional[str]:
    """Optional override from PYILC_BACKEND environment variable."""
    v = os.environ.get("PYILC_BACKEND", None)
    if v is None or v.strip() == "":
        return None
    return v.strip().lower()
