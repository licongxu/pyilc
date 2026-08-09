#!/usr/bin/env python
"""
Benchmark constrained-ILC weight computation: CPU (numba/numpy) vs GPU (jax/cupy).

Each backend is timed in a fresh subprocess so OpenBLAS/numba thread pools do not
corrupt process state across backends.

Usage (with project venv activated)::

    python benchmarks/bench_ilc_weights.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

CONFIGS = [
    {"label": "nside64_F6_C1", "n_pix": 12 * 64**2, "n_freq": 6, "n_comp": 1},
    {"label": "nside128_F6_C2", "n_pix": 12 * 128**2, "n_freq": 6, "n_comp": 2},
    {"label": "nside256_F9_C1", "n_pix": 12 * 256**2, "n_freq": 9, "n_comp": 1},
    {"label": "nside512_F9_C2", "n_pix": 12 * 512**2, "n_freq": 9, "n_comp": 2},
]


WORKER = r"""
import json, sys, time, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
import numpy as np
from pyilc.ilc_linalg import compute_ilc_weights_from_cov

payload = json.loads(sys.stdin.read())
backend = payload["backend"]
n_pix = payload["n_pix"]
n_freq = payload["n_freq"]
n_comp = payload["n_comp"]
seed = payload.get("seed", 0)
reps = payload.get("reps", 1)
save_weights = payload.get("save_weights", False)
out_path = payload.get("weights_path")

rng = np.random.default_rng(seed)
X = rng.standard_normal((n_pix, n_freq, n_freq))
cov = X @ np.transpose(X, (0, 2, 1)) + n_freq * np.eye(n_freq)
A = rng.standard_normal((n_freq, n_comp))
A[:, 0] = np.abs(A[:, 0]) + 0.5
cov = cov.astype(np.float64)
A = A.astype(np.float64)

# warmup / JIT
_w, used = compute_ilc_weights_from_cov(cov[: min(512, n_pix)], A, backend=backend)

times = []
w = None
for _ in range(reps):
    t0 = time.perf_counter()
    w, used = compute_ilc_weights_from_cov(cov, A, backend=backend)
    t1 = time.perf_counter()
    times.append(t1 - t0)

result = {
    "backend_used": used,
    "times_s": times,
    "best_s": min(times),
    "mean_s": sum(times) / len(times),
    "max_abs_w": float(np.max(np.abs(w))),
}
if save_weights and out_path:
    np.save(out_path, w)
    result["weights_path"] = out_path
print(json.dumps(result))
"""


def run_backend(backend: str, cfg: dict, save_weights: bool = False, weights_path: str | None = None):
    payload = {
        "backend": backend,
        "n_pix": cfg["n_pix"],
        "n_freq": cfg["n_freq"],
        "n_comp": cfg["n_comp"],
        "seed": 0,
        "reps": 1 if cfg["n_pix"] > 1_000_000 else 2,
        "save_weights": save_weights,
        "weights_path": weights_path,
    }
    env = os.environ.copy()
    env.setdefault("OPENBLAS_NUM_THREADS", "8")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    # Help CuPy find pip-installed NVIDIA CUDA libs if present
    try:
        import pathlib
        base = pathlib.Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia"
        if base.is_dir():
            libs = [str(p / "lib") for p in base.iterdir() if (p / "lib").is_dir()]
            if libs:
                env["LD_LIBRARY_PATH"] = ":".join(libs) + (
                    (":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else ""
                )
    except Exception:
        pass

    proc = subprocess.run(
        [sys.executable, "-c", WORKER],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        return {
            "error": proc.stderr[-2000:] if proc.stderr else f"exit {proc.returncode}",
            "stdout": proc.stdout[-1000:] if proc.stdout else "",
        }
    # last line is JSON
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def main():
    out_dir = Path(os.environ.get("PYILC_BENCH_OUT", "benchmarks/results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover backends in a lightweight parent process
    from pyilc.ilc_linalg import available_backends

    backends_info = available_backends()
    print("Available backends:", backends_info)

    run_backends = []
    if backends_info["numba"]:
        run_backends.append("numba")
    run_backends.append("numpy")
    if backends_info["jax"]:
        run_backends.append("jax")
    if backends_info["cupy"]:
        run_backends.append("cupy")

    results = {"backends": backends_info, "configs": []}

    header = f"{'config':<22} {'backend':<8} {'best_s':>10} {'speedup_vs_numba':>16} {'max|dw|':>12}"
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for cfg in CONFIGS:
            row = dict(cfg)
            row["runs"] = {}
            ref_t = None
            ref_path = td / f"{cfg['label']}_ref.npy"
            for b in run_backends:
                save = b in ("numba", "numpy") and ref_t is None
                wpath = str(ref_path) if save else (
                    str(td / f"{cfg['label']}_{b}.npy") if b in ("jax", "cupy") else None
                )
                save_w = save or b in ("jax", "cupy")
                stats = run_backend(b, cfg, save_weights=save_w, weights_path=wpath)
                if "error" in stats:
                    print(f"{cfg['label']:<22} {b:<8} FAILED")
                    print(stats["error"][:400])
                    row["runs"][b] = stats
                    continue
                if b == "numba" or (b == "numpy" and ref_t is None):
                    ref_t = stats["best_s"]
                maxdiff = None
                if ref_path.exists() and wpath and Path(wpath).exists() and b not in ("numba",):
                    if b == "numpy" and run_backends[0] == "numpy":
                        maxdiff = 0.0
                    else:
                        w = np.load(wpath)
                        wr = np.load(ref_path)
                        maxdiff = float(np.max(np.abs(w - wr)))
                elif b in ("numba", "numpy"):
                    maxdiff = 0.0
                speedup = (ref_t / stats["best_s"]) if ref_t and stats["best_s"] > 0 else None
                print(
                    f"{cfg['label']:<22} {stats['backend_used']:<8} "
                    f"{stats['best_s']:10.4f} "
                    f"{(speedup if speedup is not None else float('nan')):16.2f} "
                    f"{(maxdiff if maxdiff is not None else float('nan')):12.3e}"
                )
                row["runs"][b] = {
                    "backend_used": stats["backend_used"],
                    "times_s": stats["times_s"],
                    "best_s": stats["best_s"],
                    "mean_s": stats["mean_s"],
                    "speedup_vs_ref": speedup,
                    "max_abs_diff_vs_ref": maxdiff,
                }
            results["configs"].append(row)

    json_path = out_dir / "ilc_weights_bench.json"
    txt_path = out_dir / "ilc_weights_bench.txt"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    lines = ["pyilc ILC weight backend benchmark", ""]
    for cfg in results["configs"]:
        lines.append(
            f"== {cfg['label']} (npix={cfg['n_pix']}, F={cfg['n_freq']}, C={cfg['n_comp']}) =="
        )
        for b, run in cfg["runs"].items():
            if "error" in run:
                lines.append(f"  {b}: ERROR {run['error'][:200]}")
            else:
                lines.append(
                    f"  {b}: best={run['best_s']:.4f}s  "
                    f"speedup_vs_ref={run['speedup_vs_ref']}  "
                    f"max|dw|={run['max_abs_diff_vs_ref']}"
                )
        lines.append("")
    txt_path.write_text("\n".join(lines))
    print(f"\nWrote {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
