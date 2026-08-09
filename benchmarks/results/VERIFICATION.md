# Verification summary (ILC weight GPU rewrite)

## Backends
{'numpy': True, 'numba': True, 'jax': True, 'jax_gpu': True, 'cupy': True, 'auto': 'jax'}

## Speedup vs numba (best of runs)

| config | numba_s | jax_s | cupy_s | jax_x | cupy_x | max|dw| jax |
|--------|--------:|------:|-------:|------:|-------:|------------:|
| nside64_F6_C1 | 0.0518 | 0.0034 | 0.0022 | 15.27 | 23.87 | 2.776e-16 |
| nside128_F6_C2 | 0.2666 | 0.0084 | 0.0095 | 31.61 | 28.19 | 6.106e-16 |
| nside256_F9_C1 | 0.9241 | 0.0633 | 0.0600 | 14.60 | 15.39 | 1.943e-16 |
| nside512_F9_C2 | 6.4247 | 1.1113 | 0.5906 | 5.78 | 10.88 | 3.053e-16 |

## pytest
See verification/pytest.txt — 12 passed.

## Shipped-path parity
See verification/shipped_path_parity.txt — historical (2,1,0) cov layout + auto GPU.
