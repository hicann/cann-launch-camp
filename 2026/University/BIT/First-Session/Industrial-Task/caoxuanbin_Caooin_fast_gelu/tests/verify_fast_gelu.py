"""Verification script for the FastGelu reference implementation."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fast_gelu import fast_gelu, scalar_fast_gelu


def reference_impl(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    flat_in = x.reshape(-1)
    flat_out = out.reshape(-1)
    for idx, value in enumerate(flat_in):
        flat_out[idx] = scalar_fast_gelu(float(value))
    return out


def check_case(name: str, x: np.ndarray, rtol: float, atol: float) -> None:
    y = fast_gelu(x)
    ref = reference_impl(x.astype(np.float32)).astype(x.dtype)

    if y.shape != x.shape:
        raise AssertionError(f"{name}: shape mismatch {y.shape} != {x.shape}")
    if y.dtype != x.dtype:
        raise AssertionError(f"{name}: dtype mismatch {y.dtype} != {x.dtype}")
    if not np.allclose(y, ref, rtol=rtol, atol=atol, equal_nan=True):
        diff = np.abs(y.astype(np.float32) - ref.astype(np.float32))
        raise AssertionError(
            f"{name}: allclose failed, max_abs_diff={diff.max()}, "
            f"rtol={rtol}, atol={atol}"
        )


def build_cases(dtype: np.dtype) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260708)
    return [
        ("vector_basic", np.array([0.0, 1.0, -1.0, 2.0], dtype=dtype)),
        ("matrix_basic", np.array([[0.5, -0.5], [1.5, -1.5]], dtype=dtype)),
        ("3d_tensor", rng.normal(0.0, 1.0, size=(3, 5, 7)).astype(dtype)),
        ("non_aligned_31", np.linspace(-4, 4, 31, dtype=dtype)),
        ("non_aligned_33", np.linspace(-4, 4, 33, dtype=dtype)),
        ("boundary_n_1", np.array([0.25], dtype=dtype)),
        ("boundary_n_10000", np.linspace(-6, 6, 10000, dtype=dtype)),
        ("4d_tensor", rng.normal(0.0, 1.0, size=(2, 3, 5, 7)).astype(dtype)),
        ("high_dim_batch", rng.normal(0.0, 1.0, size=(2, 2, 3, 4, 5, 7)).astype(dtype)),
        ("zeros", np.zeros((4, 4), dtype=dtype)),
        ("empty_tensor", np.array([], dtype=dtype)),
        (
            "boundary_values",
            np.array([-10.0, -6.0, -1e-3, 0.0, 1e-3, 6.0, 10.0], dtype=dtype),
        ),
        (
            "large_values",
            np.array([-50.0, -20.0, -12.0, 12.0, 20.0, 50.0], dtype=dtype),
        ),
    ]


def main() -> None:
    configs = {
        np.float32: {"rtol": 1e-4, "atol": 1e-4},
        np.float16: {"rtol": 1e-3, "atol": 1e-3},
    }

    total = 0
    for dtype, tol in configs.items():
        for name, x in build_cases(dtype):
            check_case(f"{dtype.__name__}:{name}", x, tol["rtol"], tol["atol"])
            total += 1

    print(f"FastGelu verification passed: {total} cases")


if __name__ == "__main__":
    main()
