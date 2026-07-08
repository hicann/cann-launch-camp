"""Reference FastGelu implementation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


ATTR = 1.702
ATTR_HALF = ATTR / 2.0
SUPPORTED_DTYPES = (np.float16, np.float32)


def _normalize_input(x: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(x)
    if array.dtype not in SUPPORTED_DTYPES:
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        else:
            raise TypeError(
                f"FastGelu only supports float16/float32, got {array.dtype!s}"
            )
    return array


def fast_gelu(x: Iterable[float] | np.ndarray) -> np.ndarray:
    """Compute FastGelu while preserving the input floating dtype."""
    array = _normalize_input(x)
    work_dtype = np.float32 if array.dtype == np.float16 else array.dtype
    work = array.astype(work_dtype, copy=False)

    abs_x = np.abs(work)
    # Both exponent arguments are mathematically non-positive:
    #   -ATTR * abs(x) <= 0
    #   ATTR_HALF * (x - abs(x)) <= 0
    # This keeps the implementation numerically stable and avoids overflow.
    exp_down = np.minimum(-ATTR * abs_x, 0.0)
    exp_up = np.minimum(ATTR_HALF * (work - abs_x), 0.0)
    div_down = 1.0 + np.exp(exp_down)
    div_up = work * np.exp(exp_up)
    result = div_up / div_down

    return result.astype(array.dtype, copy=False)


def scalar_fast_gelu(x: float) -> float:
    abs_x = abs(x)
    div_down = 1.0 + math.exp(-ATTR * abs_x)
    div_up = x * math.exp(ATTR_HALF * (x - abs_x))
    return div_up / div_down


def main() -> None:
    sample = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    output = fast_gelu(sample)
    print("input :", sample)
    print("output:", output)


if __name__ == "__main__":
    main()
