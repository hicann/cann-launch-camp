#!/usr/bin/env python3
"""Generate LessEqual binary inputs and exact boolean golden outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def case(name, dtype, x1, x2):
    return name, np.asarray(x1, dtype=dtype), np.asarray(x2, dtype=dtype)


def build_cases():
    rng = np.random.default_rng(20260711)
    return [
        case("float16_basic", np.float16, [1, 2, 3, 4], [2, 2, 2, 2]),
        case("float32_non_aligned", np.float32,
             rng.normal(size=37), rng.normal(size=37)),
        case("int32_basic", np.int32, [1, 5, 3, 7], [2, 4, 6, 8]),
        case("int8_boundaries", np.int8,
             [-128, -1, 0, 1, 127], [-128, 0, 0, 0, 127]),
        case("scalar_broadcast", np.float32, 2.0,
             np.arange(35, dtype=np.float32).reshape(5, 7)),
        case("vector_matrix_broadcast", np.float32,
             [[1, 2], [3, 4]], [2, 3]),
        case("high_rank_broadcast", np.int32,
             rng.integers(-100, 100, size=(2, 1, 3, 1), dtype=np.int32),
             rng.integers(-100, 100, size=(1, 4, 1, 5), dtype=np.int32)),
        case("float_special", np.float32,
             [-np.inf, -0.0, 0.0, np.inf, np.nan],
             [-np.inf, 0.0, -0.0, np.inf, np.nan]),
        case("empty", np.float16,
             np.empty((0, 3), dtype=np.float16),
             np.empty((1, 3), dtype=np.float16)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("test/data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, x1, x2 in build_cases():
        golden = np.less_equal(x1, x2).astype(bool)
        x1_path = args.output_dir / f"{name}_x1.bin"
        x2_path = args.output_dir / f"{name}_x2.bin"
        golden_path = args.output_dir / f"{name}_golden.bin"
        x1.tofile(x1_path)
        x2.tofile(x2_path)
        golden.tofile(golden_path)
        manifest.append({
            "name": name,
            "dtype": x1.dtype.name,
            "x1_shape": list(x1.shape),
            "x2_shape": list(x2.shape),
            "output_shape": list(golden.shape),
            "x1": x1_path.name,
            "x2": x2_path.name,
            "golden": golden_path.name,
        })

    manifest_path = args.output_dir / "cases.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} cases in {args.output_dir}")


if __name__ == "__main__":
    main()
