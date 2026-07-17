#!/usr/bin/env python3
"""Verify generated LessEqual golden files against tf.math.less_equal."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("test/data/cases.json"))
    args = parser.parse_args()

    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for item in cases:
        dtype = np.dtype(item["dtype"])
        x1 = np.fromfile(args.manifest.parent / item["x1"], dtype=dtype).reshape(item["x1_shape"])
        x2 = np.fromfile(args.manifest.parent / item["x2"], dtype=dtype).reshape(item["x2_shape"])
        golden = np.fromfile(args.manifest.parent / item["golden"], dtype=bool)
        tensorflow_result = tf.math.less_equal(
            tf.convert_to_tensor(x1), tf.convert_to_tensor(x2)).numpy().reshape(-1)
        mismatch = np.flatnonzero(golden != tensorflow_result)
        if mismatch.size:
            failures.append(
                f"{item['name']}: {mismatch.size} mismatches, first index {mismatch[0]}")
        else:
            print(f"PASS {item['name']}")

    if failures:
        for message in failures:
            print(f"FAIL {message}")
        raise SystemExit(1)
    print(f"All {len(cases)} cases exactly match tf.math.less_equal")


if __name__ == "__main__":
    main()
