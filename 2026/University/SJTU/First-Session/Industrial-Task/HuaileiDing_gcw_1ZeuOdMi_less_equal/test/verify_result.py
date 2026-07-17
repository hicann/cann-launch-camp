#!/usr/bin/env python3
"""Compare NPU LessEqual outputs with generated boolean golden files."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("test/data/cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("test/output"))
    args = parser.parse_args()

    cases = json.loads(args.manifest.read_text(encoding="utf-8"))
    failed = []
    for item in cases:
        golden_path = args.manifest.parent / item["golden"]
        actual_path = args.output_dir / f"{item['name']}_y.bin"
        if not actual_path.is_file():
            failed.append(f"{item['name']}: missing {actual_path}")
            continue

        golden = np.fromfile(golden_path, dtype=bool)
        actual = np.fromfile(actual_path, dtype=bool)
        if actual.shape != golden.shape:
            failed.append(
                f"{item['name']}: element count {actual.size}, expected {golden.size}")
            continue
        mismatch = np.flatnonzero(actual != golden)
        if mismatch.size:
            failed.append(
                f"{item['name']}: {mismatch.size} mismatches, first index {mismatch[0]}")
        else:
            print(f"PASS {item['name']}")

    if failed:
        for message in failed:
            print(f"FAIL {message}")
        raise SystemExit(1)
    print(f"All {len(cases)} cases passed")


if __name__ == "__main__":
    main()
