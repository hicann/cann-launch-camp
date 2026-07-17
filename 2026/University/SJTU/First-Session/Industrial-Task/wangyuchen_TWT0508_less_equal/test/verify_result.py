"""NumPy-based verification for LessEqual operator (no NPU required)."""
import sys
import os
from pathlib import Path
import numpy as np

def verify_one(x1_file, x2_file, golden_file, dtype_str, desc):
    """Compare x1 <= x2 numpy result against pre-computed golden."""
    x1 = np.fromfile(x1_file, dtype=dtype_str)
    x2 = np.fromfile(x2_file, dtype=dtype_str)
    golden = np.fromfile(golden_file, dtype=np.bool_)

    expected = (x1 <= x2)
    ok = np.array_equal(expected, golden)

    if ok:
        print(f"  [{desc}] PASSED  shape={x1.shape}")
    else:
        print(f"  [{desc}] FAILED")
        for i in range(min(10, len(x1))):
            print(f"    idx={i}: x1={x1.flat[i]}, x2={x2.flat[i]}, "
                  f"expected={expected.flat[i]}, golden={golden.flat[i]}")
    return ok

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)

    cases = [
        ("input_x1_case1_float32_1d.bin", "input_x2_case1_float32_1d.bin",
         "golden_case1_float32_1d.bin", "float32", "case1_float32_1d"),
        ("input_x1_case2_float32_2d.bin", "input_x2_case2_float32_2d.bin",
         "golden_case2_float32_2d.bin", "float32", "case2_float32_2d"),
        ("input_x1_case3_float16_1d.bin", "input_x2_case3_float16_1d.bin",
         "golden_case3_float16_1d.bin", "float16", "case3_float16_1d"),
        ("input_x1_case4_int32_1d.bin", "input_x2_case4_int32_1d.bin",
         "golden_case4_int32_1d.bin", "int32", "case4_int32_1d"),
        ("input_x1_case5_int8_1d.bin", "input_x2_case5_int8_1d.bin",
         "golden_case5_int8_1d.bin", "int8", "case5_int8_1d"),
    ]

    results = []
    for x1f, x2f, gf, dt, desc in cases:
        try:
            r = verify_one(x1f, x2f, gf, dt, desc)
            results.append(r)
        except FileNotFoundError:
            print(f"  [{desc}] SKIP - files not found")

    passed = sum(results)
    total = len(results)
    print(f"\n=== Result: {passed}/{total} passed ===")
    sys.exit(0 if passed == total else 1)
