#!/usr/bin/env python3
"""
verify_result.py - 验证 LessEqual 算子的输出结果

将算子输出与 golden (预期值) 进行逐元素比对。
"""

import sys
import numpy as np


def verify_result(output_path, golden_path, dtype_str='float32',
                  tolerance=0):
    """
    验证算子输出

    Args:
        output_path: 算子输出二进制文件路径
        golden_path: 预期值二进制文件路径
        dtype_str: 输入数据类型 (用于打印上下文)
        tolerance: 允许的不一致元素数量 (0 表示完全匹配)

    Returns:
        bool: 验证是否通过
    """
    output = np.fromfile(output_path, dtype=np.uint8).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.uint8).reshape(-1)

    if output.shape != golden.shape:
        print(f"[ERROR] Shape mismatch: output={output.shape}, "
              f"golden={golden.shape}")
        return False

    n = output.size
    mismatches = np.where(output != golden)[0]
    num_errors = len(mismatches)

    if num_errors == 0:
        print(f"[PASS] All {n} elements match! (tolerance={tolerance})")
        return True

    print(f"[FAIL] {num_errors} / {n} elements mismatch "
          f"(tolerance={tolerance})")
    print(f"  First {min(20, num_errors)} mismatches:")
    for idx in mismatches[:20]:
        print(f"    [{idx}] output={output[idx]}, "
              f"golden={golden[idx]}")

    return num_errors <= tolerance


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 verify_result.py <output.bin> <golden.bin> "
              "[tolerance]")
        sys.exit(1)

    output_file = sys.argv[1]
    golden_file = sys.argv[2]
    tol = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    success = verify_result(output_file, golden_file, tolerance=tol)
    sys.exit(0 if success else 1)
