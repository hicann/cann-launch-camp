#!/usr/bin/env python3
"""
gen_data.py - 为 LessEqual 算子生成测试数据

生成随机输入张量和预期输出 (golden)。
支持 float16, float32, int32, int8 四种数据类型。
"""

import numpy as np
import argparse
import os
import sys


def gen_test_data(dtype_str, shape, broadcast=False):
    """
    生成测试数据

    Args:
        dtype_str: 'float16', 'float32', 'int32', 'int8'
        shape: 输出 shape (广播后)
        broadcast: 是否使用不同 shape (广播场景)

    Returns:
        (x1, x2, golden) 三组 numpy 数组
    """
    np_dtype = {
        'float16': np.float16,
        'float32': np.float32,
        'int32': np.int32,
        'int8': np.int8,
    }[dtype_str]

    # 生成输入数据
    shape1 = list(shape)
    shape2 = list(shape)

    if broadcast:
        # 对最后一个维度进行广播测试
        if len(shape2) >= 2:
            shape2[-2] = 1
        elif len(shape2) == 1:
            shape2[0] = 1

    if dtype_str in ('float16', 'float32'):
        x1 = np.random.uniform(-100, 100, shape1).astype(np_dtype)
        x2 = np.random.uniform(-100, 100, shape2).astype(np_dtype)
    elif dtype_str == 'int32':
        x1 = np.random.randint(-1000, 1000, shape1, dtype=np_dtype)
        x2 = np.random.randint(-1000, 1000, shape2, dtype=np_dtype)
    elif dtype_str == 'int8':
        x1 = np.random.randint(-100, 100, shape1, dtype=np_dtype)
        x2 = np.random.randint(-100, 100, shape2, dtype=np_dtype)

    # 计算 golden
    golden = (x1.astype(np.float64) <= x2.astype(np.float64)).astype(np.uint8)

    return x1, x2, golden


def save_binary(data, filename):
    """保存 numpy 数组为二进制文件"""
    data.tofile(filename)
    print(f"  Saved {filename}: shape={data.shape}, dtype={data.dtype}, "
          f"size={data.nbytes} bytes")


def main():
    parser = argparse.ArgumentParser(
        description='LessEqual算子测试数据生成')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float16', 'float32', 'int32', 'int8'],
                        help='数据类型 (default: float32)')
    parser.add_argument('--shape', type=str, default='8,2048',
                        help='输出 shape, 逗号分隔 (default: 8,2048)')
    parser.add_argument('--broadcast', action='store_true',
                        help='启用广播测试')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录 (default: 当前目录)')
    args = parser.parse_args()

    shape = [int(s) for s in args.shape.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating LessEqual test data:")
    print(f"  dtype={args.dtype}, shape={shape}, broadcast={args.broadcast}")

    x1, x2, golden = gen_test_data(args.dtype, shape, args.broadcast)

    save_binary(x1, os.path.join(args.output_dir, 'input_x1.bin'))
    save_binary(x2, os.path.join(args.output_dir, 'input_x2.bin'))
    save_binary(golden, os.path.join(args.output_dir, 'golden.bin'))

    # 保存元信息
    meta_path = os.path.join(args.output_dir, 'meta.txt')
    with open(meta_path, 'w') as f:
        f.write(f"dtype={args.dtype}\n")
        f.write(f"x1_shape={list(x1.shape)}\n")
        f.write(f"x2_shape={list(x2.shape)}\n")
        f.write(f"output_shape={list(golden.shape)}\n")
        f.write(f"total_elements={golden.size}\n")
        f.write(f"x1_dtype={x1.dtype}\n")
        f.write(f"x2_dtype={x2.dtype}\n")
        f.write(f"golden_dtype={golden.dtype}\n")
    print(f"  Saved {meta_path}")

    print("=" * 50)
    print("Data generation completed!")
    print(f"Files in {args.output_dir}:")
    for f in os.listdir(args.output_dir):
        print(f"  {f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
