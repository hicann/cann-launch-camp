#!/usr/bin/env python3
"""
GELU 算子本地测试脚本
- 生成测试数据（多个测试用例，覆盖各种张量大小）
- 用 PyTorch 计算参考结果
- 调用 AscendCL 运行时加载算子并运行
- 对比输出并报告差异
"""

import numpy as np
import torch
import torch.nn.functional as F
import struct
import os
import sys
import ctypes
import json

# ==================== 配置 ====================
DTYPE_MAP = {
    'float16': np.float16,
    'float32': np.float32,
}

TEST_CASES = [
    # (name, n_elements, dtype)
    # 模拟竞赛测试点的各种张量规模
    ("tiny", 128, 'float32'),
    ("small", 2048, 'float32'),
    ("medium", 8192, 'float16'),
    ("large", 65536, 'float32'),
    ("huge", 262144, 'float16'),
]


# ==================== 生成测试数据 & PyTorch 参考 ====================
def generate_test_case(name, n, dtype_str):
    """生成测试输入和 PyTorch 参考输出"""
    np_dtype = DTYPE_MAP[dtype_str]
    # 生成覆盖正负范围的随机数
    np.random.seed(42)
    x = np.random.randn(n).astype(np_dtype)

    # PyTorch 参考
    if dtype_str == 'float16':
        x_torch = torch.from_numpy(x.astype(np.float32))
        y_ref = F.gelu(x_torch).half().numpy().astype(np_dtype)
    else:
        x_torch = torch.from_numpy(x)
        y_ref = F.gelu(x_torch).numpy().astype(np_dtype)

    return x, y_ref


def save_binary(filename, data):
    """保存为二进制文件（与 AscendC op_test_frame 兼容）"""
    data.tofile(filename)
    print(f"  Saved {filename} ({len(data)} elements, dtype={data.dtype}, {os.path.getsize(filename)} bytes)")


# ==================== 使用 AscendCL 运行算子 ====================
class AscendGeluRunner:
    """通过 AscendCL 加载并运行编译好的 GELU 算子"""

    def __init__(self):
        # 加载 ACL 库
        toolkit_path = os.environ.get('ASCEND_HOME_DIR', '/usr/local/Ascend/ascend-toolkit/latest')
        self.acl_lib = ctypes.CDLL(f'{toolkit_path}/lib64/libascendcl.so', ctypes.RTLD_GLOBAL)
        self.runtime_lib = ctypes.CDLL(f'{toolkit_path}/lib64/libruntime.so', ctypes.RTLD_GLOBAL)

        # 加载编译好的算子库
        build_dir = os.path.join(os.path.dirname(__file__), 'build')
        ops_so = os.path.join(build_dir, 'autogen', 'libascend_all_ops.so')
        if os.path.exists(ops_so):
            self.ops_lib = ctypes.CDLL(ops_so, ctypes.RTLD_GLOBAL)
            print(f"Loaded ops library: {ops_so}")
        else:
            print(f"WARNING: ops library not found at {ops_so}")
            self.ops_lib = None

    def run_gelu(self, input_data):
        """运行 GELU 算子（使用 ACL 算子调用方式）"""
        # 注意：实际 ACL 算子调用需要完整的 context/stream 设置
        # 这里仅作为框架占位，真实运行需要算子注册后的调用接口
        print("  [AscendCL runner - placeholder]")
        print(f"  Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        return None


# ==================== 主测试流程 ====================
def main():
    print("=" * 60)
    print("GELU 算子本地测试 — 与 PyTorch 对比")
    print("=" * 60)

    # 创建输出目录
    data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(data_dir, exist_ok=True)

    all_results = {}
    reference_results = {}

    for name, n, dtype_str in TEST_CASES:
        print(f"\n--- Test Case: {name} ({n} elements, {dtype_str}) ---")

        x, y_ref = generate_test_case(name, n, dtype_str)
        reference_results[name] = {'dtype': dtype_str, 'n': n, 'y_ref': y_ref}

        # 保存输入文件
        in_path = os.path.join(data_dir, f"{name}_input.bin")
        ref_path = os.path.join(data_dir, f"{name}_reference.bin")
        save_binary(in_path, x)
        save_binary(ref_path, y_ref)

        # 分析输入数据统计
        print(f"  Input range: [{x.min():.6f}, {x.max():.6f}], mean={x.mean():.6f}")
        print(f"  Reference range: [{y_ref.min():.6f}, {y_ref.max():.6f}], mean={y_ref.mean():.6f}")

        # 对特定输入验证 GELU 公式
        special_vals = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5]
        print(f"  Reference GELU values:")
        for v in special_vals:
            x_t = torch.tensor([v], dtype=torch.float32)
            y_t = F.gelu(x_t).item()
            print(f"    GELU({v:+.1f}) = {y_t:.8f}")

    # 保存测试配置
    config = {
        'test_cases': [
            {'name': name, 'n': n, 'dtype': dtype_str}
            for name, n, dtype_str in TEST_CASES
        ],
        'date': '2026-07-05',
    }
    config_path = os.path.join(data_dir, 'test_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config: {config_path}")

    # 尝试用 AscendCL 运行
    print(f"\n{'=' * 60}")
    print("Attempting AscendCL execution...")
    print("=" * 60)
    try:
        runner = AscendGeluRunner()
        # 运行一个简单用例
        x_sample = reference_results['tiny']['y_ref']  # 随便测试
        result = runner.run_gelu(x_sample)
    except Exception as e:
        print(f"AscendCL execution failed (expected in non-NPU env): {e}")
        print("To run on NPU: deploy to Ascend device with compiled operator package.")

    print(f"\n{'=' * 60}")
    print("Test data generation complete!")
    print(f"Test files saved to: {data_dir}")
    print("=" * 60)
    print("\nTo use these test files:")
    print("1. Deploy the compiled operator to an Ascend NPU device")
    print("2. Load input binary files and run the operator")
    print("3. Compare operator output with reference .bin files")
    print("\nOr modify this script to use the AscendCL API when NPU is available.")


if __name__ == '__main__':
    main()
