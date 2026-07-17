#!/usr/bin/env python3
"""
在 NPU 上真正运行 GELU 自定义算子并与 PyTorch 参考值对比。
"""
import ctypes
import numpy as np
import torch
import torch.nn.functional as F
import os, sys

# ==================== 常量 ====================
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_ND = 2
ACL_MEMCPY_H2D = 1
ACL_MEMCPY_D2H = 2
ACL_MEM_HUGE_FIRST = 0

ASCEND_HOME = "/usr/local/Ascend/cann-8.5.0"
OPAPI_SO = os.path.abspath("build/packages/vendors/custom/op_api/lib/libcust_opapi.so")

TEST_CASES = [
    ("tiny",   128,     'float32'),
    ("small",  2048,    'float32'),
    ("medium", 8192,    'float16'),
    ("large",  65536,   'float32'),
    ("huge",   262144,  'float16'),
]


# ==================== 获取函数指针 ====================
class LibFuncs:
    """从各个 so 中获取所需的 C 函数"""
    def __init__(self):
        # 加载依赖库（RTLD_GLOBAL 使符号全局可见）
        ctypes.CDLL(f"{ASCEND_HOME}/lib64/libc_sec.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(f"{ASCEND_HOME}/lib64/libruntime.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(f"{ASCEND_HOME}/lib64/libnnopbase.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(f"{ASCEND_HOME}/lib64/libacl_rt.so", ctypes.RTLD_GLOBAL)
        ctypes.CDLL(f"{ASCEND_HOME}/lib64/libascendcl.so", ctypes.RTLD_GLOBAL)

        # 加载自定义算子库（触发 aclnnGelu 注册）
        self.opapi = ctypes.CDLL(OPAPI_SO, ctypes.RTLD_GLOBAL)

        # 从各库获取函数指针
        self._get("aclInit",                     "libascendcl.so",    [ctypes.c_char_p], ctypes.c_int32)
        self._get("aclrtSetDevice",              "libascendcl.so",    [ctypes.c_int32], ctypes.c_int32)
        self._get("aclrtCreateContext",           "libascendcl.so",    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32], ctypes.c_int32)
        self._get("aclrtCreateStream",            "libascendcl.so",    [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int32)
        self._get("aclrtMalloc",                 "libascendcl.so",    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_int32], ctypes.c_int32)
        self._get("aclrtMemcpy",                 "libascendcl.so",    [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32], ctypes.c_int32)
        self._get("aclrtFree",                   "libascendcl.so",    [ctypes.c_void_p], ctypes.c_int32)
        self._get("aclrtSynchronizeStream",       "libascendcl.so",    [ctypes.c_void_p], ctypes.c_int32)
        self._get("aclrtDestroyStream",           "libascendcl.so",    [ctypes.c_void_p], ctypes.c_int32)
        self._get("aclrtDestroyContext",          "libascendcl.so",    [ctypes.c_void_p], ctypes.c_int32)
        self._get("aclrtResetDevice",            "libascendcl.so",    [ctypes.c_int32], ctypes.c_int32)
        self._get("aclFinalize",                 "libascendcl.so",    [], ctypes.c_int32)

        # aclCreateTensor 来自 libnnopbase.so
        self._get("aclCreateTensor",             "libnnopbase.so",    [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64,
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p
        ], ctypes.c_void_p)
        self._get("aclDestroyTensor",            "libnnopbase.so",    [ctypes.c_void_p], ctypes.c_int32)

        # aclnnGelu 来自 libcust_opapi.so
        self.opapi.aclnnGeluGetWorkspaceSize.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)
        ]
        self.opapi.aclnnGeluGetWorkspaceSize.restype = ctypes.c_int32
        self.opapi.aclnnGelu.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p
        ]
        self.opapi.aclnnGelu.restype = ctypes.c_int32

        print("  [OK] 加载所有库")

    def _get(self, name, lib_name, argtypes, restype):
        """从指定 so 获取函数并设置签名"""
        lib_path = f"{ASCEND_HOME}/lib64/{lib_name}"
        try:
            lib = ctypes.CDLL(lib_path, ctypes.RTLD_GLOBAL)
        except:
            lib = ctypes.CDLL(lib_path)
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        setattr(self, name, fn)


# ==================== 在 NPU 上运行 GELU ====================
def run_gelu(x_np, F, stream):
    acl_dtype = ACL_FLOAT16 if x_np.dtype == np.float16 else ACL_FLOAT
    n = x_np.size
    nbytes = x_np.nbytes

    def chk(ret, msg):
        if ret != 0:
            raise RuntimeError(f"{msg} 失败, ret={ret}")

    # 分配 device 内存
    dev_input = ctypes.c_void_p()
    dev_output = ctypes.c_void_p()
    chk(F.aclrtMalloc(ctypes.byref(dev_input), nbytes, ACL_MEM_HUGE_FIRST), "malloc input")
    chk(F.aclrtMalloc(ctypes.byref(dev_output), nbytes, ACL_MEM_HUGE_FIRST), "malloc output")

    # host -> device
    chk(F.aclrtMemcpy(dev_input, nbytes,
                       ctypes.c_char_p(x_np.tobytes()), nbytes,
                       ACL_MEMCPY_H2D), "memcpy H2D")

    # 创建 ACL Tensor
    shape = (ctypes.c_int64 * 1)(n)
    tensor_in = F.aclCreateTensor(shape, 1, acl_dtype, shape, 1, ACL_ND, None, 0, dev_input)
    tensor_out = F.aclCreateTensor(shape, 1, acl_dtype, shape, 1, ACL_ND, None, 0, dev_output)
    assert tensor_in and tensor_out, "aclCreateTensor failed"

    # GetWorkspaceSize
    ws_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p()
    chk(F.opapi.aclnnGeluGetWorkspaceSize(tensor_in, tensor_out,
                                           ctypes.byref(ws_size),
                                           ctypes.byref(executor)), "GetWorkspaceSize")

    # workspace
    workspace = ctypes.c_void_p()
    if ws_size.value > 0:
        chk(F.aclrtMalloc(ctypes.byref(workspace), ws_size.value, ACL_MEM_HUGE_FIRST), "malloc ws")

    # 执行 + 同步
    chk(F.opapi.aclnnGelu(workspace, ws_size.value, executor, stream), "aclnnGelu")
    chk(F.aclrtSynchronizeStream(stream), "sync")

    # device -> host
    out_buf = (ctypes.c_byte * nbytes)()
    chk(F.aclrtMemcpy(out_buf, nbytes, dev_output, nbytes, ACL_MEMCPY_D2H), "memcpy D2H")
    y_np = np.frombuffer(out_buf, dtype=x_np.dtype).copy()

    # 清理
    F.aclDestroyTensor(tensor_in)
    F.aclDestroyTensor(tensor_out)
    F.aclrtFree(dev_input)
    F.aclrtFree(dev_output)
    if ws_size.value > 0:
        F.aclrtFree(workspace)

    return y_np


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print("GELU 自定义算子 — NPU 实测")
    print("=" * 60)
    os.environ['ASCEND_CUSTOM_OPP_PATH'] = os.path.abspath("build/packages/vendors")
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ['ASCEND_CUSTOM_OPP_PATH']}\n")

    print("[1/3] 初始化 AscendCL...")
    F = LibFuncs()
    def check(r, m):
        if r != 0:
            raise RuntimeError(f"{m}, ret={r}")

    check(F.aclInit(None), "aclInit")
    check(F.aclrtSetDevice(0), "setDevice")
    ctx = ctypes.c_void_p()
    stream = ctypes.c_void_p()
    check(F.aclrtCreateContext(ctypes.byref(ctx), 0), "createContext")
    check(F.aclrtCreateStream(ctypes.byref(stream)), "createStream")
    print("  [OK]\n")

    print("[2/3] 在 NPU 上运行 GELU 算子...\n")
    all_pass = True
    for name, n, dtype_str in TEST_CASES:
        print(f"  --- {name} ({n}, {dtype_str}) ---")
        np.random.seed(42)
        if dtype_str == 'float16':
            x_np = np.random.randn(n).astype(np.float32).astype(np.float16)
        else:
            x_np = np.random.randn(n).astype(np.float32)

        with torch.no_grad():
            x_t = torch.from_numpy(x_np.astype(np.float32) if dtype_str == 'float16' else x_np)
            y_ref = F_torch(x_t).numpy()
            if dtype_str == 'float16':
                y_ref = y_ref.astype(np.float16)

        try:
            y_npu = run_gelu(x_np, F, stream)
        except Exception as e:
            print(f"  [FAIL] 异常: {e}")
            all_pass = False
            continue

        yd = float(np.max(np.abs(y_ref.astype(np.float32) - y_npu.astype(np.float32))))
        rtol = 1e-2 if dtype_str == 'float16' else 1e-5
        passed = yd < rtol
        if not passed:
            all_pass = False
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} | max_diff={yd:.2e} | rtol={rtol:.0e}")
        for i in [n//2, n//2+1, n//2+2]:
            rf = float(y_ref.flat[i]) if dtype_str=='float32' else float(np.float32(y_ref.flat[i]))
            nu = float(y_npu.flat[i]) if dtype_str=='float32' else float(np.float32(y_npu.flat[i]))
            print(f"    [{i}] ref={rf:.6f}  npu={nu:.6f}  diff={abs(rf-nu):.2e}")

    print("\n[3/3] 清理...")
    F.aclrtDestroyStream(stream)
    F.aclrtDestroyContext(ctx)
    F.aclrtResetDevice(0)
    F.aclFinalize()

    print("\n" + "=" * 60)
    if all_pass:
        print("✅ 全部通过！GELU 自定义算子可在 NPU 上正常运行。")
    else:
        print("❌ 部分失败。")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    # 让 torch 不报 core 相关警告
    import warnings
    warnings.filterwarnings("ignore")
    # 引用时赋值
    F_torch = torch.nn.functional.gelu
    sys.exit(main())
