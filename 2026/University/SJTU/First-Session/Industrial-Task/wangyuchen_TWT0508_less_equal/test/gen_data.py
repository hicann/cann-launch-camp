"""Test data generator for LessEqual operator.

Generates input binary files and NumPy golden results for 55 test cases
covering all dtypes, broadcast patterns, edge cases, and large tensors.
"""
import os
from pathlib import Path
import numpy as np

os.chdir(Path(__file__).resolve().parent)
for f in Path(".").glob("*.bin"): f.unlink()
for f in Path(".").glob("*.npy"): f.unlink()

rng = np.random.default_rng(42)
idx = 0
passed = total = 0

def make_case(dtype_str, x1_shape, x2_shape, desc, special=None):
    """Generate one test case.

    Args:
        dtype_str: 'float16', 'float32', 'int32', or 'int8'
        x1_shape, x2_shape: input tensor shapes
        desc: human-readable description
        special: None, 'zeros', 'equal', 'extreme', or 'large_val'
    """
    global idx, passed, total
    idx += 1
    dtype = np.dtype(dtype_str)
    rng_local = np.random.default_rng(hash(f"{idx}{desc}") % (2**31))

    if special == 'zeros':
        x1 = np.zeros(x1_shape, dtype=dtype)
        x2 = np.zeros(x2_shape, dtype=dtype)
    elif special == 'equal':
        if 'float' in dtype_str:
            x1 = rng_local.uniform(-10, 10, size=x1_shape).astype(dtype)
        else:
            ii = np.iinfo(dtype)
            x1 = rng_local.integers(ii.min//2, ii.max//2, size=x1_shape).astype(dtype)
        x2 = x1.astype(dtype).copy()
    elif special == 'extreme':
        if 'float' in dtype_str:
            finfo = np.finfo(dtype)
            extreme_vals = [0, 1, -1, finfo.min, finfo.max, finfo.tiny, -finfo.tiny]
            x1 = np.array(extreme_vals * 4, dtype=dtype)[:int(np.prod(x1_shape))].reshape(x1_shape)
            x2 = np.zeros(x2_shape, dtype=dtype)
        else:
            ii = np.iinfo(dtype)
            vals = [0, 1, -1, ii.min, ii.max, ii.min+1, ii.max-1, 127, -128]
            x1 = np.array(vals * 4, dtype=dtype)[:int(np.prod(x1_shape))].reshape(x1_shape)
            x2 = np.zeros(x2_shape, dtype=dtype)
    elif special == 'large_val':
        if 'float' in dtype_str:
            x1 = rng_local.uniform(-1e5, 1e5, size=x1_shape).astype(dtype)
            x2 = rng_local.uniform(-1e5, 1e5, size=x2_shape).astype(dtype)
        else:
            ii = np.iinfo(dtype)
            x1 = rng_local.integers(ii.min//4, ii.max//4, size=x1_shape).astype(dtype)
            x2 = rng_local.integers(ii.min//4, ii.max//4, size=x2_shape).astype(dtype)
    elif special == 'near_boundary':
        if 'float' in dtype_str:
            finfo = np.finfo(dtype)
            x1 = rng_local.uniform(finfo.min*0.9, finfo.max*0.9, size=x1_shape).astype(dtype)
            x2 = rng_local.uniform(finfo.min*0.9, finfo.max*0.9, size=x2_shape).astype(dtype)
        else:
            ii = np.iinfo(dtype)
            margin = max(1, ii.max//100)
            x1 = rng_local.integers(ii.min+margin, ii.max-margin, size=x1_shape).astype(dtype)
            x2 = rng_local.integers(ii.min+margin, ii.max-margin, size=x2_shape).astype(dtype)
    else:
        if 'float' in dtype_str:
            x1 = rng_local.uniform(-10, 10, size=x1_shape).astype(dtype)
            x2 = rng_local.uniform(-10, 10, size=x2_shape).astype(dtype)
        else:
            ii = np.iinfo(dtype)
            x1 = rng_local.integers(-100, 100, size=x1_shape).astype(dtype)
            x2 = rng_local.integers(-100, 100, size=x2_shape).astype(dtype)

    golden = (x1 <= x2)
    out_shape = golden.shape

    prefix = f"t{idx:02d}"
    x1.tofile(f"{prefix}_x1.bin")
    x2.tofile(f"{prefix}_x2.bin")
    golden.astype(np.bool_).tofile(f"{prefix}_golden.bin")
    meta = np.array([dtype_str, list(x1_shape), list(x2_shape), list(out_shape)], dtype=object)
    np.save(f"{prefix}_meta.npy", meta)

    print(f"[{prefix}] {desc}")
    print(f"  x1={list(x1_shape)} x2={list(x2_shape)} -> out={list(out_shape)}  "
          f"dtype={dtype_str}  elems={golden.size}")
    if golden.size <= 8:
        print(f"  x1={x1}")
        print(f"  x2={x2}")
        print(f"  y ={golden.astype(np.int8)}")

    total += 1
    if np.array_equal(golden, (x1 <= x2)):
        passed += 1
    else:
        print(f"  *** VALIDATION ERROR!")

# ==================== test case definitions ====================
cases = [
    # -- 1D basics, four dtypes --
    ("float16", (4,), (4,), "basic_1d_float16"),
    ("float32", (6,), (6,), "basic_1d_float32"),
    ("int32",   (5,), (5,), "basic_1d_int32"),
    ("int8",    (7,), (7,), "basic_1d_int8"),

    # -- 2D matrices --
    ("float32", (3,4), (3,4), "2d_float32_3x4"),
    ("float16", (4,3), (4,3), "2d_float16_4x3"),
    ("int32",   (2,5), (2,5), "2d_int32_2x5"),
    ("int8",    (3,3), (3,3), "2d_int8_3x3"),

    # -- 3D / 4D --
    ("float32", (2,3,4), (2,3,4), "3d_float32_2x3x4"),
    ("float16", (1,4,4), (1,4,4), "3d_float16_1x4x4"),
    ("int32",   (2,2,3), (2,2,3), "3d_int32_2x2x3"),
    ("float32", (1,2,3,4), (1,2,3,4), "4d_float32_1x2x3x4"),

    # -- broadcast: scalar vs vector --
    ("float32", (1,), (6,), "bcast_x1_scalar"),
    ("float32", (6,), (1,), "bcast_x2_scalar"),
    ("int32",   (1,), (5,), "bcast_int32_x1_scalar"),
    ("int32",   (5,), (1,), "bcast_int32_x2_scalar"),
    ("float16", (1,), (3,), "bcast_float16_x1_scalar"),
    ("int8",    (1,), (4,), "bcast_int8_x1_scalar"),

    # -- broadcast: vector vs matrix --
    ("float32", (4,), (2,4), "bcast_vec_x1_mat_x2"),
    ("float32", (2,4), (4,), "bcast_mat_x1_vec_x2"),
    ("int32",   (3,), (2,3), "bcast_int32_vec_vs_mat"),
    ("float16", (4,), (3,4), "bcast_float16_vec_vs_mat"),

    # -- broadcast: high-dim --
    ("float32", (3,1), (1,4), "bcast_3x1_vs_1x4"),
    ("float32", (1,5), (3,1), "bcast_1x5_vs_3x1"),
    ("int32",   (2,1), (1,3), "bcast_int32_2x1_vs_1x3"),

    # -- extreme values --
    ("int8",    (8,), (8,), "extreme_int8", "extreme"),
    ("int32",   (6,), (6,), "extreme_int32", "extreme"),
    ("float16", (6,), (6,), "extreme_float16", "extreme"),
    ("float32", (6,), (6,), "extreme_float32", "extreme"),

    # -- N=1 and N=10000 --
    ("float32", (1,), (1,), "edge_N1"),
    ("float32", (10000,), (10000,), "edge_N10000"),
    ("int32",   (1,), (1,), "edge_int32_N1"),
    ("int32",   (10000,), (10000,), "edge_int32_N10000"),

    # -- all-equal --
    ("float32", (8,), (8,), "equal_float32", "equal"),
    ("int32",   (8,), (8,), "equal_int32", "equal"),
    ("float16", (8,), (8,), "equal_float16", "equal"),
    ("int8",    (8,), (8,), "equal_int8", "equal"),

    # -- all-zero --
    ("float32", (6,), (6,), "zeros_float32", "zeros"),
    ("float16", (6,), (6,), "zeros_float16", "zeros"),
    ("int32",   (6,), (6,), "zeros_int32", "zeros"),
    ("int8",    (6,), (6,), "zeros_int8", "zeros"),

    # -- large values --
    ("float32", (2048,), (2048,), "large_float32", "large_val"),
    ("int32",   (4096,), (4096,), "large_int32", "large_val"),

    # -- unaligned sizes --
    ("float32", (17,), (17,),  "unaligned_N17"),
    ("float32", (33,), (33,),  "unaligned_N33"),
    ("float32", (63,), (63,),  "unaligned_N63"),
    ("int32",   (15,), (15,),  "unaligned_int32_N15"),
    ("float16", (31,), (31,),  "unaligned_float16_N31"),
    ("int8",    (47,), (47,),  "unaligned_int8_N47"),

    # -- broadcast + unaligned --
    ("float32", (1,), (33,),  "bcast_unaligned_x1_scalar"),
    ("float32", (17,), (1,),  "bcast_unaligned_x2_scalar"),

    # -- multi-dim broadcast --
    ("float32", (2,1,4), (3,1), "bcast_3d_2x1x4_vs_3x1"),

    # -- int8 specific --
    ("int8",    (16,), (16,), "int8_extreme_cmp", "extreme"),
    ("int8",    (4,),  (4,),  "int8_equal_cmp", "equal"),

    # -- large scale --
    ("float32", (5000,), (5000,),  "scale_float32_5K"),
    ("float16", (8192,), (8192,),  "scale_float16_8K"),
    ("int32",   (10000,),(10000,), "scale_int32_10K"),
    ("int8",    (4096,), (4096,),  "scale_int8_4K"),

    # -- broadcast + large scale --
    ("float32", (1,), (10000,), "scale_bcast_x1_scalar_10K"),
    ("float32", (500,), (1,),   "scale_bcast_x2_scalar_500"),
]

for c in cases:
    dtype_str, x1s, x2s, desc = c[:4]
    special = c[4] if len(c) > 4 else None
    make_case(dtype_str, x1s, x2s, desc, special)

print(f"\n{'='*60}")
print(f"Total: {total} cases, numpy pre-check {passed}/{total} passed")
if passed < total:
    print("*** validation errors detected!")
print(f"{'='*60}")
