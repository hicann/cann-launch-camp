# LessEqual Operator

Author: Yuan Haotong

GitCode account: yht20050302

## Overview

This directory contains the CANN custom operator implementation for `LessEqual`.
The operator compares two tensors element by element and writes a boolean-style
result for whether `x1 <= x2`.

## Implementation

The implementation is split into a host-side tiling function and an Ascend C
kernel. The operator supports same-shape `float16`, `float`, `int32`, and `int8`
inputs, and writes `bool` output. It does not call the built-in `Compare`
instruction. Instead, each data type uses vector arithmetic to manually produce
`0` or `1` for every element.

### Host-Side Tiling

The host side first checks the input data type and verifies that `x1` and `x2`
have the same shape. It then flattens the tensor to `total` elements and chooses
`blockDim` according to the data type and total size:

- `float16`: uses one AI Core. This avoids extra multi-core scheduling overhead
  for the target float16 cases where a single core is faster.
- `int32`: uses as many AI Cores as possible, capped by `total`, because the
  int32 arithmetic path is heavier and benefits from parallelism.
- `float`: uses all available AI Cores when `total >= 4097`; smaller cases keep
  the generic small-tensor policy to avoid unnecessary launch-side overhead.
- Other small tensors with `total <= 1024`: use one AI Core.
- Medium tensors with `total <= 4096`: use all available AI Cores, capped by
  `total`.
- Larger tensors: target about `1024` elements per core and cap the result by
  the available AI Core count.

After `blockDim` is selected, the host side splits the flattened tensor as evenly
as possible:

- `baseBlockLength = total / blockDim`
- `tailBlockNum = total % blockDim`
- the first `tailBlockNum` cores process `baseBlockLength + 1` elements
- the remaining cores process `baseBlockLength` elements

For each core, the maximum block length is used to choose `tileLength`. Small
blocks use a fixed `tileLength` of `256`, which reduces the relative overhead of
tiny tiles. The generic small-block threshold is `128` elements, while `int32`
uses a wider threshold of `216` elements because its manual arithmetic path needs
more temporary buffers and benefits from avoiding overly small tiles. For larger
blocks, the host side binary-searches the largest `32`-aligned tile length that
fits in UB after accounting for input queues, output queues, and the temporary
calculation buffers required by that data type.

The tiling data passed to the kernel contains `total`, `baseBlockLength`,
`tailBlockNum`, `tileNum`, `tileLength`, `lastTileLength`, buffer count, and the
aligned input/output buffer sizes. No workspace is used.

### Kernel Data Movement

Each AI Core computes its own contiguous slice from the flattened tensor. The
kernel derives the core-local offset and length from `baseBlockLength` and
`tailBlockNum`, then loops over tiles:

1. copy one contiguous tile of `x1` and `x2` from GM to local queues
2. compute the boolean result in vector registers/temporary buffers
3. copy the `int8_t` boolean output tile back to GM

Full tiles use normal `DataCopy`. Tail tiles use `DataCopyPad` so the vector
calculation length remains `32`-byte aligned while only valid elements are
written back.

### Kernel Arithmetic By Data Type

The kernel avoids `Compare` and manually maps the relation `x1 <= x2` to `0` or
`1`:

- `float16`: computes `max(x1, x2)`, then uses `abs(x2 - max(x1, x2))`. This is
  positive when `x1 > x2` and `0` when `x1 <= x2`. The value is clamped to the
  minimum fp16 accuracy, multiplied up, shifted by `-1`, and passed through
  `abs` to invert the mask into bool-style `0` or `1`.
- `float`: follows the same idea as `float16`, but uses fp32 constants and a
  larger scaling sequence before casting through `half` to `int8_t` output.
- `int8`: casts both inputs to `half` first. One branch computes whether
  `x2 - min(x1, x2)` is positive, and another branch handles equality by
  clamping `abs(x1 - x2)` into a `0/1` mask. Adding the two masks gives `1` for
  `x1 < x2` and `x1 == x2`, and `0` for `x1 > x2`.
- `int32`: keeps the main comparison arithmetic in `int32`. It similarly builds
  one mask for `x1 < x2` from `x2 - min(x1, x2)` and another mask for equality
  from the clamped squared difference. The result is then cast through `float`
  and `half` before writing the `int8_t` bool output.

## Directory Layout

- `op_host/`: host-side registration and tiling implementation
- `op_kernel/`: Ascend C kernel implementation
- `tests/`: ACLNN functional test and test documentation
- `run_less_equal_test.sh`: one-command build and test script
- `PERFORMANCE.md`: performance experiments and tuning notes
- `CMakeLists.txt`: top-level CMake build entry

## Build And Test

Run on a machine with CANN and NPU runtime installed:

```bash
cd 2026/University/SJTU/First-Session/Industrial-Task/YuanHaotong_yht20050302_less_equal
bash run_less_equal_test.sh
```

If your CANN installation is not in the default path used by the script, set the
environment variables before running:

```bash
export ASCEND_HOME=/path/to/cann
export ASCEND_TOOLKIT_ENV=/path/to/ascend-toolkit/set_env.sh
export DEVICE_ID=0
bash run_less_equal_test.sh
```

The expected result is that all ACLNN test cases print `[PASS]`.
