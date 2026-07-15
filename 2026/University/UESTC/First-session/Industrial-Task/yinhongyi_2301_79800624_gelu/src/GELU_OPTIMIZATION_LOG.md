# GELU Custom Operator Implementation and Optimization Log

This document records the implementation path for the AscendC GELU custom operator, including the stable passing baseline and optimization attempts that were tried. It is intended as handoff material for rebuilding the project from a clean template or asking another AI/engineer to continue the work.

## 1. Problem Summary

Operator: `Gelu`

Reference behavior: PyTorch `torch.nn.functional.gelu` default mode.

Formula:

```text
gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Supported input/output:

- Input: `input_x`
- Output: `output`
- Dtype: `float16`, `float32`
- Format: `ND`
- Shape: arbitrary positive-dimensional tensor, output shape equals input shape
- Need to support non-32-byte-aligned element counts

Precision requirement observed from the problem statement:

- `float32`: strict enough that tanh approximation may fail after hidden tests change
- `float16`: exact erf path is safest; tanh approximation can produce Wrong Answer on newer tests

Important conclusion:

**Use the exact erf formula as the correctness baseline. Do not use tanh approximation or `AscendC::Gelu` unless willing to risk hidden-test Wrong Answer.**

## 2. Files to Implement

Core files:

- `code/op_host/gelu.cpp`
- `code/op_kernel/gelu.cpp`
- `code/op_kernel/gelu_tiling.h`

Usually all three should be submitted together, especially when `GeluTilingData` changes. Submitting only host or only kernel after changing the tiling struct can cause compile errors or platform crashes.

## 3. Stable Passing Baseline

### 3.1 Host Side

The host side should:

1. Register op name `Gelu`.
2. Support `ge::DT_FLOAT16` and `ge::DT_FLOAT`.
3. Infer output shape equal to input shape.
4. Infer output dtype equal to input dtype.
5. Compute tiling by 32-byte blocks, not raw element count only.
6. Store original `inputNum` in tiling data so the kernel can avoid reading/writing padded tail elements.
7. Use `platform.GetCoreNum()` for block dimension.

Key stable tiling idea:

```cpp
uint32_t inputLengthAlign32 = ((inputNum * typeLength + blockSize - 1) / blockSize) * blockSize;
uint32_t inputBlockNum = inputLengthAlign32 / blockSize;
coreNum = std::min(coreNum, inputBlockNum);
coreNum = std::max(coreNum, static_cast<uint32_t>(1));
context->SetBlockDim(coreNum);
```

UB layout for stable version:

- input queue
- output queue
- temp buffer

So host should calculate tile size using `ubTensorNum = 3` and `bufferNum = 2`.

### 3.2 Tiling Struct

Stable struct:

```cpp
struct GeluTilingData {
    uint32_t totalDataNum;
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t finalBigTileNum;
    uint32_t finalSmallTileNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t tailCoreNum;
};
```

Notes:

- Some fields are not strictly needed by the latest kernel because tile counts are recomputed in kernel.
- Keeping them is safe and matches a previously passing structure.
- If removing fields, update host and kernel together.

### 3.3 Kernel Side

Stable kernel behavior:

1. Split work according to `smallCoreDataNum`, `bigCoreDataNum`, and `tailCoreNum`.
2. Clamp each core's data range by `totalDataNum`.
3. Copy input tile from GM to UB.
4. Compute exact GELU:

```cpp
tmp = x * 0.7071067811865475244;
tmp = erf(tmp);
tmp = tmp + 1;
y = x * tmp;
y = y * 0.5;
```

5. Copy output tile from UB to GM.

Important:

- Use a separate `tmpBuffer`.
- Do not reuse `yLocal` as the temp buffer for `Erf`/`Mul` in-place. This caused platform crash/all Fail in testing.

## 4. Correctness Bugs Found and Fixes

### 4.1 Non-32-byte-aligned Shape Wrong Answer

Symptom:

- Some tests passed, but non-aligned case had about 5-6% output error.

Cause:

- Work split was based on raw element count without carefully handling 32-byte data-copy alignment and true tail length.

Fix:

- Split by aligned 32-byte block count.
- Store true `totalDataNum`.
- In kernel, clamp each core:

```cpp
if (globalBufferIndex >= totalDataNum) {
    coreDataNum = 0;
} else if (globalBufferIndex + coreDataNum > totalDataNum) {
    coreDataNum = totalDataNum - globalBufferIndex;
}
```

### 4.2 Tiling Struct Mismatch

Symptom:

Compile error similar to:

```text
error: struct GeluTilingData has no member named finalBigTileNum
```

Cause:

- Host file and `gelu_tiling.h` were not submitted in sync.
- Or host was reverted but tiling header was still from a later experiment.

Fix:

- Always submit `op_host/gelu.cpp`, `op_kernel/gelu.cpp`, and `op_kernel/gelu_tiling.h` together after changing tiling fields.

### 4.3 Platform Script Crash

Symptom:

```text
UnboundLocalError: local variable 'output' referenced before assignment
```

Cause observed in experiments:

- Often not a normal Wrong Answer.
- Usually means compile/runtime crashed before the judge script collected output.
- Triggered by risky kernel API usage or struct mismatch.

Known trigger:

- Reusing output tensor as temporary buffer in-place:

```cpp
Erf(yLocal, yLocal, count);
Mul(yLocal, xLocal, yLocal, count);
```

Fix:

- Return to separate `tmpBuffer`.

## 5. Optimization Experiments Tried

### 5.1 `AscendC::Gelu`

Result:

- It can be faster on some cases.
- It caused Wrong Answer or timeout in others after hidden tests changed.

Conclusion:

- Not recommended for stable final submission.
- Likely not bit/precision-compatible enough with PyTorch default exact GELU under current tests.

### 5.2 Tanh Approximation

Formula tried:

```text
0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Result:

- Improved score on earlier tests.
- Failed newer tests with about 5% Wrong Answer.

Conclusion:

- Do not use for current hidden tests unless the problem explicitly accepts `approximate="tanh"`.

### 5.3 `BUFFER_NUM = 1`

Result:

- Caused timeout from early tests.

Conclusion:

- Keep `BUFFER_NUM = 2`.

### 5.4 `GetCoreNumAiv()`

Result:

- Performance became worse than `GetCoreNum()` in previous attempts.

Conclusion:

- Use `platform.GetCoreNum()` for current baseline.

### 5.5 Reduce-based Shortcut

Idea:

- Use `ReduceMax`/`ReduceMin` to detect all-positive or all-negative tiles and skip `Erf` for extreme values.

Result:

- Compile errors involving half scalar comparison and API use.
- Later attempts caused platform crash/all Fail.

Conclusion:

- Not recommended unless API usage is tested locally with CANN and dtype-specialized carefully.

### 5.6 Compare/Select/GetCmpMask

Idea:

- Use masks to select approximate outputs for large positive/negative x.

Observation:

- Compare/Select alone does not avoid computing `Erf` unless the compute path is split.
- If `Erf` is still computed for all elements, Select only adds overhead.

Conclusion:

- Not useful unless implementing a reliable branch that truly skips `Erf` for whole tiles or selected regions.

### 5.7 Reusing Output Tensor as Temp Buffer

Idea:

- Remove `tmpBuffer`, use `yLocal` for intermediate values, reduce UB from 3 tensors to 2 tensors, increase tile size.

Result:

- Platform crash/all Fail with `UnboundLocalError`.

Conclusion:

- Do not do in-place `Erf`/`Mul` with output tensor as temp buffer in this project.

### 5.8 Small-input Core Limit

Idea:

- Limit core count so every core gets at least several 32-byte blocks.

Example:

```cpp
const uint32_t minBlockPerCore = 8;
uint32_t coreNumByWorkload = (inputBlockNum + minBlockPerCore - 1) / minBlockPerCore;
coreNum = std::min(coreNum, coreNumByWorkload);
```

Risk:

- This only changes host side and should be safer than kernel math changes.
- But if this is applied while `gelu_tiling.h` is out of sync, compile errors can be misattributed to the core-limit change.

Recommendation:

- Try only after restoring the stable three-file baseline.
- If testing this, submit at least host plus the current matching tiling header.

## 6. Recommended Rebuild Flow From Empty Template

### Phase 1: Make It Compile

1. Define `GeluTilingData`.
2. Register `Gelu` in host.
3. Implement shape and dtype inference.
4. Implement minimal tiling.
5. Implement kernel skeleton with `Init`, `Process`, `CopyIn`, `Compute`, `CopyOut`.

Do not optimize yet.

### Phase 2: Make It Correct

1. Implement exact erf formula.
2. Support both `float16` and `float32` through `KernelGelu<DTYPE_INPUT_X>`.
3. Split by 32-byte-aligned block count.
4. Clamp by true `totalDataNum`.
5. Test non-32-aligned shapes.

Expected result:

- All tests should Pass, though not necessarily high score.

### Phase 3: Stabilize Submission

Before every submit:

1. Check `gelu_tiling.h` fields match host writes.
2. Check kernel reads the same fields.
3. Ensure no leftover experimental branches:

```text
computeMode
Tanh
AscendC::Gelu
ReduceMax
ReduceMin
GetValue
Compare
Select
GetCmpMask
```

4. Submit all three core files together.

### Phase 4: Conservative Optimization

Try only one change per submission.

Recommended order:

1. Host-side core count tuning.
2. Tile size tuning while keeping three UB buffers.
3. Remove unused tiling fields only if host/header/kernel are submitted together.
4. Only then consider dtype-specialized paths.

Avoid:

- Approximate tanh formula.
- `AscendC::Gelu`.
- In-place temp/output reuse.
- Reduce shortcuts without local compile validation.

## 7. Current Safe Baseline Checklist

Use this checklist before handing the project to another AI:

- [ ] `code/op_kernel/gelu_tiling.h` has `totalDataNum`, core data fields, tile fields, tail fields.
- [ ] `code/op_host/gelu.cpp` uses `ubTensorNum = 3`.
- [ ] `code/op_kernel/gelu.cpp` has separate `tmpBuffer`.
- [ ] Kernel formula uses `AscendC::Erf`.
- [ ] No tanh approximation.
- [ ] No `AscendC::Gelu`.
- [ ] No reduce shortcut.
- [ ] Non-aligned tail is clamped by `totalDataNum`.
- [ ] Submit host, kernel, and tiling header together.

## 8. Suggested Prompt for Restarting With Another AI

```text
I have a CANN AscendC GELU custom operator template. Please implement it from scratch using the exact PyTorch default GELU formula:

gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

Requirements:
- op name: Gelu
- input: input_x
- output: output
- dtype: float16 and float32
- format: ND
- output shape and dtype equal input
- support non-32-byte-aligned shapes

Please follow GELU_OPTIMIZATION_LOG.md. First build a stable passing baseline:
- split work by 32-byte aligned blocks
- keep true totalDataNum and clamp tail in kernel
- use BUFFER_NUM = 2
- use three UB areas: input queue, output queue, tmpBuffer
- do not use tanh approximation, AscendC::Gelu, ReduceMax/ReduceMin shortcuts, or in-place output-as-temp optimization

After it passes, optimize only one small host/tiling change at a time and keep a changelog.
```

## 9. 2026-07-08 Optimization Update

This section records the later performance exploration after rebuilding a clean passing baseline.

### 9.1 Clean Rebuild and Current Mainline

A clean implementation was rebuilt under the standard template layout:

- `code/op_host/gelu.cpp`
- `code/op_kernel/gelu.cpp`
- `code/op_kernel/gelu_tiling.h`
- `code/op_kernel/tiling_key_gelu.h`

The current mainline was intentionally rolled back to the best stable `tile4096` version:

- `float16`: sigmoid-form approximation
- `float32`: exact `AscendC::Erf`
- `tileDataNum` capped at `4096`
- `BUFFER_NUM = 2`
- three UB areas: input, output, temp
- no `computeMode`
- no A&S branch
- no `Vec*` APIs

Current rollback package:

```text
gelu_rollback_tile4096.zip
```

Representative result for this family:

```text
test 1: about 3.5-3.8 us
test 2: about 11.3 us
test 3: about 27.3 us
test 4: about 5.0 us
test 5: about 20.8 us
test 6: about 128-129 us
```

The online judge has noticeable timing variance. The same package can produce different per-test times across submissions, so compare changes by repeated submissions instead of a single run.

### 9.2 `tileDataNum` Sweep

After fp16 sigmoid was introduced, tile size became the useful tuning knob.

Tested caps:

```text
2048
3072
3584
4096
4608
5120
6144
```

Observed conclusion:

```text
4096 was the best overall.
3584, 4608, 5120 were tested and were not better.
2048 and larger fp16-only caps were also not better in the observed runs.
```

Practical recommendation:

```cpp
tileDataNum = std::min(tileDataNum, static_cast<uint32_t>(4096));
```

Keep `ubTensorNum = 3` for the current best mainline.

### 9.3 fp16 Sigmoid Approximation

A CSDN/Juejin-style GELU approximation was tested for fp16:

```text
y = x / (1 + exp(-1.595769122 * (x + 0.0455399241 * x^3)))
```

Offline error analysis:

- `float32`: max absolute error around `7.46e-4`, too large for the stated fp32 tolerance.
- `float16`: fits the looser fp16 tolerance in the judge tests.

Result:

- Pass.
- Clear speedup on fp16-heavy test points, especially test 1 and test 4.
- Became part of the current best mainline.

Do not apply this approximation blindly to fp32.

### 9.4 Pipeline Experiment

Idea:

- Use existing `BUFFER_NUM = 2` to prefetch tiles and overlap copy/compute/writeback.

Result:

- Passed.
- Slower than the non-pipelined version.

Conclusion:

```text
Do not keep the pipeline version. Simple CopyIn -> Compute -> CopyOut per tile is faster for this workload.
```

### 9.5 Core Count Experiments

Tested:

- Full `platform.GetCoreNum()`
- `platform.GetCoreNumAiv()`
- small-input core limit, including `minBlockPerCore = 4` and earlier `8`

Observed:

- `GetCoreNum()` remains the best stable choice.
- `GetCoreNumAiv()` did not improve the score.
- `minBlockPerCore = 8` was too conservative.
- `minBlockPerCore = 4` did not beat the current tile4096 mainline.

Recommendation:

```cpp
uint32_t coreNum = platform.GetCoreNum();
coreNum = std::min(coreNum, inputBlockNum);
coreNum = std::max(coreNum, static_cast<uint32_t>(1));
```

### 9.6 A&S erf Approximation for fp32

A blog suggested Abramowitz & Stegun erf approximation:

```text
erf(x) ~= sign(x) * (1 - tau * exp(-x^2))
t = 1 / (1 + p * |x|)
tau = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
p  = 0.3275911
a1 = 0.254829592
a2 = -0.284496736
a3 = 1.421413741
a4 = -1.453152027
a5 = 1.061405429
```

The first attempt copied the blog's `Vec*` API style:

```text
VecAbs
VecMul
VecSub
VecExp
VecReciprocal
VecSelect
GetPhyAddr
```

Compile result:

```text
Vec* identifiers were undeclared in the judge environment.
GetPhyAddr() returned uint64_t, not a usable float*.
```

Conclusion:

```text
The blog's raw Vec* API style is not directly usable in this template.
```

A second version implemented the same A&S formula using standard AscendC tensor APIs:

```text
Abs
Muls
Adds
Mul
Exp
Div
Duplicate
```

Result:

- Passed.
- Improved fp32-heavy test points:

```text
test 2: about 11.3 us -> about 10.3-10.5 us
test 3: about 27.3 us -> about 24.2-24.4 us
test 5: about 20.8 us -> about 18.3-18.7 us
test 6: about 128-129 us -> about 110-111 us
```

Downside:

- fp16/small test points became worse or unstable.
- Extra UB buffers and additional branch structure made the total score not clearly better than tile4096 mainline.

Conclusion:

```text
A&S fp32 is a valid passable direction and improves large/fp32-heavy points,
but it is not the current final mainline because it hurts smaller/fp16 points
and does not close enough of the gap to the best leaderboard times.
```

### 9.7 Split-Class A&S Experiment

Idea:

- Split the kernel into separate half and float classes.
- Keep half class free of fp32 A&S buffers and code.

Result:

- Prepared and tested as an experiment.
- Did not become the final mainline.

Conclusion:

```text
Splitting classes did not justify replacing the simpler tile4096 mainline.
```

### 9.8 Large-Data Hybrid Experiment

Idea:

- Keep current `tile4096` mainline for small data.
- Use A&S fp32 only for large data, selected by host-side `totalDataNum`.

Tested thresholds:

```text
float32 && totalDataNum >= 32768
float32 && totalDataNum >= 65536
```

Representative results:

```text
threshold 32768:
test 1: 3.95 us
test 2: 10.33 us
test 3: 24.35 us
test 4: 4.97 us
test 5: 18.55 us
test 6: 110.65 us

threshold 65536:
test 1: 3.48 us
test 2: 10.22 us
test 3: 23.85 us
test 4: 5.23 us
test 5: 18.31 us
test 6: 110.28 us
```

Result:

- Both passed.
- They did improve fp32-heavy points compared with exact-erf tile4096.
- They still did not provide a clearly better overall final candidate because of timing variance and remaining gap to best times.

Decision:

```text
Stop pursuing large/small hybrid for now.
Keep the main workspace rolled back to tile4096.
```

### 9.9 Current Recommendation

Use this as the final safe-performance version unless new information appears:

```text
gelu_rollback_tile4096.zip
```

Keep these notes for future experiments:

- fp16 sigmoid approximation is worth keeping.
- `tileDataNum = 4096` is the best observed tile cap.
- Pipeline is not worth keeping.
- `GetCoreNum()` is better than `GetCoreNumAiv()` here.
- A&S fp32 approximation passes and improves large fp32 points, but it is not the current final mainline.
- Blog `Vec*` API code cannot be copied directly into this judge environment.
- The judge timing has visible run-to-run variance; repeat submissions before trusting small differences.
