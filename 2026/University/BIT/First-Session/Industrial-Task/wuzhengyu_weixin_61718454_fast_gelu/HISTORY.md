# FastGelu Optimization History

This document records the main optimization path for the submitted FastGelu custom operator.

## Baseline

The initial accepted implementation used the original FastGelu expression and a simple tiling strategy.

Baseline online result:

```text
4.72 / 4.72 / 9.08 / 8.56 / 12.80 us
sum = 39.88 us
```

## Formula Simplification

The formula was changed to:

```cpp
y = x * sigmoid(1.702x)
```

This removed several vector operations from the stable expression, including extra `Abs`, `Exp`, `Add`, and `Div` style work. It became the best formula basis for the remaining experiments.

## General Tiling Improvements

The implementation then moved through several tiling policies:

- fewer cores for small inputs;
- dtype-aware tile sizes;
- segmented AIV core selection;
- physical AI Core cap experiments.

These steps improved the total time from about `39.88 us` to about `27.98 us`.

## Small-Shape TilingKey Path

The biggest late-stage gain came from adding an `IS_SMALL_SHAPE` TilingKey.

Host-side tiling now selects between:

```text
generic path: multi-core generalized tiling
small path:   single-core, single-tile TBuf kernel
```

This avoids generic core splitting, queue setup, tile loops, and offset calculations for small tensors.

Representative V11 result:

```text
2.88 / 3.46 / 6.70 / 6.36 / 7.52 us
sum = 26.92 us
```

## Dtype-Aware 24U Threshold

The final submitted version keeps the same `DT_X + IS_SMALL_SHAPE` TilingKey shape, but selects the small path by dtype-aware work size.

```cpp
constexpr uint32_t SMALL_SHAPE_BASE_THRESHOLD = CORE_SPLIT_ELEM_NUM * 24U;

static uint32_t GetSmallShapeThreshold(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT16 ? SMALL_SHAPE_BASE_THRESHOLD * 2U : SMALL_SHAPE_BASE_THRESHOLD;
}
```

This means:

```text
fp32 threshold = CORE_SPLIT_ELEM_NUM * 24U
fp16 threshold = CORE_SPLIT_ELEM_NUM * 48U
```

Best ranking result:

```text
2.50 / 3.34 / 6.10 / 6.36 / 7.68 us
sum = 25.98 us
```

## Failed or Rejected Experiments

### Too-low dtype-aware threshold

`22U` made some medium shapes fall back to the generic path.

```text
2.82 / 3.36 / 6.84 / 6.94 / 8.14 us
sum = 28.10 us
```

### Too-high dtype-aware threshold

`26U` and `32U` pushed larger shapes into the single-core small path and slowed the large test cases.

`26U`:

```text
2.72 / 3.42 / 7.00 / 7.08 / 8.22 us
sum = 28.44 us
```

### Small aligned `DataCopy` TilingKey

An extra small-shape aligned TilingKey was tested so 32B-aligned small tensors used `DataCopy` instead of `DataCopyPad`.

Result:

```text
3.70 / 2.92 / 6.96 / 7.50 / 8.06 us
sum = 29.14 us
```

The extra specialization did not pay off, so the final code keeps the simpler `DataCopyPad` small path.

### Removing `PipeBarrier<PIPE_ALL>()` in the small path

Directly deleting the small-path barrier caused wrong answers:

```text
test 1 wrong answer: 91.11%
test 2 wrong answer: 76.95%
test 3/4/5 pass
```

This shows the small path needs an MTE2-to-Vector synchronization after `DataCopyPad`. The final implementation keeps the safe barrier.

### Lookup table or original formula

Replacing `Sigmoid` with a lookup table or reverting to the original expanded formula was rejected. Both approaches would add indexing, interpolation, extra vector operations, more temporary buffers, and more numerical risk.

## Final Decision

The submitted version keeps:

```text
formula:      y = x * sigmoid(1.702x)
TilingKey:    DT_X + IS_SMALL_SHAPE
threshold:    dtype-aware 24U
small path:   single-core TBuf + DataCopyPad + required barrier
generic path: segmented multi-core tiling
```
