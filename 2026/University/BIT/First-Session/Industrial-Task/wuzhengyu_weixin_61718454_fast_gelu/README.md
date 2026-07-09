# FastGelu Custom Operator

## Submitter

- Name: wuzhengyu
- GitCode account: weixin_61718454
- Task: fast_gelu

## Result

Best CANNJudge ranking record:

```text
2.50 / 3.34 / 6.10 / 6.36 / 7.68 us
sum = 25.98 us
```

The submitted implementation passes all 5 test cases.

## Main Optimizations

1. Replaced the original stable FastGelu expression with:

   ```cpp
   y = x * sigmoid(1.702x)
   ```

2. Added a `DT_X + IS_SMALL_SHAPE` TilingKey so host-side tiling selects a specialized small-shape kernel before launch.

3. Used a dtype-aware small-shape threshold:

   ```cpp
   fp32: CORE_SPLIT_ELEM_NUM * 24U
   fp16: CORE_SPLIT_ELEM_NUM * 24U * 2U
   ```

4. Added a single-core small-shape path using `TBuf`, one tile, and fewer queue/loop/core-distribution overheads.

## Directory Layout

```text
code/
  CMakeLists.txt
  op_host/
    CMakeLists.txt
    fast_gelu.cpp
  op_kernel/
    CMakeLists.txt
    fast_gelu.cpp
    fast_gelu_tiling.h
    tiling_key_fast_gelu.h
  tests/
    test_fast_gelu_static.py
  tools/
    verify_fast_gelu.py
    allnight_fastgelu_pipeline.py
FASTGELU_REQUIREMENTS.md
HISTORY.md
```

## Verification

Static verification:

```powershell
d:\py\Anaconda3\python.exe code\tools\verify_fast_gelu.py
```

Expected result:

```text
Ran 9 tests
OK
```

See `HISTORY.md` for the detailed optimization process and failed experiments.
