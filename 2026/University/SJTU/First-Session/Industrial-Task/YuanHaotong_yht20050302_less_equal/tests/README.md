# LessEqual ACLNN Test

This test calls the generated `aclnnLessEqual` API and compares NPU output with a CPU reference.

## Cases

- `same-shape-small`: no broadcast, fewer than one tile
- `same-shape-one-tile`: no broadcast, exactly one tile
- `same-shape-tail-tile`: no broadcast, final tile is partial
- `same-shape-multicore`: no broadcast, enough elements for multiple AI Cores
- `same-shape-2d`: no broadcast, 2D tensor
- `same-shape-4d`: no broadcast, 4D tensor

Each case runs for `float16`, `float`, `int32`, and `int8`.

## Build And Run

The recommended path is to run the repository script from this operator directory:

```bash
bash run_less_equal_test.sh
```

For a manual run, use commands equivalent to the script below from this operator directory on a machine with CANN and NPU runtime.

```bash
set -e

ASCEND_HOME=${ASCEND_HOME:-/opt/home/developer/Ascend/cann-9.0.0}
ASCEND_TOOLKIT_ENV=${ASCEND_TOOLKIT_ENV:-/opt/home/developer/Ascend/ascend-toolkit/set_env.sh}
DEVICE_ID=${DEVICE_ID:-0}

source "$ASCEND_TOOLKIT_ENV"

rm -rf build
cmake -S . -B build
cmake --build build --target binary -j2
cmake --build build --target optiling -j2
cmake --build build --target custom_ascendc_cust_opapi -j2
mkdir -p build/scripts
touch build/version.info

if [ ! -e build/op_host/liboptiling.so ]; then
  for optiling_lib in build/op_host/liboptiling.so build/op_host/lib*_optiling.so build/op_host/lib*optiling*.so build/op_host/lib*opmaster*.so; do
    if [ -f "$optiling_lib" ]; then
      ln -s "$(basename "$optiling_lib")" build/op_host/liboptiling.so 2>/dev/null || cp "$optiling_lib" build/op_host/liboptiling.so
      break
    fi
  done
fi

if ! cmake --install build; then
  if ! find build/packages/vendors/custom -type f | grep -qi 'less_equal\|LessEqual'; then
    echo "cmake install failed and LessEqual custom OPP files were not generated." >&2
    exit 1
  fi
  echo "cmake install failed, continuing with generated build/packages/vendors/custom." >&2
fi

g++ -std=c++17 tests/less_equal_acl_test.cpp \
  -I"$ASCEND_HOME/include" \
  -Ibuild/autogen \
  -L"$ASCEND_HOME/lib64" \
  -Lbuild/packages/vendors/custom/op_api/lib \
  -Lbuild/op_host \
  -Wl,-rpath,"$ASCEND_HOME/lib64" \
  -Wl,-rpath,$PWD/build/packages/vendors/custom/op_api/lib \
  -Wl,-rpath,$PWD/build/op_host \
  -lcust_opapi -lascendcl -lnnopbase \
  -o build/less_equal_acl_test

# Make the generated custom op files visible to runtime.
export ASCEND_CUSTOM_OPP_PATH=$PWD/build/packages/vendors/custom
export LD_LIBRARY_PATH=$PWD/build/packages/vendors/custom/op_api/lib:$PWD/build/op_host:$LD_LIBRARY_PATH

npu-smi info
./build/less_equal_acl_test "$DEVICE_ID"
```

If your CANN installation uses another path, override `ASCEND_HOME` and `ASCEND_TOOLKIT_ENV`, for example:

```bash
ASCEND_HOME=/usr/local/Ascend/cann-9.0.1/x86_64-linux \
ASCEND_TOOLKIT_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
bash run_less_equal_test.sh
```

Expected output is a list of `[PASS] ... mismatch=0` lines. Any `[FAIL]` line means the first mismatching element is printed and the program exits with code `2`.
