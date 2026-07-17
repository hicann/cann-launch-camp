#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ASCEND_HOME="${ASCEND_HOME:-/opt/home/developer/Ascend/cann-9.0.0}"
ASCEND_TOOLKIT_ENV="${ASCEND_TOOLKIT_ENV:-/opt/home/developer/Ascend/ascend-toolkit/set_env.sh}"
DEVICE_ID="${DEVICE_ID:-0}"

if [[ ! -f "${ASCEND_TOOLKIT_ENV}" ]]; then
  echo "CANN env script not found: ${ASCEND_TOOLKIT_ENV}" >&2
  echo "Set ASCEND_TOOLKIT_ENV=/path/to/set_env.sh and rerun." >&2
  exit 1
fi

if [[ ! -d "${ASCEND_HOME}" ]]; then
  echo "ASCEND_HOME not found: ${ASCEND_HOME}" >&2
  echo "Set ASCEND_HOME=/path/to/cann-9.0.0 and rerun." >&2
  exit 1
fi

source "${ASCEND_TOOLKIT_ENV}"

rm -rf build
cmake -S . -B build
cmake --build build --target binary -j2
cmake --build build --target optiling -j2
cmake --build build --target custom_ascendc_cust_opapi -j2

# Some CANN package templates install an optional scripts directory even when
# the template did not generate one. Keep install non-fatal if the custom OPP
# package we need for ACLNN tests is already present.
mkdir -p build/scripts
touch build/version.info

# CANN 9 package install rules look for the conventional liboptiling.so name.
# Some toolkit builds emit a differently named tiling/master library, so provide
# the expected compatibility name before install.
if [[ ! -e build/op_host/liboptiling.so ]]; then
  for optiling_lib in \
    build/op_host/liboptiling.so \
    build/op_host/lib*_optiling.so \
    build/op_host/lib*optiling*.so \
    build/op_host/lib*opmaster*.so; do
    if [[ -f "${optiling_lib}" ]]; then
      ln -s "$(basename "${optiling_lib}")" build/op_host/liboptiling.so 2>/dev/null \
        || cp "${optiling_lib}" build/op_host/liboptiling.so
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

if ! find build/packages/vendors/custom -type f | grep -qi 'less_equal\|LessEqual'; then
  echo "LessEqual custom OPP files were not found under build/packages/vendors/custom." >&2
  exit 1
fi

g++ -std=c++17 tests/less_equal_acl_test.cpp \
  -I"${ASCEND_HOME}/include" \
  -Ibuild/autogen \
  -L"${ASCEND_HOME}/lib64" \
  -Lbuild/packages/vendors/custom/op_api/lib \
  -Lbuild/op_host \
  -Wl,-rpath,"${ASCEND_HOME}/lib64" \
  -Wl,-rpath,"${SCRIPT_DIR}/build/packages/vendors/custom/op_api/lib" \
  -Wl,-rpath,"${SCRIPT_DIR}/build/op_host" \
  -lcust_opapi -lascendcl -lnnopbase \
  -o build/less_equal_acl_test

export ASCEND_CUSTOM_OPP_PATH="${SCRIPT_DIR}/build/packages/vendors/custom"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/build/packages/vendors/custom/op_api/lib:${SCRIPT_DIR}/build/op_host:${LD_LIBRARY_PATH:-}"

echo "ASCEND_HOME=${ASCEND_HOME}"
echo "ASCEND_CUSTOM_OPP_PATH=${ASCEND_CUSTOM_OPP_PATH}"
echo "DEVICE_ID=${DEVICE_ID}"

"${SCRIPT_DIR}/build/less_equal_acl_test" "${DEVICE_ID}"
