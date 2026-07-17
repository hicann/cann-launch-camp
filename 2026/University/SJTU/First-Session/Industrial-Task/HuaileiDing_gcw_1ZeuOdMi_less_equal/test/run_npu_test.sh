#!/usr/bin/env bash
set -euo pipefail

SUBMISSION_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_DIR="${SUBMISSION_DIR}/src"
BUILD_DIR="${PROJECT_DIR}/build"
ASCEND_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
DEVICE_ID="${1:-0}"

if [[ ! -f "${BUILD_DIR}/autogen/aclnn_less_equal.h" ||
      ! -f "${BUILD_DIR}/libcust_opapi.so" ]]; then
    echo "Build the operator first: cmake -S src -B src/build && cmake --build src/build -j"
    exit 1
fi

g++ -std=c++17 -O2 "${SUBMISSION_DIR}/test/aclnn_runner.cpp" \
    -I"${BUILD_DIR}/autogen" \
    -I"${ASCEND_PATH}/include" \
    -L"${BUILD_DIR}" \
    -L"${ASCEND_PATH}/lib64" \
    -Wl,-rpath,"${BUILD_DIR}" \
    -Wl,-rpath,"${ASCEND_PATH}/lib64" \
    -lcust_opapi -lnnopbase -lascendcl \
    -o "${BUILD_DIR}/less_equal_npu_test"

export ASCEND_CUSTOM_OPP_PATH="${BUILD_DIR}/tmp/vendors/custom${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
export LD_LIBRARY_PATH="${BUILD_DIR}:${BUILD_DIR}/autogen:${BUILD_DIR}/op_host:${ASCEND_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

"${BUILD_DIR}/less_equal_npu_test" "${DEVICE_ID}"
