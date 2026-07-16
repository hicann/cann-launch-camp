#!/bin/bash
# LessEqual 算子：编译 + 运行测试（不修改工程 CMakeLists，使用 SHARED 产物）
#
# SHARED 构建产物：
#   - 算子实现/kernel 二进制/op_info: build/tmp/vendors/custom  (指向 ASCEND_CUSTOM_OPP_PATH)
#   - aclnn 二段式接口库:            build/libcust_opapi.so
#   - aclnn 头文件:                  build/autogen/aclnn_less_equal.h
# 默认 `make` 不编译 kernel 二进制，必须执行 `make binary`。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

: "${ASCEND_HOME_PATH:=/home/developer/Ascend/cann-8.5.2}"
BUILD_DIR="${SCRIPT_DIR}/build"
VENDOR_DIR="${BUILD_DIR}/tmp/vendors/custom"

echo "=========================================="
echo "LessEqual 算子构建"
echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"
echo "=========================================="

# [1/3] 编译算子工程 + kernel 二进制
echo ""
echo "[1/3] 编译算子（含 make binary 编译 kernel）..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}"
make -j"$(nproc)"
make binary -j"$(nproc)"
cd "${SCRIPT_DIR}"

# [2/3] 编译测试程序（链接 build 下的 aclnn 库）
echo ""
echo "[2/3] 编译测试程序..."
g++ -std=c++17 -o "${BUILD_DIR}/less_equal_test" test/less_equal_test.cpp \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_HOME_PATH}/include/acl" \
    -I"${ASCEND_HOME_PATH}/include/aclnn" \
    -I"${BUILD_DIR}/autogen" \
    -L"${ASCEND_HOME_PATH}/lib64" -L"${BUILD_DIR}" \
    -lascendcl -lnnopbase -lcust_opapi -ldl \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" -Wl,-rpath,"${BUILD_DIR}"

# [3/3] 运行测试
echo ""
echo "[3/3] 运行测试..."
export ASCEND_CUSTOM_OPP_PATH="${VENDOR_DIR}"
"${BUILD_DIR}/less_equal_test"
