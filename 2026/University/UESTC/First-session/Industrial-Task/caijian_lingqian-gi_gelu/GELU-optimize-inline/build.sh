#!/bin/bash
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "please source set_env.sh first"
    exit 1
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*

target=package
if [ "$1"x != ""x ]; then target=$1; fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_BINARY_PACKAGE=True \
    -DASCEND_COMPUTE_UNIT=ascend910b \
    -Dvendor_name=custom \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}" \
    -DENABLE_CROSS_COMPILE=False
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
# CPack 需要 binary/config 目录
mkdir -p "$BUILD_DIR/op_kernel/ascendc_kernels/binary/config"
echo "aicore_arch=ascend910b" > "$BUILD_DIR/op_kernel/ascendc_kernels/binary/config/ccec.cfg"
cmake --build "$BUILD_DIR" --target $target -j$(nproc)
