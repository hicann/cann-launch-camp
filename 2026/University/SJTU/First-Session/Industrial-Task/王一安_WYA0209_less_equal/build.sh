#!/bin/bash
# LessEqual 算子构建脚本
# 用法: ./build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "===== LessEqual 算子构建 ====="
echo "构建目录: ${BUILD_DIR}"

# 清理旧的构建
if [ -d "${BUILD_DIR}" ]; then
    echo "清理旧构建..."
    rm -rf "${BUILD_DIR}"
fi

# 创建构建目录
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# CMake 配置
echo "运行 CMake 配置..."
cmake ..

# 编译
echo "开始编译..."
make -j$(nproc)

echo "===== 构建完成 ====="
echo "输出目录: ${BUILD_DIR}"
ls -la "${BUILD_DIR}"
