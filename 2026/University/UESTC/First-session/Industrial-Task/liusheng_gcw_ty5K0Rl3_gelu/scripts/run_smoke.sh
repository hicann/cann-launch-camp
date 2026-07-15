#!/usr/bin/env bash
set -euo pipefail

# Lightweight PR smoke check.
# It verifies that required source/docs/scripts exist and, when a build
# directory is present, that CMake generated build files are available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"

required_files=(
  "${ROOT_DIR}/README.md"
  "${ROOT_DIR}/CMakeLists.txt"
  "${ROOT_DIR}/op_host/CMakeLists.txt"
  "${ROOT_DIR}/op_host/gelu.cpp"
  "${ROOT_DIR}/op_kernel/CMakeLists.txt"
  "${ROOT_DIR}/op_kernel/gelu.cpp"
  "${ROOT_DIR}/op_kernel/gelu_tiling.h"
  "${ROOT_DIR}/op_kernel/tiling_key_gelu.h"
  "${ROOT_DIR}/scripts/build.sh"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "error: missing required file: ${file}" >&2
    exit 1
  fi
done

if [[ -d "${BUILD_DIR}" ]]; then
  if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "warning: build directory exists but CMakeCache.txt is missing: ${BUILD_DIR}"
    echo "warning: run ./scripts/build.sh after initializing the CANN/Ascend Toolkit environment."
  else
    echo "Build directory detected: ${BUILD_DIR}"
  fi
else
  echo "Build directory not found: ${BUILD_DIR}"
  echo "Run ./scripts/build.sh before submitting if your reviewer requires local build artifacts."
fi

echo "Smoke check passed."
