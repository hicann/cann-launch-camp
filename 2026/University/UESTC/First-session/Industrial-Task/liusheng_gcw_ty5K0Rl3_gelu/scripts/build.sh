#!/usr/bin/env bash
set -euo pipefail

# Build helper for the AscendC custom Gelu operator.
# Usage:
#   ./scripts/build.sh [build_dir]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-${ROOT_DIR}/build}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake is not available in PATH" >&2
  exit 1
fi

echo "Project root: ${ROOT_DIR}"
echo "Build dir:    ${BUILD_DIR}"

if ! cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"; then
  echo "error: CMake configure failed." >&2
  echo "hint: initialize the CANN/Ascend Toolkit environment, for example:" >&2
  echo "      source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  echo "      or set ASC_DIR/CMAKE_PREFIX_PATH to the directory containing ASCConfig.cmake." >&2
  exit 1
fi

cmake --build "${BUILD_DIR}"

echo "Build completed."
