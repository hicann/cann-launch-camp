#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

command -v cmake >/dev/null 2>&1 || {
  echo "错误：未找到 cmake，请安装 CMake 3.16 或更高版本。" >&2
  exit 1
}

mkdir -p "${BUILD_DIR}"
: > "${BUILD_DIR}/.less_equal_build_dir"
cmake -S "${ROOT_DIR}/code" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo "构建完成：${BUILD_DIR}"

