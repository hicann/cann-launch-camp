#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

required_files=(
  "README.md"
  "code/CMakeLists.txt"
  "code/op_host/CMakeLists.txt"
  "code/op_host/less_equal.cpp"
  "code/op_kernel/CMakeLists.txt"
  "code/op_kernel/less_equal.cpp"
  "code/op_kernel/less_equal_tiling.h"
  "code/op_kernel/tiling_key_less_equal.h"
  "scripts/build.sh"
  "scripts/clean.sh"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -f "${ROOT_DIR}/${file}" ]]; then
    echo "缺少文件：${file}" >&2
    missing=1
  fi
done

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

echo "提交目录检查通过。"

