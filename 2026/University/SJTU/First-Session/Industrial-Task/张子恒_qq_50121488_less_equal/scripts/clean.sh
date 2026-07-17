#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
MARKER_FILE="${BUILD_DIR}/.less_equal_build_dir"

case "${BUILD_DIR}" in
  "${ROOT_DIR}"|"/"|"")
    echo "拒绝清理不安全的目录：${BUILD_DIR}" >&2
    exit 1
    ;;
esac

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "构建目录不存在：${BUILD_DIR}"
  exit 0
fi

if [[ ! -f "${MARKER_FILE}" ]]; then
  echo "拒绝清理未由本项目脚本创建的目录：${BUILD_DIR}" >&2
  exit 1
fi

rm -rf -- "${BUILD_DIR}"
echo "已清理：${BUILD_DIR}"
