#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_PATH="${PROJECT_ROOT}/${BUILD_DIR}"

if [[ -d "${BUILD_PATH}" ]]; then
    rm -rf "${BUILD_PATH}"
    echo "Removed: ${BUILD_PATH}"
else
    echo "Nothing to clean: ${BUILD_PATH}"
fi

