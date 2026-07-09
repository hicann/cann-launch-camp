#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/code/build"

cmake -S "${SCRIPT_DIR}/code" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j
