#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="Release"
CANN_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --cann-path)
            CANN_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/build.sh [--build-dir build] [--build-type Release] [--cann-path PATH]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_PATH="${PROJECT_ROOT}/${BUILD_DIR}"

if [[ -n "${CANN_PATH}" ]]; then
    export ASCEND_HOME_PATH="${CANN_PATH}"
    export ASCEND_CANN_PACKAGE_PATH="${CANN_PATH}"
fi

if [[ -z "${ASCEND_HOME_PATH:-}" && -z "${ASCEND_CANN_PACKAGE_PATH:-}" ]]; then
    echo "CANN environment is not configured. Source set_env.sh first or pass --cann-path." >&2
    exit 1
fi

CMAKE_ARGS=(-S "${PROJECT_ROOT}" -B "${BUILD_PATH}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}")
if [[ -n "${CANN_PATH}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=${CANN_PATH}")
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_PATH}" --config "${BUILD_TYPE}"

echo "Build finished: ${BUILD_PATH}"
