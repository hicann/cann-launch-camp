#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"

if [[ ! -f "$SRC_DIR/CMakeLists.txt" ]]; then
    echo "ERROR: 找不到 $SRC_DIR/CMakeLists.txt" >&2
    exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: 当前环境中找不到 cmake" >&2
    exit 1
fi

load_cann_environment() {
    local candidates=(
        "$HOME/Ascend/cann-9.0.0/set_env.sh"
        "$HOME/Ascend/ascend-toolkit/latest/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
        "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    )

    local script

    for script in "${candidates[@]}"; do
        if [[ -f "$script" ]]; then
            echo "Loading CANN environment: $script"
            set +u
            source "$script"
            set -u
            return 0
        fi
    done

    echo "ERROR: 未找到 CANN set_env.sh" >&2
    echo "请先手动加载 CANN 环境后重试" >&2
    return 1
}

load_cann_environment

if [[ "$CLEAN_BUILD" == "1" ]]; then
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "Source directory: $SRC_DIR"
echo "Build directory : $BUILD_DIR"
echo "Compute unit    : ascend910b"

cmake \
    -S "$SRC_DIR" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release

cmake \
    --build "$BUILD_DIR" \
    --parallel "$(nproc)"

echo
echo "LESS EQUAL BUILD: PASS"
echo "Build output: $BUILD_DIR"
