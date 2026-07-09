#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/code"

mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
