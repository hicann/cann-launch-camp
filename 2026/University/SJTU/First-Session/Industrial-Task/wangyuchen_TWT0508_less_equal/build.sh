#!/bin/bash
# Build and test script for LessEqual operator
set -e

# Locate CANN installation
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND=$ASCEND_HOME_PATH
elif [ -d "$HOME/Ascend/cann-8.5.2" ]; then
    _ASCEND=$HOME/Ascend/cann-8.5.2
elif [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND=$HOME/Ascend/ascend-toolkit/latest
elif [ -d "/usr/local/Ascend/ascend-toolkit/latest" ]; then
    _ASCEND=/usr/local/Ascend/ascend-toolkit/latest
else
    echo "ERROR: cannot find Ascend installation, please set ASCEND_HOME_PATH"
    exit 1
fi

echo "=== Ascend: $_ASCEND ==="
source $_ASCEND/bin/setenv.bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# ---- build ----
echo "=== cleaning ==="
rm -rf build
mkdir -p build && cd build

echo "=== cmake ==="
cmake .. -DCMAKE_SKIP_RPATH=TRUE

echo "=== compiling ==="
make -j$(nproc) 2>&1 | tee build.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "=== build failed, see build.log ==="
    exit 1
fi
echo "=== build succeeded ==="
cd ..

# ---- test ----
if [ -d test ]; then
    echo ""
    echo "=== running local numpy tests ==="
    cd test
    python3 gen_data.py
    echo ""
    echo "=========================================="
    echo "  build and test completed, ready to submit"
    echo "=========================================="
    cd ..
fi
