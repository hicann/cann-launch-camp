#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${SCRIPT_DIR}/code"
BUILD_DIR="${BUILD_DIR:-${SOURCE_DIR}/build}"
CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"

if [[ ! -f "${CANN_ENV_SCRIPT}" ]]; then
    echo "CANN environment script not found: ${CANN_ENV_SCRIPT}" >&2
    echo "Set CANN_ENV_SCRIPT to the actual set_env.sh path." >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CANN_ENV_SCRIPT}"

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found after loading the CANN environment." >&2
    exit 1
fi

if [[ -z "${BUILD_JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        BUILD_JOBS="$(nproc)"
    else
        BUILD_JOBS=8
    fi
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
