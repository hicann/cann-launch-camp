#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="${ROOT_DIR}/code"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

load_cann_env() {
    local env_file="$1"
    if [[ ! -f "${env_file}" ]]; then
        return 1
    fi
    echo "Loading CANN environment: ${env_file}"
    set +u
    # shellcheck disable=SC1090
    source "${env_file}"
    set -u
}

if [[ -n "${CANN_ENV:-}" ]]; then
    load_cann_env "${CANN_ENV}" || {
        echo "CANN_ENV does not point to a file: ${CANN_ENV}" >&2
        exit 1
    }
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    loaded=0
    for candidate in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "${HOME}/Ascend/ascend-toolkit/set_env.sh"; do
        if load_cann_env "${candidate}"; then
            loaded=1
            break
        fi
    done
    if [[ "${loaded}" -eq 0 ]]; then
        echo "CANN set_env.sh was not found." >&2
        echo "Set CANN_ENV=/path/to/ascend-toolkit/set_env.sh and run again." >&2
        exit 1
    fi
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake was not found in PATH." >&2
    exit 1
fi

jobs="${JOBS:-}"
if [[ -z "${jobs}" ]]; then
    jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
fi

cmake_args=(
    -S "${SOURCE_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    cmake_args+=("-DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}")
fi

echo "CANN home: ${ASCEND_HOME_PATH:-not set}"
echo "Build directory: ${BUILD_DIR}"
cmake "${cmake_args[@]}"

build_log="${BUILD_DIR}/build.log"
set +e
cmake --build "${BUILD_DIR}" --parallel "${jobs}" 2>&1 | tee "${build_log}"
build_status="${PIPESTATUS[0]}"
set -e

if [[ "${build_status}" -ne 0 ]] || \
   grep -Eq '(^|[[:space:]])error:|\[ERROR\]|Kernel Compilation Error|not generated!' "${build_log}"; then
    echo "Build failed. See ${build_log}" >&2
    exit 1
fi

echo "Build completed. Generated libraries/packages:"
find "${BUILD_DIR}" -maxdepth 5 -type f \( -name '*.so' -o -name '*.run' \) -print
