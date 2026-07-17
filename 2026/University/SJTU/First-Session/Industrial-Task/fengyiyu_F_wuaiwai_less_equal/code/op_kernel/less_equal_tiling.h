/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

constexpr uint32_t LE_MAX_DIM = 16;

enum LessEqualMode : uint32_t {
    LE_MODE_FAST = 0,
    LE_MODE_BCAST = 1,
};

struct LessEqualTilingData {
    uint32_t mode;
    uint32_t totalElements;
    uint32_t tileSize;
    uint32_t blockDim;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;
    uint32_t outShape[LE_MAX_DIM];
    uint32_t x1Stride[LE_MAX_DIM];
    uint32_t x2Stride[LE_MAX_DIM];
};
