#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdint>
#include <limits>

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {

static inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    return static_cast<uint32_t>((static_cast<uint64_t>(a) + b - 1) / b);
}

static inline uint32_t AlignUpU(uint32_t a, uint32_t m) {
    return (m == 0) ? a : static_cast<uint32_t>(((static_cast<uint64_t>(a) + m - 1) / m) * m);


static bool SafeMultiplyU64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (num_cores == 0) num_cores = 1;

    const gert::Tensor* tensor_x1 = context->GetRequiredInputTensor(0);
    const gert::Tensor* tensor_x2 = context->GetRequiredInputTensor(1);

    if (tensor_x1 == nullptr || tensor_x2 == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    ge::DataType dtype_x2 = tensor_x2->GetDataType();

    if (dtype_x1 != dtype_x2) {
        return ge::GRAPH_FAILED;
    }

    if (dtype_x1 != ge::DT_FLOAT16 && dtype_x1 != ge::DT_FLOAT &&
        dtype_x1 != ge::DT_INT32 && dtype_x1 != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    const gert::Shape& x1s = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& x2s = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape& ys = context->GetOutputShape(0)->GetStorageShape();

    LessEqualTilingData* tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t outElems = ys.GetShapeSize();

    if (outElems == 0) {
        tiling->mode = 0;
        tiling->totalLen = 0;
        tiling->tileLen = 0;
        tiling->blockDim = 1;
        tiling->perCore = 0;
        tiling->ndim = 0;
        tiling->lastDimLen = 0;
        tiling->totalRows = 0;
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }
        context->SetBlockDim(1);
        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    size_t n1 = x1s.GetDimNum();
    size_t n2 = x2s.GetDimNum();

    bool fast = (n1 == n2);
    if (fast) {
        for (size_t i = 0; i < n1; ++i) {
            if (x1s.GetDim(i) != x2s.GetDim(i)) {
                fast = false;
                break;
            }
        }
    }

    int dtype_size = ge::GetSizeByDataType(dtype_x1);
    if (dtype_size <= 0) {
        return ge::GRAPH_FAILED;
    }

    // -------- 计算 tileLen --------
    uint32_t perElemUB;
    if (dtype_x1 == ge::DT_INT32) {
        perElemUB = 27;
    } else if (dtype_x1 == ge::DT_FLOAT) {
        perElemUB = 25;
    } else {
        perElemUB = 15;
    }

    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    constexpr uint64_t UB_RESERVE = 8192;
    uint64_t usable_ub = (ub_size > UB_RESERVE) ? ub_size - UB_RESERVE : ub_size;

    const uint32_t BLOCK = 32;
    uint32_t perBlockUB = perElemUB * (BLOCK / static_cast<uint32_t>(dtype_size));
    uint32_t tileBlockNum = static_cast<uint32_t>(usable_ub) / perBlockUB;
    if (tileBlockNum == 0) tileBlockNum = 1;

    uint32_t tileLen = tileBlockNum * (BLOCK / static_cast<uint32_t>(dtype_size));

    constexpr uint32_t TILE_LEN_CAP = 16384;
    if (tileLen > TILE_LEN_CAP) {
        tileLen = TILE_LEN_CAP;
    }

    uint32_t total = static_cast<uint32_t>(outElems);
    if (tileLen > total) tileLen = total;

    // -------- Fast 模式 --------
    if (fast) {
        uint32_t blockDim = 1;
        uint32_t perCore = 32;

        const bool useFullCore = (dtype_x1 == ge::DT_FLOAT16 || dtype_x1 == ge::DT_INT8);

        if (useFullCore) {
            perCore = AlignUpU(CeilDivU(total, num_cores), 32U);
            if (perCore == 0) perCore = 32U;
            blockDim = CeilDivU(total, perCore);
        } else {
            constexpr uint32_t TARGET_PER_CORE = 8192;
            blockDim = CeilDivU(total, TARGET_PER_CORE);
            blockDim = std::min(blockDim, num_cores);
            if (blockDim == 0) blockDim = 1;
            perCore = AlignUpU(CeilDivU(total, blockDim), 32U);
        }

        if (blockDim == 0) blockDim = 1;

        tiling->mode = 0;
        tiling->totalLen = total;
        tiling->tileLen = tileLen;
        tiling->blockDim = blockDim;
        tiling->perCore = perCore;
        tiling->ndim = 0;
        tiling->lastDimLen = 0;
        tiling->totalRows = 0;

        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }

        context->SetBlockDim(blockDim);
        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // -------- 广播模式 --------
    size_t nmax = (n1 > n2) ? n1 : n2;
    int64_t d1[64], d2[64], od[64];

    for (size_t i = 0; i < nmax; ++i) {
        int64_t a = (i + n1 >= nmax) ? x1s.GetDim(i - (nmax - n1)) : 1;
        int64_t b = (i + n2 >= nmax) ? x2s.GetDim(i - (nmax - n2)) : 1;

        if (a < 0 || b < 0) return ge::GRAPH_FAILED;

        d1[i] = a;
        d2[i] = b;

        if (a == b) {
            od[i] = a;
        } else if (a == 1) {
            od[i] = b;
        } else if (b == 1) {
            od[i] = a;
        } else {
            return ge::GRAPH_FAILED;
        }
    }

    int64_t s1[64], s2[64];

    {
        int64_t acc1 = 1;
        for (int i = static_cast<int>(nmax) - 1; i >= 0; --i) {
            s1[i] = (d1[i] == 1) ? 0 : acc1;
            if (d1[i] != 1) acc1 *= d1[i];
        }

        int64_t acc2 = 1;
        for (int i = static_cast<int>(nmax) - 1; i >= 0; --i) {
            s2[i] = (d2[i] == 1) ? 0 : acc2;
            if (d2[i] != 1) acc2 *= d2[i];
        }
    }

    uint32_t cShape[64];
    int64_t cs1[64], cs2[64];
    int cnt = 0;

    for (int i = static_cast<int>(nmax) - 1; i >= 0; --i) {
        if (cnt > 0) {
            int j = cnt - 1;
            bool bothBcast = (s1[i] == 0 && s2[i] == 0 && cs1[j] == 0 && cs2[j] == 0);
            bool contig = (s1[i] != 0 && cs1[j] != 0 && s1[i] == cs1[j] * static_cast<int64_t>(cShape[j]) &&
                           s2[i] != 0 && cs2[j] != 0 && s2[i] == cs2[j] * static_cast<int64_t>(cShape[j]));

            if (bothBcast) {
                cShape[j] = static_cast<uint32_t>(static_cast<int64_t>(cShape[j]) * od[i]);
                continue;
            }
            if (contig) {
                cShape[j] = static_cast<uint32_t>(static_cast<int64_t>(cShape[j]) * od[i]);
                cs1[j] = s1[i];
                cs2[j] = s2[i];
                continue;
            }
        }

        cShape[cnt] = static_cast<uint32_t>(od[i]);
        cs1[cnt] = s1[i];
        cs2[cnt] = s2[i];
        ++cnt;
    }

    int ndim = cnt;

    uint32_t fShape[64];
    int64_t fs1[64], fs2[64];

    for (int i = 0; i < ndim; ++i) {
        fShape[i] = cShape[ndim - 1 - i];
        fs1[i] = cs1[ndim - 1 - i];
        fs2[i] = cs2[ndim - 1 - i];
    }

    while (ndim > static_cast<int>(LE_MAX_DIM)) {
        fShape[1] = static_cast<uint32_t>(static_cast<int64_t>(fShape[0]) * fShape[1]);
        for (int i = 1; i < ndim; ++i) {
            fShape[i - 1] = fShape[i];
            fs1[i - 1] = fs1[i];
            fs2[i - 1] = fs2[i];
        }
        --ndim;
    }

    uint32_t lastDimLen = fShape[ndim - 1];
    uint32_t totalRows = 1;

    for (int i = 0; i < ndim - 1; ++i) {
        uint64_t product = 0;
        if (!SafeMultiplyU64(totalRows, fShape[i], product)) {
            return ge::GRAPH_FAILED;
        }
        totalRows = static_cast<uint32_t>(product);
    }

    constexpr uint64_t BROADCAST_TARGET_PER_BLOCK = 2048;
    uint64_t desired = (static_cast<uint64_t>(total) + BROADCAST_TARGET_PER_BLOCK - 1) / BROADCAST_TARGET_PER_BLOCK;
    uint32_t blockDim = static_cast<uint32_t>(std::min(desired, static_cast<uint64_t>(num_cores)));
    if (blockDim == 0) blockDim = 1;

    uint32_t perCoreRows = CeilDivU(totalRows, blockDim);
    if (perCoreRows == 0) perCoreRows = 1;

    tiling->mode = 1;
    tiling->totalLen = total;
    tiling->tileLen = tileLen;
    tiling->blockDim = blockDim;
    tiling->perCore = perCoreRows;
    tiling->ndim = static_cast<uint32_t>(ndim);
    tiling->lastDimLen = lastDimLen;
    tiling->totalRows = totalRows;

    for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
        if (static_cast<int>(i) < ndim) {
            tiling->outShape[i] = fShape[i];
            tiling->x1Stride[i] = fs1[i];
            tiling->x2Stride[i] = fs2[i];
        } else {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }
    }

    context->SetBlockDim(blockDim);
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) ws[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context) {
    if (context == nullptr) return GRAPH_FAILED;

    const gert::Shape* s1 = context->GetInputShape(0);
    const gert::Shape* s2 = context->GetInputShape(1);
    gert::Shape* output = context->GetOutputShape(0);

    if (s1 == nullptr || s2 == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }

    size_t n1 = s1->GetDimNum();
    size_t n2 = s2->GetDimNum();
    size_t rank = (n1 > n2) ? n1 : n2;

    output->SetDimNum(rank);

    for (size_t i = 0; i < rank; ++i) {
        int64_t dim1 = (i < n1) ? s1->GetDim(n1 - 1 - i) : 1;
        int64_t dim2 = (i < n2) ? s2->GetDim(n2 - 1 - i) : 1;

        if (dim1 < 0 || dim2 < 0) return GRAPH_FAILED;

        int64_t outputDim = 0;
        if (dim1 == dim2) {
            outputDim = dim1;
        } else if (dim1 == 1) {
            outputDim = dim2;
        } else if (dim2 == 1) {
            outputDim = dim1;
        } else {
            return GRAPH_FAILED;
        }

        output->SetDim(rank - 1 - i, outputDim);
    }

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    if (context == nullptr) return GRAPH_FAILED;
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LessEqual : public OpDef {
public:
    explicit LessEqual(const char* name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(LessEqual);

}  // namespace ops