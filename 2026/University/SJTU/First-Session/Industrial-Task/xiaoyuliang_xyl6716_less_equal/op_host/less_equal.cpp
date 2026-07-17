#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {

static inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    return static_cast<uint32_t>((static_cast<uint64_t>(a) + b - 1) / b);
}

static inline uint32_t AlignUpU(uint32_t a, uint32_t m) {
    return (m == 0) ? a : ((a + m - 1) / m) * m;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (num_cores == 0) num_cores = 1;

    const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    const gert::Shape &x1s = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape &x2s = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape &ys  = context->GetOutputShape(0)->GetStorageShape();
    int64_t outElems = ys.GetShapeSize();

    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();

    if (outElems == 0) {
        tiling->mode = 0;
        tiling->totalElements = 0;
        tiling->tileSize = 0;
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
        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    size_t n1 = x1s.GetDimNum();
    size_t n2 = x2s.GetDimNum();

    bool fast = (n1 == n2);
    if (fast) {
        for (size_t i = 0; i < n1; ++i) {
            if (x1s.GetDim(i) != x2s.GetDim(i)) { fast = false; break; }
        }
    }

    const uint32_t BLOCK = 32;
    int dtype_size = ge::GetSizeByDataType(dtype_x1);

    uint32_t perElemUB;
    if (dtype_x1 == ge::DT_INT32) {
        perElemUB = 27;
    } else if (dtype_x1 == ge::DT_FLOAT) {
        perElemUB = 25;
    } else {
        perElemUB = 15;
    }

    uint32_t perBlockUB = perElemUB * (BLOCK / static_cast<uint32_t>(dtype_size));
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    uint32_t tileBlockNum = static_cast<uint32_t>(ub_size) / perBlockUB;
    if (tileBlockNum == 0) tileBlockNum = 1;

    uint32_t tileSize = tileBlockNum * (BLOCK / static_cast<uint32_t>(dtype_size));

    if (fast) {
        uint32_t total = static_cast<uint32_t>(outElems);
        if (tileSize > total) tileSize = total;

        uint32_t perCore = AlignUpU(CeilDivU(total, num_cores), 32u);
        if (perCore == 0) perCore = 32u;
        uint32_t blockDim = CeilDivU(total, perCore);
        if (blockDim == 0) blockDim = 1;

        tiling->mode = 0;
        tiling->totalElements = total;
        tiling->tileSize = tileSize;
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
        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    size_t nmax = (n1 > n2) ? n1 : n2;
    int64_t d1[64], d2[64], od[64];
    for (size_t i = 0; i < nmax; ++i) {
        int64_t a = (i + n1 >= nmax) ? x1s.GetDim(i - (nmax - n1)) : 1;
        int64_t b = (i + n2 >= nmax) ? x2s.GetDim(i - (nmax - n2)) : 1;
        d1[i] = a;
        d2[i] = b;
        od[i] = (a == 1) ? b : a;
    }

    int64_t s1[64], s2[64];
    {
        int64_t acc1 = 1;
        for (int i = (int)nmax - 1; i >= 0; --i) {
            s1[i] = (d1[i] == 1) ? 0 : acc1;
            if (d1[i] != 1) acc1 *= d1[i];
        }
        int64_t acc2 = 1;
        for (int i = (int)nmax - 1; i >= 0; --i) {
            s2[i] = (d2[i] == 1) ? 0 : acc2;
            if (d2[i] != 1) acc2 *= d2[i];
        }
    }

    uint32_t cShape[64];
    int64_t  cs1[64], cs2[64];
    int cnt = 0;
    for (int i = (int)nmax - 1; i >= 0; --i) {
        if (cnt > 0) {
            int j = cnt - 1;
            bool bothBcast = (s1[i] == 0 && s2[i] == 0 && cs1[j] == 0 && cs2[j] == 0);
            bool contig =
                (s1[i] != 0 && cs1[j] != 0 && s1[i] == cs1[j] * (int64_t)cShape[j]) &&
                (s2[i] != 0 && cs2[j] != 0 && s2[i] == cs2[j] * (int64_t)cShape[j]);
            if (bothBcast) {
                cShape[j] = (uint32_t)((int64_t)cShape[j] * od[i]);
                continue;
            }
            if (contig) {
                cShape[j] = (uint32_t)((int64_t)cShape[j] * od[i]);
                cs1[j] = s1[i];
                cs2[j] = s2[i];
                continue;
            }
        }
        cShape[cnt] = (uint32_t)od[i];
        cs1[cnt] = s1[i];
        cs2[cnt] = s2[i];
        cnt++;
    }

    int ndim = cnt;
    uint32_t fShape[64];
    int64_t  fs1[64], fs2[64];
    for (int i = 0; i < ndim; ++i) {
        fShape[i] = cShape[ndim - 1 - i];
        fs1[i] = cs1[ndim - 1 - i];
        fs2[i] = cs2[ndim - 1 - i];
    }

    while (ndim > (int)LE_MAX_DIM) {
        fShape[1] = (uint32_t)((int64_t)fShape[0] * fShape[1]);
        for (int i = 1; i < ndim; ++i) {
            fShape[i - 1] = fShape[i];
            fs1[i - 1] = fs1[i];
            fs2[i - 1] = fs2[i];
        }
        ndim--;
    }

    uint32_t lastDimLen = fShape[ndim - 1];
    uint32_t totalRows = 1;
    for (int i = 0; i < ndim - 1; ++i) totalRows *= fShape[i];

    uint32_t perCore = CeilDivU(totalRows, num_cores);
    if (perCore == 0) perCore = 1;
    uint32_t blockDim = CeilDivU(totalRows, perCore);
    if (blockDim == 0) blockDim = 1;

    tiling->mode = 1;
    tiling->totalElements = static_cast<uint32_t>(outElems);
    tiling->tileSize = tileSize;
    tiling->blockDim = blockDim;
    tiling->perCore = perCore;
    tiling->ndim = (uint32_t)ndim;
    tiling->lastDimLen = lastDimLen;
    tiling->totalRows = totalRows;
    for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
        if ((int)i < ndim) {
            tiling->outShape[i] = fShape[i];
            tiling->x1Stride[i] = (int32_t)fs1[i];
            tiling->x2Stride[i] = (int32_t)fs2[i];
        } else {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }
    }
    context->SetBlockDim(blockDim);
    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *s1 = context->GetInputShape(0);
    const gert::Shape *s2 = context->GetInputShape(1);
    gert::Shape *y = context->GetOutputShape(0);
    size_t n1 = s1->GetDimNum();
    size_t n2 = s2->GetDimNum();
    size_t n = (n1 > n2) ? n1 : n2;
    y->SetDimNum(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t dd1 = (i < n1) ? s1->GetDim(n1 - 1 - i) : 1;
        int64_t dd2 = (i < n2) ? s2->GetDim(n2 - 1 - i) : 1;
        int64_t o = (dd1 == 1) ? dd2 : dd1;
        y->SetDim(n - 1 - i, o);
    }
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class LessEqual : public OpDef {
public:
    explicit LessEqual(const char *name) : OpDef(name) {
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops
