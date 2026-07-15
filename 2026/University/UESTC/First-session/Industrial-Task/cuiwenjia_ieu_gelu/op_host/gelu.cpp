#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static int64_t CeilAlign(int64_t v, int64_t a) { return CeilDiv(v, a) * a; }

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = platform.GetCoreNumAiv();
    if (coreNum <= 0) coreNum = 1;

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) ubSize = 196608;

    const gert::Tensor *input = context->GetRequiredInputTensor(0);
    ge::DataType dt = input->GetDataType();
    int64_t total = static_cast<int64_t>(input->GetShapeSize());
    int64_t elemSize = ge::GetSizeByDataType(dt);

    uint32_t tilingKey = static_cast<uint32_t>(dt);
    ASCENDC_TPL_SEL_PARAM(context, tilingKey);

    // Staged min elements per core for different tensor sizes
    int64_t minPerCore = 512;
    if (total > 8192) minPerCore = 1024;
    if (total > 262144) minPerCore = 768;

    // Adaptive core count
    int64_t usedCores = std::min(coreNum, std::max<int64_t>(1, CeilDiv(total, minPerCore)));

    // Base block length, aligned to 64 elements
    int64_t blockAlign = 64;
    int64_t blockLen = (total == 0) ? 0 : CeilDiv(total, usedCores);
    blockLen = CeilAlign(blockLen, blockAlign);

    // Compute actual core count from aligned block
    int64_t realCores = (total == 0) ? 1 : CeilDiv(total, blockLen);
    realCores = std::max<int64_t>(1, std::min(coreNum, realCores));

    // Tile size: 5 buffers * elemSize bytes/element
    int64_t maxTileByUb = (static_cast<int64_t>(ubSize) - 2048) / (5 * elemSize);
    maxTileByUb = CeilAlign(maxTileByUb, blockAlign);
    int64_t maxTile = (dt == ge::DT_FLOAT) ? 4096 : 8192;
    int64_t tileLen = std::max<int64_t>(blockAlign, std::min<int64_t>(maxTile, maxTileByUb));
    if (blockLen > 0) {
        tileLen = std::min(tileLen, CeilAlign(blockLen, blockAlign));
        tileLen = std::max<int64_t>(blockAlign, tileLen);
    }

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = static_cast<uint32_t>(total);
    tiling->blockLength = static_cast<uint32_t>(blockLen);
    tiling->tileLength = static_cast<uint32_t>(tileLen);

    context->SetBlockDim(static_cast<uint32_t>(realCores));

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x = context->GetInputShape(0);
    gert::Shape *y = context->GetOutputShape(0);
    *y = *x;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}
