#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCores = platform.GetCoreNumAiv();

    const gert::Tensor* input = context->GetRequiredInputTensor(0);
    uint32_t totalLength = input->GetShapeSize();

    // 固定每个 tile 处理 2048 个元素，简单高效
    constexpr uint32_t FIXED_TILE_LEN = 2048;
    uint32_t tileLength = FIXED_TILE_LEN;

    // 根据总长度和核心数计算实际使用的 core 数
    uint32_t neededCores = (totalLength + tileLength - 1) / tileLength;
    uint32_t blockDim = (neededCores < (uint32_t)numCores) ? neededCores : (uint32_t)numCores;
    if (blockDim < 1) blockDim = 1;

    // 每个 core 分配的元素数，并对齐到 32B
    uint32_t rawPerCore = (totalLength + blockDim - 1) / blockDim;
    constexpr uint32_t ALIGN = 32;
    uint32_t blockLength = ((rawPerCore + ALIGN - 1) / ALIGN) * ALIGN;

    GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength  = tileLength;

    context->SetBlockDim(blockDim);
    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;

    // 【修正点】先定义变量，再传入宏，避免宏展开错误
    uint32_t DT_INPUT_X = static_cast<uint32_t>(input->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// 形状推导与数据类型推导
namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

// 算子定义
namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

} // namespace ops