#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor* tensor_input_x = context->GetRequiredInputTensor(0);

    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    uint32_t length = tensor_input_x->GetShapeSize();

    if (length == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t DT_INPUT_X = 0;

    if (dtype_input_x == ge::DT_FLOAT16) {
        DT_INPUT_X = static_cast<uint32_t>(ge::DT_FLOAT16);
    } else if (dtype_input_x == ge::DT_FLOAT) {
        DT_INPUT_X = static_cast<uint32_t>(ge::DT_FLOAT);
    } else {
        return ge::GRAPH_FAILED;
    }

    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length;

    // 用满所有 AIV 核，但确保每个核至少处理 512 个元素
    uint32_t blockDim = num_cores_aiv;
    if (blockDim == 0) {
        blockDim = 1;
    }

    uint32_t maxCores = (length + 1023) / 1024;
    if (blockDim > maxCores) {
        blockDim = maxCores;
        if (blockDim == 0) {
            blockDim = 1;
        }
    }

    // 确保每个核的 blockLength 是 32-byte 对齐的（提升 DataCopy 性能）
    // float32: 8 元素对齐, float16: 16 元素对齐
    uint32_t alignSize = (dtype_input_x == ge::DT_FLOAT) ? 8 : 16;
    while (blockDim > 1 && (length / blockDim) % alignSize != 0) {
        blockDim--;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }

    context->SetBlockDim(blockDim);

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

} // namespace ops