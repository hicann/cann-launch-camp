#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t MAX_BLOCK_DIM = 8;
constexpr uint32_t TILE_LENGTH = 1024;

constexpr uint32_t DTYPE_FLOAT32 = 0;
constexpr uint32_t DTYPE_FLOAT16 = 1;
constexpr uint32_t DTYPE_BFLOAT16 = 2;

static uint32_t GetBestBlockDim(uint32_t totalLength, uint32_t alignNum)
{
    if (totalLength == 0) {
        return 1;
    }

    for (uint32_t blockDim = MAX_BLOCK_DIM; blockDim > 0; --blockDim) {
        if ((totalLength % blockDim == 0) &&
            (((totalLength / blockDim) % alignNum) == 0)) {
            return blockDim;
        }
    }

    return 1;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);

    uint64_t dataSize = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= x1_shape->GetStorageShape().GetDim(i);
    }

    uint32_t totalLength = static_cast<uint32_t>(dataSize);

    auto inputDataType = context->GetInputDesc(0)->GetDataType();

    uint32_t dataType = DTYPE_FLOAT32;
    uint32_t alignNum = 8;   // float32: 32B / 4B = 8 elements

    if (inputDataType == ge::DT_FLOAT16) {
        dataType = DTYPE_FLOAT16;
        alignNum = 16;      // float16: 32B / 2B = 16 elements
    } else if (inputDataType == ge::DT_BF16) {
        dataType = DTYPE_BFLOAT16;
        alignNum = 16;      // bfloat16: 32B / 2B = 16 elements
    } else {
        dataType = DTYPE_FLOAT32;
        alignNum = 8;
    }

    uint32_t blockDim = GetBestBlockDim(totalLength, alignNum);
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;

    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = TILE_LENGTH;
    tiling->dataType = dataType;

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}
