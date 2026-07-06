#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t MAX_BLOCK_DIM = 8;
constexpr uint32_t TILE_LENGTH = 1024;

enum DataType : uint32_t {
    DTYPE_FLOAT32 = 0,
    DTYPE_FLOAT16 = 1,
    DTYPE_BFLOAT16 = 2
};

static uint32_t GetBestBlockDim(uint32_t totalLength, uint32_t alignNum) {
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

static uint32_t GetDataTypeId(ge::DataType dataType) {
    if (dataType == ge::DT_FLOAT16) {
        return DTYPE_FLOAT16;
    } else if (dataType == ge::DT_BF16) {
        return DTYPE_BFLOAT16;
    } else {
        return DTYPE_FLOAT32;
    }
}

static uint32_t GetAlignNum(ge::DataType dataType) {
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        return 16;  // 32B / 2B = 16 elements
    } else {
        return 8;   // 32B / 4B = 8 elements
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    LogSigmoidCustomTilingData* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);

    // 计算总数据量
    uint64_t dataSize = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        dataSize *= xShape->GetStorageShape().GetDim(i);
    }
    uint32_t totalLength = static_cast<uint32_t>(dataSize);

    // 获取数据类型信息
    ge::DataType inputDataType = context->GetInputDesc(0)->GetDataType();
    uint32_t dataType = GetDataTypeId(inputDataType);
    uint32_t alignNum = GetAlignNum(inputDataType);

    // 计算最优Block数量
    uint32_t blockDim = GetBestBlockDim(totalLength, alignNum);
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;

    // 填充Tiling数据
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = TILE_LENGTH;
    tiling->dataType = dataType;

    // 设置Block数量
    context->SetBlockDim(blockDim);

    // 设置Workspace大小
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name) {
        // 输入定义
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 输出定义
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // 设置Shape和DataType推导函数
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // 设置AICore配置
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);

}  // namespace ops
