#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

// 最多使用的 AI Core 数量
constexpr uint32_t MAX_BLOCK_DIM = 8;

// 按 32 个元素对齐。
// float32、float16 和 bfloat16 使用该对齐长度都没有问题。
constexpr uint32_t ALIGN_NUM = 32;

// 单个 Tile 最多处理 4096 个元素
constexpr uint32_t DEFAULT_TILE_LENGTH = 4096;

// 向上对齐
static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

// 向上取整除法
static uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取 Tiling 参数存储空间
    LogSigmoidCustomTilingData* tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    // 获取输入张量 Shape
    const gert::StorageShape* xShape = context->GetInputShape(0);

    if (tiling == nullptr || xShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& storageShape = xShape->GetStorageShape();

    // ---------------------------------------------------------
    // 1. 计算输入张量的总元素数量
    // ---------------------------------------------------------
    uint64_t dataSize64 = 1;

    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        int64_t dim = storageShape.GetDim(i);

        if (dim <= 0) {
            dataSize64 = 0;
            break;
        }

        dataSize64 *= static_cast<uint64_t>(dim);
    }

    // 当前 Tiling 结构体使用 uint32_t 保存长度
    if (dataSize64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dataSize = static_cast<uint32_t>(dataSize64);

    // ---------------------------------------------------------
    // 2. 计算使用的 AI Core 数量
    // ---------------------------------------------------------
    uint32_t blockDim = 1;
    uint32_t blockLength = 0;
    uint32_t tileLength = 0;

    if (dataSize > 0) {
        /*
         * 每个 AI Core 至少分配一个 ALIGN_NUM 数据块。
         *
         * 例如：
         * dataSize = 1024
         * blockDim = 8
         *
         * dataSize = 33
         * blockDim = 2
         */
        uint32_t alignedBlockCount = CeilDiv(dataSize, ALIGN_NUM);

        blockDim = std::min(MAX_BLOCK_DIM, alignedBlockCount);

        // 防止 blockDim 为 0
        blockDim = std::max(blockDim, 1U);

        // -----------------------------------------------------
        // 3. 计算每个核处理的数据长度
        // -----------------------------------------------------
        uint32_t rawBlockLength = CeilDiv(dataSize, blockDim);

        // 每个核的起始位置按照对齐后的 blockLength 计算
        blockLength = AlignUp(rawBlockLength, ALIGN_NUM);

        // -----------------------------------------------------
        // 4. 计算单个 Tile 的长度
        // -----------------------------------------------------
        tileLength = std::min(DEFAULT_TILE_LENGTH, blockLength);
    }

    // ---------------------------------------------------------
    // 5. 写入 Tiling 参数
    // ---------------------------------------------------------
    tiling->size = dataSize;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    // 设置实际启动的 AI Core 数量
    context->SetBlockDim(blockDim);

    // 本算子不需要额外的 Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);

    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);

    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输出 Shape 与输入 Shape 相同
    *yShape = *xShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // 输出数据类型与输入数据类型相同
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT,
                ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT,
                ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore()
            .AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);

}  // namespace ops
