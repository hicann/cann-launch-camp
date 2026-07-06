
#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto coreNum = ascendcPlatform.GetCoreNum();

        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        auto dataType = context->GetInputDesc(0)->GetDataType();
        uint32_t typeSize = 0;
        ge::TypeUtils::GetDataTypeLength(dataType, typeSize);
        const uint32_t BLOCK_SIZE = 32;

        // 单核
        coreNum = 1;
        context->SetBlockDim(coreNum);

        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        const uint32_t BUFFER_NUM = 2;

        // 基础缓冲区：输入队列2 + 输出队列2 + tmpOne + tmpNegX = 6 个 T 类型
        uint32_t baseBuffers = BUFFER_NUM * 2 + 2;
        // bfloat16 需要 4 个 float 缓冲区（floatX, floatY, floatOne, floatNegX）
        uint32_t extraFloatBuffers = (dataType == ge::DT_BF16) ? 4 : 0;
        uint32_t bytesPerElement = baseBuffers * typeSize + extraFloatBuffers * sizeof(float);
        uint32_t maxTileElems = ubSize / bytesPerElement;
        if (maxTileElems == 0) maxTileElems = 1;

        // 按 32B 对齐（向下）
        uint32_t maxTileBytes = (maxTileElems * typeSize) / BLOCK_SIZE * BLOCK_SIZE;
        uint32_t tileDataNum = maxTileBytes / typeSize;
        if (tileDataNum == 0) tileDataNum = 1;

        // 计算 tile 个数和最后一个 tile 的数据数
        uint32_t tileNum = (inputNum + tileDataNum - 1) / tileDataNum;
        uint32_t tailDataNum = inputNum % tileDataNum;
        if (tailDataNum == 0) tailDataNum = tileDataNum;

        // 单核时 small/big 一致
        uint32_t coreDataNum = inputNum;
        uint32_t finalTileNum = tileNum;

        auto* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
        tiling->smallCoreDataNum = coreDataNum;
        tiling->bigCoreDataNum = coreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = tailDataNum;
        tiling->bigTailDataNum = tailDataNum;
        tiling->finalSmallTileNum = finalTileNum;
        tiling->finalBigTileNum = finalTileNum;
        tiling->tailBlockNum = 0; // 单核无尾巴

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
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
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LogSigmoidCustom);
}  // namespace ops
