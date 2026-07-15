#include <algorithm>

#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        // 获取平台核心数和输入规模，后续按 32B block 对齐进行切分。
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNum();
        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint32_t inputLength = inputNum * typeLength;

        const uint32_t BLOCK_SIZE = 32;
        uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        uint32_t maxCoreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);

        // 小规模输入减少并行核数，避免调度和尾块开销过高。
        uint32_t minElemsPerCore = 512;
        if (typeLength == 2) {
            minElemsPerCore = inputNum <= 4096 ? 512 : 1024;
        } else {
            minElemsPerCore = inputNum <= 8192 ? 256 : 512;
        }
        uint32_t plannedCoreNum = (inputNum + minElemsPerCore - 1) / minElemsPerCore;
        coreNum = std::min(maxCoreNum, plannedCoreNum);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));

        uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
        context->SetBlockDim(coreNum);

        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        // UB 中预留输入、输出和临时计算空间；双 buffer 路径按 BUFFER_NUM 估算 tile 容量。
        const uint32_t BUFFER_NUM = 2;
        uint32_t ubDataNumber = 5;
        uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
        tileBlockNum = tileBlockNum == 0 ? 1 : tileBlockNum;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

        // 先计算小核配置，再在 block 数量加一后得到大核配置。
        uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

        everyCoreInputBlockNum += 1;
        uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        // tile 数很少时使用 TBuf 路径，减少队列管理开销。
        bool useTbufPath = finalBigTileNum <= 2 && finalSmallTileNum <= 2;

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->tailBlockNum = tailBlockNum;

        uint32_t DT_INPUT_X = static_cast<uint32_t>(context->GetInputDesc(0)->GetDataType());
        // 根据输入 dtype 和 tiling 策略选择 Device 侧模板实例。
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X, useTbufPath);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        // Gelu 为逐元素算子，输出 shape 与输入保持一致。
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        // 输出 dtype 与输入 dtype 保持一致。
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Gelu : public OpDef
    {
    public:
        explicit Gelu(const char *name) : OpDef(name)
        {
            // 注册算子输入输出约束，并绑定 shape/dtype 推导和 tiling 函数。
            this->Input("self")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
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
