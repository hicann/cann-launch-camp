

#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>

namespace optiling
{

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreCount = platform.GetCoreNum();

        uint32_t elemNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

        auto dtype = context->GetInputDesc(0)->GetDataType();

        if (dtype != ge::DT_FLOAT16 &&
            dtype != ge::DT_FLOAT &&
            dtype != ge::DT_BF16)
        {
            return ge::GRAPH_FAILED;
        }

        uint32_t dtypeBytes = 0;
        ge::TypeUtils::GetDataTypeLength(dtype, dtypeBytes);

        const uint32_t BLOCK_BYTES = 32;
        uint32_t totalBytes = elemNum * dtypeBytes;
        uint32_t alignedBytes = (totalBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        uint32_t totalBlockNum = alignedBytes / BLOCK_BYTES;

        coreCount = std::min(coreCount, totalBlockNum);
        coreCount = std::max(coreCount, static_cast<uint32_t>(1));
        context->SetBlockDim(coreCount);

        uint32_t baseBlockPerCore = totalBlockNum / coreCount;
        uint32_t extraCoreNum = totalBlockNum % coreCount;

        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const uint32_t BUFFER_NUM = 2;

        uint32_t calcBytesPerElem = 0;
        if (dtype == ge::DT_FLOAT)
        {
            calcBytesPerElem = sizeof(float); // 存 sigmoid 中间结果
        }
        else
        {
            calcBytesPerElem = sizeof(float) * 2; // xFloat + yFloat
        }

        uint32_t ubBytesPerElem =
            BUFFER_NUM * dtypeBytes + // inputQueue
            BUFFER_NUM * dtypeBytes + // outputQueue
            calcBytesPerElem;         // calcBuf

        uint32_t tileElemNum = ubSize / ubBytesPerElem;

        uint32_t elemPerBlock = BLOCK_BYTES / dtypeBytes;

        tileElemNum = tileElemNum / elemPerBlock * elemPerBlock;
        tileElemNum = std::max(tileElemNum, elemPerBlock);

        uint32_t tileBlockNum = tileElemNum / elemPerBlock;

        uint32_t normalBlockPerCore = baseBlockPerCore;
        uint32_t largeBlockPerCore = baseBlockPerCore + 1;

        uint32_t normalCoreElemNum = normalBlockPerCore * elemPerBlock;
        uint32_t largeCoreElemNum = largeBlockPerCore * elemPerBlock;

        uint32_t normalTileNum = (normalBlockPerCore + tileBlockNum - 1) / tileBlockNum;
        uint32_t largeTileNum = (largeBlockPerCore + tileBlockNum - 1) / tileBlockNum;

        uint32_t normalTailElemNum = normalCoreElemNum - (normalTileNum - 1) * tileElemNum;
        uint32_t largeTailElemNum = largeCoreElemNum - (largeTileNum - 1) * tileElemNum;

        LogSigmoidCustomTilingData *tiling =
            context->GetTilingData<LogSigmoidCustomTilingData>();

        tiling->normalCoreElemNum = normalCoreElemNum;
        tiling->largeCoreElemNum = largeCoreElemNum;
        tiling->tileElemNum = tileElemNum;
        tiling->normalTileNum = normalTileNum;
        tiling->largeTileNum = largeTileNum;
        tiling->normalTailElemNum = normalTailElemNum;
        tiling->largeTailElemNum = largeTailElemNum;
        tiling->largeCoreNum = extraCoreNum;

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

}

namespace ge
{

    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        auto inputType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputType);
        return ge::GRAPH_SUCCESS;
    }

}

namespace ops
{

    class LogSigmoidCustom : public OpDef
    {
    public:
        explicit LogSigmoidCustom(const char *name) : OpDef(name)
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

            this->AICore().SetTiling(optiling::TilingFunc);

            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(LogSigmoidCustom);

}