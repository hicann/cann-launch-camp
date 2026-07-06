#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        // 获取硬件平台实例
        auto platformInst = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t aicCoreNum = platformInst.GetCoreNum();

        // 统计输入张量的总元素数量
        uint32_t totalElemCount = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

        // 获取单个元素占用的字节大小
        uint32_t elemByteSize = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), elemByteSize);

        // 按32字节为单位做对齐处理，计算对齐后的总数据块数
        const uint32_t ALIGN_UNIT_BYTES = 32;
        uint32_t totalByteLen = totalElemCount * elemByteSize;
        uint32_t alignedByteLen = ((totalByteLen + ALIGN_UNIT_BYTES - 1) / ALIGN_UNIT_BYTES) * ALIGN_UNIT_BYTES;
        uint32_t totalAlignBlocks = alignedByteLen / ALIGN_UNIT_BYTES;

        // 自适应调整启用的核数：不超过总数据块数，最少保留1个核
        aicCoreNum = std::min(aicCoreNum, totalAlignBlocks);
        aicCoreNum = std::max(aicCoreNum, static_cast<uint32_t>(1));

        // 核间数据均分：基础块数 + 余数块由前N个核额外承担
        uint32_t baseBlocksPerCore = totalAlignBlocks / aicCoreNum;
        uint32_t remainBlockCount = totalAlignBlocks % aicCoreNum;
        context->SetBlockDim(aicCoreNum);

        // 获取单个AI Core的UB缓存总容量
        uint64_t ubCapacity = 0;
        platformInst.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubCapacity);

        // 估算单元素在UB中占用的空间：双缓冲队列 + 输入输出两路 + float临时计算缓存
        const uint32_t PIPE_BUF_DEPTH = 2;
        const uint32_t IO_QUEUE_NUM = 2;
        const uint32_t TMP_BUF_NUM = 2;

        uint32_t ubBytePerElem =
            IO_QUEUE_NUM * PIPE_BUF_DEPTH * elemByteSize + TMP_BUF_NUM * sizeof(float);

        // 计算单次tile最多可处理的元素数，并按32字节对齐
        uint32_t maxElemPerTile = static_cast<uint32_t>(ubCapacity / ubBytePerElem);
        uint32_t elemPerAlignBlock = ALIGN_UNIT_BYTES / elemByteSize;
        uint32_t baseTileElemCount = (maxElemPerTile / elemPerAlignBlock) * elemPerAlignBlock;
        baseTileElemCount = std::max(baseTileElemCount, elemPerAlignBlock);

        uint32_t tileAlignBlocks = baseTileElemCount * elemByteSize / ALIGN_UNIT_BYTES;
        tileAlignBlocks = std::max(tileAlignBlocks, static_cast<uint32_t>(1));

        // 计算基础核（无余数块）的处理参数
        uint32_t baseCoreElemTotal = baseBlocksPerCore * ALIGN_UNIT_BYTES / elemByteSize;
        uint32_t baseTileLoopBase = baseBlocksPerCore / tileAlignBlocks;
        uint32_t baseCoreTileTotal =
            (baseBlocksPerCore % tileAlignBlocks) == 0 ? baseTileLoopBase : baseTileLoopBase + 1;
        baseCoreTileTotal = std::max(baseCoreTileTotal, static_cast<uint32_t>(1));

        uint32_t baseTailElemCount = baseCoreElemTotal - baseTileElemCount * baseTileLoopBase;
        baseTailElemCount = baseTailElemCount == 0 ? baseTileElemCount : baseTailElemCount;

        // 计算满载核（带余数块）的处理参数：比基础核多1个对齐块
        baseBlocksPerCore += 1;
        uint32_t fullCoreElemTotal = baseBlocksPerCore * ALIGN_UNIT_BYTES / elemByteSize;
        uint32_t fullTileLoopBase = baseBlocksPerCore / tileAlignBlocks;
        uint32_t fullCoreTileTotal =
            (baseBlocksPerCore % tileAlignBlocks) == 0 ? fullTileLoopBase : fullTileLoopBase + 1;
        fullCoreTileTotal = std::max(fullCoreTileTotal, static_cast<uint32_t>(1));

        uint32_t fullTailElemCount = fullCoreElemTotal - baseTileElemCount * fullTileLoopBase;
        fullTailElemCount = fullTailElemCount == 0 ? baseTileElemCount : fullTailElemCount;

        // 将切分参数写入tiling结构体，供Kernel侧读取使用
        LogSigmoidCustomTilingData* tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
        tiling->baseCoreElemTotal = baseCoreElemTotal;
        tiling->fullCoreElemTotal = fullCoreElemTotal;
        tiling->fullCoreTileTotal = fullCoreTileTotal;
        tiling->baseCoreTileTotal = baseCoreTileTotal;
        tiling->baseTileElemCount = baseTileElemCount;
        tiling->baseTailElemCount = baseTailElemCount;
        tiling->fullTailElemCount = fullTailElemCount;
        tiling->remainBlockCount = remainBlockCount;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        // 逐元素算子，输出shape与输入完全一致
        const gert::Shape* xShape = context->GetInputShape(0);
        gert::Shape* yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        // 输出数据类型与输入保持一致
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class LogSigmoidCustom : public OpDef {
    public:
        explicit LogSigmoidCustom(const char* name) : OpDef(name)
        {
            // 注册输入x：必选参数，支持ND格式下三种数据类型
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });

            // 注册输出y：类型与格式与输入保持一致
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });

            // 绑定shape与数据类型推导函数
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            // 绑定AI Core切分函数，声明支持的硬件型号
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };

    // 注册算子到系统
    OP_ADD(LogSigmoidCustom);
}