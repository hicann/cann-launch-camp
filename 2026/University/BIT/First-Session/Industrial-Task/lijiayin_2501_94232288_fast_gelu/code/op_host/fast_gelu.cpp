// Host侧Tiling实现 - 优化版
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // ========== 1. 获取平台信息 ==========
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t coreNum = platform.GetCoreNumAiv();
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // ========== 2. 获取输入Tensor信息 ==========
        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        ge::DataType dtypeX = tensorX->GetDataType();
        uint32_t typeSize = ge::GetSizeByDataType(dtypeX);
        uint32_t totalLength = tensorX->GetShapeSize();

        // ========== 3. 配置Tiling Key ==========
        uint32_t DT_X = static_cast<uint32_t>(dtypeX);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        const uint32_t BLOCK_SIZE = 32;
        const uint32_t BUFFER_NUM = 2;

        // ========== 4. 计算核数 ==========
        uint32_t totalBytes = totalLength * typeSize;
        // 按32B对齐
        uint32_t alignTotalBytes = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        uint32_t alignBlockNum = alignTotalBytes / BLOCK_SIZE;

        // 每个核至少处理1个block
        uint32_t usedCoreNum = std::min(static_cast<uint32_t>(coreNum), alignBlockNum);
        usedCoreNum = std::max(usedCoreNum, 1u);
        context->SetBlockDim(usedCoreNum);

        // ========== 5. 精确计算Tile大小 (最大化UB利用率) ==========
        uint32_t ubBlockNum = ubSize / BLOCK_SIZE;

        // UB分配计算:
        //   - 输入队列(双缓冲): BUFFER_NUM * tileDataNum * typeSize
        //   - 输出队列(双缓冲): BUFFER_NUM * tileDataNum * typeSize
        //   - FasterGelu临时buffer: tmpSize
        // 总计: (BUFFER_NUM * 2) * tileDataNum * typeSize + tmpSize
        //
        // 临时buffer估算: FasterGelu内部按float精度计算,
        //   对于float输入: tmpSize ≈ tileDataNum * sizeof(float) * 2 = tileDataNum * 8
        //   对于half输入:  内部提升到float, tmpSize ≈ tileDataNum * sizeof(float) * 2 = tileDataNum * 8
        // 所以: tmpSize = tileDataNum * 8 (向上32B对齐)
        //
        // 代入: 4 * tileDataNum * typeSize + tileDataNum * 8 <= ubSize
        //   => tileDataNum * (4 * typeSize + 8) <= ubSize
        //   => tileDataNum = ubSize / (4 * typeSize + 8)  (需32B对齐)

        // 临时buffer占用的block数 per element: 8 / 32 = 0.25
        // IO占用的block数 per element: 4 * typeSize / 32
        // 合计 per element: (4 * typeSize + 8) / 32
        // 最大tile元素数 = ubBlockNum * 32 / (4 * typeSize + 8)

        // 使用整数计算避免浮点
        uint32_t tileDataNum = ubBlockNum * BLOCK_SIZE / (BUFFER_NUM * 2 * typeSize + 8);
        // 32B对齐: tileDataNum * typeSize 必须是32的倍数
        uint32_t alignUnit = BLOCK_SIZE / typeSize;  // 对于float=8, 对于half=16
        tileDataNum = (tileDataNum / alignUnit) * alignUnit;
        tileDataNum = std::max(tileDataNum, alignUnit);  // 至少能传一个block

        // 计算临时buffer大小
        uint32_t tmpSize = tileDataNum * 8;  // float内部精度, 8 bytes per element
        tmpSize = (tmpSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        // UB溢出保护: 如果计算结果超出UB, 缩小tile
        uint32_t totalUbNeeded = BUFFER_NUM * 2 * tileDataNum * typeSize + tmpSize;
        while (totalUbNeeded > ubSize && tileDataNum > alignUnit) {
            tileDataNum -= alignUnit;
            tmpSize = tileDataNum * 8;
            tmpSize = (tmpSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            totalUbNeeded = BUFFER_NUM * 2 * tileDataNum * typeSize + tmpSize;
        }

        // ========== 6. 大/小核负载均衡 ==========
        // 按32B block分配: 每核baseBlockNum个block, 前tailBlockNum个核多1个block
        uint32_t baseBlockNum = alignBlockNum / usedCoreNum;
        uint32_t tailBlockNum = alignBlockNum % usedCoreNum;

        // --- 小核参数 ---
        uint32_t smallCoreBlockNum = baseBlockNum;
        uint32_t smallCoreDataNum = smallCoreBlockNum * BLOCK_SIZE / typeSize;
        uint32_t smallTileNum = smallCoreBlockNum / (tileDataNum * typeSize / BLOCK_SIZE);
        // 换算: 每个tile占 tileDataNum * typeSize / BLOCK_SIZE 个block
        uint32_t tileBlockNum = tileDataNum * typeSize / BLOCK_SIZE;
        uint32_t smallRemainBlock = smallCoreBlockNum % tileBlockNum;
        uint32_t finalSmallTileNum = (smallRemainBlock == 0) ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
        smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

        // --- 大核参数(比小核多1个block) ---
        uint32_t bigCoreBlockNum = baseBlockNum + 1;
        uint32_t bigCoreDataNum = bigCoreBlockNum * BLOCK_SIZE / typeSize;
        uint32_t bigTileNum = bigCoreBlockNum / tileBlockNum;
        uint32_t bigRemainBlock = bigCoreBlockNum % tileBlockNum;
        uint32_t finalBigTileNum = (bigRemainBlock == 0) ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

        // ========== 7. 填充Tiling结构体 ==========
        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->length = totalLength;
        tiling->blockNum = usedCoreNum;
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->smallTileNum = finalSmallTileNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->bigTileNum = finalBigTileNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->tailBlockNum = tailBlockNum;
        tiling->tmpSize = tmpSize;

        // ========== 8. 配置Workspace ==========
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class FastGelu : public OpDef {
    public:
        explicit FastGelu(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}  // namespace ops
