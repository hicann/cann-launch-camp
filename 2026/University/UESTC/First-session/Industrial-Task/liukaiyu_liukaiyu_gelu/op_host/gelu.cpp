// ==========================================================
//  Host侧Tiling实现 — GELU激活函数 (分级tile + 大核/小核)
// ==========================================================
//  策略:
//    1. 核间: 按256B对齐均分字节 → 反推核数, 余数摊入大核
//      前tailBlockNum个核 = 大核 bigCoreElems (多1元素)
//      剩余核 = 小核 smallCoreElems
//    2. 核内: tileDataNum按总数据量分级设定
//       (受UB容量约束, 分级表中系数可调)
//      尾tile做32B上对齐
//    3. GELU公式: gelu(x) = 0.5 * x * (1 + erf(x / √2))

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // ---- 1. 获取平台 & 输入信息 ----
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t totalCoreNum = platform.GetCoreNumAiv();

        const gert::Tensor *tensorInput = context->GetRequiredInputTensor(0);
        ge::DataType dtype = tensorInput->GetDataType();
        uint32_t typeLen   = ge::GetSizeByDataType(dtype);
        uint32_t totalElems = tensorInput->GetShapeSize();

        // ---- 2. TilingKey ----
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // ---- 3. 32B对齐粒度 ----
        const uint32_t alignElems = 32 / typeLen;     // fp32→8, fp16→16

        // ---- 4. 核数确定: 256B对齐均分字节反推 ----
        uint32_t totalBytes = totalElems * typeLen;
        uint32_t avgBytes = (totalBytes + totalCoreNum - 1) / totalCoreNum;
        uint32_t alignedAvgBytes = (avgBytes + 255) / 256 * 256;
        uint32_t coreNum = (totalBytes + alignedAvgBytes - 1) / alignedAvgBytes;
        if (coreNum > totalCoreNum) coreNum = totalCoreNum;
        if (coreNum < 1) coreNum = 1;

        // ---- 5. tileDataNum: 按总数据量分级 (DEFAULT_SLOT * 系数) ----
        const uint32_t DEFAULT_SLOT = alignElems;     // 32B元素数
        uint32_t tileDataNum;
        uint32_t totalDataScale = totalElems * typeLen;   // 总字节数
        if (totalDataScale < 128) {
            tileDataNum = DEFAULT_SLOT * 320;               // 极小
        } else if (totalDataScale < 512) {
            tileDataNum = DEFAULT_SLOT * 1040;              // 小
        } else if (totalDataScale < 2048) {
            tileDataNum = DEFAULT_SLOT * 1040;              // 中小
        } else {
            tileDataNum = DEFAULT_SLOT * 1020;               // 中/大
        }

        // ---- 6. 核间: 大核/小核模型 ----
        uint32_t baseElemsPerCore = totalElems / coreNum;
        uint32_t tailBlockNum     = totalElems % coreNum;       // 大核个数
        uint32_t bigCoreElems     = baseElemsPerCore + (tailBlockNum > 0 ? 1 : 0);
        uint32_t smallCoreElems   = baseElemsPerCore;

        context->SetBlockDim(coreNum);

        // ---- 7. 大核核内切分 ----
        uint32_t bigTileNum = bigCoreElems / tileDataNum;
        uint32_t finalBigTileNum = (bigCoreElems % tileDataNum == 0)
                                   ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreElems - tileDataNum * bigTileNum;
        if (bigTailDataNum == 0) bigTailDataNum = tileDataNum;
        bigTailDataNum = ((bigTailDataNum + alignElems - 1) / alignElems) * alignElems;

        // ---- 8. 小核核内切分 ----
        uint32_t smallTileNum = smallCoreElems / tileDataNum;
        uint32_t finalSmallTileNum = (smallCoreElems % tileDataNum == 0)
                                     ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreElems - tileDataNum * smallTileNum;
        if (smallTailDataNum == 0) smallTailDataNum = tileDataNum;
        smallTailDataNum = ((smallTailDataNum + alignElems - 1) / alignElems) * alignElems;

        // ---- 9. 填充Tiling结构体 ----
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->smallCoreDataNum   = smallCoreElems;
        tiling->bigCoreDataNum     = bigCoreElems;
        tiling->finalBigTileNum    = finalBigTileNum;
        tiling->finalSmallTileNum  = finalSmallTileNum;
        tiling->tileDataNum        = tileDataNum;
        tiling->smallTailDataNum   = smallTailDataNum;
        tiling->bigTailDataNum     = bigTailDataNum;
        tiling->tailBlockNum       = tailBlockNum;
        tiling->bufferNum          = 2;
        tiling->usedCoreNum        = coreNum;

        // ---- 10. workspace ----
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
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
}  // namespace ops