// Host侧Tiling实现：小输入单核单发 + 大输入多核流水（双 kernel 运行时分派）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        uint32_t ubDataNumber = 3;   // 同时驻留 x / 中间 / y 三块 buffer
        uint32_t BLOCK_SIZE = 32;

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        uint32_t tileBlockNum = (ub_size / BLOCK_SIZE) / ubDataNumber;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtype_size_x;   // 单核单 tile 容量（3 buffer 平分 UB 后每块元素数）
        uint32_t totalBytes = length_x * dtype_size_x;
        uint32_t totalBlocks = (totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE;

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->totalLength = length_x;

        // 小输入判据：整块能塞进单核一个 tile（基线大 kernel 已在用同样的 3×tileDataNum，验证过不溢出 UB）→ 单核单发
        if (length_x > 0 && length_x <= tileDataNum) {
            tiling->isSmallShape = 1;
            // 单发 buffer 用「向上对齐到 32B 的元素数」；DataCopyPad 只搬 length_x 个
            tiling->tileDataNum = (totalBlocks * BLOCK_SIZE) / dtype_size_x;
            tiling->bigCoreDataNum    = 0;
            tiling->smallCoreDataNum  = 0;
            tiling->finalBigTileNum   = 0;
            tiling->finalSmallTileNum = 0;
            tiling->bigTailDataNum    = 0;
            tiling->smallTailDataNum  = 0;
            tiling->tailBlockNum      = 0;
            context->SetBlockDim(1);   // 关键：小输入只起 1 个核，固定开销只付一次
        } else {
            // 大输入：沿用验证过的多核负载均衡方案（原样保留）
            tiling->isSmallShape = 0;
            coreNum = std::min(coreNum, totalBlocks);  // 别开多余的核
            coreNum = std::max(coreNum, static_cast<uint32_t>(1));
            uint32_t tailBlockNum = totalBlocks % coreNum; //大核数
            uint32_t smallBlocks = totalBlocks / coreNum;  //小核处理块数
            uint32_t bigBlocks = smallBlocks + 1;          //大核处理块数

            uint32_t smallCoreDataNum = smallBlocks * BLOCK_SIZE / dtype_size_x; //小核处理总元素数
            uint32_t bigCoreDataNum = bigBlocks * BLOCK_SIZE / dtype_size_x;     //大核处理总元素数

            uint32_t smallTileNum = smallBlocks / tileBlockNum;
            uint32_t finalSmallTileNum = (smallCoreDataNum % tileDataNum == 0) ? smallTileNum : smallTileNum + 1;
            uint32_t bigTileNum = bigBlocks / tileBlockNum;
            uint32_t finalBigTileNum = (bigCoreDataNum % tileDataNum == 0) ? bigTileNum : bigTileNum + 1;

            uint32_t smallTailDataNum = smallCoreDataNum % tileDataNum;
            smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;
            uint32_t bigTailDataNum = bigCoreDataNum % tileDataNum;
            bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

            tiling->bigCoreDataNum    = bigCoreDataNum;
            tiling->smallCoreDataNum  = smallCoreDataNum;
            tiling->tileDataNum       = tileDataNum;
            tiling->finalBigTileNum   = finalBigTileNum;
            tiling->finalSmallTileNum = finalSmallTileNum;
            tiling->bigTailDataNum    = bigTailDataNum;
            tiling->smallTailDataNum  = smallTailDataNum;
            tiling->tailBlockNum      = tailBlockNum;
            context->SetBlockDim(coreNum);
        }

        // 配置 tiling key（只按数据类型区分）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 配置 workspace 大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x = context->GetInputShape(0);
        *context->GetOutputShape(0) = *x;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
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
