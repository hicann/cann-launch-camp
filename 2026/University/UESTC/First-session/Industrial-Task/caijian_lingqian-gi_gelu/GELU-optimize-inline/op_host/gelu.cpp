// Host 侧 Tiling 实现 — EleWise 规范版
// 多核对齐 512 元素，UB 对齐 256B，动态核数分配，区分首/尾 block
#include "register/op_def_registry.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
        ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);
        uint32_t elemSize = (dtype == ge::DT_FLOAT16) ? 2 : 4;

        // ---- 常量 ----
        constexpr uint32_t BLOCK_ALIGN_BYTES = 256;   // 多核对齐粒度（字节，与 UB_ALIGN_BYTES 一致）
        constexpr uint32_t UB_ALIGN_BYTES = 256;      // UB 对齐粒度（字节）
        constexpr uint32_t MIN_CORE_BYTES = 8192;     // 每核最少处理字节数（v29: 4096→8192，增大每核负载摊薄 SCALAR 开销）
        constexpr uint32_t MAX_CORE_NUM = 24;         // 910B4 最大核数
        constexpr uint32_t UB_SIZE = 192 * 1024;      // UB 总容量 192KB

        // 1. 动态计算核数：ceil(totalBytes / MIN_CORE_BYTES)
        uint64_t totalBytes = static_cast<uint64_t>(totalLength) * elemSize;
        uint32_t coreNum = static_cast<uint32_t>((totalBytes + MIN_CORE_BYTES - 1) / MIN_CORE_BYTES);
        if (coreNum < 1) coreNum = 1;
        if (coreNum > MAX_CORE_NUM) coreNum = MAX_CORE_NUM;
        context->SetBlockDim(coreNum);

        // 2. 每核基础元素数，256 字节对齐（与 UB 对齐粒度一致）
        uint32_t alignElements = BLOCK_ALIGN_BYTES / elemSize;
        uint32_t blockFormer = ((totalLength + coreNum - 1) / coreNum + alignElements - 1) / alignElements * alignElements;
        if (blockFormer < alignElements) blockFormer = alignElements;

        // 3. 总 block 数
        uint32_t blockNum = (totalLength + blockFormer - 1) / blockFormer;
        if (blockNum < 1) blockNum = 1;

        // 4. UB 切分：按总 UB 容量计算最大元素数
        //    TQue 深度=2 (双缓冲): 4*sizeof(T) + tmpBufs*sizeof(float) 字节/元素
        //    float16: 4*2 + 3*4 = 20B/elem (3 tmpBufs: xFp32, bufA/y, bufB)
        //    float32: 4*4 + 2*4 = 24B/elem (2 tmpBufs: bufA, bufB; v20 条件省 tmpBuf3)。留 5% 余量
        constexpr uint32_t UB_MARGIN_NUMER = 95;
        constexpr uint32_t UB_MARGIN_DENOM = 100;
        uint32_t bufferDivisor = (dtype == ge::DT_FLOAT16) ? 20 : 24;
        uint32_t maxUbElements = UB_SIZE * UB_MARGIN_NUMER / UB_MARGIN_DENOM / bufferDivisor;

        uint32_t ubAlignElements = UB_ALIGN_BYTES / elemSize;
        uint32_t ubFormer = (maxUbElements / ubAlignElements) * ubAlignElements;
        if (ubFormer < ubAlignElements) ubFormer = ubAlignElements;
        if (ubFormer > blockFormer) ubFormer = blockFormer;
        if (ubFormer < 1) ubFormer = 1;

        // 5. 首 block 的 UB 循环与尾段
        uint32_t ubLoopOfFormerBlock = blockFormer / ubFormer;
        uint32_t ubTailOfFormerBlock = blockFormer % ubFormer;

        // 6. 末 block 的 UB 循环与尾段（末 block 可能比 blockFormer 短）
        uint32_t tailBlockElements = totalLength - (blockNum - 1) * blockFormer;
        uint32_t ubLoopOfTailBlock = tailBlockElements / ubFormer;
        uint32_t ubTailOfTailBlock = tailBlockElements % ubFormer;

        // 填充 Tiling 数据
        GeluTilingData *t = context->GetTilingData<GeluTilingData>();
        t->dim0 = totalLength;
        t->coreNum = coreNum;
        t->blockFormer = blockFormer;
        t->blockNum = blockNum;
        t->ubFormer = ubFormer;
        t->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
        t->ubTailOfFormerBlock = ubTailOfFormerBlock;
        t->ubLoopOfTailBlock = ubLoopOfTailBlock;
        t->ubTailOfTailBlock = ubTailOfTailBlock;

        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        size_t *ws = context->GetWorkspaceSizes(1);
        if (ws) ws[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x").ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output").ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}  // namespace ops
