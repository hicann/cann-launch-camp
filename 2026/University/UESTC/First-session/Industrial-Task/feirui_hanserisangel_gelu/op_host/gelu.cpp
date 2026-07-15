#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // ======== 1. 获取平台信息 ========
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();          // AI Core 总数
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);  // UB 大小

    // ======== 2. 获取输入信息 ========
    int64_t  inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();  // 元素个数
    ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = ge::GetSizeByDataType(dtype);   // 每个元素的字节数
    uint64_t inputLength = inputNum * typeLength;          // 输入数据总字节数

    // ======== 3. 配置 TilingKey 模板参数（区分 float16 / float32） ========
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);

    // ======== 4. 32B 对齐 & 核间数据拆分 ========
    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLengthAlgin32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    // 限制核数不超过数据块数，保证每个核至少处理 1 个 32B 块
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));

    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;  // 每个核基础 32B 块数
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;          // 余数块数 → 前 N 核各多拿 1 块
    context->SetBlockDim(coreNum);

    // ======== 5. 核内数据切分（基于 UB 容量） ========
    const uint32_t BUFFER_NUM = 2;
    const uint32_t ubDataNumber = 6;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;  // 单 tile 元素个数
    // tile 扩大至原来的 1.7×（ubDataNumber 7 vs 原始12: 12/7≈1.7×）

    // ======== 5.5 Erf 临时空间（CANN 910b 无 GetErfMaxMinTmpSize，按 1×f32 TBuf 估算） ========
    uint32_t erfTmpBufSize = tileDataNum * sizeof(float);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    // ======== 6. 小核参数 ========
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum == 0)
                                     ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // ======== 7. 大核参数（比小核多 1 个 32B 块） ========
    uint32_t bigEveryCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigEveryCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = bigEveryCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigEveryCoreInputBlockNum % tileBlockNum == 0)
                                   ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // ======== 8. 填充 Tiling 结构体 ========
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->smallCoreDataNum  = smallCoreDataNum;
    tiling->bigCoreDataNum    = bigCoreDataNum;
    tiling->finalBigTileNum   = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum       = tileDataNum;
    tiling->smallTailDataNum  = smallTailDataNum;
    tiling->bigTailDataNum    = bigTailDataNum;
    tiling->tailBlockNum      = tailBlockNum;

    // ======== 9. 配置 workspace（当前不使用） ========
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// ======== Shape 推导 & Dtype 推导 ========
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

// ======== 算子原型注册 ========
namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops

