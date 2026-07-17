#include "../op_kernel/gelu_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE_BYTE = 32;
constexpr uint32_t MAX_UB_TILE_BYTE = 96 * 1024;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GeluTilingData tiling;

    const gert::StorageShape* xShapePtr = context->GetInputShape(0);
    if (xShapePtr == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape& xShape = xShapePtr->GetStorageShape();

    uint64_t totalLengthU64 = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i)
        totalLengthU64 *= xShape.GetDim(i);

    uint32_t totalLength = static_cast<uint32_t>(totalLengthU64);
    if (totalLength == 0) totalLength = 1;

    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeSize = (dtype == ge::DT_FLOAT16) ? 2U : 4U;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) coreNum = 1;

    uint32_t alignNum = BLOCK_SIZE_BYTE / dtypeSize;   // float32:8, float16:16
    if (alignNum == 0) alignNum = 1;

    // 多核切分：former 部分保持 32B 对齐，尾核处理剩余部分
    uint32_t avgLength = (totalLength + coreNum - 1) / coreNum;
    uint32_t alignLength = ((avgLength + alignNum - 1) / alignNum) * alignNum;
    if (alignLength == 0) alignLength = alignNum;

    uint32_t actualCoreNum = (totalLength + alignLength - 1) / alignLength;
    if (actualCoreNum == 0) actualCoreNum = 1;
    if (actualCoreNum > coreNum) actualCoreNum = coreNum;

    uint32_t formerNum = 0, formerLength = alignLength;
    uint32_t tailNum = 1, tailLength = totalLength;

    if (actualCoreNum > 1) {
        formerNum = actualCoreNum - 1;
        uint32_t formerTotal = formerNum * formerLength;
        if (formerTotal >= totalLength) {
            formerNum = 0; formerLength = 0;
            tailLength = totalLength; actualCoreNum = 1;
        } else {
            tailLength = totalLength - formerTotal;
        }
    }

    /*
     * UB 预算（每元素字节数）：
     * float32：inQueue(2*4) + outQueue(2*4) + mask(1) + negZero(4) = 21
     * float16：inQueue(2*2) + outQueue(2*2) + mask(1) + negZero(2) + floatTmp(4) = 15
     * 统一用 6*dtypeSize + 5 确保不溢出。
     */
    uint32_t bytesPerElemBudget = 6 * dtypeSize + 5;
    if (dtype == ge::DT_FLOAT16) bytesPerElemBudget += 4;   // float 临时区
    if (bytesPerElemBudget == 0) bytesPerElemBudget = 1;

    uint32_t tileLength = MAX_UB_TILE_BYTE / bytesPerElemBudget;
    tileLength = (tileLength / alignNum) * alignNum;
    if (tileLength == 0) tileLength = alignNum;

    tiling.set_totalLength(totalLength);
    tiling.set_formerNum(formerNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailNum(tailNum);
    tiling.set_tailLength(tailLength);
    tiling.set_tileLength(tileLength);

    context->SetBlockDim(actualCoreNum);

    if (dtype == ge::DT_FLOAT16) context->SetTilingKey(2);
    else context->SetTilingKey(1);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) return GRAPH_FAILED;
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char* name) : OpDef(name) {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops