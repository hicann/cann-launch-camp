// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include "graph/utils/type_utils.h"
#include "../op_kernel/gelu_tiling.h"

#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    uint64_t ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);


 
    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
    uint32_t length_input_x = tensor_input_x->GetShapeSize();
 
     std::vector<int64_t> shapeVec = {length_input_x};
    ge::Shape srcShape(shapeVec);
    uint32_t minValue = AscendC::GetGeluMinTmpSize(srcShape, sizeof(float));
    ub_size-=minValue;

    if (length_input_x == 0) {
        length_input_x = 1;
    }
    uint32_t inputLength = length_input_x * dtype_size_input_x;
 
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);
 
    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);
 
    uint32_t ubDataNumber = 16;
    uint32_t tileBlockNum = (ub_size / BLOCK_SIZE) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtype_size_input_x;
 
    // 小核
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / dtype_size_input_x;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
 
    // 大核（比小核多一个 BLOCK）
    uint32_t bigCoreInputBlockNum = everyCoreInputBlockNum + 1;
    uint32_t bigCoreDataNum = bigCoreInputBlockNum * BLOCK_SIZE / dtype_size_input_x;
    uint32_t bigTileNum = bigCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (bigCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - (tileDataNum * bigTileNum);
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
 
    // tilling 结构体填充
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
 
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *intputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *intputShape;
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

