// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"


#include "../op_kernel/gelu_tiling.h"

#include "../op_kernel/tiling_key_gelu.h"


constexpr int32_t BUFFER_NUM = 2;

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {


        // 示例: 获取平台信息
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        auto coreNum = ascendcPlatform.GetCoreNum();
        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        // 示例: 获取算子输入数组信息
        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        // ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType(); // 获取数据类型
        typeLength = ge::GetSizeByDataType(dtype_input_x); // 获取数据类型的字长

        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        // auto dtype_input_x = context->GetInputDesc(0)->GetDataType();
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        uint32_t inputLength = inputNum * typeLength;
        const uint32_t BLOCK_SIZE = 32;
        uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
        uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
        
        // 根据数据类型调整 ubDataNumber
        // uint32_t ubDataNumber = (typeLength == 2) ? 3 : 4; 
        uint32_t ubDataNumber = 3;
        uint32_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber / BUFFER_NUM;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
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
        
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->tailBlockNum = tailBlockNum;

        // 配置启动核数
        context->SetBlockDim(coreNum);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;

    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
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

