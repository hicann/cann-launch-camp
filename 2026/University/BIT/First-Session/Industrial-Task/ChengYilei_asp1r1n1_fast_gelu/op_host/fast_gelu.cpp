// Host侧Tiling实现
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
        uint32_t ubDataNumber = 3;
        uint32_t BLOCK_SIZE = 32;
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType(); // 获取数据类型
        int dtype_size_x = ge::GetSizeByDataType(dtype_x); // 获取数据类型的字长
        uint64_t length_x = static_cast<uint64_t>(tensor_x->GetShapeSize()); // 获取元素个数（64位防大张量溢出）

        uint32_t tileBlockNum = (ub_size / BLOCK_SIZE ) / ubDataNumber; //每个tile的块数
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / dtype_size_x; //每个tile的元素数
        uint64_t totalBytes = static_cast<uint64_t>(length_x) * dtype_size_x; //输入总字节数（64位乘法防溢出）
        uint32_t totalBlocks = static_cast<uint32_t>((totalBytes + BLOCK_SIZE - 1) / BLOCK_SIZE); //输入所需总块数

        coreNum = std::min(coreNum, totalBlocks);  // 别开多余的核
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));
        uint32_t tailBlockNum = totalBlocks % coreNum; //大核数
        uint32_t smallBlocks = totalBlocks / coreNum; //小核处理块数
        uint32_t bigBlocks = smallBlocks + 1; //大核处理块数

        uint32_t smallCoreDataNum = smallBlocks * BLOCK_SIZE / dtype_size_x; //小核处理总元素数
        uint32_t bigCoreDataNum = bigBlocks * BLOCK_SIZE / dtype_size_x; //大核处理总元素数

        uint32_t smallTileNum = smallBlocks / tileBlockNum;//小核的tile数
        uint32_t finalSmallTileNum = (smallCoreDataNum % tileDataNum == 0) ? smallTileNum : smallTileNum + 1;

        uint32_t bigTileNum = bigBlocks / tileBlockNum;//大核的tile数
        uint32_t finalBigTileNum = (bigCoreDataNum % tileDataNum == 0) ? bigTileNum : bigTileNum + 1;
        
        uint32_t smallTailDataNum = smallCoreDataNum % tileDataNum;  //小核尾部处理的元素数
        smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum; 
        uint32_t bigTailDataNum = bigCoreDataNum % tileDataNum;  //大核尾部处理的元素数
        bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum; 

        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->bigCoreDataNum    = bigCoreDataNum;
        tiling->smallCoreDataNum  = smallCoreDataNum;
        tiling->tileDataNum       = tileDataNum;
        tiling->finalBigTileNum   = finalBigTileNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->bigTailDataNum    = bigTailDataNum;
        tiling->smallTailDataNum  = smallTailDataNum;
        tiling->tailBlockNum      = tailBlockNum;

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
