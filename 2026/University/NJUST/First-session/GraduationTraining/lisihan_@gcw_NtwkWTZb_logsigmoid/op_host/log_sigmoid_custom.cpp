#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "../op_kernel/tiling_key_log_sigmoid_custom.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // ---- 步骤1：获取平台信息（UB大小、可用核数） ----
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();

    // ---- 步骤2：获取输入数据大小（元素数 × 单元素字节数） ----
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint32_t inputLength = inputNum * typeLength;

    // ---- 步骤3：获取数据类型用于模板参数赋值 ----
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    ge::DataType dtype_y = context->GetOutputDesc(0)->GetDataType();
    uint32_t D_T_X = static_cast<int>(dtype_x);  // float16/float/bf16
    uint32_t D_T_Y = static_cast<int>(dtype_y);  // 输出类型与输入相同

    // ---- 步骤4：32字节对齐 ----
    // 硬件要求UB上数据必须32B对齐，以32B为最小处理粒度
    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLengthAlgin32 =
        (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);

    // ---- 步骤5：核间数据拆分 ----
    // 确定实际使用的核数（每个核至少处理1个32B块）
    coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    // 每个核基础的32B块数，以及余数（前tailBlockNum个核各多分1块，成为"大核"）
    uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
    context->SetBlockDim(coreNum);

    // ---- 步骤6：核内数据切分（受UB大小约束） ----
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // LogSigmoid算子UB同时驻留：1个输入队列 + 1个输出队列 + 2个TBuf = 4块
    // bf16类型：队列(2B) + TBuf(float=4B) = 等效6块（TBuf占双倍空间）
    uint32_t ubDataNumber = (dtype_x == ge::DT_BF16) ? 6 : 4;
    // 单buffer最多容纳的32B块数与对应的元素个数
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

    // ---- 步骤7：小核的tile批次和尾块计算 ----
    // smallCoreDataNum = 每个核基础的32B块数 × 32B / 单元素字节数
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    // 若不能整除需要多一次循环处理剩余数据
    uint32_t finalSmallTileNum =
        (everyCoreInputBlockNum % tileBlockNum == 0)
            ? smallTileNum : smallTileNum + 1;
    // 尾块元素数（整除时尾块等于tile大小，否则为余数）
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = (smallTailDataNum == 0) ? tileDataNum : smallTailDataNum;

    // ---- 步骤8：大核的tile批次和尾块计算 ----
    // 大核比小核多1个32B块
    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum =
        (everyCoreInputBlockNum % tileBlockNum == 0)
            ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = (bigTailDataNum == 0) ? tileDataNum : bigTailDataNum;

    // ---- 步骤9：将Tiling参数写入结构体 ----
    LogSigmoidCustomTilingData *tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();
    tiling->smallCoreDataNum  = smallCoreDataNum;
    tiling->bigCoreDataNum    = bigCoreDataNum;
    tiling->tileDataNum       = tileDataNum;
    tiling->smallTailDataNum  = smallTailDataNum;
    tiling->bigTailDataNum    = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum   = finalBigTileNum;
    tiling->tailBlockNum      = tailBlockNum;

    // ---- 步骤10：配置模板参数（自动生成TilingKey） ----
    ASCENDC_TPL_SEL_PARAM(context, D_T_X, D_T_Y);

    // ---- 步骤11：设置workspace大小 ----
    // 系统workspace（框架API预留）
    size_t sysWorkspaceSize =
        static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // 用户workspace：预留单tile的float临时数据空间
    size_t usrWorkspaceSize = tileDataNum * sizeof(float);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize + usrWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling


namespace ge {
// ---- Shape推导函数 ----
// LogSigmoid是逐元素操作，输出shape与输入shape相同
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

// ---- Dtype推导函数 ----
// 输出数据类型与输入数据类型一致
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
// ---- 算子原型注册 ----
// 注册LogSigmoidCustom算子，支持ND格式的float16/float32/bf16类型
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        // 输入x：支持三种数据类型，ND格式
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        // 输出y：数据类型与输入一致，ND格式
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);
}  // namespace ops
