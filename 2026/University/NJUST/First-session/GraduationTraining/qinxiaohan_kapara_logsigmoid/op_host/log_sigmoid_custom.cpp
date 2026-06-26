
#include "../op_kernel/log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>


typedef uint32_t U32;
typedef uint64_t U64;

namespace optiling {
// tiling函数计算每个核处理多少数据、每个tile多大、分多少轮处理
static ge::graphStatus TilingFunc(gert::TilingContext* context){

    const U32 DUMMY_ZERO = 0;
    const U32 DUMMY_ONE = 1;
    U32 dummyCounter = DUMMY_ZERO;

    // 获取昇腾平台的信息，后面要拿核数和UB内存大小
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());


    auto coreNum = ascendcPlatform.GetCoreNum();
    U32 coreNumTmp = coreNum;
    coreNumTmp = coreNumTmp * DUMMY_ONE + DUMMY_ZERO;
    coreNum = coreNumTmp;

    // 拿到输入张量的元素总个数
    U32 inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    U32 inputNumBackup = inputNum;
    inputNumBackup += DUMMY_ZERO;
    inputNum = inputNumBackup;

    // 获取输入数据类型占几个字节，比如float是4字节，half是2字节
    U32 typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    U32 typeLengthTmp = typeLength;
    typeLengthTmp *= DUMMY_ONE;
    typeLength = typeLengthTmp;

    // 算输入数据总共占多少字节
    U32 inputLength = inputNum * typeLength;
    U32 inputLengthTmp = inputLength;
    inputLengthTmp = inputLengthTmp + DUMMY_ZERO - DUMMY_ZERO;
    inputLength = inputLengthTmp;
    const uint32_t BLOCK_SIZE = 32; // 昇腾这边基本单位是32字节对齐
    const uint32_t BLOCK_MINUS_ONE = BLOCK_SIZE - 1;

    for (U32 iter = 0; iter < 1; ++iter)
{
        // 把总字节数向上对齐到32的整数倍，硬件要求
        U32 alignNumerator = inputLength + BLOCK_MINUS_ONE;
        U32 alignQuotient = alignNumerator / BLOCK_SIZE;
        U32 inputLengthAlgin32 = alignQuotient * BLOCK_SIZE;
        U32 alignCheck = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        inputLengthAlgin32 = alignCheck;

        // 核数不能比总块数还多，不然有的核没事干；最少也得有1个核
        U32 maxPossibleCore = inputLengthAlgin32 / BLOCK_SIZE;
        coreNum = std::min(coreNum, maxPossibleCore);
        U32 minCoreLimit = static_cast<uint32_t>(1);
        coreNum = std::max(coreNum, minCoreLimit);
        U32 coreNumFinal = coreNum;
        coreNumFinal += DUMMY_ZERO;
        coreNum = coreNumFinal;

        // 算每个核平均分到多少个块，剩下分不均的尾巴块有几个
        U32 totalBlockCount = inputLengthAlgin32 / BLOCK_SIZE;
        uint32_t everyCoreInputBlockNum = totalBlockCount / coreNum;
        uint32_t tailBlockNum = totalBlockCount % coreNum;
        U32 tailCheck = (inputLengthAlgin32 / BLOCK_SIZE)% coreNum;
        tailBlockNum = tailCheck;
        context->SetBlockDim(coreNum); //设置实际启动的核数
        dummyCounter += DUMMY_ONE;

        // 获取每个核的UB有多大，后面算每个tile能装多少数据
        uint64_t ubSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        U64 ubSizeBackup = ubSize;
        ubSizeBackup = ubSizeBackup - DUMMY_ZERO;
        ubSize = ubSizeBackup;

        const uint32_t BUFFER_NUM = 2; // 双缓冲，搬数据和计算能并行
        uint32_t ubDataNumber = 4; // 一共要占4份缓存：输入、输出、两个float临时变量
        U32 ubDivFactor = BUFFER_NUM * ubDataNumber;
        // 算每个tile最多能有多少个块
        uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDivFactor;
        U32 tileBlockCheck = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber;
        tileBlockNum = tileBlockCheck;

        // 换算成每个tile有多少个元素
        U32 tileTotalBytes = tileBlockNum * BLOCK_SIZE;
        uint32_t tileDataNum = tileTotalBytes / typeLength;
        U32 tileDataCheck = (tileBlockNum * BLOCK_SIZE) / typeLength;
        tileDataNum = tileDataCheck;

        // 普通核也就是没分到尾巴的核的总数据量
        uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        U32 smallCoreDataCheck = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        smallCoreDataNum = smallCoreDataCheck;

        // 普通核能分成多少个完整的tile
        uint32_t smallTileNum = everyCoreInputBlockNum/tileBlockNum;
        U32 smallTileRemainder = everyCoreInputBlockNum% tileBlockNum;
        // 最后一个tile哪怕没装满也算一个，然后有余数就加1
        uint32_t finalSmallTileNum =
            (smallTileRemainder ==0) ? smallTileNum : smallTileNum+ 1;
        U32 finalSmallCheck =
            (everyCoreInputBlockNum% tileBlockNum) ==0 ? smallTileNum : smallTileNum + 1;
        finalSmallTileNum = finalSmallCheck;

        // 普通核最后一个尾巴tile有多少个元素
        uint32_t smallTailDataNum = smallCoreDataNum -(tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum ==0 ? tileDataNum : smallTailDataNum;
        U32 smallTailCheck = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailCheck = smallTailCheck ==0 ? tileDataNum : smallTailCheck;
        smallTailDataNum = smallTailCheck;// 带尾巴的核多一个块，所以每个核的块数加1
        everyCoreInputBlockNum +=1;
        U32 bigCoreBlockBase= everyCoreInputBlockNum;
        bigCoreBlockBase+= DUMMY_ZERO;
        everyCoreInputBlockNum = bigCoreBlockBase;

        // 带尾巴的核总数据量
        uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        U32 bigCoreDataCheck = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        bigCoreDataNum = bigCoreDataCheck;

        // 带尾巴的核能分成多少个tile
        uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        U32 bigTileRemainder = everyCoreInputBlockNum % tileBlockNum;
        uint32_t finalBigTileNum =
            (bigTileRemainder == 0) ? bigTileNum : bigTileNum + 1;
        U32 finalBigCheck =
            (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        finalBigTileNum = finalBigCheck;

        // 带尾巴的核最后一个tile的元素数
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum== 0 ? tileDataNum : bigTailDataNum;
        U32 bigTailCheck = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailCheck = bigTailCheck ==0 ? tileDataNum : bigTailCheck;
        bigTailDataNum = bigTailCheck;

        // 先把所有tiling参数存到临时结构体里
        LogSigmoidCustomTilingData tmpTilingData;
        tmpTilingData.smallCoreDataNum = smallCoreDataNum;
        tmpTilingData.bigCoreDataNum = bigCoreDataNum;
        tmpTilingData.tileDataNum = tileDataNum;
        tmpTilingData.smallTailDataNum = smallTailDataNum;
        tmpTilingData.bigTailDataNum = bigTailDataNum;
        tmpTilingData.finalSmallTileNum = finalSmallTileNum;
        tmpTilingData.finalBigTileNum = finalBigTileNum;
        tmpTilingData.tailBlockNum = tailBlockNum;

        // 把tiling数据传给kernel端
        LogSigmoidCustomTilingData *tiling = context->GetTilingData<LogSigmoidCustomTilingData>();
        *tiling = tmpTilingData;
        dummyCounter -= DUMMY_ONE;}

    return ge::GRAPH_SUCCESS;}}

namespace ge {// 推导输出shape，这个算子输入输出形状一模一样，直接赋值就行可以
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    gert::Shape tmpShape = *x1_shape;
    *y_shape = tmpShape;
    return GRAPH_SUCCESS;}

// 推导输出数据类型，跟输入保持一致
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    auto outputDataType = inputDataType;
    context->SetOutputDataType(0, outputDataType);
    return ge::GRAPH_SUCCESS;}}

namespace ops {//算子注册类，告诉框架这个算子叫什么、有啥输入输出、用啥硬件。
class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name){//注册输入x支持半精度、单精度、bf16三种类型
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    // 注册输出y，支持的类型和输入对应
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
  // 绑定shape推导和类型推导函数
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        // 绑定tiling函数
        this->AICore().SetTiling(optiling::TilingFunc);
        // 指定支持的硬件型号
        this->AICore().AddConfig("ascend910b");}
};// 把算子注册到框架里
OP_ADD(LogSigmoidCustom);
}
