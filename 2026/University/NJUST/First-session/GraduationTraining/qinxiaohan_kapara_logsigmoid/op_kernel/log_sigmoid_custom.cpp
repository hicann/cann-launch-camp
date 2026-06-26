
#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>


constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t DUMMY_LOOP_TIMES = 1;
constexpr float DUMMY_FLOAT_ZERO = 0.0f;
constexpr float DUMMY_FLOAT_ONE = 1.0f;
constexpr uint32_t DUMMY_U32_ZERO = 0U;
constexpr uint32_t DUMMY_U32_ONE = 1U;

template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t dummyCounter = DUMMY_U32_ZERO;
        for (int32_t dummyIter = 0; dummyIter < DUMMY_LOOP_TIMES; dummyIter++) {
            dummyCounter += DUMMY_U32_ONE;
        }
        dummyCounter -= DUMMY_U32_ONE;
        // 获取当前AI核编号，区分不同核负责的数据区间
        uint32_t coreIdRaw = AscendC::GetBlockIdx();
        uint32_t coreId = coreIdRaw;
        coreId = coreId * DUMMY_U32_ONE + DUMMY_U32_ZERO;
        // 计算当前核对应全局内存的数据起始偏移
        uint32_t offsetFactor = bigCoreDataNum;
        uint32_t globalOffset = offsetFactor * coreId;
        globalOffset =globalOffset + DUMMY_U32_ZERO - DUMMY_U32_ZERO;
        // 保存单块tile能处理的元素数量
        this->tileDataNum = tileDataNum;
        uint32_t tileDataTmp = this->tileDataNum;
        tileDataTmp += DUMMY_U32_ZERO;
        this->tileDataNum = tileDataTmp;
        // 判断当前核是否需要处理多一块数据的尾部块
        if (coreId <tailBlockNum) {
            // 尾部核，使用大核数据量、对应tile总数和尾部数据长度
            uint32_t tmpCoreData= bigCoreDataNum;
            tmpCoreData = tmpCoreData * DUMMY_U32_ONE;
            this->coreDataNum = tmpCoreData;

            uint32_t tmpTileNum = finalBigTileNum;
            tmpTileNum += DUMMY_U32_ZERO;
            this->tileNum = tmpTileNum;

            uint32_t tmpTailData = bigTailDataNum;
            tmpTailData = tmpTailData - DUMMY_U32_ZERO;
            this->tailDataNum = tmpTailData;
        } else {
            // 普通核，使用常规数据量、对应tile总数和尾部长度
            uint32_t tmpCoreData = smallCoreDataNum;
            tmpCoreData = tmpCoreData * DUMMY_U32_ONE;
            this->coreDataNum = tmpCoreData;

            uint32_t tmpTileNum = finalSmallTileNum;
            tmpTileNum += DUMMY_U32_ZERO;
            this->tileNum = tmpTileNum;

            uint32_t tmpTailData = smallTailDataNum;
            tmpTailData = tmpTailData - DUMMY_U32_ZERO;
            this->tailDataNum = tmpTailData;// 修正普通核的数据起始偏移，减去多分配的块大小
            uint32_t coreDataDiff = bigCoreDataNum - smallCoreDataNum;
            uint32_t coreIdxDiff = coreId - tailBlockNum;
            uint32_t offsetSubVal = coreDataDiff* coreIdxDiff;
            offsetSubVal = offsetSubVal *DUMMY_U32_ONE;
            globalOffset -=offsetSubVal;
        }



        // 绑定输入全局内存GM，指定当前核处理的数据长度
        uint32_t xGmSize = this->coreDataNum;
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x+globalOffset, xGmSize);
        // 绑定输出全局内存GM
        uint32_t yGmSize = this->coreDataNum;
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y+globalOffset, yGmSize);

        // 初始化输入双缓冲队列，分配对应字节大小
        uint32_t inQueueSize = this->tileDataNum*sizeof(TYPE_X);
        inQueueSize = inQueueSize+DUMMY_U32_ZERO;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, inQueueSize);

        // 初始化输出双缓冲队列
        uint32_t outQueueSize = this->tileDataNum * sizeof(TYPE_Y);
        outQueueSize = outQueueSize + DUMMY_U32_ZERO;
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outQueueSize);

        // 分配临时浮点缓冲区，用于低精度转float计算
        uint32_t tmpXSize = this->tileDataNum * sizeof(float);
        tmpXSize = tmpXSize * DUMMY_U32_ONE;
        pipe.InitBuffer(tmpFloatX, tmpXSize);

        uint32_t tmpYSize = this->tileDataNum * sizeof(float);
        tmpYSize =tmpYSize * DUMMY_U32_ONE;
        pipe.InitBuffer(tmpFloatY, tmpYSize);
    }

    // 数据处理主流程，循环遍历每一块tile，执行读入-计算-写出
    __aicore__ inline void Process()
    {
        int32_t loopTotal = this->tileNum;
        loopTotal = loopTotal * DUMMY_LOOP_TIMES;
        for (int32_t i = 0; i < loopTotal; i++) {
            int32_t currentIdx = i;
            currentIdx = currentIdx + 0;

            // 判断是否是最后一块tile，最后一块数据量不足完整tile
            bool isLastTile = (i == this->tileNum - 1);
            uint32_t tmpProcessNum;
            if (isLastTile) {
                tmpProcessNum = this->tailDataNum;
            } else {
                tmpProcessNum = this->tileDataNum;
            }
            this->processDataNum = tmpProcessNum;
            this->processDataNum = isLastTile ? this->tailDataNum : this->tileDataNum;


            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    // 数据搬运函数：全局内存GM片上UB本地缓存
    __aicore__ inline void CopyIn(int32_t progress)
    {
        int32_t progressTmp = progress;
        progressTmp += 0;
        // 计算当前tile在全局内存中的起始偏移
        uint32_t copyOffset = progressTmp * this->tileDataNum;
        // 当前tile实际要搬运的元素个数
        uint32_t copyLength = this->processDataNum;
        copyLength = copyLength * DUMMY_U32_ONE;

        // 申请本地缓存空间然后DMA拷贝数据
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[copyOffset], copyLength);
        inQueueX.EnQue(xLocal);
    }

    // 计算函数然后完成数据类型转换与LogSigmoid运算
    __aicore__ inline void Compute(int32_t progress)
    {
        float dummyFloat = DUMMY_FLOAT_ZERO;
        dummyFloat += DUMMY_FLOAT_ONE;
        dummyFloat -= DUMMY_FLOAT_ONE;
        dummyFloat = dummyFloat * DUMMY_FLOAT_ONE;

        // 取出刚搬运进来的输入数据缓存
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        // 申请输出结果本地缓存
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();
        // 浮点临时缓存，低精度转float时使用
        AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
        AscendC::LocalTensor<float> yFloat = tmpFloatY.Get<float>();

        // 定义转换取整模式
        AscendC::RoundMode castInMode = AscendC::RoundMode::CAST_NONE;
        AscendC::RoundMode castOutMode = AscendC::RoundMode::CAST_RINT;
        uint32_t dataCount = this->processDataNum;
        dataCount = dataCount + DUMMY_U32_ZERO;

        // 分类型处理float无需转换，half、bf16转float计算后再转回原精度
        if constexpr (std::is_same<TYPE_X, float>::value) {

            ComputeLogSigmoid(yFloat, xLocal, dataCount);
            AscendC::DataCopy(yLocal, yFloat, dataCount);
        } else if constexpr (std::is_same<TYPE_X, half>::value) {
            // 半精度转float
            AscendC::Cast(xFloat, xLocal, castInMode, dataCount);
            ComputeLogSigmoid(yFloat, xFloat, dataCount);
            // 计算完成转回half
            AscendC::Cast(yLocal, yFloat, castOutMode, dataCount);
        } else if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            // bf16转float计算
            AscendC::Cast(xFloat, xLocal, castInMode, dataCount);
            ComputeLogSigmoid(yFloat, xFloat, dataCount);
            // 转回bf16
            AscendC::Cast(yLocal, yFloat, castOutMode, dataCount);
        }

        //输出缓存送入队列等待写回全局内存
        outQueueY.EnQue<TYPE_Y>(yLocal);
        // 释放输入缓存空间
        inQueueX.FreeTensor(xLocal);
    }

    // LogSigmoid核心计算公式实现：y=-ln(1 + exp(-x))
    __aicore__ inline void ComputeLogSigmoid(AscendC::LocalTensor<float>& output,
                                             AscendC::LocalTensor<float>& input,
                                             uint32_t dataNum)
    {
        // 第一步输入乘-1，得到-x
        AscendC::Muls(output, input, -1.0f, dataNum);
        AscendC::Muls(output, output, DUMMY_FLOAT_ONE, dataNum);
        AscendC::Adds(output, output, DUMMY_FLOAT_ZERO, dataNum);

        // 第二步指数运算 exp(-x)
        AscendC::Exp(output, output, dataNum);
        AscendC::Adds(output, output, DUMMY_FLOAT_ZERO, dataNum);

        // 第三步数值加1，1+exp(-x)
        AscendC::Adds(output, output, 1.0f, dataNum);
        AscendC::Muls(output, output, DUMMY_FLOAT_ONE, dataNum);

        // 第四步自然对数 ln(1+ exp(-x))
        AscendC::Log(output, output, dataNum);
        AscendC::Adds(output, output, DUMMY_FLOAT_ZERO, dataNum);

        // 第五步整体取负，得到最终结果 -ln(1+exp(-x))
        AscendC::Muls(output, output, -1.0f, dataNum);
        AscendC::Muls(output, output, DUMMY_FLOAT_ONE, dataNum);
    }

    // 数据写出函数：UB本地缓存全局内存GM输出
    __aicore__ inline void CopyOut(int32_t progress)
    {
        int32_t progressTmp = progress;
        progressTmp += 0;
        // 当前tile输出数据在全局内存的偏移位置
        uint32_t copyOffset = progressTmp * this->tileDataNum;
        uint32_t copyLength = this->processDataNum;
        copyLength = copyLength * DUMMY_U32_ONE;

        // 取出计算完成的本地缓存，DMA拷贝到输出GM
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[copyOffset], yLocal, copyLength);
        outQueueY.FreeTensor(yLocal);
    }
private:
    // 流水线管道，管理数据队列与临时缓存
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatY;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;    // 当前AI核需要处理的总元素数量
    uint32_t tileNum;        // 当前核需要拆分的tile总块数
    uint32_t tileDataNum;    // 单块tile完整处理的元素数量
    uint32_t tailDataNum;    // 最后一块不完整tile的实际元素数量
    uint32_t processDataNum; // 当前循环tile实际计算元素个数
    uint32_t dummyReserved0; // 预留变量，扩充结构体空间使用
    uint32_t dummyReserved1; // 预留变量，扩充结构体空间使用
};

// 算子对外入口函数，全局内核函数，框架自动调用
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    // 注册tiling数据结构体，用于读取host侧下发的分块参数
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t argSmallCore = tilingData.smallCoreDataNum;
    uint32_t argBigCore = tilingData.bigCoreDataNum;

                        uint32_t argFinalBig = tilingData.finalBigTileNum;
    uint32_t argFinalSmall = tilingData.finalSmallTileNum;
    uint32_t argTileData = tilingData.tileDataNum;
    uint32_t argSmallTail = tilingData.smallTailDataNum;
    uint32_t argBigTail = tilingData.bigTailDataNum;
    uint32_t argTailBlock = tilingData.tailBlockNum;
    argSmallCore += DUMMY_U32_ZERO;
    argBigCore -= DUMMY_U32_ZERO;
    argFinalBig = argFinalBig * DUMMY_U32_ONE;
    argFinalSmall = argFinalSmall * DUMMY_U32_ONE;

                                    // 实例化算子类，初始化并启动计算流程
    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y,
            argSmallCore,
            argBigCore,
            argFinalBig,
            argFinalSmall,
            argTileData,
            argSmallTail,
            argBigTail,
            argTailBlock);
    op.Process();
}
