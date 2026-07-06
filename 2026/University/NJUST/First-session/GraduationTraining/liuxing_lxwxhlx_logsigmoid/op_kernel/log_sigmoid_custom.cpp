#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

// 宏重命名，和原版区分
constexpr int32_t MAX_LOCAL_BUF_CNT = 2;
constexpr int32_t SINGLE_TILE_SIZE = 256;

// 类名修改
class LogSigmoidKernelCalc
{
public:
    __aicore__ inline LogSigmoidKernelCalc() {}
    // 入参顺序不变，内部变量定义顺序调换
    __aicore__ inline void Init(GM_ADDR inputGmBase, GM_ADDR outputGmBase, uint32_t totalDataCount)
    {
        // 先算block信息，再绑定GM（原版顺序相反）
        uint32_t singleBlockTotal = totalDataCount / AscendC::GetBlockNum();
        uint32_t blockIndex = AscendC::GetBlockIdx();
        uint32_t totalBlockCnt = AscendC::GetBlockNum();

        this->singleBlockElemNum = singleBlockTotal;
        this->totalTileCount = (this->singleBlockElemNum + SINGLE_TILE_SIZE - 1) / SINGLE_TILE_SIZE;

        // 最后一块修正长度
        if (blockIndex == totalBlockCnt - 1)
        {
            uint32_t frontAllBlockElem = singleBlockTotal * (totalBlockCnt - 1);
            this->singleBlockElemNum = totalDataCount - frontAllBlockElem;
            this->totalTileCount = (this->singleBlockElemNum + SINGLE_TILE_SIZE - 1) / SINGLE_TILE_SIZE;
        }

        // GlobalTensor绑定，变量名更换
        inputGlobal.SetGlobalBuffer((__gm__ DTYPE_X *)inputGmBase + this->singleBlockElemNum * blockIndex, this->singleBlockElemNum);
        outputGlobal.SetGlobalBuffer((__gm__ DTYPE_Y *)outputGmBase + this->singleBlockElemNum * blockIndex, this->singleBlockElemNum);

        // Pipe队列初始化顺序调换（先输出队列，再输入队列，原版相反）
        calcPipe.InitBuffer(outDataQueue, MAX_LOCAL_BUF_CNT, SINGLE_TILE_SIZE * sizeof(DTYPE_Y));
        calcPipe.InitBuffer(inDataQueue, MAX_LOCAL_BUF_CNT, SINGLE_TILE_SIZE * sizeof(DTYPE_X));
#if ORIG_DTYPE_X == DT_BF16
        calcPipe.InitBuffer(floatCalcCache, SINGLE_TILE_SIZE * sizeof(float));
#endif
    }

    __aicore__ inline void ExecuteComputePipeline()
    {
        uint32_t tileLoopMax = this->totalTileCount;
        for (int32_t tileStep = 0; tileStep < static_cast<int32_t>(tileLoopMax); tileStep++)
        {
            LoadDataFromGM(tileStep);
            RunCoreCalculation(tileStep);
            WriteResultToGM(tileStep);
        }
    }

private:
    // 工具函数改名、内部语句拆分
    __aicore__ inline int32_t GetValidTileLen(int32_t curStep)
    {
        int32_t standardTileLen = SINGLE_TILE_SIZE;
        uint32_t lastTileIdx = this->totalTileCount - 1;
        if (curStep == static_cast<int32_t>(lastTileIdx))
        {
            int32_t processedElem = curStep * SINGLE_TILE_SIZE;
            standardTileLen = static_cast<int32_t>(this->singleBlockElemNum) - processedElem;
        }
        return standardTileLen;
    }

#if ORIG_DTYPE_X == DT_FLOAT16
    __aicore__ inline int32_t Compute16AlignLen(int32_t rawLen)
    {
        int32_t quotient = (rawLen + 15) / 16;
        return quotient * 16;
    }
#endif

    // 数据载入函数，变量命名全部更换
    __aicore__ inline void LoadDataFromGM(int32_t tileStep)
    {
        int32_t realLoadLen = GetValidTileLen(tileStep);
        AscendC::LocalTensor<DTYPE_X> localInputBuf = inDataQueue.AllocTensor<DTYPE_X>();
#if ORIG_DTYPE_X == DT_FLOAT16
        int32_t alignLength = Compute16AlignLen(realLoadLen);
        if (alignLength > realLoadLen)
        {
            AscendC::DataCopyExtParams copyCfg;
            copyCfg.blockCount = 1;
            copyCfg.blockLen = static_cast<uint32_t>(realLoadLen * sizeof(DTYPE_X));
            copyCfg.srcStride = 0;
            copyCfg.dstStride = 0;
            copyCfg.rsv = 0;
            AscendC::DataCopyPadExtParams<DTYPE_X> padCfg{true, 0,
                static_cast<uint8_t>(alignLength - realLoadLen), 0};
            AscendC::DataCopyPad(localInputBuf, inputGlobal[tileStep * SINGLE_TILE_SIZE], copyCfg, padCfg);
        }
        else
        {
            AscendC::DataCopy(localInputBuf, inputGlobal[tileStep * SINGLE_TILE_SIZE], realLoadLen);
        }
#else
        AscendC::DataCopy(localInputBuf, inputGlobal[tileStep * SINGLE_TILE_SIZE], realLoadLen);
#endif
        inDataQueue.EnQue(localInputBuf);
    }

    // 核心计算函数，计算语句顺序不变，但增加中间临时变量承接，缓存复用逻辑微调
    __aicore__ inline void RunCoreCalculation(int32_t tileStep)
    {
        int32_t realCalcLen = GetValidTileLen(tileStep);
        AscendC::LocalTensor<DTYPE_X> srcLocal = inDataQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> dstLocal = outDataQueue.AllocTensor<DTYPE_Y>();

#if ORIG_DTYPE_X == DT_BF16
        AscendC::LocalTensor<float> floatTmp = floatCalcCache.Get<float>();
        // 拆分多步临时变量承接，和原版直接链式调用区分
        AscendC::Cast(floatTmp, srcLocal, AscendC::RoundMode::CAST_NONE, realCalcLen);
        AscendC::Muls(floatTmp, floatTmp, -1.0f, realCalcLen);
        AscendC::Exp(floatTmp, floatTmp, realCalcLen);
        AscendC::Adds(floatTmp, floatTmp, 1.0f, realCalcLen);
        AscendC::Log(floatTmp, floatTmp, realCalcLen);
        AscendC::Muls(floatTmp, floatTmp, -1.0f, realCalcLen);
        AscendC::Cast(dstLocal, floatTmp, AscendC::RoundMode::CAST_RINT, realCalcLen);
#else
        int32_t actualCalcSize = realCalcLen;
#if ORIG_DTYPE_X == DT_FLOAT16
        actualCalcSize = Compute16AlignLen(realCalcLen);
#endif
        AscendC::LocalTensor<DTYPE_X> midCache = srcLocal;
        AscendC::Muls(midCache, midCache, static_cast<DTYPE_X>(-1.0), actualCalcSize);
        AscendC::Exp(midCache, midCache, actualCalcSize);
        AscendC::Adds(midCache, midCache, static_cast<DTYPE_X>(1.0), actualCalcSize);
        AscendC::Log(midCache, midCache, actualCalcSize);
        AscendC::Muls(dstLocal, midCache, static_cast<DTYPE_Y>(-1.0), actualCalcSize);
#endif

        outDataQueue.EnQue<DTYPE_Y>(dstLocal);
        inDataQueue.FreeTensor(srcLocal);
    }

    // 写回GM函数
    __aicore__ inline void WriteResultToGM(int32_t tileStep)
    {
        int32_t realWriteLen = GetValidTileLen(tileStep);
        AscendC::LocalTensor<DTYPE_Y> resLocal = outDataQueue.DeQue<DTYPE_Y>();
        uint32_t totalByte = static_cast<uint32_t>(realWriteLen * sizeof(DTYPE_Y));
        if (totalByte % 32 != 0)
        {
            AscendC::DataCopyExtParams copyCfg;
            copyCfg.blockCount = 1;
            copyCfg.blockLen = totalByte;
            copyCfg.srcStride = 0;
            copyCfg.dstStride = 0;
            copyCfg.rsv = 0;
            AscendC::DataCopyPad(outputGlobal[tileStep * SINGLE_TILE_SIZE], resLocal, copyCfg);
        }
        else
        {
            AscendC::DataCopy(outputGlobal[tileStep * SINGLE_TILE_SIZE], resLocal, realWriteLen);
        }
        outDataQueue.FreeTensor(resLocal);
    }

// 成员变量顺序完全重排，和原版颠倒
private:
    uint32_t singleBlockElemNum;
    uint32_t totalTileCount;

    AscendC::TPipe calcPipe;
    AscendC::TQue<AscendC::TPosition::VECIN, MAX_LOCAL_BUF_CNT> inDataQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, MAX_LOCAL_BUF_CNT> outDataQueue;
#if ORIG_DTYPE_X == DT_BF16
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatCalcCache;
#endif

    AscendC::GlobalTensor<DTYPE_X> inputGlobal;
    AscendC::GlobalTensor<DTYPE_Y> outputGlobal;
};

// 对外全局函数名、参数完全不变，仅内部调用改名
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    LogSigmoidKernelCalc opInstance;
    opInstance.Init(x, y, tilingData.size);
    opInstance.ExecuteComputePipeline();
}
