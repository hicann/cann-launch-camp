#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t PIPE_BUF_DEPTH = 2;

template<typename TYPE_X, typename TYPE_Y>
class LogSigmoidKernel {
public:
    __aicore__ inline LogSigmoidKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t baseCoreElemTotal,
        uint32_t fullCoreElemTotal, uint32_t fullCoreTileTotal,
        uint32_t baseCoreTileTotal, uint32_t baseTileElemCount,
        uint32_t baseTailElemCount, uint32_t fullTailElemCount,
        uint32_t remainBlockCount)
    {
        // 获取当前核的编号
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // 计算当前核在全局内存中的起始元素偏移
        uint32_t coreElemOffset = fullCoreElemTotal * blockIdx;
        this->baseTileElem = baseTileElemCount;

        if (blockIdx < remainBlockCount) {
            // 满载核分支：使用满载核的参数
            this->coreTotalElem = fullCoreElemTotal;
            this->tileLoopCount = fullCoreTileTotal;
            this->tailTileElem = fullTailElemCount;
        }
        else {
            // 基础核分支：使用基础核参数，修正全局偏移量
            this->coreTotalElem = baseCoreElemTotal;
            this->tileLoopCount = baseCoreTileTotal;
            this->tailTileElem = baseTailElemCount;
            coreElemOffset -= (fullCoreElemTotal - baseCoreElemTotal) * (blockIdx - remainBlockCount);
        }

        // 绑定当前核负责的输入、输出全局内存段
        xGlobal.SetGlobalBuffer((__gm__ TYPE_X*)x + coreElemOffset, this->coreTotalElem);
        yGlobal.SetGlobalBuffer((__gm__ TYPE_Y*)y + coreElemOffset, this->coreTotalElem);

        // 初始化UB中的队列缓存与临时计算空间
        pipe.InitBuffer(xInQueue, PIPE_BUF_DEPTH, this->baseTileElem * sizeof(TYPE_X));
        pipe.InitBuffer(yOutQueue, PIPE_BUF_DEPTH, this->baseTileElem * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpBufA, this->baseTileElem * sizeof(float));
        pipe.InitBuffer(tmpBufB, this->baseTileElem * sizeof(float));
    }

    __aicore__ inline void Run()
    {
        // 按tile循环处理数据，最后一次处理尾块数据
        for (int32_t i = 0; i < this->tileLoopCount; i++) {
            this->curTileElem = this->baseTileElem;
            if (i == this->tileLoopCount - 1) {
                this->curTileElem = this->tailTileElem;
            }
            LoadInput(i);
            Calculate(i);
            StoreOutput(i);
        }
    }

private:
    __aicore__ inline void LoadInput(int32_t step)
    {
        // 从全局内存搬运输入数据到UB
        AscendC::LocalTensor<TYPE_X> xUbTensor = xInQueue.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xUbTensor, xGlobal[step * this->baseTileElem], this->curTileElem);
        xInQueue.EnQue(xUbTensor);
    }

    __aicore__ inline void Calculate(int32_t step)
    {
        AscendC::LocalTensor<TYPE_X> xUbTensor = xInQueue.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yUbTensor = yOutQueue.AllocTensor<TYPE_Y>();

        // LogSigmoid计算：y = -log(1 + exp(-x))
        AscendC::LocalTensor<float> calcBufA = tmpBufA.Get<float>();
        AscendC::LocalTensor<float> calcBufB = tmpBufB.Get<float>();

        // 输入统一转成float做中间计算，保证精度
        if constexpr (std::is_same_v<TYPE_X, float>) {
            AscendC::Muls(calcBufA, xUbTensor, 1.0f, this->curTileElem);
        }
        else {
            AscendC::Cast(calcBufA, xUbTensor, AscendC::RoundMode::CAST_NONE, this->curTileElem);
        }

        // 计算 -x
        AscendC::Muls(calcBufA, calcBufA, -1.0f, this->curTileElem);
        // 计算 exp(-x)
        AscendC::Exp(calcBufA, calcBufA, this->curTileElem);
        // 计算 1 + exp(-x)
        AscendC::Adds(calcBufA, calcBufA, 1.0f, this->curTileElem);
        // 计算 log(1 + exp(-x))
        AscendC::Log(calcBufA, calcBufA, this->curTileElem);
        // 计算最终结果 -log(1 + exp(-x))
        AscendC::Muls(calcBufB, calcBufA, -1.0f, this->curTileElem);

        // 将float结果转换为输出对应类型
        if constexpr (std::is_same_v<TYPE_Y, float>) {
            AscendC::Muls(yUbTensor, calcBufB, 1.0f, this->curTileElem);
        }
        else {
            AscendC::Cast(yUbTensor, calcBufB, AscendC::RoundMode::CAST_RINT, this->curTileElem);
        }

        yOutQueue.EnQue<TYPE_Y>(yUbTensor);
        xInQueue.FreeTensor(xUbTensor);
    }

    __aicore__ inline void StoreOutput(int32_t step)
    {
        // 将计算结果从UB写回全局内存
        AscendC::LocalTensor<TYPE_Y> yUbTensor = yOutQueue.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGlobal[step * this->baseTileElem], yUbTensor, this->curTileElem);
        yOutQueue.FreeTensor(yUbTensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPE_BUF_DEPTH> xInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPE_BUF_DEPTH> yOutQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufA, tmpBufB;
    AscendC::GlobalTensor<TYPE_X> xGlobal;
    AscendC::GlobalTensor<TYPE_Y> yGlobal;
    uint32_t coreTotalElem;
    uint32_t tileLoopCount;
    uint32_t baseTileElem;
    uint32_t tailTileElem;
    uint32_t curTileElem;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    LogSigmoidKernel<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.baseCoreElemTotal,
        tilingData.fullCoreElemTotal, tilingData.fullCoreTileTotal,
        tilingData.baseCoreTileTotal, tilingData.baseTileElemCount,
        tilingData.baseTailElemCount, tilingData.fullTailElemCount,
        tilingData.remainBlockCount);
    op.Run();
}