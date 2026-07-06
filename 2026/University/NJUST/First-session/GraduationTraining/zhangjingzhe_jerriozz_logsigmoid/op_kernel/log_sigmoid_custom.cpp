#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelLogSigmoid
{
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, LogSigmoidCustomTilingData tilingData)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t dataNum;
        uint32_t offset;

        if (blockIdx < tilingData.tailBlockNum)
        {
            dataNum = tilingData.bigCoreDataNum;
            offset = blockIdx * dataNum;
        }
        else
        {
            dataNum = tilingData.smallCoreDataNum;
            offset = tilingData.tailBlockNum * tilingData.bigCoreDataNum + (blockIdx - tilingData.tailBlockNum) * dataNum;
        }
        totalLength = dataNum;
        this->tilingData = tilingData;

        xGm.SetGlobalBuffer((__gm__ T *)x + offset, dataNum);
        yGm.SetGlobalBuffer((__gm__ T *)y + offset, dataNum);

        pipe.InitBuffer(inQueue, BUFFER_NUM, tilingData.tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tilingData.tileDataNum * sizeof(T));

        pipe.InitBuffer(calcBuf, tilingData.tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t tileNum = (totalLength + tilingData.tileDataNum - 1) / tilingData.tileDataNum;
        for (uint32_t i = 0; i < tileNum; i++)
        {
            uint32_t length = min(tilingData.tileDataNum, totalLength - i * tilingData.tileDataNum);
            CopyIn(i, length);
            Compute(length);
            CopyOut(i, length);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t length)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * tilingData.tileDataNum], length);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();

        // 核心修改：使用 TBuf 获取计算 Tensor
        LocalTensor<float> tmp = calcBuf.Get<float>();

        if constexpr (std::is_same_v<T, float>)
        {
            DataCopy(tmp, xLocal, length);
        }
        else
        {
            Cast(tmp, xLocal, RoundMode::CAST_NONE, length);
        }

        // 数学计算： LogSigmoid(x) = -ln(1 + exp(-x))
        Muls(tmp, tmp, -1.0f, length);
        Exp(tmp, tmp, length);
        Adds(tmp, tmp, 1.0f, length);
        Ln(tmp, tmp, length);
        Muls(tmp, tmp, -1.0f, length);

        // 从 tmp (float) 写回输出 yLocal
        if constexpr (std::is_same_v<T, float>)
        {
            DataCopy(yLocal, tmp, length);
        }
        else
        {
            Cast(yLocal, tmp, RoundMode::CAST_ROUND, length);
        }

        inQueue.FreeTensor(xLocal);
        outQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t length)
    {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        DataCopy(yGm[progress * tilingData.tileDataNum], yLocal, length);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    // 核心修改：从 TQue 变更为 TBuf，因为 VEC 计算阶段无需异步队列机制
    TBuf<TPosition::VECCALC> calcBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t totalLength;

    LogSigmoidCustomTilingData tilingData;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSigmoid<DTYPE_X> op;
    op.Init(x, y, tilingData);
    op.Process(); // 移除无用的传参，因已保存至成员变量
}