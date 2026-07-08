#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

template<typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum, uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;
        this->tileDataNum = tileDataNum;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) *
                                 (blockIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

        // 四队列双缓冲：输入、临时1、临时2、输出
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tempQueue1, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tempQueue2, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum],
                          this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> tmp1 = tempQueue1.AllocTensor<T>();
        AscendC::LocalTensor<T> tmp2 = tempQueue2.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        // 严格按公式：output = x * exp(0.851*(x-|x|)) / (1 + exp(-1.702*|x|))
        AscendC::Abs(tmp1, xLocal, this->processDataNum);

        // 分母部分
        AscendC::Muls(tmp2, tmp1, static_cast<T>(-1.702), this->processDataNum);
        AscendC::Exp(tmp2, tmp2, this->processDataNum);
        AscendC::Adds(tmp2, tmp2, static_cast<T>(1.0), this->processDataNum);

        // 分子部分
        AscendC::Sub(yLocal, xLocal, tmp1, this->processDataNum);
        AscendC::Muls(yLocal, yLocal, static_cast<T>(0.851), this->processDataNum);
        AscendC::Exp(yLocal, yLocal, this->processDataNum);
        AscendC::Mul(yLocal, xLocal, yLocal, this->processDataNum);

        // 除法
        AscendC::Div(yLocal, yLocal, tmp2, this->processDataNum);

        outQueueY.EnQue<T>(yLocal);
        tempQueue2.FreeTensor(tmp2);
        tempQueue1.FreeTensor(tmp1);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal,
                          this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, tempQueue1, tempQueue2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm, yGm;
    uint32_t coreDataNum, tileNum, tileDataNum, tailDataNum, processDataNum;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                     GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y,
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum, tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum, tiling_data.tailBlockNum);
    op.Process();
}