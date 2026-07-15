#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr float INV_SQRT2 = 0.7071067811865475244f;
constexpr float HALF = 0.5f;
constexpr float ONE  = 1.0f;

template<typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t finalSmallTileNum,
                                uint32_t finalBigTileNum,
                                uint32_t tailBlockNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }

        uint32_t remaining = totalLength - globalBufferIndex;
        if (remaining <= 0) {
            this->coreDataNum = 0;
            this->tileNum = 0;
            return;
        }
        if (this->coreDataNum > remaining) {
            this->coreDataNum = remaining;
        }

        this->tileDataNum = tileDataNum;
        this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
        uint32_t fullTileCount = this->coreDataNum / this->tileDataNum;
        uint32_t tail = this->coreDataNum - fullTileCount * this->tileDataNum;
        this->tailDataNum = (tail == 0) ? this->tileDataNum : tail;

        this->prefill = (BUFFER_NUM < this->tileNum) ? BUFFER_NUM : this->tileNum;

        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->tileNum == 0) return;

        // 预填充：连续装入前prefill个tile
        for (int32_t i = 0; i < this->prefill; i++) {
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn(i);
        }

        // 流水线主循环：算tile i的同时MTE搬运tile i+BUFFER_NUM
        for (int32_t i = 0; i < this->tileNum; i++) {
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            Compute();
            CopyOut();

            int32_t next = i + BUFFER_NUM;
            if (next < this->tileNum) {
                this->processDataNum = (next == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
                CopyIn(next);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        AscendC::Muls(yLocal, xLocal, static_cast<T>(INV_SQRT2), this->processDataNum);
        AscendC::Erf(yLocal, yLocal, this->processDataNum);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(ONE), this->processDataNum);
        AscendC::Muls(yLocal, yLocal, static_cast<T>(HALF), this->processDataNum);
        AscendC::Mul(yLocal, xLocal, yLocal, this->processDataNum);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[this->outProgress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
        this->outProgress++;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t coreDataNum;
    int32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    int32_t prefill;
    int32_t outProgress = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(x, y, tilingData.totalLength,
            tilingData.smallCoreDataNum,
            tilingData.bigCoreDataNum,
            tilingData.tileDataNum,
            tilingData.smallTailDataNum,
            tilingData.bigTailDataNum,
            tilingData.finalSmallTileNum,
            tilingData.finalBigTileNum,
            tilingData.tailBlockNum);
    op.Process();
}
