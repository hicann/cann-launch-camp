// Kernel 侧核函数实现 — 泛化 Tiling 驱动：按 Tiling 参数直接判断大/小核，循环批次处理
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x,
                                GM_ADDR output,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;

        this->tileDataNum = tileDataNum;

        if (blockIdx < tailBlockNum) {
            // 大核：多 1 个 32B 块
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 小核：基准分配
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + globalBufferIndex,
                            this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + globalBufferIndex,
                            this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM,
                        this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM,
                        this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuffer,
                        this->tileDataNum * sizeof(DT_INPUT_X));
    }

    // 核内流水线：按 Tiling 批次循环处理，最后一次用尾块大小
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal =
            inQueueX.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(xLocal,
                          xGm[progress * this->tileDataNum],
                          this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute() {
        AscendC::LocalTensor<DT_INPUT_X> xLocal =
            inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal =
            outQueueY.AllocTensor<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> tmpLocal =
            tmpBuffer.Get<DT_INPUT_X>();

        // GELU(x) = 0.5 * x * (1 + erf(x / √2)),  √2 ≈ 1.41421356
        AscendC::Muls(tmpLocal, xLocal,
                      static_cast<DT_INPUT_X>(0.7071067811865476),
                      this->processDataNum);
        AscendC::Erf(tmpLocal, tmpLocal, this->processDataNum);
        AscendC::Adds(tmpLocal, tmpLocal,
                      static_cast<DT_INPUT_X>(1.0),
                      this->processDataNum);
        AscendC::Muls(tmpLocal, tmpLocal,
                      static_cast<DT_INPUT_X>(0.5),
                      this->processDataNum);
        AscendC::Mul(yLocal, xLocal, tmpLocal, this->processDataNum);

        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_INPUT_X> yLocal =
            outQueueY.DeQue<DT_INPUT_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum],
                          yLocal,
                          this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x,
                                GM_ADDR output,
                                GM_ADDR workspace,
                                GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;

    op.Init(input_x, output,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);

    op.Process();
}
