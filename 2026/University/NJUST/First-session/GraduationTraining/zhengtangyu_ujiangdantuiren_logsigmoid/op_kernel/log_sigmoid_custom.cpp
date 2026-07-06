#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y,
        uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
        uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
        uint32_t tileDataNum, uint32_t smallTailDataNum,
        uint32_t bigTailDataNum, uint32_t tailBlockNum)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) *
                                 (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));

        if constexpr (std::is_same<TYPE, bfloat16_t>::value) {
            pipe.InitBuffer(tmpFloat0, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloat1, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < this->tileNum; ++i) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        auto xLocal = inQueueX.AllocTensor<TYPE>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        auto xLocal = inQueueX.DeQue<TYPE>();
        auto yLocal = outQueueY.AllocTensor<TYPE>();

        if constexpr (std::is_same<TYPE, bfloat16_t>::value) {
            auto f0 = tmpFloat0.Get<float>();
            auto f1 = tmpFloat1.Get<float>();
            AscendC::Cast(f0, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            // y = -log(1 + exp(-x))
            AscendC::Muls(f1, f0, -1.0f, this->processDataNum);  // -x
            AscendC::Exp(f1, f1, this->processDataNum);
            AscendC::Adds(f1, f1, 1.0f, this->processDataNum);
            AscendC::Log(f1, f1, this->processDataNum);
            AscendC::Muls(f1, f1, -1.0f, this->processDataNum);
            AscendC::Cast(yLocal, f1, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            // 直接计算：y = -log(1 + exp(-x))
            AscendC::Muls(yLocal, xLocal, TYPE(-1.0), this->processDataNum);
            AscendC::Exp(yLocal, yLocal, this->processDataNum);
            AscendC::Adds(yLocal, yLocal, TYPE(1.0), this->processDataNum);
            AscendC::Log(yLocal, yLocal, this->processDataNum);
            AscendC::Muls(yLocal, yLocal, TYPE(-1.0), this->processDataNum);
        }

        outQueueY.EnQue<TYPE>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        auto yLocal = outQueueY.DeQue<TYPE>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpFloat0, tmpFloat1;  // only for bf16
    AscendC::GlobalTensor<TYPE> xGm, yGm;
    uint32_t coreDataNum, tileNum, tileDataNum, tailDataNum, processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSigmoid<DTYPE_X> op;
    op.Init(x, y,
            tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
            tilingData.tileDataNum, tilingData.smallTailDataNum,
            tilingData.bigTailDataNum, tilingData.tailBlockNum);
    op.Process();
}
