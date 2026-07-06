#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

template<typename T>
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
        uint32_t tailBlockNum) {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ T*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmpOne, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(tmpNegX, this->tileDataNum * sizeof(T));

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatY, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatOne, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatNegX, this->tileDataNum * sizeof(float));
        }

        // 初始化 tmpOne 为全1
        AscendC::LocalTensor<T> oneLocal = tmpOne.Get<T>();
        AscendC::Duplicate(oneLocal, static_cast<T>(1.0f), this->tileDataNum);
        tmpOne.FreeTensor(oneLocal);
    }

    __aicore__ inline void Process() {
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
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<T> oneLocal = tmpOne.Get<T>();

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            // 获取 float 临时缓冲区
            AscendC::LocalTensor<float> floatX = tmpFloatX.Get<float>();
            AscendC::LocalTensor<float> floatY = tmpFloatY.Get<float>();
            AscendC::LocalTensor<float> floatOne = tmpFloatOne.Get<float>();
            AscendC::LocalTensor<float> floatNegX = tmpFloatNegX.Get<float>();


            AscendC::Cast(floatX, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            // 初始化 float 类型的全 1 张量
            AscendC::Duplicate(floatOne, 1.0f, this->processDataNum);

            // 执行 LogSigmoid 计算： -log(1 + exp(-x))
            AscendC::Muls(floatNegX, floatX, -1.0f, this->processDataNum);        // -x
            AscendC::Exp(floatY, floatNegX, this->processDataNum);                // exp(-x)
            AscendC::Add(floatY, floatOne, floatY, this->processDataNum);         // 1 + exp(-x)
            AscendC::Log(floatY, floatY, this->processDataNum);                   // log(1 + exp(-x))
            AscendC::Muls(floatY, floatY, -1.0f, this->processDataNum);           // -log(...)


            AscendC::Cast(yLocal, floatY, AscendC::RoundMode::CAST_RINT, this->processDataNum);

            // 释放 float 临时缓冲区
            tmpFloatX.FreeTensor(floatX);
            tmpFloatY.FreeTensor(floatY);
            tmpFloatOne.FreeTensor(floatOne);
            tmpFloatNegX.FreeTensor(floatNegX);

        }
        else {
            // float / half 类型直接计算（
            AscendC::LocalTensor<T> negX = tmpNegX.Get<T>();
            AscendC::Muls(negX, xLocal, static_cast<T>(-1.0f), this->processDataNum);
            AscendC::Exp(negX, negX, this->processDataNum);
            AscendC::Add(yLocal, oneLocal, negX, this->processDataNum);
            AscendC::Log(yLocal, yLocal, this->processDataNum);
            AscendC::Muls(yLocal, yLocal, static_cast<T>(-1.0f), this->processDataNum);
            tmpNegX.FreeTensor(negX);
        }

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
        tmpOne.FreeTensor(oneLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpOne;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpNegX;

    // bfloat16 专用 float 缓冲区
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatOne;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatNegX;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSigmoid<DTYPE_X> op;
    op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
        tilingData.finalBigTileNum, tilingData.finalSmallTileNum, tilingData.tileDataNum,
        tilingData.smallTailDataNum, tilingData.bigTailDataNum, tilingData.tailBlockNum);
    op.Process();
}
