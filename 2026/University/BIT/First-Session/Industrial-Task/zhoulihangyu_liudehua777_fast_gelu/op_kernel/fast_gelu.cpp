#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; 

template <typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t alignNum, uint32_t blockLength, uint32_t tileNum, uint32_t tileLength, uint32_t lastTileLength) {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->lastTileLength = lastTileLength;

        uint32_t core_id = GetBlockIdx();
        this->coreOffset = core_id * blockLength;
        
        if (this->coreOffset >= totalLength) {
            this->tileNum = 0;
        }

        xGm.SetGlobalBuffer((__gm__ T*)x + this->coreOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + this->coreOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(calcBuf1, this->tileLength * sizeof(T));
        pipe.InitBuffer(calcBuf2, this->tileLength * sizeof(T));
        pipe.InitBuffer(calcBuf3, this->tileLength * sizeof(T)); 
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->tileNum; i++) {
            uint32_t currentLength = (i == this->tileNum - 1) ? this->lastTileLength : this->tileLength;
            CopyIn(i, currentLength);
            Compute(i, currentLength);
            CopyOut(i, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> tmp1 = calcBuf1.Get<T>();
        LocalTensor<T> tmp2 = calcBuf2.Get<T>();
        LocalTensor<T> tmp3 = calcBuf3.Get<T>();

        Abs(tmp1, xLocal, length);

        Muls(tmp2, tmp1, (T)(-1.702), length);
        Exp(tmp2, tmp2, length);
        Adds(tmp2, tmp2, (T)(1.0), length);

        Sub(tmp3, xLocal, tmp1, length);
        Muls(tmp3, tmp3, (T)(0.851), length);
        Exp(tmp3, tmp3, length);

        Mul(tmp1, xLocal, tmp3, length);
        Div(yLocal, tmp1, tmp2, length);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopy(yGm[progress * this->tileLength], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> calcBuf1;
    TBuf<QuePosition::VECCALC> calcBuf2;
    TBuf<QuePosition::VECCALC> calcBuf3;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t coreOffset;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.alignNum, tiling_data.blockLength, tiling_data.tileNum, tiling_data.tileLength, tiling_data.lastTileLength);
    op.Process();
}