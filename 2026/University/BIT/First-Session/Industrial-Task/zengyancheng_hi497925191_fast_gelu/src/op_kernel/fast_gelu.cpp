// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, FastGeluTilingData tiling) {
        uint32_t coreIdx = GetBlockIdx();

        if (coreIdx < tiling.tailBlockNum) {
            coreDataNum = tiling.bigCoreDataNum;
            finalTileNum = tiling.finalBigTileNum;
            tailDataNum = tiling.bigTailDataNum;
            gmOffset = coreIdx * tiling.bigCoreDataNum;
        } else {
            coreDataNum = tiling.smallCoreDataNum;
            finalTileNum = tiling.finalSmallTileNum;
            tailDataNum = tiling.smallTailDataNum;
            gmOffset = tiling.tailBlockNum * tiling.bigCoreDataNum
                       + (coreIdx - tiling.tailBlockNum) * tiling.smallCoreDataNum;
        }
        tileDataNum = tiling.tileDataNum;
        totalDataNum = tiling.totalDataNum;

        if (coreDataNum == 0) {
            return;
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + gmOffset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + gmOffset, coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (coreDataNum == 0) {
            return;
        }
        int32_t loopCount = finalTileNum - 1;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i * tileDataNum, tileDataNum);
            Compute(tileDataNum);
            CopyOut(i * tileDataNum, tileDataNum);
        }

        uint32_t lastOffset = loopCount * tileDataNum;
        uint32_t validLen = totalDataNum - gmOffset - lastOffset;
        if (validLen == 0) return;
        if (validLen > tailDataNum) validLen = tailDataNum;

        CopyInLast(lastOffset, validLen, tailDataNum);
        Compute(tailDataNum);
        CopyOutLast(lastOffset, validLen, tailDataNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[offset], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInLast(uint32_t offset, uint32_t validLen, uint32_t alignedLen) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        if (validLen == alignedLen) {
            DataCopy(xLocal, xGm[offset], alignedLen);
        } else {
            DataCopyExtParams copyInParams(1, static_cast<uint32_t>(validLen * sizeof(DT_X)), 0, 0, 0);
            DataCopyPadExtParams<DT_X> padParams{true, 0, 0, (DT_X)0};
            DataCopyPad(xLocal, xGm[offset], copyInParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    // FastGelu: x / (1 + exp(-1.702 * x))
    __aicore__ inline void Compute(int32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        Muls(yLocal, xLocal, (DT_X)(-1.702), length);
        Exp(yLocal, yLocal, length);
        Adds(yLocal, yLocal, (DT_X)1.0, length);
        Div(yLocal, xLocal, yLocal, length);

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[offset], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutLast(uint32_t offset, uint32_t validLen, uint32_t alignedLen) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        if (validLen == alignedLen) {
            DataCopy(yGm[offset], yLocal, alignedLen);
        } else {
            DataCopyExtParams copyOutParams(1, static_cast<uint32_t>(validLen * sizeof(DT_X)), 0, 0, 0);
            DataCopyPad(yGm[offset], yLocal, copyOutParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<DT_X> xGm, yGm;
    uint32_t coreDataNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t finalTileNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t gmOffset = 0;
    uint32_t totalDataNum = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
