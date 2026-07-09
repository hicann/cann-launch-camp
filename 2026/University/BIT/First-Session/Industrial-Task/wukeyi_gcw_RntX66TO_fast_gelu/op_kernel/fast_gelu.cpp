// Kernel侧核函数实现：FastGelu逐元素计算
// 公式：y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
// 固定每块处理的元素数。1024同时适合float16/float32，UB占用较小，尾块由DataCopyPad处理。
constexpr uint32_t TILE_LENGTH = 1024;

template <typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t length) {
        this->totalLength = length;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), length);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), length);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(absBuf, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(expBuf, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(denBuf, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(numBuf, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->totalLength == 0) {
            return;
        }
        uint32_t loopCount = (this->totalLength + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t offset = i * TILE_LENGTH;
            uint32_t count = this->totalLength - offset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> absX = absBuf.Get<T>();
        LocalTensor<T> expPart = expBuf.Get<T>();
        LocalTensor<T> denominator = denBuf.Get<T>();
        LocalTensor<T> numerator = numBuf.Get<T>();

        // 1. absX = |x|
        Abs(absX, xLocal, count);
        // 2. expPart = exp(0.851 * (x - |x|))
        Sub(expPart, xLocal, absX, count);
        Muls(expPart, expPart, static_cast<T>(0.851f), count);
        Exp(expPart, expPart, count);
        // 3. denominator = 1 + exp(-1.702 * |x|)
        Muls(denominator, absX, static_cast<T>(-1.702f), count);
        Exp(denominator, denominator, count);
        Adds(denominator, denominator, static_cast<T>(1.0f), count);
        // 4. y = x * expPart / denominator
        Mul(numerator, xLocal, expPart, count);
        Div(yLocal, numerator, denominator, count);

        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> absBuf;
    TBuf<QuePosition::VECCALC> expBuf;
    TBuf<QuePosition::VECCALC> denBuf;
    TBuf<QuePosition::VECCALC> numBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t totalLength{0};
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tilingData.length);
    op.Process();
}