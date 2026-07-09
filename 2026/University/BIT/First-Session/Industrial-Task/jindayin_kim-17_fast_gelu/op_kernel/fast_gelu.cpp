// Kernel侧核函数实现：FastGelu逐元素计算
// 公式：y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
// 2048比1024减少循环次数，同时UB占用仍然较安全；尾块由DataCopyPad处理。
constexpr uint32_t TILE_LENGTH = 2048;

template <typename T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t blockDim) {
        const uint32_t blockIdx = GetBlockIdx();
        if (blockDim == 0) {
            blockDim = 1;
        }

        // 将一维展平后的总数据按AI Core均分。
        // 前 remainder 个核多处理1个元素，保证所有元素都覆盖且不重叠。
        const uint32_t base = totalLength / blockDim;
        const uint32_t remainder = totalLength % blockDim;
        this->coreLength = base + (blockIdx < remainder ? 1 : 0);
        this->coreOffset = blockIdx * base + (blockIdx < remainder ? blockIdx : remainder);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }
        const uint32_t loopCount = (this->coreLength + TILE_LENGTH - 1) / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; ++i) {
            const uint32_t localOffset = i * TILE_LENGTH;
            uint32_t count = this->coreLength - localOffset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }
            CopyIn(this->coreOffset + localOffset, count);
            Compute(count);
            CopyOut(this->coreOffset + localOffset, count);
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
        LocalTensor<T> expBuf = tmpBuf.Get<T>();

        // FastGelu simplifies to y = x / (1 + exp(-1.702 * x)).
        Muls(expBuf, xLocal, static_cast<T>(-1.702f), count);
        Exp(expBuf, expBuf, count);
        Adds(expBuf, expBuf, static_cast<T>(1.0f), count);
        Div(yLocal, xLocal, expBuf, count);

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
    TBuf<QuePosition::VECCALC> tmpBuf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t coreOffset{0};
    uint32_t coreLength{0};
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tilingData, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tilingData.length, tilingData.blockDim);
    op.Process();
}
