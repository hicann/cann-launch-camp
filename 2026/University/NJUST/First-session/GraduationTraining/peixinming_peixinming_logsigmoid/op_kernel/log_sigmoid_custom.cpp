#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 2048;

template <typename T>
__aicore__ inline void CopyInPad(LocalTensor<T>& dst, GlobalTensor<T>& src, uint32_t offset, uint32_t len)
{
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(len * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPadExtParams<T> padParams{
        false,
        0,
        0,
        static_cast<T>(0)
    };

    DataCopyPad(dst, src[offset], copyParams, padParams);
}

template <typename T>
__aicore__ inline void CopyOutPad(GlobalTensor<T>& dst, LocalTensor<T>& src, uint32_t offset, uint32_t len)
{
    DataCopyExtParams copyParams{
        1,
        static_cast<uint32_t>(len * sizeof(T)),
        0,
        0,
        0
    };

    DataCopyPad(dst[offset], src, copyParams);
}

template <typename T>
struct CastInToFloat {
    __aicore__ inline static void Do(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t len)
    {
        Cast(dst, src, RoundMode::CAST_NONE, len);
    }
};

template <>
struct CastInToFloat<float> {
    __aicore__ inline static void Do(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len)
    {
        Adds(dst, src, 0.0f, len);
    }
};

template <typename T>
struct CastFloatToOut {
    __aicore__ inline static void Do(LocalTensor<T>& dst, LocalTensor<float>& src, uint32_t len)
    {
        Cast(dst, src, RoundMode::CAST_NONE, len);
    }
};

template <>
struct CastFloatToOut<float> {
    __aicore__ inline static void Do(LocalTensor<float>& dst, LocalTensor<float>& src, uint32_t len)
    {
        Adds(dst, src, 0.0f, len);
    }
};

template <>
struct CastFloatToOut<bfloat16_t> {
    __aicore__ inline static void Do(LocalTensor<bfloat16_t>& dst, LocalTensor<float>& src, uint32_t len)
    {
        Cast(dst, src, RoundMode::CAST_RINT, len);
    }
};

template <typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize)
    {
        this->totalSize = totalSize;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        uint32_t baseLen = totalSize / blockNum;
        uint32_t tail = totalSize % blockNum;

        this->blockLen = baseLen + static_cast<uint32_t>(blockIdx < tail);
        this->blockOffset = blockIdx * baseLen + ((blockIdx < tail) ? blockIdx : tail);

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + this->blockOffset, this->blockLen);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + this->blockOffset, this->blockLen);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(TYPE_Y));

        pipe.InitBuffer(xFloatBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(resBuf, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->blockLen; offset += TILE_LENGTH) {
            uint32_t len = TILE_LENGTH;
            if (offset + TILE_LENGTH > this->blockLen) {
                len = this->blockLen - offset;
            }

            CopyIn(offset, len);
            Compute(len);
            CopyOut(offset, len);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        CopyInPad<TYPE_X>(xLocal, xGm, offset, len);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<float> res = resBuf.Get<float>();

        // xFloat = float(x)
        CastInToFloat<TYPE_X>::Do(xFloat, xLocal, len);

        // tmp = abs(x)
        Abs(tmp, xFloat, len);

        // tmp = -abs(x)
        Muls(tmp, tmp, -1.0f, len);

        // tmp = exp(-abs(x))
        Exp(tmp, tmp, len);

        // tmp = 1 + exp(-abs(x))
        Adds(tmp, tmp, 1.0f, len);

        // tmp = log(1 + exp(-abs(x)))
        Ln(tmp, tmp, len);

        // res = min(x, 0)
        Mins(res, xFloat, 0.0f, len);

        // res = min(x, 0) - log(1 + exp(-abs(x)))
        Sub(res, res, tmp, len);

        // y = cast(res)
        CastFloatToOut<TYPE_Y>::Do(yLocal, res, len);

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        CopyOutPad<TYPE_Y>(yGm, yLocal, offset, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY;

    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> resBuf;

    GlobalTensor<TYPE_X> xGm;
    GlobalTensor<TYPE_Y> yGm;

    uint32_t totalSize = 0;
    uint32_t blockOffset = 0;
    uint32_t blockLen = 0;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.size);
    op.Process();
}
