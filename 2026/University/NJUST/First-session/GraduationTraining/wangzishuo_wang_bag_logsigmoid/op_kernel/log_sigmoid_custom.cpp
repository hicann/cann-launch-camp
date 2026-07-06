#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t ALIGN_BYTES = 32;

template <typename T>
class KernelLogSigmoidCustom {
public:
    __aicore__ inline KernelLogSigmoidCustom()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t blockOffset,
        uint32_t currentBlockLength,
        uint32_t tileLength)
    {
        blockLength_ = currentBlockLength;
        tileLength_ = tileLength;

        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x) + blockOffset,
            currentBlockLength);

        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y) + blockOffset,
            currentBlockLength);

        // 输入、输出缓冲区
        pipe_.InitBuffer(
            inQueueX_,
            BUFFER_NUM,
            tileLength * sizeof(T));

        pipe_.InitBuffer(
            outQueueY_,
            BUFFER_NUM,
            tileLength * sizeof(T));

        // float32计算临时缓冲区
        pipe_.InitBuffer(
            tmpBuf0_,
            tileLength * sizeof(float));

        pipe_.InitBuffer(
            tmpBuf1_,
            tileLength * sizeof(float));

        pipe_.InitBuffer(
            tmpBuf2_,
            tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0 || tileLength_ == 0) {
            return;
        }

        uint32_t loopCount =
            (blockLength_ + tileLength_ - 1) / tileLength_;

        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t tileOffset = i * tileLength_;
            uint32_t remainLength = blockLength_ - tileOffset;

            uint32_t currentTileLength =
                remainLength > tileLength_
                    ? tileLength_
                    : remainLength;

            CopyIn(tileOffset, currentTileLength);
            Compute(currentTileLength);
            CopyOut(tileOffset, currentTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t tileOffset,
        uint32_t currentTileLength)
    {
        LocalTensor<T> xLocal =
            inQueueX_.AllocTensor<T>();

        uint32_t copyBytes =
            currentTileLength * sizeof(T);

        /*
         * 对齐数据使用普通DataCopy。
         * 尾块不对齐时才使用DataCopyPad。
         */
        if ((copyBytes % ALIGN_BYTES) == 0) {
            DataCopy(
                xLocal,
                xGm_[tileOffset],
                currentTileLength);
        } else {
            DataCopyExtParams copyParams{
                1,
                copyBytes,
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

            DataCopyPad(
                xLocal,
                xGm_[tileOffset],
                copyParams,
                padParams);
        }

        inQueueX_.EnQue<T>(xLocal);
    }

    __aicore__ inline void Compute(
        uint32_t currentTileLength)
    {
        LocalTensor<T> xLocal =
            inQueueX_.DeQue<T>();

        LocalTensor<T> yLocal =
            outQueueY_.AllocTensor<T>();

        LocalTensor<float> tmp0 =
            tmpBuf0_.Get<float>();

        LocalTensor<float> minX =
            tmpBuf1_.Get<float>();

        LocalTensor<float> work =
            tmpBuf2_.Get<float>();

        /*
         * 稳定公式：
         *
         * logSigmoid(x)
         * = min(x, 0) - ln(1 + exp(-abs(x)))
         */

        if constexpr (
            AscendC::Std::is_same<T, float>::value) {
            // minX = min(x, 0)
            Mins(
                minX,
                xLocal,
                0.0f,
                currentTileLength);

            // work = abs(x)
            Abs(
                work,
                xLocal,
                currentTileLength);
        } else {
            // half/bfloat16_t -> float
            Cast(
                tmp0,
                xLocal,
                RoundMode::CAST_NONE,
                currentTileLength);

            /*
             * 后续Mins和Abs都会读取tmp0，
             * 必须等待Cast完成。
             */
            PipeBarrier<PIPE_V>();

            // minX = min(x, 0)
            Mins(
                minX,
                tmp0,
                0.0f,
                currentTileLength);

            // work = abs(x)
            Abs(
                work,
                tmp0,
                currentTileLength);
        }

        /*
         * work由Abs写入，下一条Muls读取work。
         */
        PipeBarrier<PIPE_V>();

        // tmp0 = -abs(x)
        Muls(
            tmp0,
            work,
            -1.0f,
            currentTileLength);

        PipeBarrier<PIPE_V>();

        // work = exp(-abs(x))
        Exp(
            work,
            tmp0,
            currentTileLength);

        PipeBarrier<PIPE_V>();

        // tmp0 = 1 + exp(-abs(x))
        Adds(
            tmp0,
            work,
            1.0f,
            currentTileLength);

        PipeBarrier<PIPE_V>();

        // work = ln(1 + exp(-abs(x)))
        Ln(
            work,
            tmp0,
            currentTileLength);

        PipeBarrier<PIPE_V>();

        if constexpr (
            AscendC::Std::is_same<T, float>::value) {
            // y = min(x,0) - log(...)
            Sub(
                yLocal,
                minX,
                work,
                currentTileLength);

            PipeBarrier<PIPE_V>();
        } else {
            // 先得到float32结果
            Sub(
                tmp0,
                minX,
                work,
                currentTileLength);

            PipeBarrier<PIPE_V>();

            // float32 -> half/bfloat16_t
            Cast(
                yLocal,
                tmp0,
                RoundMode::CAST_ROUND,
                currentTileLength);

            PipeBarrier<PIPE_V>();
        }

        outQueueY_.EnQue<T>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t tileOffset,
        uint32_t currentTileLength)
    {
        LocalTensor<T> yLocal =
            outQueueY_.DeQue<T>();

        uint32_t copyBytes =
            currentTileLength * sizeof(T);

        if ((copyBytes % ALIGN_BYTES) == 0) {
            DataCopy(
                yGm_[tileOffset],
                yLocal,
                currentTileLength);
        } else {
            DataCopyExtParams copyParams{
                1,
                copyBytes,
                0,
                0,
                0
            };

            DataCopyPad(
                yGm_[tileOffset],
                yLocal,
                copyParams);
        }

        outQueueY_.FreeTensor(yLocal);
    }

private:
    uint32_t blockLength_ = 0;
    uint32_t tileLength_ = 0;

    TPipe pipe_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    TBuf<QuePosition::VECCALC> tmpBuf0_;
    TBuf<QuePosition::VECCALC> tmpBuf1_;
    TBuf<QuePosition::VECCALC> tmpBuf2_;
};

extern "C" __global__ __aicore__
void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        LogSigmoidCustomTilingData);

    GET_TILING_DATA(
        tilingData,
        tiling);

    uint32_t blockIdx = GetBlockIdx();

    uint32_t blockOffset =
        blockIdx * tilingData.blockLength;

    if (blockOffset >= tilingData.size) {
        return;
    }

    uint32_t remainLength =
        tilingData.size - blockOffset;

    uint32_t currentBlockLength =
        remainLength > tilingData.blockLength
            ? tilingData.blockLength
            : remainLength;

    if (currentBlockLength == 0 ||
        tilingData.tileLength == 0) {
        return;
    }

    KernelLogSigmoidCustom<DTYPE_X> op;

    op.Init(
        x,
        y,
        blockOffset,
        currentBlockLength,
        tilingData.tileLength);

    op.Process();
}
