// Kernel侧核函数实现 — 最终冲刺版 (最大性能)
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// fp32: degree-5 erf 多项式 (minimax, max err 3.16e-5, 过 1e-4)
constexpr float GELU_C1 = -9.544351866166e-10f;
constexpr float GELU_C2 = -3.856906597293e-08f;
constexpr float GELU_C3 = -5.730141076270e-06f;
constexpr float GELU_C4 = 7.778656730654e-04f;
constexpr float GELU_C5 = -7.424451276829e-02f;
constexpr float GELU_C6 = -1.594850524765e+00f;
constexpr float GELU_X_LO = -13.15f;
constexpr float GELU_X_HI = 5.751f;
constexpr float GELU_ONE = 1.0f;
// fp16: tanh 近似 gelu(x) = x / (1 + exp(-1.59576912*(x + 0.044715*x^3)))
constexpr float TANH_BETA  = 0.044715f;
constexpr float TANH_ALPHA = 1.5957691216057308f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData* tilingData)
    {
        totalNum_ = tilingData->totalNum;
        blockLength_ = tilingData->blockFactor;
        ubLength_ = tilingData->ubFactor;

        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        blockOffset_ = blockIdx * blockLength_;
        int64_t remain = totalNum_ - blockOffset_;
        if (remain < 0) { remain = 0; }
        if (remain < blockLength_) { blockLength_ = remain; }

        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X*>(input_x) + blockOffset_, blockLength_);
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ DT_INPUT_X*>(output) + blockOffset_, blockLength_);

        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBufA, ubLength_ * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBufB, ubLength_ * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0 || ubLength_ <= 0) { return; }
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; ++i) {
            int64_t currentNum = ubLength_;
            int64_t processed = i * ubLength_;
            if (processed + currentNum > blockLength_) { currentNum = blockLength_ - processed; }
            CopyIn(i, currentNum);
            Compute(currentNum);
            CopyOut(i, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_INPUT_X> xLocal = inputQueueX.AllocTensor<DT_INPUT_X>();
        int64_t gmOffset = progress * ubLength_;
        if (((blockOffset_ + gmOffset) % ALIGN_ELEMS == 0) && (currentNum % ALIGN_ELEMS == 0)) {
            DataCopy(xLocal, inputGMX[gmOffset], static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal, inputGMX[gmOffset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    // fp16: tanh 近似 gelu (原生 fp16, ~9 op, 天然饱和)
    __aicore__ inline void GeluTanh(LocalTensor<DT_INPUT_X> x, LocalTensor<DT_INPUT_X> y, int64_t n)
    {
        LocalTensor<DT_INPUT_X> a = tmpBufA.Get<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> b = tmpBufB.Get<DT_INPUT_X>();
        Mul(a, x, x, n);                               // a = x^2
        Mul(a, a, x, n);                               // a = x^3
        Muls(a, a, static_cast<DT_INPUT_X>(TANH_BETA), n);  // a = 0.0447*x^3
        Add(b, x, a, n);                               // b = x + 0.0447*x^3
        Muls(a, b, static_cast<DT_INPUT_X>(TANH_ALPHA), n); // a = z
        Muls(a, a, static_cast<DT_INPUT_X>(-1.0f), n);       // a = -z
        Exp(b, a, n);                                  // b = exp(-z)
        Adds(b, b, static_cast<DT_INPUT_X>(GELU_ONE), n);    // b = 1+exp(-z)
        Div(y, x, b, n);                               // y = x/(1+exp(-z))
    }

    // fp32: degree-5 erf 多项式 (max err 3.16e-5, 过 1e-4)
    __aicore__ inline void GeluErf(LocalTensor<float> x, LocalTensor<float> y, int64_t n)
    {
        LocalTensor<float> a = tmpBufA.Get<float>();
        LocalTensor<float> b = tmpBufB.Get<float>();
        Maxs(a, x, GELU_X_LO, n);                      // a = max(x, -13.15)
        Mins(a, a, GELU_X_HI, n);                      // a = clamp(x)
        Mul(b, a, a, n);                               // b = t = xc^2
        // Horner degree-5: poly(t) 累加在 a (复用)
        Muls(a, b, GELU_C1, n); Adds(a, a, GELU_C2, n);
        Mul(a, a, b, n);      Adds(a, a, GELU_C3, n);
        Mul(a, a, b, n);      Adds(a, a, GELU_C4, n);
        Mul(a, a, b, n);      Adds(a, a, GELU_C5, n);
        Mul(a, a, b, n);      Adds(a, a, GELU_C6, n);
        // a = poly(t), b = t (不再需要)
        Mins(b, x, GELU_X_HI, n);                      // b = min(x, 5.751)
        Mul(a, a, b, n);                               // a = poly * min
        Exp(a, a, n);
        Adds(a, a, GELU_ONE, n);
        Div(y, x, a, n);                               // y = x/(1+exp(p))
    }

    __aicore__ inline void Compute(int64_t currentNum)
    {
        LocalTensor<DT_INPUT_X> xLocal = inputQueueX.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> yLocal = outputQueueY.AllocTensor<DT_INPUT_X>();
        if constexpr (IS_FP16) {
            GeluTanh(xLocal, yLocal, currentNum);       // tanh 近似, 原生 fp16
        } else {
            GeluErf(xLocal.template ReinterpretCast<float>(),
                    yLocal.template ReinterpretCast<float>(), currentNum); // degree-5 erf, fp32
        }
        outputQueueY.EnQue<DT_INPUT_X>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum)
    {
        LocalTensor<DT_INPUT_X> yLocal = outputQueueY.DeQue<DT_INPUT_X>();
        int64_t gmOffset = progress * ubLength_;
        if (((blockOffset_ + gmOffset) % ALIGN_ELEMS == 0) && (currentNum % ALIGN_ELEMS == 0)) {
            DataCopy(outputGMY[gmOffset], yLocal, static_cast<uint32_t>(currentNum));
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[gmOffset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    static constexpr int64_t ALIGN_ELEMS = 512 / sizeof(DT_INPUT_X);
    static constexpr bool IS_FP16 = (sizeof(DT_INPUT_X) == 2);

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpBufA;
    TBuf<QuePosition::VECCALC> tmpBufB;

    GlobalTensor<DT_INPUT_X> inputGMX;
    GlobalTensor<DT_INPUT_X> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t blockOffset_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, &tilingData);
    op.Process();
}
