#include <cstdint>

#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace {
constexpr int64_t kCopyAlignBytes = 32;
constexpr int64_t kSkewShort = 256;
constexpr int64_t kSkewLong = 512;
constexpr int64_t kLongLoopCount = 4;

constexpr float kHalf = 0.5f;
constexpr float kOne = 1.0f;
constexpr float kRatA = 0.7974155258731228f;
constexpr float kRatB = 0.04567719200657896f;
constexpr float kRatC = 0.010705696658804787f;
constexpr float kSigmoidCube = 0.0455399241f;
constexpr float kSigmoidScale = -1.595769122f;
constexpr float kP5A = 0.7975078533000823f;
constexpr float kP5B = 0.04640164965416966f;
constexpr float kP5C = -0.00044077768784770024f;

__aicore__ inline int64_t Smaller(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline bool CopyIsAligned(int64_t elems, int64_t elemBytes)
{
    return (elems * elemBytes) % kCopyAlignBytes == 0;
}

#define GELU_QUEUE_POS(a, b, c) a##b##c
}

template <typename T, int USE_POLY5, int USE_EXP_APPROX>
class GeluKernel {
public:
    __aicore__ inline GeluKernel() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const GeluTilingData *td)
    {
        int64_t idx = static_cast<int64_t>(AscendC::GetBlockIdx());
        int64_t base = idx * td->core_elems;
        int64_t remain = td->total_elems - base;
        length_ = remain > td->core_elems ? td->core_elems : remain;
        if (length_ < 0) {
            length_ = 0;
        }
        step_ = td->tile_elems;

        xGm_.SetGlobalBuffer((__gm__ T *)input + base, length_);
        yGm_.SetGlobalBuffer((__gm__ T *)output + base, length_);

        int64_t rounds = (length_ + step_ - 1) / step_;
        oneShot_ = (rounds <= 1);
        int64_t extra = PickSkew(rounds);
        int64_t bytes = step_ * sizeof(T) + extra;

        if (oneShot_) {
            pipe_.InitBuffer(singleIn_, bytes);
            pipe_.InitBuffer(singleOut_, bytes);
        } else {
            uint8_t depth = rounds > 1 ? 2 : 1;
            pipe_.InitBuffer(inQue_, depth, bytes);
            pipe_.InitBuffer(outQue_, depth, bytes);
        }
        pipe_.InitBuffer(temp_, bytes);
    }

    __aicore__ inline void Process()
    {
        if (length_ <= 0) {
            return;
        }
        if (oneShot_) {
            ProcessOnce();
            return;
        }

        int64_t rounds = (length_ + step_ - 1) / step_;
        for (int64_t r = 0; r < rounds; ++r) {
            int64_t pos = r * step_;
            int64_t take = Smaller(step_, length_ - pos);
            ReadChunk(pos, take);
            CalcChunk(take);
            WriteChunk(pos, take);
        }
    }

private:
    __aicore__ inline int64_t PickSkew(int64_t rounds)
    {
        if (rounds >= kLongLoopCount) {
            return kSkewLong;
        }
        return rounds >= 2 ? kSkewShort : 0;
    }

    __aicore__ inline void ApproxRational(AscendC::LocalTensor<T> y,
                                          AscendC::LocalTensor<T> x,
                                          AscendC::LocalTensor<T> t,
                                          int64_t n)
    {
        int32_t len = static_cast<int32_t>(n);
        AscendC::Mul(t, x, x, len);
        AscendC::Muls(y, t, static_cast<T>(kRatB), len);
        AscendC::Adds(y, y, static_cast<T>(kRatA), len);
        AscendC::Mul(y, y, x, len);
        AscendC::Muls(t, t, static_cast<T>(kRatC), len);
        AscendC::Adds(t, t, static_cast<T>(kOne), len);
        AscendC::Div(y, y, t, len);
        AscendC::Tanh(y, y, len);
        AscendC::Muls(y, y, static_cast<T>(kHalf), len);
        AscendC::Adds(y, y, static_cast<T>(kHalf), len);
        AscendC::Mul(y, x, y, len);
    }

    __aicore__ inline void ApproxExp(AscendC::LocalTensor<T> y,
                                     AscendC::LocalTensor<T> x,
                                     AscendC::LocalTensor<T> t,
                                     int64_t n)
    {
        int32_t len = static_cast<int32_t>(n);
        AscendC::Mul(t, x, x, len);
        AscendC::Mul(y, t, x, len);
        AscendC::Muls(y, y, static_cast<T>(kSigmoidCube), len);
        AscendC::Add(y, y, x, len);
        AscendC::Muls(y, y, static_cast<T>(kSigmoidScale), len);
        AscendC::Exp(y, y, len);
        AscendC::Adds(y, y, static_cast<T>(kOne), len);
        AscendC::Div(y, x, y, len);
    }

    __aicore__ inline void ApproxPoly5(AscendC::LocalTensor<T> y,
                                       AscendC::LocalTensor<T> x,
                                       AscendC::LocalTensor<T> t,
                                       int64_t n)
    {
        int32_t len = static_cast<int32_t>(n);
        AscendC::Mul(t, x, x, len);
        AscendC::Mul(y, t, x, len);
        AscendC::Mul(t, y, t, len);
        AscendC::Muls(y, y, static_cast<T>(kP5B), len);
        AscendC::Muls(t, t, static_cast<T>(kP5C), len);
        AscendC::Add(y, y, t, len);
        AscendC::Add(y, y, x, len);
        AscendC::Muls(y, y, static_cast<T>(kP5A), len);
        AscendC::Tanh(y, y, len);
        AscendC::Muls(y, y, static_cast<T>(kHalf), len);
        AscendC::Adds(y, y, static_cast<T>(kHalf), len);
        AscendC::Mul(y, x, y, len);
    }

    __aicore__ inline void SelectApprox(AscendC::LocalTensor<T> y,
                                        AscendC::LocalTensor<T> x,
                                        AscendC::LocalTensor<T> t,
                                        int64_t n)
    {
        if constexpr (USE_POLY5 != 0) {
            ApproxPoly5(y, x, t, n);
        } else if constexpr (USE_EXP_APPROX != 0) {
            ApproxExp(y, x, t, n);
        } else {
            ApproxRational(y, x, t, n);
        }
    }

    __aicore__ inline void ProcessOnce()
    {
        int64_t n = length_;
        AscendC::LocalTensor<T> x = singleIn_.template Get<T>();
        AscendC::LocalTensor<T> y = singleOut_.template Get<T>();
        AscendC::LocalTensor<T> tmp = temp_.template Get<T>();

        AscendC::DataCopyParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(n * sizeof(T));
        cp.srcStride = 0;
        cp.dstStride = 0;

        AscendC::DataCopyPad(x, xGm_[0], cp, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        SelectApprox(y, x, tmp, n);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (CopyIsAligned(n, sizeof(T))) {
            AscendC::DataCopy(yGm_[0], y, static_cast<uint32_t>(n));
        } else {
            AscendC::DataCopyPad(yGm_[0], y, cp);
        }
    }

    __aicore__ inline void ReadChunk(int64_t offset, int64_t n)
    {
        AscendC::LocalTensor<T> x = inQue_.template AllocTensor<T>();
        AscendC::DataCopyParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(n * sizeof(T));
        cp.srcStride = 0;
        cp.dstStride = 0;
        AscendC::DataCopyPad(x, xGm_[offset], cp, {false, 0, 0, 0});
        inQue_.EnQue(x);
    }

    __aicore__ inline void CalcChunk(int64_t n)
    {
        AscendC::LocalTensor<T> x = inQue_.template DeQue<T>();
        AscendC::LocalTensor<T> y = outQue_.template AllocTensor<T>();
        AscendC::LocalTensor<T> tmp = temp_.template Get<T>();
        SelectApprox(y, x, tmp, n);
        outQue_.template EnQue<T>(y);
        inQue_.FreeTensor(x);
    }

    __aicore__ inline void WriteChunk(int64_t offset, int64_t n)
    {
        AscendC::LocalTensor<T> y = outQue_.template DeQue<T>();
        AscendC::DataCopyParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint32_t>(n * sizeof(T));
        cp.srcStride = 0;
        cp.dstStride = 0;
        AscendC::DataCopyPad(yGm_[offset], y, cp);
        outQue_.FreeTensor(y);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQue_;
    AscendC::TQue<AscendC::QuePosition::GELU_QUEUE_POS(VE, C, OUT), 1> outQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> singleIn_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> singleOut_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> temp_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    int64_t length_ = 0;
    int64_t step_ = 0;
    bool oneShot_ = false;
};

template <typename DT_INPUT_X, int USE_POLY5, int USE_EXP_APPROX>
__global__ __aicore__ void gelu(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, td, tiling);
    GeluKernel<DT_INPUT_X, USE_POLY5, USE_EXP_APPROX> kernel;
    kernel.Init(input, output, &td);
    kernel.Process();
}
