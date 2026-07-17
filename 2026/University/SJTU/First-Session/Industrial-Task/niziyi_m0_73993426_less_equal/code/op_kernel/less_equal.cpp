#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {

__aicore__ inline uint32_t LessEqualMin(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t LessEqualAlignUp(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

template <typename T>
class LessEqualVectorWorkspace;

template <>
class LessEqualVectorWorkspace<half> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<half> &x1,
        const AscendC::LocalTensor<half> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t computeCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const half scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, computeCount);
        }
        if (x2Scalar) {
            const half scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, computeCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1, x2, AscendC::CMPMODE::LE, computeCount);
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Duplicate(result, static_cast<half>(1.0), computeCount);
        AscendC::Select(result, mask, result, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, computeCount);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspace<float> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<float> &x1,
        const AscendC::LocalTensor<float> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t compareCount = LessEqualAlignUp(count, 64U);
        const uint32_t outputCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const float scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, compareCount);
        }
        if (x2Scalar) {
            const float scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, compareCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1, x2, AscendC::CMPMODE::LE, compareCount);
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Duplicate(result, static_cast<half>(1.0), outputCount);
        AscendC::Select(result, mask, result, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, outputCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, outputCount);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspace<int8_t> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(x1HalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(x2HalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<int8_t> &x1,
        const AscendC::LocalTensor<int8_t> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t computeCount = LessEqualAlignUp(count, 256U);
        AscendC::LocalTensor<half> x1Half = x1HalfBuf_.Get<half>();
        AscendC::LocalTensor<half> x2Half = x2HalfBuf_.Get<half>();
        if (x1Scalar) {
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE, 1U);
            const half scalar = x1Half.GetValue(0);
            AscendC::Duplicate(x1Half, scalar, computeCount);
        } else {
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE, computeCount);
        }
        if (x2Scalar) {
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE, 1U);
            const half scalar = x2Half.GetValue(0);
            AscendC::Duplicate(x2Half, scalar, computeCount);
        } else {
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE, computeCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1Half, x2Half, AscendC::CMPMODE::LE, computeCount);
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Duplicate(result, static_cast<half>(1.0), computeCount);
        AscendC::Select(result, mask, result, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, computeCount);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1HalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2HalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspace<int32_t> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(maxIntBuf_, tileLength * sizeof(int32_t));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<int32_t> &x1,
        const AscendC::LocalTensor<int32_t> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t compareCount = LessEqualAlignUp(count, 64U);
        const uint32_t outputCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const int32_t scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, compareCount);
        }
        if (x2Scalar) {
            const int32_t scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, compareCount);
        }
        AscendC::LocalTensor<int32_t> maxLocal = maxIntBuf_.Get<int32_t>();
        AscendC::Max(maxLocal, x1, x2, static_cast<int32_t>(compareCount));
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, maxLocal, x2, AscendC::CMPMODE::EQ, compareCount);
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Duplicate(result, static_cast<half>(1.0), outputCount);
        AscendC::Select(result, mask, result, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, outputCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, outputCount);
    }

private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxIntBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};


// Dedicated workspace used only by the SCH_MODE=LINEAR_DB_ONES binary.
// The normal binary continues to use the original V28/V29a workspace above,
// so its class layout and per-tile instruction sequence remain unchanged.
template <typename T>
class LessEqualVectorWorkspaceOnes;

template <>
class LessEqualVectorWorkspaceOnes<half> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(onesHalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::Duplicate(ones, static_cast<half>(1.0), tileLength);
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<half> &x1,
        const AscendC::LocalTensor<half> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t computeCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const half scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, computeCount);
        }
        if (x2Scalar) {
            const half scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, computeCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1, x2, AscendC::CMPMODE::LE, computeCount);
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Select(result, mask, ones, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, computeCount);
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspaceOnes<float> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(onesHalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::Duplicate(ones, static_cast<half>(1.0), tileLength);
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<float> &x1,
        const AscendC::LocalTensor<float> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t compareCount = LessEqualAlignUp(count, 64U);
        const uint32_t outputCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const float scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, compareCount);
        }
        if (x2Scalar) {
            const float scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, compareCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1, x2, AscendC::CMPMODE::LE, compareCount);
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Select(result, mask, ones, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, outputCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, outputCount);
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspaceOnes<int8_t> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(x1HalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(x2HalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(onesHalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::Duplicate(ones, static_cast<half>(1.0), tileLength);
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<int8_t> &x1,
        const AscendC::LocalTensor<int8_t> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t computeCount = LessEqualAlignUp(count, 256U);
        AscendC::LocalTensor<half> x1Half = x1HalfBuf_.Get<half>();
        AscendC::LocalTensor<half> x2Half = x2HalfBuf_.Get<half>();
        if (x1Scalar) {
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE, 1U);
            const half scalar = x1Half.GetValue(0);
            AscendC::Duplicate(x1Half, scalar, computeCount);
        } else {
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE, computeCount);
        }
        if (x2Scalar) {
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE, 1U);
            const half scalar = x2Half.GetValue(0);
            AscendC::Duplicate(x2Half, scalar, computeCount);
        } else {
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE, computeCount);
        }
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, x1Half, x2Half, AscendC::CMPMODE::LE, computeCount);
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Select(result, mask, ones, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, computeCount);
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1HalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2HalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <>
class LessEqualVectorWorkspaceOnes<int32_t> {
public:
    __aicore__ inline void Init(AscendC::TPipe &pipe, uint32_t tileLength)
    {
        const uint32_t maskBytes = LessEqualAlignUp((tileLength + 7U) / 8U, 32U);
        pipe.InitBuffer(maskBuf_, maskBytes);
        pipe.InitBuffer(maxIntBuf_, tileLength * sizeof(int32_t));
        pipe.InitBuffer(onesHalfBuf_, tileLength * sizeof(half));
        pipe.InitBuffer(resultHalfBuf_, tileLength * sizeof(half));
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::Duplicate(ones, static_cast<half>(1.0), tileLength);
    }

    __aicore__ inline void Run(
        const AscendC::LocalTensor<int32_t> &x1,
        const AscendC::LocalTensor<int32_t> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t compareCount = LessEqualAlignUp(count, 64U);
        const uint32_t outputCount = LessEqualAlignUp(count, 128U);
        if (x1Scalar) {
            const int32_t scalar = x1.GetValue(0);
            AscendC::Duplicate(x1, scalar, compareCount);
        }
        if (x2Scalar) {
            const int32_t scalar = x2.GetValue(0);
            AscendC::Duplicate(x2, scalar, compareCount);
        }
        AscendC::LocalTensor<int32_t> maxLocal = maxIntBuf_.Get<int32_t>();
        AscendC::Max(maxLocal, x1, x2, static_cast<int32_t>(compareCount));
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::Compare(mask, maxLocal, x2, AscendC::CMPMODE::EQ, compareCount);
        AscendC::LocalTensor<half> ones = onesHalfBuf_.Get<half>();
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();
        AscendC::Select(result, mask, ones, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, outputCount);
        AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE, outputCount);
    }
private:
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxIntBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> onesHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
};

template <typename T, bool USE_ONES>
struct LessEqualWorkspaceSelector;

template <typename T>
struct LessEqualWorkspaceSelector<T, false> {
    using Type = LessEqualVectorWorkspace<T>;
};

template <typename T>
struct LessEqualWorkspaceSelector<T, true> {
    using Type = LessEqualVectorWorkspaceOnes<T>;
};

template <typename T, uint8_t BUFFER_NUM, bool USE_ONES>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling,
                                AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        totalLength_ = tiling.totalLength;
        innerLength_ = tiling.innerLength;
        outerLength_ = tiling.outerLength;
        workPerCore_ = tiling.workPerCore;
        tileLength_ = tiling.tileLength;
        rank_ = tiling.rank;
        mode_ = tiling.mode;
        coreId_ = AscendC::GetBlockIdx();

        if (mode_ == LESS_EQUAL_MODE_LINEAR) {
            x1Stride_[0] = tiling.x1Stride[0];
            x2Stride_[0] = tiling.x2Stride[0];
        } else {
            for (uint32_t i = 0U; i < rank_; ++i) {
                outShape_[i] = tiling.outShape[i];
                x1Stride_[i] = tiling.x1Stride[i];
                x2Stride_[i] = tiling.x2Stride[i];
            }
        }

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y));

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, BUFFER_NUM, tileLength_ * sizeof(uint8_t));

        vectorWorkspace_.Init(*pipe_, tileLength_);
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0U) {
            return;
        }
        if (mode_ == LESS_EQUAL_MODE_LINEAR) {
            ProcessLinear();
        } else if (mode_ == LESS_EQUAL_MODE_REUSE_X1) {
            ProcessReuseX1();
        } else if (mode_ == LESS_EQUAL_MODE_REUSE_X2) {
            ProcessReuseX2();
        } else {
            ProcessRows();
        }
    }

private:
    __aicore__ inline void CopyInput1(uint32_t offset, uint32_t validCount,
                                      bool scalarInput)
    {
        AscendC::LocalTensor<T> local = x1Queue_.AllocTensor<T>();
        const uint32_t copyCount = scalarInput ? 1U : validCount;
        const uint32_t byteCount = copyCount * sizeof(T);
        if (!scalarInput && ((offset * sizeof(T)) % 32U == 0U) &&
            (byteCount % 32U == 0U)) {
            AscendC::DataCopy(local, x1Gm_[offset], copyCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, x1Gm_[offset], copyParams, padParams);
        }
        x1Queue_.EnQue<T>(local);
    }

    __aicore__ inline void CopyInput2(uint32_t offset, uint32_t validCount,
                                      bool scalarInput)
    {
        AscendC::LocalTensor<T> local = x2Queue_.AllocTensor<T>();
        const uint32_t copyCount = scalarInput ? 1U : validCount;
        const uint32_t byteCount = copyCount * sizeof(T);
        if (!scalarInput && ((offset * sizeof(T)) % 32U == 0U) &&
            (byteCount % 32U == 0U)) {
            AscendC::DataCopy(local, x2Gm_[offset], copyCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, byteCount, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, x2Gm_[offset], copyParams, padParams);
        }
        x2Queue_.EnQue<T>(local);
    }

    __aicore__ inline void ComputeAndCopy(
        const AscendC::LocalTensor<T> &x1Local,
        const AscendC::LocalTensor<T> &x2Local,
        uint32_t outputOffset,
        uint32_t validCount,
        bool x1Scalar,
        bool x2Scalar)
    {
        AscendC::LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        vectorWorkspace_.Run(x1Local, x2Local, yLocal, validCount,
                             x1Scalar, x2Scalar);
        yQueue_.EnQue<uint8_t>(yLocal);
        AscendC::LocalTensor<uint8_t> ready = yQueue_.DeQue<uint8_t>();

        if (((outputOffset & 31U) == 0U) && ((validCount & 31U) == 0U)) {
            AscendC::DataCopy(yGm_[outputOffset], ready, validCount);
        } else {
            AscendC::DataCopyExtParams copyParams{1, validCount, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[outputOffset], ready, copyParams);
        }
        yQueue_.FreeTensor(ready);
    }

    __aicore__ inline void ProcessSegment(uint32_t x1Offset, uint32_t x2Offset,
                                          uint32_t outputOffset,
                                          uint32_t validCount,
                                          bool x1Scalar, bool x2Scalar)
    {
        CopyInput1(x1Offset, validCount, x1Scalar);
        CopyInput2(x2Offset, validCount, x2Scalar);
        AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        ComputeAndCopy(x1Local, x2Local, outputOffset, validCount,
                       x1Scalar, x2Scalar);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ProcessLinear()
    {
        const uint32_t start = coreId_ * workPerCore_;
        if (start >= totalLength_) {
            return;
        }
        const uint32_t coreLength = LessEqualMin(workPerCore_, totalLength_ - start);
        const bool x1Scalar = x1Stride_[0] == 0U;
        const bool x2Scalar = x2Stride_[0] == 0U;
        if (coreLength <= tileLength_) {
            ProcessSegment(x1Scalar ? 0U : start,
                           x2Scalar ? 0U : start,
                           start, coreLength, x1Scalar, x2Scalar);
            return;
        }
        uint32_t done = 0U;
        while (done < coreLength) {
            const uint32_t valid = LessEqualMin(tileLength_, coreLength - done);
            ProcessSegment(x1Scalar ? 0U : start + done,
                           x2Scalar ? 0U : start + done,
                           start + done, valid, x1Scalar, x2Scalar);
            done += valid;
        }
    }

    __aicore__ inline void InitRowOffsets(uint32_t row,
                                          uint32_t &x1Offset,
                                          uint32_t &x2Offset,
                                          uint32_t index[LESS_EQUAL_MAX_RANK])
    {
        x1Offset = 0U;
        x2Offset = 0U;
        for (uint32_t i = 0U; i < rank_; ++i) {
            index[i] = 0U;
        }
        uint32_t value = row;
        for (int32_t i = static_cast<int32_t>(rank_) - 2; i >= 0; --i) {
            const uint32_t dim = outShape_[i];
            const uint32_t coordinate = dim == 0U ? 0U : value % dim;
            value = dim == 0U ? 0U : value / dim;
            index[i] = coordinate;
            x1Offset += coordinate * x1Stride_[i];
            x2Offset += coordinate * x2Stride_[i];
        }
    }

    __aicore__ inline void AdvanceRow(uint32_t &x1Offset,
                                      uint32_t &x2Offset,
                                      uint32_t index[LESS_EQUAL_MAX_RANK])
    {
        for (int32_t i = static_cast<int32_t>(rank_) - 2; i >= 0; --i) {
            ++index[i];
            if (index[i] < outShape_[i]) {
                x1Offset += x1Stride_[i];
                x2Offset += x2Stride_[i];
                return;
            }
            index[i] = 0U;
            x1Offset -= x1Stride_[i] * (outShape_[i] - 1U);
            x2Offset -= x2Stride_[i] * (outShape_[i] - 1U);
        }
    }

    __aicore__ inline void ProcessRows()
    {
        const uint32_t rowStart = coreId_ * workPerCore_;
        if (rowStart >= outerLength_) {
            return;
        }
        const uint32_t rowCount = LessEqualMin(workPerCore_, outerLength_ - rowStart);
        uint32_t x1RowOffset = 0U;
        uint32_t x2RowOffset = 0U;
        uint32_t index[LESS_EQUAL_MAX_RANK];
        InitRowOffsets(rowStart, x1RowOffset, x2RowOffset, index);
        const bool x1InnerScalar = x1Stride_[rank_ - 1U] == 0U;
        const bool x2InnerScalar = x2Stride_[rank_ - 1U] == 0U;

        for (uint32_t row = 0U; row < rowCount; ++row) {
            uint32_t col = 0U;
            while (col < innerLength_) {
                const uint32_t valid = LessEqualMin(tileLength_, innerLength_ - col);
                ProcessSegment(x1RowOffset + (x1InnerScalar ? 0U : col),
                               x2RowOffset + (x2InnerScalar ? 0U : col),
                               (rowStart + row) * innerLength_ + col,
                               valid, x1InnerScalar, x2InnerScalar);
                col += valid;
            }
            if (row + 1U < rowCount) {
                AdvanceRow(x1RowOffset, x2RowOffset, index);
            }
        }
    }

    __aicore__ inline void ProcessReuseX1()
    {
        const uint32_t rowStart = coreId_ * workPerCore_;
        if (rowStart >= outerLength_) {
            return;
        }
        const uint32_t rowCount = LessEqualMin(workPerCore_, outerLength_ - rowStart);
        uint32_t col = 0U;
        while (col < innerLength_) {
            const uint32_t valid = LessEqualMin(tileLength_, innerLength_ - col);
            CopyInput1(col, valid, false);
            AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
            for (uint32_t row = 0U; row < rowCount; ++row) {
                CopyInput2((rowStart + row) * innerLength_ + col, valid, false);
                AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
                ComputeAndCopy(x1Local, x2Local,
                               (rowStart + row) * innerLength_ + col,
                               valid, false, false);
                x2Queue_.FreeTensor(x2Local);
            }
            x1Queue_.FreeTensor(x1Local);
            col += valid;
        }
    }

    __aicore__ inline void ProcessReuseX2()
    {
        const uint32_t rowStart = coreId_ * workPerCore_;
        if (rowStart >= outerLength_) {
            return;
        }
        const uint32_t rowCount = LessEqualMin(workPerCore_, outerLength_ - rowStart);
        uint32_t col = 0U;
        while (col < innerLength_) {
            const uint32_t valid = LessEqualMin(tileLength_, innerLength_ - col);
            CopyInput2(col, valid, false);
            AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
            for (uint32_t row = 0U; row < rowCount; ++row) {
                CopyInput1((rowStart + row) * innerLength_ + col, valid, false);
                AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
                ComputeAndCopy(x1Local, x2Local,
                               (rowStart + row) * innerLength_ + col,
                               valid, false, false);
                x1Queue_.FreeTensor(x1Local);
            }
            x2Queue_.FreeTensor(x2Local);
            col += valid;
        }
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x1Queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x2Queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQueue_;
    typename LessEqualWorkspaceSelector<T, USE_ONES>::Type vectorWorkspace_;

    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;

    uint32_t totalLength_ = 0U;
    uint32_t innerLength_ = 0U;
    uint32_t outerLength_ = 0U;
    uint32_t workPerCore_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t rank_ = 1U;
    uint32_t mode_ = LESS_EQUAL_MODE_LINEAR;
    uint32_t coreId_ = 0U;
    uint32_t outShape_[LESS_EQUAL_MAX_RANK] = {};
    uint32_t x1Stride_[LESS_EQUAL_MAX_RANK] = {};
    uint32_t x2Stride_[LESS_EQUAL_MAX_RANK] = {};
};

}  // namespace

constexpr uint32_t LessEqualStaticAlign32(uint32_t value)
{
    return (value + 31U) & ~31U;
}


class KernelLessEqualStaticHalf {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling)
    {
        totalLength_ = tiling.totalLength;
        workPerCore_ = tiling.workPerCore;
        coreId_ = AscendC::GetBlockIdx();
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y));
        x1Raw_ = reinterpret_cast<__gm__ uint16_t *>(x1);
        x2Raw_ = reinterpret_cast<__gm__ uint16_t *>(x2);
        yRaw_ = reinterpret_cast<__gm__ uint8_t *>(y);
    }

    __aicore__ inline void Process()
    {
        const uint32_t start = coreId_ * workPerCore_;
        if (start >= totalLength_) {
            return;
        }
        const uint32_t length =
            LessEqualMin(workPerCore_, totalLength_ - start);

        AscendC::LocalTensor<half> x1Local(
            AscendC::TPosition::VECCALC, X1_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<half> x2Local(
            AscendC::TPosition::VECCALC, X2_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> yLocal(
            AscendC::TPosition::VECCALC, Y_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> maskLocal(
            AscendC::TPosition::VECCALC, MASK_ADDR, MASK_BYTES);
        AscendC::LocalTensor<half> onesLocal(
            AscendC::TPosition::VECCALC, ONES_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<half> resultLocal(
            AscendC::TPosition::VECCALC, RESULT_ADDR, TILE_ELEMENTS);

        const uint32_t vectorLength = length / 32U * 32U;
        if (vectorLength != 0U) {
            const uint32_t constantCount = LessEqualAlignUp(
                LessEqualMin(vectorLength, TILE_ELEMENTS), 128U);
            AscendC::Duplicate(onesLocal, static_cast<half>(1.0),
                               constantCount);
        }

        uint32_t progress = 0U;
        while (progress < vectorLength) {
            const uint32_t validCount = LessEqualMin(
                TILE_ELEMENTS, vectorLength - progress);
            const uint32_t computeCount =
                LessEqualAlignUp(validCount, 128U);
            AscendC::DataCopy(x1Local, x1Gm_[start + progress],
                              validCount);
            AscendC::DataCopy(x2Local, x2Gm_[start + progress],
                              validCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);

            AscendC::Compare(maskLocal, x1Local, x2Local,
                             AscendC::CMPMODE::LE, computeCount);
            AscendC::Select(
                resultLocal, maskLocal, onesLocal, static_cast<half>(0.0),
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
            AscendC::Cast(yLocal, resultLocal,
                          AscendC::RoundMode::CAST_NONE, computeCount);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::DataCopy(yGm_[start + progress], yLocal, validCount);
            progress += validCount;
            if (progress < vectorLength) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            }
        }

        for (uint32_t i = vectorLength; i < length; ++i) {
            yRaw_[start + i] = LessEqualHalfBits(
                x1Raw_[start + i], x2Raw_[start + i]);
        }
    }

private:
    __aicore__ inline static uint8_t LessEqualHalfBits(uint16_t a,
                                                       uint16_t b)
    {
        constexpr uint16_t SIGN_MASK = 0x8000U;
        constexpr uint16_t ABS_MASK = 0x7FFFU;
        constexpr uint16_t INF_CODE = 0x7C00U;
        const uint16_t absA = a & ABS_MASK;
        const uint16_t absB = b & ABS_MASK;
        if (absA > INF_CODE || absB > INF_CODE) {
            return 0U;
        }
        if ((absA | absB) == 0U) {
            return 1U;
        }
        const uint16_t keyA = (a & SIGN_MASK) != 0U
            ? static_cast<uint16_t>(~a)
            : static_cast<uint16_t>(a ^ SIGN_MASK);
        const uint16_t keyB = (b & SIGN_MASK) != 0U
            ? static_cast<uint16_t>(~b)
            : static_cast<uint16_t>(b ^ SIGN_MASK);
        return keyA <= keyB ? 1U : 0U;
    }

private:
    static constexpr uint32_t TILE_ELEMENTS = 4096U;
    static constexpr int32_t EVENT_ID = 0;
    static constexpr uint32_t X1_ADDR = 0U;
    static constexpr uint32_t X2_ADDR =
        LessEqualStaticAlign32(X1_ADDR + TILE_ELEMENTS * sizeof(half));
    static constexpr uint32_t Y_ADDR =
        LessEqualStaticAlign32(X2_ADDR + TILE_ELEMENTS * sizeof(half));
    static constexpr uint32_t MASK_BYTES =
        LessEqualStaticAlign32((TILE_ELEMENTS + 7U) / 8U);
    static constexpr uint32_t MASK_ADDR =
        LessEqualStaticAlign32(Y_ADDR + TILE_ELEMENTS * sizeof(uint8_t));
    static constexpr uint32_t ONES_ADDR =
        LessEqualStaticAlign32(MASK_ADDR + MASK_BYTES);
    static constexpr uint32_t RESULT_ADDR =
        LessEqualStaticAlign32(ONES_ADDR + TILE_ELEMENTS * sizeof(half));

    AscendC::GlobalTensor<half> x1Gm_;
    AscendC::GlobalTensor<half> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    __gm__ uint16_t *x1Raw_ = nullptr;
    __gm__ uint16_t *x2Raw_ = nullptr;
    __gm__ uint8_t *yRaw_ = nullptr;
    uint32_t totalLength_ = 0U;
    uint32_t workPerCore_ = 0U;
    uint32_t coreId_ = 0U;
};


// V31: minimal binary for exact-shape FP16/INT8 workloads that are already
// split so each core owns a single segment.  It intentionally does not copy
// rank/outShape/stride arrays and has no mode, scalar-broadcast, ROW or REUSE
// branches.  This targets the fixed Scalar overhead visible in TP1/TP4.
template <typename T>
class KernelLessEqualFastLight {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling,
                                AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        totalLength_ = tiling.totalLength;
        workPerCore_ = tiling.workPerCore;
        tileLength_ = tiling.tileLength;
        coreId_ = AscendC::GetBlockIdx();

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y));

        pipe_->InitBuffer(x1Queue_, 1, tileLength_ * sizeof(T));
        pipe_->InitBuffer(x2Queue_, 1, tileLength_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, 1, tileLength_ * sizeof(uint8_t));

        const uint32_t maskBytes =
            LessEqualAlignUp((tileLength_ + 7U) / 8U, 32U);
        pipe_->InitBuffer(maskBuf_, maskBytes);
        pipe_->InitBuffer(resultHalfBuf_, tileLength_ * sizeof(half));

        if constexpr (std::is_same_v<T, int8_t>) {
            pipe_->InitBuffer(x1HalfBuf_, tileLength_ * sizeof(half));
            pipe_->InitBuffer(x2HalfBuf_, tileLength_ * sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t start = coreId_ * workPerCore_;
        if (start >= totalLength_) {
            return;
        }

        const uint32_t validCount =
            LessEqualMin(workPerCore_, totalLength_ - start);

        AscendC::LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        CopyInput(x1Local, x1Gm_, start, validCount);
        x1Queue_.EnQue<T>(x1Local);

        AscendC::LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        CopyInput(x2Local, x2Gm_, start, validCount);
        x2Queue_.EnQue<T>(x2Local);

        x1Local = x1Queue_.DeQue<T>();
        x2Local = x2Queue_.DeQue<T>();

        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue_.AllocTensor<uint8_t>();
        Compute(x1Local, x2Local, yLocal, validCount);
        yQueue_.EnQue<uint8_t>(yLocal);

        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);

        AscendC::LocalTensor<uint8_t> ready =
            yQueue_.DeQue<uint8_t>();
        if (((start & 31U) == 0U) && ((validCount & 31U) == 0U)) {
            AscendC::DataCopy(yGm_[start], ready, validCount);
        } else {
            AscendC::DataCopyExtParams params{1, validCount, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[start], ready, params);
        }
        yQueue_.FreeTensor(ready);
    }

private:
    __aicore__ inline void CopyInput(
        const AscendC::LocalTensor<T> &dst,
        const AscendC::GlobalTensor<T> &src,
        uint32_t start, uint32_t count)
    {
        const uint32_t byteCount = count * sizeof(T);
        if (((start * sizeof(T)) & 31U) == 0U &&
            (byteCount & 31U) == 0U) {
            AscendC::DataCopy(dst, src[start], count);
        } else {
            AscendC::DataCopyExtParams params{1, byteCount, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(dst, src[start], params, pad);
        }
    }

    __aicore__ inline void Compute(
        const AscendC::LocalTensor<T> &x1,
        const AscendC::LocalTensor<T> &x2,
        const AscendC::LocalTensor<uint8_t> &y,
        uint32_t count)
    {
        AscendC::LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
        AscendC::LocalTensor<half> result = resultHalfBuf_.Get<half>();

        if constexpr (std::is_same_v<T, half>) {
            const uint32_t computeCount =
                LessEqualAlignUp(count, 128U);
            AscendC::Compare(mask, x1, x2, AscendC::CMPMODE::LE,
                             computeCount);
            AscendC::Duplicate(result, static_cast<half>(1.0),
                               computeCount);
            AscendC::Select(
                result, mask, result, static_cast<half>(0.0),
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                computeCount);
            AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE,
                          computeCount);
        } else {
            static_assert(std::is_same_v<T, int8_t>,
                          "FAST_LIGHT supports only FP16 and INT8");
            const uint32_t computeCount =
                LessEqualAlignUp(count, 256U);
            AscendC::LocalTensor<half> x1Half =
                x1HalfBuf_.Get<half>();
            AscendC::LocalTensor<half> x2Half =
                x2HalfBuf_.Get<half>();
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE,
                          computeCount);
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE,
                          computeCount);
            AscendC::Compare(mask, x1Half, x2Half,
                             AscendC::CMPMODE::LE, computeCount);
            AscendC::Duplicate(result, static_cast<half>(1.0),
                               computeCount);
            AscendC::Select(
                result, mask, result, static_cast<half>(0.0),
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                computeCount);
            AscendC::Cast(y, result, AscendC::RoundMode::CAST_NONE,
                          computeCount);
        }
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x1Queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x2Queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQueue_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1HalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2HalfBuf_;

    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;

    uint32_t totalLength_ = 0U;
    uint32_t workPerCore_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t coreId_ = 0U;
};

template <typename DT_X1, uint32_t SCH_MODE>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    if constexpr (SCH_MODE == LESS_EQUAL_SCH_STATIC_HALF) {
        if constexpr (std::is_same_v<DT_X1, half>) {
            AscendC::InitSocState();
            KernelLessEqualStaticHalf op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else {
            AscendC::TPipe pipe;
            KernelLessEqual<DT_X1, 1U, false> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        }
    } else if constexpr (SCH_MODE == LESS_EQUAL_SCH_LINEAR_DB_ONES) {
        AscendC::TPipe pipe;
        KernelLessEqual<DT_X1, 2U, true> op;
        op.Init(x1, x2, y, tilingData, &pipe);
        op.Process();
    } else if constexpr (SCH_MODE == LESS_EQUAL_SCH_FAST_LIGHT) {
        if constexpr (std::is_same_v<DT_X1, half> ||
                      std::is_same_v<DT_X1, int8_t>) {
            AscendC::TPipe pipe;
            KernelLessEqualFastLight<DT_X1> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        } else {
            // Host never selects FAST_LIGHT for FP32/INT32.  Keep a valid
            // fallback because the TilingKey generator forms the Cartesian
            // product of dtype and schedule mode.
            AscendC::TPipe pipe;
            KernelLessEqual<DT_X1, 1U, false> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        }
    } else {
        AscendC::TPipe pipe;
        KernelLessEqual<DT_X1, 1U, false> op;
        op.Init(x1, x2, y, tilingData, &pipe);
        op.Process();
    }
}
