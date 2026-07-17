// Kernel implementation of LessEqual.
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {
constexpr uint32_t BOOL_BYTES_PER_ELEMENT = 1;
constexpr uint32_t VECTOR_REPEAT_BYTES = 256;
constexpr uint32_t VECIN_BUF = 2;       // x1/x2 input queues
constexpr uint32_t VECOUT_BUF_FP16 = 3; // fp16: deeper VECOUT for MTE3 overlap
constexpr uint32_t VECOUT_BUF_OTHER = 2; // other types

template <typename T>
struct IsHalfType {
    static constexpr bool value = false;
};
template <>
struct IsHalfType<half> {
    static constexpr bool value = true;
};

template <typename T>
struct IsFloatType {
    static constexpr bool value = false;
};
template <>
struct IsFloatType<float> {
    static constexpr bool value = true;
};

template <typename T>
struct IsInt8Type {
    static constexpr bool value = false;
};
template <>
struct IsInt8Type<int8_t> {
    static constexpr bool value = true;
};

template <typename T>
struct IsInt32Type {
    static constexpr bool value = false;
};
template <>
struct IsInt32Type<int32_t> {
    static constexpr bool value = true;
};

template <typename T>
struct SupportsVectorPath {
    static constexpr bool value = IsHalfType<T>::value || IsFloatType<T>::value ||
                                  IsInt8Type<T>::value || IsInt32Type<T>::value;
};
}  // namespace

template <typename T>
class KernelLessEqual {
public:
    // VECOUT depth varies by type: fp16 gets 3 for better MTE3 overlap.
    static constexpr uint32_t OUTQUEUE_DEPTH = IsHalfType<T>::value ? VECOUT_BUF_FP16 : VECOUT_BUF_OTHER;

    __aicore__ inline KernelLessEqual() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        tiling_ = tiling;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1, tiling_.x1Length);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2, tiling_.x2Length);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tiling_.totalLength);

        blockStart_ = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling_.blockLength;
        blockEnd_ = blockStart_ + tiling_.blockLength;
        if (blockEnd_ > tiling_.totalLength) {
            blockEnd_ = tiling_.totalLength;
        }

        pipe_.InitBuffer(x1Queue_, VECIN_BUF, tiling_.tileLength * sizeof(T));
        pipe_.InitBuffer(x2Queue_, VECIN_BUF, tiling_.tileLength * sizeof(T));
        pipe_.InitBuffer(outQueue_, OUTQUEUE_DEPTH, tiling_.tileLength * BOOL_BYTES_PER_ELEMENT);

        if constexpr (SupportsVectorPath<T>::value) {
            pipe_.InitBuffer(compareMaskBuf_, AlignUp32((tiling_.tileLength + 7U) / 8U));
        }
        // Shared half workspace for MaterializeMaskAsBool and constant-output folding.
        pipe_.InitBuffer(tempHalf1Buf_, tiling_.tileLength * sizeof(half));
        if constexpr (IsInt8Type<T>::value) {
            // int8 is exactly representable by half. Two temporary half tensors remove all
            // per-element integer comparisons while preserving exact LessEqual semantics.
            pipe_.InitBuffer(tempHalf2Buf_, tiling_.tileLength * sizeof(half));
        } else if constexpr (IsInt32Type<T>::value) {
            // Exact int32 LessEqual uses min(a, b) == a. The int32 workspace is reused as
            // the half 0/1 workspace after Compare has consumed the minimum values.
            pipe_.InitBuffer(tempInt32Buf_, tiling_.tileLength * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockStart_ >= blockEnd_) {
            return;
        }

        if (tiling_.broadcastMode == LESS_EQUAL_SAME_SHAPE) {
            ProcessSameShapePipelined();
        } else if (tiling_.broadcastMode == LESS_EQUAL_X1_SCALAR) {
            ProcessX1ScalarPipelined();
        } else if (tiling_.broadcastMode == LESS_EQUAL_X2_SCALAR) {
            ProcessX2ScalarPipelined();
        } else {
            ProcessGeneralBroadcast();
        }
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) & ~31U;
    }

    __aicore__ inline uint32_t CurrentTileLength(uint64_t tileStart) const
    {
        const uint64_t remaining = blockEnd_ - tileStart;
        return remaining < tiling_.tileLength ? static_cast<uint32_t>(remaining) : tiling_.tileLength;
    }

    __aicore__ inline uint32_t VectorAlignedCount(uint32_t elementCount) const
    {
        // All vector comparison paths operate on half or float. int8 is first converted to half.
        const uint32_t sourceBytes = IsFloatType<T>::value ? sizeof(float) : sizeof(half);
        const uint32_t elementsPerRepeat = VECTOR_REPEAT_BYTES / sourceBytes;
        return ((elementCount + elementsPerRepeat - 1U) / elementsPerRepeat) * elementsPerRepeat;
    }

    __aicore__ inline bool CanUseAlignedCopy(uint64_t elementOffset, uint32_t elementCount,
                                             uint32_t bytesPerElement) const
    {
        const uint64_t byteOffset = elementOffset * bytesPerElement;
        const uint64_t byteCount = static_cast<uint64_t>(elementCount) * bytesPerElement;
        return ((byteOffset & 31U) == 0U) && ((byteCount & 31U) == 0U);
    }

    __aicore__ inline void EnqueueX1(uint64_t inputOffset, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> inputLocal = x1Queue_.AllocTensor<T>();
        if (CanUseAlignedCopy(inputOffset, elementCount, sizeof(T))) {
            AscendC::DataCopy(inputLocal, x1Gm_[inputOffset], elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(inputLocal, x1Gm_[inputOffset], copyParams, padParams);
        }
        x1Queue_.EnQue(inputLocal);
    }

    __aicore__ inline void EnqueueX2(uint64_t inputOffset, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> inputLocal = x2Queue_.AllocTensor<T>();
        if (CanUseAlignedCopy(inputOffset, elementCount, sizeof(T))) {
            AscendC::DataCopy(inputLocal, x2Gm_[inputOffset], elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams = {false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(inputLocal, x2Gm_[inputOffset], copyParams, padParams);
        }
        x2Queue_.EnQue(inputLocal);
    }

    __aicore__ inline void EnqueueScalarX1(T value, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> local = x1Queue_.AllocTensor<T>();
        FillScalarLocal(local, value, elementCount);
        x1Queue_.EnQue(local);
    }

    __aicore__ inline void EnqueueScalarX2(T value, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> local = x2Queue_.AllocTensor<T>();
        FillScalarLocal(local, value, elementCount);
        x2Queue_.EnQue(local);
    }

    __aicore__ inline void FillScalarLocal(AscendC::LocalTensor<T> &local, T value, uint32_t elementCount)
    {
        if constexpr (IsHalfType<T>::value || IsFloatType<T>::value || IsInt32Type<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::Duplicate(local, value, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            uint32_t i = 0;
            for (; i + 3U < elementCount; i += 4U) {
                local.SetValue(i, value);
                local.SetValue(i + 1U, value);
                local.SetValue(i + 2U, value);
                local.SetValue(i + 3U, value);
            }
            for (; i < elementCount; ++i) {
                local.SetValue(i, value);
            }
        }
    }

    __aicore__ inline void EnqueueOutput(AscendC::LocalTensor<uint8_t> &outputLocal)
    {
        outQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOutQueued(uint64_t outputOffset, uint32_t elementCount)
    {
        AscendC::LocalTensor<uint8_t> readyLocal = outQueue_.template DeQue<uint8_t>();
        if (CanUseAlignedCopy(outputOffset, elementCount, BOOL_BYTES_PER_ELEMENT)) {
            AscendC::DataCopy(yGm_[outputOffset], readyLocal, elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, elementCount * BOOL_BYTES_PER_ELEMENT, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[outputOffset], readyLocal, copyParams);
        }
        outQueue_.FreeTensor(readyLocal);
    }

    __aicore__ inline void MaterializeMaskAsBool(AscendC::LocalTensor<half> &workHalf,
                                                 AscendC::LocalTensor<uint8_t> &outputLocal,
                                                 uint32_t alignedCount)
    {
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();

        // Select consumes Compare's packed mask directly. Reusing workHalf as both source
        // and destination avoids another tile-sized temporary tensor.
        AscendC::Duplicate(workHalf, static_cast<half>(1.0), alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_NONE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputePair(AscendC::LocalTensor<T> &x1Local,
                                        AscendC::LocalTensor<T> &x2Local,
                                        AscendC::LocalTensor<uint8_t> &outputLocal,
                                        uint32_t elementCount)
    {
        if constexpr (IsHalfType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(compareMask, x1Local, x2Local, AscendC::CMPMODE::LE, alignedCount);
            // Duplicate uses tempHalf1Buf_ (no data dep with compareMask / x1Local / x2Local),
            // so it can overlap with Compare — no PipeBarrier needed here.
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf,
                          AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (IsFloatType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(compareMask, x1Local, x2Local, AscendC::CMPMODE::LE, alignedCount);
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf,
                          AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        } else if constexpr (IsInt8Type<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<half> x1Half = tempHalf1Buf_.Get<half>();
            AscendC::LocalTensor<half> x2Half = tempHalf2Buf_.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(compareMask, x1Half, x2Half, AscendC::CMPMODE::LE, alignedCount);
            // Duplicate reuses x1Half (already consumed by Compare) — no data dep with Compare.
            AscendC::Duplicate(x1Half, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(x1Half, compareMask, x1Half, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, x1Half,
                          AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // For every signed int32 pair: a <= b iff min(a, b) == a. Both operations are
            // native vector instructions on Atlas A2, so the comparison stays exact for the
            // complete int32 range and avoids all float conversions and scalar mask merging.
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<int32_t> minimum = tempInt32Buf_.Get<int32_t>();
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Min(minimum, x1Local, x2Local, static_cast<int32_t>(alignedCount));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, minimum, x1Local, AscendC::CMPMODE::EQ, alignedCount);
            // tempHalf1Buf_ is independent of min/compareMask/x1Local — Duplicate can overlap.
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf,
                          AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeX1Scalar(T x1Scalar,
                                            AscendC::LocalTensor<T> &x2Local,
                                            AscendC::LocalTensor<uint8_t> &outputLocal,
                                            uint32_t elementCount)
    {
        if constexpr (IsHalfType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            // x1Scalar <= x2  <=>  x2 >= x1Scalar
            AscendC::CompareScalar(compareMask, x2Local, x1Scalar, AscendC::CMPMODE::GE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x2Local, outputLocal, alignedCount);
        } else if constexpr (IsFloatType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::CompareScalar(compareMask, x2Local, x1Scalar, AscendC::CMPMODE::GE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        } else if constexpr (IsInt8Type<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<half> x2Half = tempHalf1Buf_.Get<half>();
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            const half scalarHalf = static_cast<half>(x1Scalar);
            AscendC::CompareScalar(compareMask, x2Half, scalarHalf, AscendC::CMPMODE::GE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x2Half, outputLocal, alignedCount);
        } else {
            // Vectorized int32 scalar: x1Scalar <= x2[i]  iff  min(x1Scalar, x2[i]) == x1Scalar
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<int32_t> scalarBuf = tempInt32Buf_.Get<int32_t>();
            AscendC::Duplicate(scalarBuf, static_cast<int32_t>(x1Scalar), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            // In-place Min: x2Local[i] = min(x1Scalar, x2Local[i]).
            // x2Local is consumed after this, safe to reuse.
            AscendC::Min(x2Local, scalarBuf, x2Local, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> mask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(mask, x2Local, scalarBuf, AscendC::CMPMODE::EQ, alignedCount);
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(workHalf, mask, workHalf, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeX2Scalar(AscendC::LocalTensor<T> &x1Local,
                                            T x2Scalar,
                                            AscendC::LocalTensor<uint8_t> &outputLocal,
                                            uint32_t elementCount)
    {
        if constexpr (IsHalfType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::CompareScalar(compareMask, x1Local, x2Scalar, AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Local, outputLocal, alignedCount);
        } else if constexpr (IsFloatType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::CompareScalar(compareMask, x1Local, x2Scalar, AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        } else if constexpr (IsInt8Type<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<half> x1Half = tempHalf1Buf_.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            const half scalarHalf = static_cast<half>(x2Scalar);
            AscendC::CompareScalar(compareMask, x1Half, scalarHalf, AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Half, outputLocal, alignedCount);
        } else {
            // Vectorized int32 scalar: x1[i] <= x2Scalar  iff  max(x1[i], x2Scalar) == x2Scalar
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<int32_t> scalarBuf = tempInt32Buf_.Get<int32_t>();
            AscendC::Duplicate(scalarBuf, static_cast<int32_t>(x2Scalar), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(x1Local, x1Local, scalarBuf, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> mask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(mask, x1Local, scalarBuf, AscendC::CMPMODE::EQ, alignedCount);
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(1.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(workHalf, mask, workHalf, static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeQueuedPair(uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.template AllocTensor<uint8_t>();
        ComputePair(x1Local, x2Local, outputLocal, elementCount);
        EnqueueOutput(outputLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeQueuedX1Scalar(T x1Scalar, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.template AllocTensor<uint8_t>();
        ComputeX1Scalar(x1Scalar, x2Local, outputLocal, elementCount);
        EnqueueOutput(outputLocal);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeQueuedX2Scalar(T x2Scalar, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.template AllocTensor<uint8_t>();
        ComputeX2Scalar(x1Local, x2Scalar, outputLocal, elementCount);
        EnqueueOutput(outputLocal);
        x1Queue_.FreeTensor(x1Local);
    }

    __aicore__ inline void ProcessSameShapePipelined()
    {
        uint64_t currentStart = blockStart_;
        uint32_t currentLength = CurrentTileLength(currentStart);
        EnqueueX1(currentStart, currentLength);
        EnqueueX2(currentStart, currentLength);

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;

        while (true) {
            const uint64_t nextStart = currentStart + currentLength;
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            if (hasNext) {
                nextLength = CurrentTileLength(nextStart);
                EnqueueX1(nextStart, nextLength);
                EnqueueX2(nextStart, nextLength);
            }

            ComputeQueuedPair(currentLength);
            if (hasPendingOutput) {
                CopyOutQueued(pendingStart, pendingLength);
            }
            pendingStart = currentStart;
            pendingLength = currentLength;
            hasPendingOutput = true;

            if (!hasNext) {
                break;
            }
            currentStart = nextStart;
            currentLength = nextLength;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void ProcessX1ScalarPipelined()
    {
        const T x1Value = x1Gm_.GetValue(0);
        uint64_t currentStart = blockStart_;
        uint32_t currentLength = CurrentTileLength(currentStart);
        if constexpr (IsInt32Type<T>::value) {
            EnqueueScalarX1(x1Value, currentLength);
        }
        EnqueueX2(currentStart, currentLength);

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;

        while (true) {
            const uint64_t nextStart = currentStart + currentLength;
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            if (hasNext) {
                nextLength = CurrentTileLength(nextStart);
                if constexpr (IsInt32Type<T>::value) {
                    EnqueueScalarX1(x1Value, nextLength);
                }
                EnqueueX2(nextStart, nextLength);
            }

            if constexpr (IsInt32Type<T>::value) {
                ComputeQueuedPair(currentLength);
            } else {
                ComputeQueuedX1Scalar(x1Value, currentLength);
            }
            if (hasPendingOutput) {
                CopyOutQueued(pendingStart, pendingLength);
            }
            pendingStart = currentStart;
            pendingLength = currentLength;
            hasPendingOutput = true;

            if (!hasNext) {
                break;
            }
            currentStart = nextStart;
            currentLength = nextLength;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void ProcessX2ScalarPipelined()
    {
        const T x2Value = x2Gm_.GetValue(0);
        uint64_t currentStart = blockStart_;
        uint32_t currentLength = CurrentTileLength(currentStart);
        EnqueueX1(currentStart, currentLength);
        if constexpr (IsInt32Type<T>::value) {
            EnqueueScalarX2(x2Value, currentLength);
        }

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;

        while (true) {
            const uint64_t nextStart = currentStart + currentLength;
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            if (hasNext) {
                nextLength = CurrentTileLength(nextStart);
                EnqueueX1(nextStart, nextLength);
                if constexpr (IsInt32Type<T>::value) {
                    EnqueueScalarX2(x2Value, nextLength);
                }
            }

            if constexpr (IsInt32Type<T>::value) {
                ComputeQueuedPair(currentLength);
            } else {
                ComputeQueuedX2Scalar(x2Value, currentLength);
            }
            if (hasPendingOutput) {
                CopyOutQueued(pendingStart, pendingLength);
            }
            pendingStart = currentStart;
            pendingLength = currentLength;
            hasPendingOutput = true;

            if (!hasNext) {
                break;
            }
            currentStart = nextStart;
            currentLength = nextLength;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void InitBroadcastOffsets(uint64_t linearIndex,
                                                 uint64_t &x1Offset, uint64_t &x2Offset) const
    {
        x1Offset = 0;
        x2Offset = 0;
        uint64_t remainder = linearIndex;
        for (uint32_t reverse = 0; reverse < tiling_.rank; ++reverse) {
            const uint32_t dimension = tiling_.rank - 1U - reverse;
            const uint64_t dimensionSize = tiling_.outputShape[dimension];
            const uint64_t coordinate = remainder % dimensionSize;
            remainder /= dimensionSize;
            x1Offset += coordinate * tiling_.x1Stride[dimension];
            x2Offset += coordinate * tiling_.x2Stride[dimension];
        }
    }

    __aicore__ inline void ProcessBroadcastSegment(uint64_t outputOffset,
                                                    uint64_t x1Offset, uint64_t x2Offset,
                                                    uint32_t elementCount,
                                                    bool x1Contiguous, bool x2Contiguous)
    {
        if (x1Contiguous && x2Contiguous) {
            EnqueueX1(x1Offset, elementCount);
            EnqueueX2(x2Offset, elementCount);
            ComputeQueuedPair(elementCount);
            CopyOutQueued(outputOffset, elementCount);
            return;
        }

        if (x1Contiguous) {
            const T x2Value = x2Gm_.GetValue(x2Offset);
            EnqueueX1(x1Offset, elementCount);
            if constexpr (IsInt32Type<T>::value) {
                EnqueueScalarX2(x2Value, elementCount);
                ComputeQueuedPair(elementCount);
            } else {
                ComputeQueuedX2Scalar(x2Value, elementCount);
            }
            CopyOutQueued(outputOffset, elementCount);
            return;
        }

        if (x2Contiguous) {
            const T x1Value = x1Gm_.GetValue(x1Offset);
            if constexpr (IsInt32Type<T>::value) {
                EnqueueScalarX1(x1Value, elementCount);
                EnqueueX2(x2Offset, elementCount);
                ComputeQueuedPair(elementCount);
            } else {
                EnqueueX2(x2Offset, elementCount);
                ComputeQueuedX1Scalar(x1Value, elementCount);
            }
            CopyOutQueued(outputOffset, elementCount);
            return;
        }

        // Both inputs are constant over the segment. Compute the comparison once,
        // then Duplicate + Cast the single-result 0.0/1.0 across the output tile.
        // This eliminates all global-memory input traffic and the entire vector
        // Compare/Select pipeline for this segment.
        const T x1Value = x1Gm_.GetValue(x1Offset);
        const T x2Value = x2Gm_.GetValue(x2Offset);
        bool constantResult = false;
        if constexpr (IsHalfType<T>::value) {
            // half lacks guaranteed scalar operator<= on AI Core; compare via float.
            constantResult = static_cast<float>(x1Value) <= static_cast<float>(x2Value);
        } else {
            constantResult = x1Value <= x2Value;
        }
        {
            AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.template AllocTensor<uint8_t>();
            constexpr uint32_t HALF_REPEAT = VECTOR_REPEAT_BYTES / sizeof(half);
            const uint32_t alignedCount =
                ((elementCount + HALF_REPEAT - 1U) / HALF_REPEAT) * HALF_REPEAT;
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            AscendC::Duplicate(workHalf, static_cast<half>(constantResult ? 1.0f : 0.0f), alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(outputLocal, workHalf,
                          AscendC::RoundMode::CAST_NONE, alignedCount);
            EnqueueOutput(outputLocal);
        }
        CopyOutQueued(outputOffset, elementCount);
    }

    __aicore__ inline void ProcessGeneralBroadcast()
    {
        const uint64_t innerLength = tiling_.innerLength;
        const bool x1Contiguous = tiling_.x1InnerContiguous != 0U;
        const bool x2Contiguous = tiling_.x2InnerContiguous != 0U;

        // Fast path: both inputs are contiguous within the inner span.
        // Data access is the same as same-shape; use full prologue-loop-epilogue
        // double-buffering for maximum DMA/compute overlap.
        if (x1Contiguous && x2Contiguous) {
            ProcessContiguousSpanPipelined(innerLength);
            return;
        }

        uint64_t segmentStart = blockStart_;
        uint64_t innerOffset = segmentStart % innerLength;
        uint64_t x1Offset = 0;
        uint64_t x2Offset = 0;
        InitBroadcastOffsets(segmentStart, x1Offset, x2Offset);

        while (segmentStart < blockEnd_) {
            const uint64_t innerRemaining = innerLength - innerOffset;
            uint64_t segmentLength = blockEnd_ - segmentStart;
            if (segmentLength > innerRemaining) {
                segmentLength = innerRemaining;
            }
            if (segmentLength > tiling_.tileLength) {
                segmentLength = tiling_.tileLength;
            }

            ProcessBroadcastSegment(segmentStart, x1Offset, x2Offset,
                                    static_cast<uint32_t>(segmentLength),
                                    x1Contiguous, x2Contiguous);
            segmentStart += segmentLength;
            innerOffset += segmentLength;
            if (segmentStart >= blockEnd_) {
                break;
            }

            if (innerOffset < innerLength) {
                if (x1Contiguous) {
                    x1Offset += segmentLength;
                }
                if (x2Contiguous) {
                    x2Offset += segmentLength;
                }
            } else {
                innerOffset = segmentStart % innerLength;
                InitBroadcastOffsets(segmentStart, x1Offset, x2Offset);
            }
        }
    }

    // Fast-path for general broadcast where both x1 and x2 are contiguous in
    // the inner span.  Within a span the access pattern is identical to the
    // same-shape case, so we use the same prologue-loop-epilogue double-buffer
    // pipeline as ProcessSameShapePipelined.  At span boundaries the linear
    // offsets are recomputed via InitBroadcastOffsets.
    __aicore__ inline void ProcessContiguousSpanPipelined(uint64_t innerLength)
    {
        uint64_t segmentStart = blockStart_;
        uint64_t innerOffset = segmentStart % innerLength;

        while (segmentStart < blockEnd_) {
            uint64_t baseX1 = 0, baseX2 = 0;
            InitBroadcastOffsets(segmentStart, baseX1, baseX2);

            uint64_t spanRemaining = innerLength - innerOffset;
            if (spanRemaining > blockEnd_ - segmentStart) {
                spanRemaining = blockEnd_ - segmentStart;
            }
            if (spanRemaining == 0) break;

            // --- Prologue: load first tile ---
            uint64_t spanPos = 0;
            uint32_t tileLen = CurrentTileLength(segmentStart);
            if (tileLen > spanRemaining) {
                tileLen = static_cast<uint32_t>(spanRemaining);
            }
            EnqueueX1(baseX1, tileLen);
            EnqueueX2(baseX2, tileLen);
            spanPos += tileLen;

            uint64_t currentStart = segmentStart;
            uint32_t currentLen = tileLen;
            uint64_t pendingStart = 0;
            uint32_t pendingLen = 0;
            bool hasPending = false;

            // --- Loop: prefetch N+1, compute N, writeback N-1 ---
            while (spanPos < spanRemaining) {
                uint32_t nextLen = CurrentTileLength(segmentStart + spanPos);
                if (nextLen > spanRemaining - spanPos) {
                    nextLen = static_cast<uint32_t>(spanRemaining - spanPos);
                }
                EnqueueX1(baseX1 + spanPos, nextLen);
                EnqueueX2(baseX2 + spanPos, nextLen);

                ComputeQueuedPair(currentLen);
                if (hasPending) {
                    CopyOutQueued(pendingStart, pendingLen);
                }
                pendingStart = currentStart;
                pendingLen = currentLen;
                hasPending = true;

                currentStart = segmentStart + spanPos;
                currentLen = nextLen;
                spanPos += nextLen;
            }

            // --- Epilogue: compute last tile, writeback all ---
            ComputeQueuedPair(currentLen);
            if (hasPending) {
                CopyOutQueued(pendingStart, pendingLen);
            }
            CopyOutQueued(currentStart, currentLen);

            segmentStart += spanPos;
            innerOffset = (innerOffset + spanPos) % innerLength;
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, VECIN_BUF> x1Queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, VECIN_BUF> x2Queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, OUTQUEUE_DEPTH> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf1Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf2Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempInt32Buf_;
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;

    LessEqualTilingData tiling_ {};
    uint64_t blockStart_ = 0;
    uint64_t blockEnd_ = 0;
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tilingData);
    op.Process();
}
