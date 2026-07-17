// Kernel implementation of LessEqual.
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {
constexpr uint32_t BOOL_BYTES_PER_ELEMENT = 1;
constexpr uint32_t VECTOR_REPEAT_BYTES = 256;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SMALL_BUFFER_NUM = 1;

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

// One-repeat static-tensor implementation. It bypasses TPipe/TQue entirely and manually
// manages UB addresses plus the two required cross-pipeline dependencies. The Host selects
// this path only when the output fits in one 256-byte vector repeat and the broadcast pattern
// is same-shape or one of the two scalar cases.
template <typename T, int PATH_KIND>
class KernelLessEqualTiny {
public:
    __aicore__ inline KernelLessEqualTiny() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        totalLength_ = static_cast<uint32_t>(tiling.totalLength);
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1, tiling.x1Length);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2, tiling.x2Length);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tiling.totalLength);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t repeatElements =
            (IsFloatType<T>::value || IsInt32Type<T>::value) ? 64U : 128U;
        const uint32_t elementCount = totalLength_;

        AscendC::LocalTensor<T> x1Local(AscendC::TPosition::VECCALC, X1_ADDR, repeatElements);
        AscendC::LocalTensor<T> x2Local(AscendC::TPosition::VECCALC, X2_ADDR, repeatElements);
        AscendC::LocalTensor<half> tempHalf1(AscendC::TPosition::VECCALC, TEMP1_ADDR, 128U);
        AscendC::LocalTensor<half> tempHalf2(AscendC::TPosition::VECCALC, TEMP2_ADDR, 128U);
        AscendC::LocalTensor<uint8_t> compareMask(AscendC::TPosition::VECCALC, MASK_ADDR, 32U);
        AscendC::LocalTensor<half> oneHalf(AscendC::TPosition::VECCALC, ONE_ADDR, 128U);
        AscendC::LocalTensor<half> zeroHalf(AscendC::TPosition::VECCALC, ZERO_ADDR, 128U);
        AscendC::LocalTensor<half> selectedHalf(AscendC::TPosition::VECCALC, SELECTED_ADDR, 128U);
        AscendC::LocalTensor<uint8_t> outputLocal(AscendC::TPosition::VECCALC, OUTPUT_ADDR, 128U);

        T x1Scalar {};
        T x2Scalar {};
        if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME) {
            CopyIn(x1Local, x1Gm_, elementCount);
            CopyIn(x2Local, x2Gm_, elementCount);
        } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR) {
            x1Scalar = x1Gm_.GetValue(0);
            CopyIn(x2Local, x2Gm_, elementCount);
        } else {
            x2Scalar = x2Gm_.GetValue(0);
            CopyIn(x1Local, x1Gm_, elementCount);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        if constexpr (IsHalfType<T>::value) {
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME) {
                AscendC::Compare(compareMask, x1Local, x2Local, AscendC::CMPMODE::LE, repeatElements);
            } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR) {
                AscendC::CompareScalar(compareMask, x2Local, x1Scalar,
                                       AscendC::CMPMODE::GE, repeatElements);
            } else {
                AscendC::CompareScalar(compareMask, x1Local, x2Scalar,
                                       AscendC::CMPMODE::LE, repeatElements);
            }
        } else if constexpr (IsFloatType<T>::value) {
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME) {
                AscendC::Compare(compareMask, x1Local, x2Local, AscendC::CMPMODE::LE, repeatElements);
            } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR) {
                AscendC::CompareScalar(compareMask, x2Local, x1Scalar,
                                       AscendC::CMPMODE::GE, repeatElements);
            } else {
                AscendC::CompareScalar(compareMask, x1Local, x2Scalar,
                                       AscendC::CMPMODE::LE, repeatElements);
            }
        } else if constexpr (IsInt8Type<T>::value) {
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME) {
                AscendC::Cast(tempHalf1, x1Local, AscendC::RoundMode::CAST_NONE, repeatElements);
                AscendC::Cast(tempHalf2, x2Local, AscendC::RoundMode::CAST_NONE, repeatElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(compareMask, tempHalf1, tempHalf2,
                                 AscendC::CMPMODE::LE, repeatElements);
            } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR) {
                AscendC::Cast(tempHalf1, x2Local, AscendC::RoundMode::CAST_NONE, repeatElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::CompareScalar(compareMask, tempHalf1, static_cast<half>(x1Scalar),
                                       AscendC::CMPMODE::GE, repeatElements);
            } else {
                AscendC::Cast(tempHalf1, x1Local, AscendC::RoundMode::CAST_NONE, repeatElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::CompareScalar(compareMask, tempHalf1, static_cast<half>(x2Scalar),
                                       AscendC::CMPMODE::LE, repeatElements);
            }
        } else {
            AscendC::LocalTensor<int32_t> scratch = tempHalf1.template ReinterpretCast<int32_t>();
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME) {
                AscendC::Min(x2Local, x1Local, x2Local, static_cast<int32_t>(repeatElements));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(compareMask, x2Local, x1Local,
                                 AscendC::CMPMODE::EQ, repeatElements);
            } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR) {
                AscendC::Maxs(scratch, x2Local, x1Scalar, static_cast<int32_t>(repeatElements));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(compareMask, scratch, x2Local,
                                 AscendC::CMPMODE::EQ, repeatElements);
            } else {
                AscendC::Mins(scratch, x1Local, x2Scalar, static_cast<int32_t>(repeatElements));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(compareMask, scratch, x1Local,
                                 AscendC::CMPMODE::EQ, repeatElements);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        MaterializeMaskAsBool(compareMask, oneHalf, zeroHalf, selectedHalf,
                              outputLocal, elementCount, repeatElements);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyOut(outputLocal, elementCount);
    }

private:
    static constexpr uint32_t X1_ADDR = 0U;
    static constexpr uint32_t X2_ADDR = 256U;
    static constexpr uint32_t TEMP1_ADDR = 512U;
    static constexpr uint32_t TEMP2_ADDR = 768U;
    static constexpr uint32_t MASK_ADDR = 1024U;
    static constexpr uint32_t ONE_ADDR = 1056U;
    static constexpr uint32_t ZERO_ADDR = 1312U;
    static constexpr uint32_t SELECTED_ADDR = 1568U;
    static constexpr uint32_t OUTPUT_ADDR = 1824U;

    __aicore__ inline bool IsAligned(uint32_t elementCount, uint32_t bytesPerElement) const
    {
        return ((static_cast<uint64_t>(elementCount) * bytesPerElement) & 31U) == 0U;
    }

    __aicore__ inline void CopyIn(AscendC::LocalTensor<T> &local,
                                  AscendC::GlobalTensor<T> &global,
                                  uint32_t elementCount)
    {
        if (IsAligned(elementCount, sizeof(T))) {
            AscendC::DataCopy(local, global[0], elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams = {
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, global[0], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<uint8_t> &local,
                                   uint32_t elementCount)
    {
        if ((elementCount & 31U) == 0U) {
            AscendC::DataCopy(yGm_[0], local, elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, elementCount * BOOL_BYTES_PER_ELEMENT, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[0], local, copyParams);
        }
    }

    __aicore__ inline void MaterializeMaskAsBool(
        AscendC::LocalTensor<uint8_t> &compareMask,
        AscendC::LocalTensor<half> &oneHalf,
        AscendC::LocalTensor<half> &zeroHalf,
        AscendC::LocalTensor<half> &selectedHalf,
        AscendC::LocalTensor<uint8_t> &outputLocal,
        uint32_t elementCount,
        uint32_t castCount)
    {
        // Mode 0 consumes one compare-register mask and requires no 8 KiB mode-1 scratch.
        AscendC::Duplicate(oneHalf, static_cast<half>(1.0), 128U);
        AscendC::Duplicate(zeroHalf, static_cast<half>(0.0), 128U);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(compareMask);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<half>(elementCount);
        const AscendC::BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
        AscendC::Select<half, AscendC::SELMODE::VSEL_CMPMASK_SPR>(
            selectedHalf, oneHalf, zeroHalf, 1U, repeatParams);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, selectedHalf, AscendC::RoundMode::CAST_NONE, castCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t totalLength_ = 0;
};

// Dedicated one-core/one-tile implementation for small same-shape tensors that span more
// than one vector repeat. It keeps the proven v8 queue-based implementation, but receives
// TPipe from the kernel entry so scalar constants can remain register-resident.
template <typename T>
class KernelLessEqualSmall {
public:
    __aicore__ inline KernelLessEqualSmall() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1, tiling_.x1Length);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2, tiling_.x2Length);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tiling_.totalLength);

        pipe_->InitBuffer(x1Queue_, SMALL_BUFFER_NUM, tiling_.tileLength * sizeof(T));
        pipe_->InitBuffer(x2Queue_, SMALL_BUFFER_NUM, tiling_.tileLength * sizeof(T));
        pipe_->InitBuffer(outQueue_, SMALL_BUFFER_NUM,
                          tiling_.tileLength * BOOL_BYTES_PER_ELEMENT);
        pipe_->InitBuffer(compareMaskBuf_, AlignUp32((tiling_.tileLength + 7U) / 8U));

        if constexpr (IsInt8Type<T>::value) {
            pipe_->InitBuffer(tempHalf1Buf_, tiling_.tileLength * sizeof(half));
            pipe_->InitBuffer(tempHalf2Buf_, tiling_.tileLength * sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t elementCount = static_cast<uint32_t>(tiling_.totalLength);
        AscendC::LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        CopyIn(x1Local, x1Gm_, elementCount);
        CopyIn(x2Local, x2Gm_, elementCount);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);

        x1Local = x1Queue_.DeQue<T>();
        x2Local = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        Compute(x1Local, x2Local, outputLocal, elementCount);
        outQueue_.EnQue(outputLocal);

        AscendC::LocalTensor<uint8_t> readyLocal = outQueue_.DeQue<uint8_t>();
        CopyOut(readyLocal, elementCount);
        outQueue_.FreeTensor(readyLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) & ~31U;
    }

    __aicore__ inline uint32_t VectorAlignedCount(uint32_t elementCount) const
    {
        const uint32_t sourceBytes = (IsFloatType<T>::value || IsInt32Type<T>::value)
            ? sizeof(float)
            : sizeof(half);
        const uint32_t elementsPerRepeat = VECTOR_REPEAT_BYTES / sourceBytes;
        return ((elementCount + elementsPerRepeat - 1U) / elementsPerRepeat) * elementsPerRepeat;
    }

    __aicore__ inline bool IsAligned(uint32_t elementCount, uint32_t bytesPerElement) const
    {
        return ((static_cast<uint64_t>(elementCount) * bytesPerElement) & 31U) == 0U;
    }

    __aicore__ inline void CopyIn(AscendC::LocalTensor<T> &local,
                                  AscendC::GlobalTensor<T> &global,
                                  uint32_t elementCount)
    {
        if (IsAligned(elementCount, sizeof(T))) {
            AscendC::DataCopy(local, global[0], elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams = {
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, global[0], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<uint8_t> &local,
                                   uint32_t elementCount)
    {
        if ((elementCount & 31U) == 0U) {
            AscendC::DataCopy(yGm_[0], local, elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, elementCount * BOOL_BYTES_PER_ELEMENT, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[0], local, copyParams);
        }
    }

    __aicore__ inline void MaterializeMaskAsBool(AscendC::LocalTensor<half> &workHalf,
                                                 AscendC::LocalTensor<uint8_t> &outputLocal,
                                                 uint32_t alignedCount)
    {
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
        AscendC::Duplicate(workHalf, static_cast<half>(1.0), alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_NONE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<T> &x1Local,
                                   AscendC::LocalTensor<T> &x2Local,
                                   AscendC::LocalTensor<uint8_t> &outputLocal,
                                   uint32_t elementCount)
    {
        const uint32_t alignedCount = VectorAlignedCount(elementCount);
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();

        if constexpr (IsHalfType<T>::value) {
            AscendC::Compare(compareMask, x1Local, x2Local,
                             AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Local, outputLocal, alignedCount);
        } else if constexpr (IsFloatType<T>::value) {
            AscendC::Compare(compareMask, x1Local, x2Local,
                             AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = x2Local.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        } else if constexpr (IsInt8Type<T>::value) {
            AscendC::LocalTensor<half> x1Half = tempHalf1Buf_.Get<half>();
            AscendC::LocalTensor<half> x2Half = tempHalf2Buf_.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, x1Half, x2Half,
                             AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Half, outputLocal, alignedCount);
        } else {
            AscendC::Min(x2Local, x1Local, x2Local, static_cast<int32_t>(alignedCount));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, x2Local, x1Local,
                             AscendC::CMPMODE::EQ, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = x2Local.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        }
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, SMALL_BUFFER_NUM> x1Queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, SMALL_BUFFER_NUM> x2Queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, SMALL_BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf1Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf2Buf_;
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    LessEqualTilingData tiling_ {};
};


// Dedicated one-core/one-tile scalar-broadcast implementation for outputs larger than
// one vector repeat and no larger than MEDIUM_SCALAR_MAX_ELEMENTS. It keeps the proven
// queue-based v9 Compute/Select/Cast chain, but allocates only the non-scalar input queue.
template <typename T, int PATH_KIND>
class KernelLessEqualMediumScalar {
public:
    __aicore__ inline KernelLessEqualMediumScalar() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1, tiling_.x1Length);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2, tiling_.x2Length);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tiling_.totalLength);

        pipe_->InitBuffer(inputQueue_, SMALL_BUFFER_NUM, tiling_.tileLength * sizeof(T));
        pipe_->InitBuffer(outQueue_, SMALL_BUFFER_NUM,
                          tiling_.tileLength * BOOL_BYTES_PER_ELEMENT);
        pipe_->InitBuffer(compareMaskBuf_, AlignUp32((tiling_.tileLength + 7U) / 8U));

        if constexpr (IsInt8Type<T>::value) {
            pipe_->InitBuffer(tempHalfBuf_, tiling_.tileLength * sizeof(half));
        } else if constexpr (IsInt32Type<T>::value) {
            pipe_->InitBuffer(tempInt32Buf_, tiling_.tileLength * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t elementCount = static_cast<uint32_t>(tiling_.totalLength);
        AscendC::LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();

        T scalar {};
        if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR) {
            scalar = x1Gm_.GetValue(0);
            CopyIn(inputLocal, x2Gm_, elementCount);
        } else {
            scalar = x2Gm_.GetValue(0);
            CopyIn(inputLocal, x1Gm_, elementCount);
        }
        inputQueue_.EnQue(inputLocal);

        inputLocal = inputQueue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        ComputeScalar(inputLocal, scalar, outputLocal, elementCount);
        outQueue_.EnQue(outputLocal);

        AscendC::LocalTensor<uint8_t> readyLocal = outQueue_.DeQue<uint8_t>();
        CopyOut(readyLocal, elementCount);
        outQueue_.FreeTensor(readyLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

private:
    __aicore__ inline uint32_t AlignUp32(uint32_t value) const
    {
        return (value + 31U) & ~31U;
    }

    __aicore__ inline uint32_t VectorAlignedCount(uint32_t elementCount) const
    {
        const uint32_t sourceBytes = (IsFloatType<T>::value || IsInt32Type<T>::value)
            ? sizeof(float)
            : sizeof(half);
        const uint32_t elementsPerRepeat = VECTOR_REPEAT_BYTES / sourceBytes;
        return ((elementCount + elementsPerRepeat - 1U) / elementsPerRepeat) * elementsPerRepeat;
    }

    __aicore__ inline bool IsAligned(uint32_t elementCount, uint32_t bytesPerElement) const
    {
        return ((static_cast<uint64_t>(elementCount) * bytesPerElement) & 31U) == 0U;
    }

    __aicore__ inline void CopyIn(AscendC::LocalTensor<T> &local,
                                  AscendC::GlobalTensor<T> &global,
                                  uint32_t elementCount)
    {
        if (IsAligned(elementCount, sizeof(T))) {
            AscendC::DataCopy(local, global[0], elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, static_cast<uint32_t>(elementCount * sizeof(T)), 0, 0, 0};
            const AscendC::DataCopyPadExtParams<T> padParams = {
                false, 0, 0, static_cast<T>(0)};
            AscendC::DataCopyPad(local, global[0], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyOut(AscendC::LocalTensor<uint8_t> &local,
                                   uint32_t elementCount)
    {
        if ((elementCount & 31U) == 0U) {
            AscendC::DataCopy(yGm_[0], local, elementCount);
        } else {
            const AscendC::DataCopyExtParams copyParams = {
                1, elementCount * BOOL_BYTES_PER_ELEMENT, 0, 0, 0};
            AscendC::DataCopyPad(yGm_[0], local, copyParams);
        }
    }

    __aicore__ inline void MaterializeMaskAsBool(AscendC::LocalTensor<half> &workHalf,
                                                 AscendC::LocalTensor<uint8_t> &outputLocal,
                                                 uint32_t alignedCount)
    {
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
        AscendC::Duplicate(workHalf, static_cast<half>(1.0), alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(workHalf, compareMask, workHalf, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outputLocal, workHalf, AscendC::RoundMode::CAST_NONE, alignedCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeScalar(AscendC::LocalTensor<T> &inputLocal,
                                         T scalar,
                                         AscendC::LocalTensor<uint8_t> &outputLocal,
                                         uint32_t elementCount)
    {
        const uint32_t alignedCount = VectorAlignedCount(elementCount);
        AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();

        if constexpr (IsHalfType<T>::value) {
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR) {
                AscendC::CompareScalar(compareMask, inputLocal, scalar,
                                       AscendC::CMPMODE::GE, alignedCount);
            } else {
                AscendC::CompareScalar(compareMask, inputLocal, scalar,
                                       AscendC::CMPMODE::LE, alignedCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(inputLocal, outputLocal, alignedCount);
        } else if constexpr (IsFloatType<T>::value) {
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR) {
                AscendC::CompareScalar(compareMask, inputLocal, scalar,
                                       AscendC::CMPMODE::GE, alignedCount);
            } else {
                AscendC::CompareScalar(compareMask, inputLocal, scalar,
                                       AscendC::CMPMODE::LE, alignedCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = inputLocal.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        } else if constexpr (IsInt8Type<T>::value) {
            AscendC::LocalTensor<half> inputHalf = tempHalfBuf_.Get<half>();
            AscendC::Cast(inputHalf, inputLocal, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            const half scalarHalf = static_cast<half>(scalar);
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR) {
                AscendC::CompareScalar(compareMask, inputHalf, scalarHalf,
                                       AscendC::CMPMODE::GE, alignedCount);
            } else {
                AscendC::CompareScalar(compareMask, inputHalf, scalarHalf,
                                       AscendC::CMPMODE::LE, alignedCount);
            }
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(inputHalf, outputLocal, alignedCount);
        } else {
            AscendC::LocalTensor<int32_t> scratch = tempInt32Buf_.Get<int32_t>();
            if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR) {
                // scalar <= tensor iff max(tensor, scalar) == tensor.
                AscendC::Maxs(scratch, inputLocal, scalar, static_cast<int32_t>(alignedCount));
            } else {
                // tensor <= scalar iff min(tensor, scalar) == tensor.
                AscendC::Mins(scratch, inputLocal, scalar, static_cast<int32_t>(alignedCount));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, scratch, inputLocal,
                             AscendC::CMPMODE::EQ, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = scratch.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        }
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, SMALL_BUFFER_NUM> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, SMALL_BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalfBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempInt32Buf_;
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    LessEqualTilingData tiling_ {};
};

template <typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling, AscendC::TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1, tiling_.x1Length);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2, tiling_.x2Length);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, tiling_.totalLength);

        blockStart_ = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling_.blockLength;
        blockEnd_ = blockStart_ + tiling_.blockLength;
        if (blockEnd_ > tiling_.totalLength) {
            blockEnd_ = tiling_.totalLength;
        }

        pipe_->InitBuffer(x1Queue_, BUFFER_NUM, tiling_.tileLength * sizeof(T));
        pipe_->InitBuffer(x2Queue_, BUFFER_NUM, tiling_.tileLength * sizeof(T));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, tiling_.tileLength * BOOL_BYTES_PER_ELEMENT);

        if constexpr (SupportsVectorPath<T>::value) {
            pipe_->InitBuffer(compareMaskBuf_, AlignUp32((tiling_.tileLength + 7U) / 8U));
        }
        if constexpr (IsFloatType<T>::value) {
            // float comparison results are materialized as half 0/1 before half -> uint8 cast.
            pipe_->InitBuffer(tempHalf1Buf_, tiling_.tileLength * sizeof(half));
        } else if constexpr (IsInt8Type<T>::value) {
            // int8 is exactly representable by half. Two temporary half tensors remove all
            // per-element integer comparisons while preserving exact LessEqual semantics.
            pipe_->InitBuffer(tempHalf1Buf_, tiling_.tileLength * sizeof(half));
            pipe_->InitBuffer(tempHalf2Buf_, tiling_.tileLength * sizeof(half));
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
        const uint32_t sourceBytes = (IsFloatType<T>::value || IsInt32Type<T>::value)
            ? sizeof(float)
            : sizeof(half);
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
        AscendC::LocalTensor<uint8_t> readyLocal = outQueue_.DeQue<uint8_t>();
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
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Local, outputLocal, alignedCount);
        } else if constexpr (IsFloatType<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(compareMask, x1Local, x2Local, AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = tempHalf1Buf_.Get<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
        } else if constexpr (IsInt8Type<T>::value) {
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<half> x1Half = tempHalf1Buf_.Get<half>();
            AscendC::LocalTensor<half> x2Half = tempHalf2Buf_.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Compare(compareMask, x1Half, x2Half, AscendC::CMPMODE::LE, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            MaterializeMaskAsBool(x1Half, outputLocal, alignedCount);
        } else {
            // Exact int32 identity: a <= b iff min(a, b) == a. x2 is dead after this
            // comparison, so use it as the in-place Min destination and then reinterpret the
            // same queue buffer as the half 0/1 workspace. This removes the tile-sized int32
            // temporary buffer used by v5 and allows a substantially larger tile.
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Min(x2Local, x1Local, x2Local, static_cast<int32_t>(alignedCount));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, x2Local, x1Local, AscendC::CMPMODE::EQ, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = x2Local.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
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
            // int32 scalar paths are handled by ComputeQueuedX1Scalar with Maxs + Compare(EQ),
            // because a scalar <= b iff max(b, scalar) == b.
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
            // int32 scalar paths are handled by ComputeQueuedX2Scalar with Mins + Compare(EQ),
            // because a <= scalar iff min(a, scalar) == a.
        }
    }

    __aicore__ inline void ComputeQueuedPair(uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        ComputePair(x1Local, x2Local, outputLocal, elementCount);
        EnqueueOutput(outputLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeQueuedX1Scalar(T x1Scalar, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        if constexpr (IsInt32Type<T>::value) {
            // scalar <= x2 iff max(x2, scalar) == x2. x1Queue is otherwise unused in this
            // path, so borrow one queue buffer as scratch without materializing the scalar.
            AscendC::LocalTensor<T> maximum = x1Queue_.AllocTensor<T>();
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Maxs(maximum, x2Local, x1Scalar, static_cast<int32_t>(alignedCount));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, maximum, x2Local, AscendC::CMPMODE::EQ, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = maximum.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
            x1Queue_.FreeTensor(maximum);
        } else {
            ComputeX1Scalar(x1Scalar, x2Local, outputLocal, elementCount);
        }
        EnqueueOutput(outputLocal);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeQueuedX2Scalar(T x2Scalar, uint32_t elementCount)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> outputLocal = outQueue_.AllocTensor<uint8_t>();
        if constexpr (IsInt32Type<T>::value) {
            // x1 <= scalar iff min(x1, scalar) == x1. Borrow x2Queue as the scratch tile.
            AscendC::LocalTensor<T> minimum = x2Queue_.AllocTensor<T>();
            const uint32_t alignedCount = VectorAlignedCount(elementCount);
            AscendC::LocalTensor<uint8_t> compareMask = compareMaskBuf_.Get<uint8_t>();
            AscendC::Mins(minimum, x1Local, x2Scalar, static_cast<int32_t>(alignedCount));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(compareMask, minimum, x1Local, AscendC::CMPMODE::EQ, alignedCount);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> workHalf = minimum.template ReinterpretCast<half>();
            MaterializeMaskAsBool(workHalf, outputLocal, alignedCount);
            x2Queue_.FreeTensor(minimum);
        } else {
            ComputeX2Scalar(x1Local, x2Scalar, outputLocal, elementCount);
        }
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
                EnqueueX2(nextStart, nextLength);
            }

            ComputeQueuedX1Scalar(x1Value, currentLength);
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
            }

            ComputeQueuedX2Scalar(x2Value, currentLength);
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
            ComputeQueuedX2Scalar(x2Value, elementCount);
            CopyOutQueued(outputOffset, elementCount);
            return;
        }

        if (x2Contiguous) {
            const T x1Value = x1Gm_.GetValue(x1Offset);
            EnqueueX2(x2Offset, elementCount);
            ComputeQueuedX1Scalar(x1Value, elementCount);
            CopyOutQueued(outputOffset, elementCount);
            return;
        }

        // Both inputs are constant over the segment. Materialize only one input, then use
        // CompareScalar for fp16/fp32/int8 instead of expanding two complete scalar tiles.
        const T x1Value = x1Gm_.GetValue(x1Offset);
        const T x2Value = x2Gm_.GetValue(x2Offset);
        EnqueueScalarX1(x1Value, elementCount);
        ComputeQueuedX2Scalar(x2Value, elementCount);
        CopyOutQueued(outputOffset, elementCount);
    }

    __aicore__ inline uint32_t BroadcastSegmentLength(uint64_t segmentStart,
                                                       uint64_t innerOffset) const
    {
        const uint64_t innerRemaining = tiling_.innerLength - innerOffset;
        uint64_t segmentLength = blockEnd_ - segmentStart;
        if (segmentLength > innerRemaining) {
            segmentLength = innerRemaining;
        }
        if (segmentLength > tiling_.tileLength) {
            segmentLength = tiling_.tileLength;
        }
        return static_cast<uint32_t>(segmentLength);
    }

    __aicore__ inline void AdvanceBroadcastCursor(uint64_t segmentLength,
                                                   uint64_t &segmentStart,
                                                   uint64_t &innerOffset,
                                                   uint64_t &x1Offset,
                                                   uint64_t &x2Offset,
                                                   bool x1Contiguous,
                                                   bool x2Contiguous) const
    {
        segmentStart += segmentLength;
        innerOffset += segmentLength;
        if (segmentStart >= blockEnd_) {
            return;
        }
        if (innerOffset < tiling_.innerLength) {
            if (x1Contiguous) {
                x1Offset += segmentLength;
            }
            if (x2Contiguous) {
                x2Offset += segmentLength;
            }
        } else {
            innerOffset = segmentStart % tiling_.innerLength;
            InitBroadcastOffsets(segmentStart, x1Offset, x2Offset);
        }
    }

    __aicore__ inline void ProcessBroadcastBothContiguousPipelined()
    {
        uint64_t currentStart = blockStart_;
        uint64_t currentInner = currentStart % tiling_.innerLength;
        uint64_t currentX1 = 0;
        uint64_t currentX2 = 0;
        InitBroadcastOffsets(currentStart, currentX1, currentX2);
        uint32_t currentLength = BroadcastSegmentLength(currentStart, currentInner);
        EnqueueX1(currentX1, currentLength);
        EnqueueX2(currentX2, currentLength);

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;
        while (true) {
            uint64_t nextStart = currentStart;
            uint64_t nextInner = currentInner;
            uint64_t nextX1 = currentX1;
            uint64_t nextX2 = currentX2;
            AdvanceBroadcastCursor(currentLength, nextStart, nextInner,
                                   nextX1, nextX2, true, true);
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            if (hasNext) {
                nextLength = BroadcastSegmentLength(nextStart, nextInner);
                EnqueueX1(nextX1, nextLength);
                EnqueueX2(nextX2, nextLength);
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
            currentInner = nextInner;
            currentX1 = nextX1;
            currentX2 = nextX2;
            currentLength = nextLength;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void ProcessBroadcastX1ContiguousPipelined()
    {
        uint64_t currentStart = blockStart_;
        uint64_t currentInner = currentStart % tiling_.innerLength;
        uint64_t currentX1 = 0;
        uint64_t currentX2 = 0;
        InitBroadcastOffsets(currentStart, currentX1, currentX2);
        uint32_t currentLength = BroadcastSegmentLength(currentStart, currentInner);
        T currentScalar = x2Gm_.GetValue(currentX2);
        EnqueueX1(currentX1, currentLength);

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;
        while (true) {
            uint64_t nextStart = currentStart;
            uint64_t nextInner = currentInner;
            uint64_t nextX1 = currentX1;
            uint64_t nextX2 = currentX2;
            AdvanceBroadcastCursor(currentLength, nextStart, nextInner,
                                   nextX1, nextX2, true, false);
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            T nextScalar {};
            if (hasNext) {
                nextLength = BroadcastSegmentLength(nextStart, nextInner);
                nextScalar = x2Gm_.GetValue(nextX2);
                EnqueueX1(nextX1, nextLength);
            }

            ComputeQueuedX2Scalar(currentScalar, currentLength);
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
            currentInner = nextInner;
            currentX1 = nextX1;
            currentX2 = nextX2;
            currentLength = nextLength;
            currentScalar = nextScalar;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void ProcessBroadcastX2ContiguousPipelined()
    {
        uint64_t currentStart = blockStart_;
        uint64_t currentInner = currentStart % tiling_.innerLength;
        uint64_t currentX1 = 0;
        uint64_t currentX2 = 0;
        InitBroadcastOffsets(currentStart, currentX1, currentX2);
        uint32_t currentLength = BroadcastSegmentLength(currentStart, currentInner);
        T currentScalar = x1Gm_.GetValue(currentX1);
        EnqueueX2(currentX2, currentLength);

        bool hasPendingOutput = false;
        uint64_t pendingStart = 0;
        uint32_t pendingLength = 0;
        while (true) {
            uint64_t nextStart = currentStart;
            uint64_t nextInner = currentInner;
            uint64_t nextX1 = currentX1;
            uint64_t nextX2 = currentX2;
            AdvanceBroadcastCursor(currentLength, nextStart, nextInner,
                                   nextX1, nextX2, false, true);
            const bool hasNext = nextStart < blockEnd_;
            uint32_t nextLength = 0;
            T nextScalar {};
            if (hasNext) {
                nextLength = BroadcastSegmentLength(nextStart, nextInner);
                nextScalar = x1Gm_.GetValue(nextX1);
                EnqueueX2(nextX2, nextLength);
            }

            ComputeQueuedX1Scalar(currentScalar, currentLength);
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
            currentInner = nextInner;
            currentX1 = nextX1;
            currentX2 = nextX2;
            currentLength = nextLength;
            currentScalar = nextScalar;
        }
        CopyOutQueued(pendingStart, pendingLength);
    }

    __aicore__ inline void ProcessBroadcastBothConstantSerial()
    {
        uint64_t segmentStart = blockStart_;
        uint64_t innerOffset = segmentStart % tiling_.innerLength;
        uint64_t x1Offset = 0;
        uint64_t x2Offset = 0;
        InitBroadcastOffsets(segmentStart, x1Offset, x2Offset);
        while (segmentStart < blockEnd_) {
            const uint32_t segmentLength = BroadcastSegmentLength(segmentStart, innerOffset);
            const T x1Value = x1Gm_.GetValue(x1Offset);
            const T x2Value = x2Gm_.GetValue(x2Offset);
            EnqueueScalarX1(x1Value, segmentLength);
            ComputeQueuedX2Scalar(x2Value, segmentLength);
            CopyOutQueued(segmentStart, segmentLength);
            AdvanceBroadcastCursor(segmentLength, segmentStart, innerOffset,
                                   x1Offset, x2Offset, false, false);
        }
    }

    __aicore__ inline void ProcessGeneralBroadcast()
    {
        const bool x1Contiguous = tiling_.x1InnerContiguous != 0U;
        const bool x2Contiguous = tiling_.x2InnerContiguous != 0U;
        if (x1Contiguous && x2Contiguous) {
            ProcessBroadcastBothContiguousPipelined();
        } else if (x1Contiguous) {
            ProcessBroadcastX1ContiguousPipelined();
        } else if (x2Contiguous) {
            ProcessBroadcastX2ContiguousPipelined();
        } else {
            ProcessBroadcastBothConstantSerial();
        }
    }

private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> compareMaskBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf1Buf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tempHalf2Buf_;
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;

    LessEqualTilingData tiling_ {};
    uint64_t blockStart_ = 0;
    uint64_t blockEnd_ = 0;
};

template <typename DT_X1, int PATH_KIND>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME ||
                  PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR ||
                  PATH_KIND == LESS_EQUAL_PATH_TINY_X2_SCALAR) {
        // Static-tensor mode must initialize the SoC global state at kernel entry.
        AscendC::InitSocState();
    }
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    if constexpr (PATH_KIND == LESS_EQUAL_PATH_TINY_SAME ||
                  PATH_KIND == LESS_EQUAL_PATH_TINY_X1_SCALAR ||
                  PATH_KIND == LESS_EQUAL_PATH_TINY_X2_SCALAR) {
        KernelLessEqualTiny<DT_X1, PATH_KIND> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else {
        // Keeping TPipe outside the kernel class enables better Scalar constant propagation.
        AscendC::TPipe pipe;
        if constexpr (PATH_KIND == LESS_EQUAL_PATH_SMALL_SAME) {
            KernelLessEqualSmall<DT_X1> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        } else if constexpr (PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X1_SCALAR ||
                             PATH_KIND == LESS_EQUAL_PATH_MEDIUM_X2_SCALAR) {
            KernelLessEqualMediumScalar<DT_X1, PATH_KIND> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        } else {
            KernelLessEqual<DT_X1> op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
        }
    }
}
