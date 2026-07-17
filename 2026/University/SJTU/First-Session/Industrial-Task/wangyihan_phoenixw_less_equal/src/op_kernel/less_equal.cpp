// Kernel-side LessEqual implementation.
#include <type_traits>

#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

template <class DT_X1, int IS_BROADCAST, int IS_SMALL>
class KernelLessEqual {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData *tiling)
    {
        tiling_ = tiling;
        tileLength_ = tiling_->tileLength;
        x1Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y, tiling_->length);
        if constexpr (IS_SMALL == 1 && std::is_same_v<DT_X1, half>) {
            x1HalfBitsGm_.SetGlobalBuffer((__gm__ uint16_t *)x1);
            x2HalfBitsGm_.SetGlobalBuffer((__gm__ uint16_t *)x2);
        }
        if constexpr (IS_SMALL == 1) {
            return;
        }

        pipe_.InitBuffer(x1Queue_, 2, tileLength_ * sizeof(DT_X1));
        pipe_.InitBuffer(x2Queue_, 2, tileLength_ * sizeof(DT_X1));
        pipe_.InitBuffer(cmpBuf_, ((tileLength_ + 7) / 8 + 31) / 32 * 32);
        pipe_.InitBuffer(outQueue_, 2, tileLength_ * sizeof(int8_t));
        pipe_.InitBuffer(onesBuf_, tileLength_ * sizeof(half));
        pipe_.InitBuffer(outHalfBuf_, tileLength_ * sizeof(half));
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe_.InitBuffer(x1CastBuf_, tileLength_ * sizeof(half));
            pipe_.InitBuffer(x2CastBuf_, tileLength_ * sizeof(half));
        }
        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe_.InitBuffer(minBuf_, tileLength_ * sizeof(int32_t));
        }
        constexpr uint32_t kElementsPerRepeat = 256 / sizeof(DT_X1);
        uint64_t maxTileCount = tileLength_;
        if constexpr (IS_BROADCAST == 0) {
            const uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
            const uint64_t maxBlockLength = (tiling_->length + blockNum - 1) / blockNum;
            maxTileCount = maxBlockLength < tileLength_ ? maxBlockLength : tileLength_;
        } else if (tiling_->rank > 0) {
            const uint64_t lastDimLength = tiling_->outShape[tiling_->rank - 1];
            maxTileCount = lastDimLength < tileLength_ ? lastDimLength : tileLength_;
        }
        const uint32_t onesCount = static_cast<uint32_t>(maxTileCount) /
            kElementsPerRepeat * kElementsPerRepeat;
        if (onesCount > 0) {
            auto onesLocal = onesBuf_.Get<half>();
            AscendC::Duplicate(onesLocal, static_cast<half>(1.0), onesCount);
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_->length == 0) {
            return;
        }
        if constexpr (IS_SMALL == 1) {
            ProcessSmall();
        } else if constexpr (IS_BROADCAST == 0) {
            ProcessSameShape();
        } else {
            ProcessBroadcast();
        }
    }

private:
    __aicore__ inline void ProcessSmall()
    {
        // For tiny tensors, UB allocation, DMA setup, and vector-mask conversion
        // cost more than the comparison itself. Host schedules this path on one core.
        for (uint64_t outIndex = 0; outIndex < tiling_->length; ++outIndex) {
            uint64_t x1Offset = outIndex;
            uint64_t x2Offset = outIndex;
            if constexpr (IS_BROADCAST == 1) {
                GetBroadcastOffsets(outIndex, x1Offset, x2Offset);
            }
            yGm_.SetValue(outIndex, CompareScalar(x1Offset, x2Offset) ? 1 : 0);
        }
    }

    __aicore__ inline void GetBroadcastOffsets(uint64_t outIndex, uint64_t &x1Offset, uint64_t &x2Offset)
    {
        x1Offset = 0;
        x2Offset = 0;
        for (int32_t dim = static_cast<int32_t>(tiling_->rank) - 1; dim >= 0; --dim) {
            const uint64_t coord = outIndex % tiling_->outShape[dim];
            outIndex /= tiling_->outShape[dim];
            x1Offset += coord * tiling_->x1Stride[dim];
            x2Offset += coord * tiling_->x2Stride[dim];
        }
    }

    __aicore__ inline bool CompareScalar(uint64_t x1Offset, uint64_t x2Offset)
    {
        if constexpr (std::is_same_v<DT_X1, half>) {
            return HalfLessEqual(x1HalfBitsGm_.GetValue(x1Offset), x2HalfBitsGm_.GetValue(x2Offset));
        } else {
            return x1Gm_.GetValue(x1Offset) <= x2Gm_.GetValue(x2Offset);
        }
    }

    __aicore__ inline void ProcessSameShape()
    {
        uint64_t start = 0;
        uint64_t end = 0;
        GetElementRange(tiling_->length, start, end);
        if (start >= end) {
            return;
        }

        auto cmpLocal = cmpBuf_.Get<uint8_t>();
        for (uint64_t offset = start; offset < end; offset += tileLength_) {
            const uint32_t count = static_cast<uint32_t>(
                (end - offset) > tileLength_ ? tileLength_ : (end - offset));
            auto x1Local = x1Queue_.AllocTensor<DT_X1>();
            auto x2Local = x2Queue_.AllocTensor<DT_X1>();
            CopyContiguous(x1Local, x1Gm_, offset, count);
            CopyContiguous(x2Local, x2Gm_, offset, count);
            x1Queue_.EnQue(x1Local);
            x2Queue_.EnQue(x2Local);
            x1Local = x1Queue_.DeQue<DT_X1>();
            x2Local = x2Queue_.DeQue<DT_X1>();
            auto outLocal = outQueue_.AllocTensor<int8_t>();
            Compute(x1Local, x2Local, cmpLocal, outLocal, count, false, false);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
            outQueue_.EnQue(outLocal);
            outLocal = outQueue_.DeQue<int8_t>();
            CopyOutput(offset, outLocal, count);
            outQueue_.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void ProcessBroadcast()
    {
        const uint64_t lastDimLength = tiling_->outShape[tiling_->rank - 1];
        const uint64_t totalRows = tiling_->length / lastDimLength;
        const uint64_t chunksPerRow = (lastDimLength + tileLength_ - 1) / tileLength_;
        const uint64_t totalChunks = totalRows * chunksPerRow;
        uint64_t chunkStart = 0;
        uint64_t chunkEnd = 0;
        GetElementRange(totalChunks, chunkStart, chunkEnd);
        if (chunkStart >= chunkEnd) {
            return;
        }

        auto cmpLocal = cmpBuf_.Get<uint8_t>();
        const uint32_t lastDim = tiling_->rank - 1;
        uint64_t row = chunkStart / chunksPerRow;
        uint64_t chunkInRow = chunkStart % chunksPerRow;
        uint64_t rowCoord[25] = {0};
        uint64_t x1Base = 0;
        uint64_t x2Base = 0;
        InitRowState(row, rowCoord, x1Base, x2Base);
        for (uint64_t chunk = chunkStart; chunk < chunkEnd; ++chunk) {
            const uint64_t col = chunkInRow * tileLength_;
            const bool x1Scalar = tiling_->x1Stride[lastDim] == 0;
            const bool x2Scalar = tiling_->x2Stride[lastDim] == 0;
            const uint32_t count = static_cast<uint32_t>(
                (lastDimLength - col) > tileLength_ ? tileLength_ : (lastDimLength - col));
            auto x1Local = x1Queue_.AllocTensor<DT_X1>();
            auto x2Local = x2Queue_.AllocTensor<DT_X1>();
            LoadBroadcastOperand(x1Local, x1Gm_, x1Base, col, count, x1Scalar);
            LoadBroadcastOperand(x2Local, x2Gm_, x2Base, col, count, x2Scalar);
            x1Queue_.EnQue(x1Local);
            x2Queue_.EnQue(x2Local);
            x1Local = x1Queue_.DeQue<DT_X1>();
            x2Local = x2Queue_.DeQue<DT_X1>();
            auto outLocal = outQueue_.AllocTensor<int8_t>();
            Compute(x1Local, x2Local, cmpLocal, outLocal, count, x1Scalar, x2Scalar);
            x1Queue_.FreeTensor(x1Local);
            x2Queue_.FreeTensor(x2Local);
            outQueue_.EnQue(outLocal);
            outLocal = outQueue_.DeQue<int8_t>();
            CopyOutput(row * lastDimLength + col, outLocal, count);
            outQueue_.FreeTensor(outLocal);
            ++chunkInRow;
            if (chunkInRow == chunksPerRow && chunk + 1 < chunkEnd) {
                chunkInRow = 0;
                ++row;
                AdvanceRowState(rowCoord, x1Base, x2Base);
            }
        }
    }

    __aicore__ inline void GetElementRange(uint64_t total, uint64_t &start, uint64_t &end)
    {
        const uint64_t blockIdx = static_cast<uint64_t>(AscendC::GetBlockIdx());
        const uint64_t blockNum = static_cast<uint64_t>(AscendC::GetBlockNum());
        const uint64_t former = total % blockNum;
        const uint64_t base = total / blockNum;
        if (blockIdx < former) {
            start = blockIdx * (base + 1);
            end = start + base + 1;
        } else {
            start = former * (base + 1) + (blockIdx - former) * base;
            end = start + base;
        }
    }

    __aicore__ inline void InitRowState(uint64_t row, uint64_t *coord,
                                        uint64_t &x1Offset, uint64_t &x2Offset)
    {
        x1Offset = 0;
        x2Offset = 0;
        for (int32_t dim = static_cast<int32_t>(tiling_->rank) - 2; dim >= 0; --dim) {
            coord[dim] = row % tiling_->outShape[dim];
            row /= tiling_->outShape[dim];
            x1Offset += coord[dim] * tiling_->x1Stride[dim];
            x2Offset += coord[dim] * tiling_->x2Stride[dim];
        }
    }

    __aicore__ inline void AdvanceRowState(uint64_t *coord, uint64_t &x1Offset, uint64_t &x2Offset)
    {
        for (int32_t dim = static_cast<int32_t>(tiling_->rank) - 2; dim >= 0; --dim) {
            ++coord[dim];
            x1Offset += tiling_->x1Stride[dim];
            x2Offset += tiling_->x2Stride[dim];
            if (coord[dim] < tiling_->outShape[dim]) {
                break;
            }
            coord[dim] = 0;
            x1Offset -= tiling_->outShape[dim] * tiling_->x1Stride[dim];
            x2Offset -= tiling_->outShape[dim] * tiling_->x2Stride[dim];
        }
    }

    __aicore__ inline void CopyContiguous(AscendC::LocalTensor<DT_X1> &local,
                                           AscendC::GlobalTensor<DT_X1> &gm,
                                           uint64_t offset, uint32_t count)
    {
        AscendC::DataCopyExtParams copyParams(1, count * sizeof(DT_X1), 0, 0, 0);
        AscendC::DataCopyPadExtParams<DT_X1> padParams(false, 0, 0, static_cast<DT_X1>(0));
        AscendC::DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void LoadBroadcastOperand(AscendC::LocalTensor<DT_X1> &local,
                                                 AscendC::GlobalTensor<DT_X1> &gm,
                                                 uint64_t base, uint64_t col,
                                                 uint32_t count, bool isScalar)
    {
        CopyContiguous(local, gm, isScalar ? base : base + col, isScalar ? 1 : count);
    }

    __aicore__ inline uint32_t VectorCount(uint32_t count)
    {
        constexpr uint32_t kVectorBytes = 256;
        constexpr uint32_t kElementsPerRepeat = kVectorBytes / sizeof(DT_X1);
        return count / kElementsPerRepeat * kElementsPerRepeat;
    }

    __aicore__ inline void Compute(AscendC::LocalTensor<DT_X1> &x1Local,
                                   AscendC::LocalTensor<DT_X1> &x2Local,
                                   AscendC::LocalTensor<uint8_t> &cmpLocal,
                                   AscendC::LocalTensor<int8_t> &outLocal,
                                   uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        const uint32_t vectorCount = VectorCount(count);
        if (vectorCount > 0) {
            ComputeVector(x1Local, x2Local, cmpLocal, outLocal, vectorCount, x1Scalar, x2Scalar);
        }
        ComputeTail(x1Local, x2Local, outLocal, vectorCount, count, x1Scalar, x2Scalar);
    }

    __aicore__ inline void ComputeVector(AscendC::LocalTensor<DT_X1> &x1Local,
                                         AscendC::LocalTensor<DT_X1> &x2Local,
                                         AscendC::LocalTensor<uint8_t> &cmpLocal,
                                         AscendC::LocalTensor<int8_t> &outLocal,
                                         uint32_t count, bool x1Scalar, bool x2Scalar)
    {
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            auto x1Half = x1CastBuf_.Get<half>();
            auto x2Half = x2CastBuf_.Get<half>();
            if (x1Scalar) {
                AscendC::Duplicate(x1Half, static_cast<half>(static_cast<float>(x1Local.GetValue(0))), count);
            } else {
                AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, count);
            }
            if (x2Scalar) {
                AscendC::Duplicate(x2Half, static_cast<half>(static_cast<float>(x2Local.GetValue(0))), count);
            } else {
                AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, count);
            }
            AscendC::Compare(cmpLocal, x1Half, x2Half, AscendC::CMPMODE::LE, count);
        } else {
            if (x1Scalar) {
                AscendC::Duplicate(x1Local, x1Local.GetValue(0), count);
            }
            if (x2Scalar) {
                AscendC::Duplicate(x2Local, x2Local.GetValue(0), count);
            }
            if constexpr (std::is_same_v<DT_X1, int32_t>) {
                auto minLocal = minBuf_.Get<int32_t>();
                AscendC::Min(minLocal, x1Local, x2Local, count);
                AscendC::Compare(cmpLocal, x1Local, minLocal, AscendC::CMPMODE::EQ, count);
            } else {
                AscendC::Compare(cmpLocal, x1Local, x2Local, AscendC::CMPMODE::LE, count);
            }
        }

        auto onesLocal = onesBuf_.Get<half>();
        auto outHalfLocal = outHalfBuf_.Get<half>();
        AscendC::Select(outHalfLocal, cmpLocal, onesLocal, static_cast<half>(0.0),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        AscendC::Cast(outLocal, outHalfLocal, AscendC::RoundMode::CAST_RINT, count);
    }

    __aicore__ inline void ComputeTail(AscendC::LocalTensor<DT_X1> &x1Local,
                                       AscendC::LocalTensor<DT_X1> &x2Local,
                                       AscendC::LocalTensor<int8_t> &outLocal,
                                       uint32_t begin, uint32_t count,
                                       bool x1Scalar, bool x2Scalar)
    {
        if constexpr (std::is_same_v<DT_X1, half>) {
            auto x1Bits = x1Local.template ReinterpretCast<uint16_t>();
            auto x2Bits = x2Local.template ReinterpretCast<uint16_t>();
            for (uint32_t i = begin; i < count; ++i) {
                const uint16_t lhs = x1Bits.GetValue(x1Scalar ? 0 : i);
                const uint16_t rhs = x2Bits.GetValue(x2Scalar ? 0 : i);
                outLocal.SetValue(i, HalfLessEqual(lhs, rhs) ? 1 : 0);
            }
        } else {
            for (uint32_t i = begin; i < count; ++i) {
                const DT_X1 lhs = x1Local.GetValue(x1Scalar ? 0 : i);
                const DT_X1 rhs = x2Local.GetValue(x2Scalar ? 0 : i);
                outLocal.SetValue(i, lhs <= rhs ? 1 : 0);
            }
        }
    }

    __aicore__ inline void CopyOutput(uint64_t offset, AscendC::LocalTensor<int8_t> &outLocal, uint32_t count)
    {
        AscendC::DataCopyPad(yGm_[offset], outLocal, AscendC::DataCopyExtParams(1, count, 0, 0, 0));
    }

    __aicore__ inline bool HalfLessEqual(uint16_t lhs, uint16_t rhs)
    {
        if (IsHalfNan(lhs) || IsHalfNan(rhs)) {
            return false;
        }
        return HalfOrderKey(lhs) <= HalfOrderKey(rhs);
    }

    __aicore__ inline bool IsHalfNan(uint16_t value)
    {
        return ((value & 0x7C00U) == 0x7C00U) && ((value & 0x03FFU) != 0);
    }

    __aicore__ inline uint32_t HalfOrderKey(uint16_t value)
    {
        const uint32_t sign = value & 0x8000U;
        const uint32_t magnitude = value & 0x7FFFU;
        if (magnitude == 0) {
            return 0x8000U;
        }
        return sign == 0 ? 0x8000U + magnitude : 0x8000U - magnitude;
    }

private:
    const LessEqualTilingData *tiling_;
    uint32_t tileLength_;
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> x1Queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> x2Queue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> cmpBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> onesBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> outHalfBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x1CastBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2CastBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> minBuf_;
    AscendC::GlobalTensor<DT_X1> x1Gm_;
    AscendC::GlobalTensor<DT_X1> x2Gm_;
    AscendC::GlobalTensor<uint16_t> x1HalfBitsGm_;
    AscendC::GlobalTensor<uint16_t> x2HalfBitsGm_;
    AscendC::GlobalTensor<int8_t> yGm_;
};

template <typename DT_X1, int IS_BROADCAST, int IS_SMALL>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    KernelLessEqual<DT_X1, IS_BROADCAST, IS_SMALL> op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}
