// Ascend C kernel for LessEqual.
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

template <typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y));

        totalLength_ = tiling.totalLength;
        rank_ = tiling.rank;
        blockNum_ = tiling.blockNum;
        tileLength_ = tiling.tileLength;
        mode_ = tiling.mode;
        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            outputDims_[i] = tiling.outputDims[i];
            x1Strides_[i] = tiling.x1Strides[i];
            x2Strides_[i] = tiling.x2Strides[i];
        }

        pipe_.InitBuffer(x1Queue_, 1, tileLength_ * sizeof(T));
        pipe_.InitBuffer(x2Queue_, 1, tileLength_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, 1, tileLength_ * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuf_, tileLength_ / 8);
        pipe_.InitBuffer(onesBuf_, tileLength_ * sizeof(half));
        pipe_.InitBuffer(selectBuf_, tileLength_ * sizeof(half));

        if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            pipe_.InitBuffer(castX1Buf_, tileLength_ * sizeof(half));
            pipe_.InitBuffer(castX2Buf_, tileLength_ * sizeof(half));
        }
        if constexpr (AscendC::IsSameType<T, int32_t>::value) {
            pipe_.InitBuffer(minBuf_, tileLength_ * sizeof(int32_t));
        }
        if (mode_ == LESS_EQUAL_MODE_X1_ROW || mode_ == LESS_EQUAL_MODE_X2_ROW) {
            pipe_.InitBuffer(rowBuf_, tileLength_ * sizeof(T));
        }

        if (totalLength_ != 0) {
            Duplicate(onesBuf_.Get<half>(), static_cast<half>(1.0f), tileLength_);
        }
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0 || GetBlockIdx() >= blockNum_) {
            return;
        }

        if (mode_ == LESS_EQUAL_MODE_X1_ROW || mode_ == LESS_EQUAL_MODE_X2_ROW) {
            ProcessRowBroadcast();
            return;
        }

        constexpr uint64_t coreAlignment = 256;
        const uint64_t rawPerCore = (totalLength_ + blockNum_ - 1) / blockNum_;
        const uint64_t elementsPerCore =
            (rawPerCore + coreAlignment - 1) / coreAlignment * coreAlignment;
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * elementsPerCore;
        if (start >= totalLength_) {
            return;
        }
        const uint64_t length =
            elementsPerCore < totalLength_ - start ? elementsPerCore : totalLength_ - start;

        if (mode_ == LESS_EQUAL_MODE_GENERAL) {
            ProcessGeneral(start, length);
        } else if (mode_ == LESS_EQUAL_MODE_X1_REPEAT ||
                   mode_ == LESS_EQUAL_MODE_X2_REPEAT) {
            ProcessRepeatRows(start, length);
        } else {
            ProcessFast(start, length);
        }
    }

private:
    __aicore__ inline uint64_t InputOffset(uint64_t outputIndex, const uint64_t *strides) const
    {
        uint64_t offset = 0;
        for (int32_t dim = static_cast<int32_t>(rank_) - 1; dim >= 0; --dim) {
            const uint64_t coord = outputIndex % outputDims_[dim];
            outputIndex /= outputDims_[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    __aicore__ inline uint32_t AlignComputeCount(uint32_t count) const
    {
        constexpr uint32_t compareElements =
            (AscendC::IsSameType<T, float>::value || AscendC::IsSameType<T, int32_t>::value) ? 64 : 128;
        return (count + compareElements - 1) / compareElements * compareElements;
    }

    __aicore__ inline void FillInput(TQue<QuePosition::VECIN, 1> &queue, GlobalTensor<T> &gm,
                                     uint64_t inputOffset, bool scalar,
                                     uint32_t count, uint32_t computeCount)
    {
        LocalTensor<T> local = queue.AllocTensor<T>();
        if (scalar) {
            if constexpr (AscendC::IsSameType<T, int8_t>::value) {
                const uint16_t byte = static_cast<uint8_t>(gm.GetValue(inputOffset));
                const int16_t packedValue = static_cast<int16_t>(byte | (byte << 8));
                Duplicate(local.template ReinterpretCast<int16_t>(), packedValue, computeCount / 2);
            } else {
                Duplicate(local, gm.GetValue(inputOffset), computeCount);
            }
        } else if ((count * sizeof(T)) % 32 == 0) {
            DataCopy(local, gm[inputOffset], count);
        } else {
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams {false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, gm[inputOffset], copyParams, padParams);
        }
        queue.EnQue(local);
    }

    __aicore__ inline void ComputeVector(uint32_t computeCount)
    {
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

        if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            LocalTensor<half> castX1 = castX1Buf_.Get<half>();
            LocalTensor<half> castX2 = castX2Buf_.Get<half>();
            Cast(castX1, x1Local, RoundMode::CAST_NONE, computeCount);
            Cast(castX2, x2Local, RoundMode::CAST_NONE, computeCount);
            PipeBarrier<PIPE_V>();
            Compare(mask, castX1, castX2, CMPMODE::LE, computeCount);
        } else if constexpr (AscendC::IsSameType<T, int32_t>::value) {
            LocalTensor<int32_t> minLocal = minBuf_.Get<int32_t>();
            Min(minLocal, x1Local, x2Local, static_cast<int32_t>(computeCount));
            PipeBarrier<PIPE_V>();
            Compare(mask, x1Local, minLocal, CMPMODE::EQ, computeCount);
        } else {
            Compare(mask, x1Local, x2Local, CMPMODE::LE, computeCount);
        }

        PipeBarrier<PIPE_V>();
        LocalTensor<half> selected = selectBuf_.Get<half>();
        Select(selected, mask, onesBuf_.Get<half>(), static_cast<half>(0.0f),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, computeCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, selected, RoundMode::CAST_RINT, computeCount);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOutput(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        if (count % 32 == 0) {
            DataCopy(yGm_[outputOffset], yLocal, count);
        } else {
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(count * sizeof(uint8_t)), 0, 0, 0};
            DataCopyPad(yGm_[outputOffset], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void RunSegment(uint64_t outputOffset, uint64_t x1Offset, uint64_t x2Offset,
                                      bool x1Scalar, bool x2Scalar, uint32_t count)
    {
        const uint32_t computeCount = AlignComputeCount(count);
        FillInput(x1Queue_, x1Gm_, x1Offset, x1Scalar, count, computeCount);
        FillInput(x2Queue_, x2Gm_, x2Offset, x2Scalar, count, computeCount);
        ComputeVector(computeCount);
        CopyOutput(outputOffset, count);
    }

    __aicore__ inline void ProcessFast(uint64_t start, uint64_t length)
    {
        const bool x1Scalar = mode_ == LESS_EQUAL_MODE_X1_SCALAR;
        const bool x2Scalar = mode_ == LESS_EQUAL_MODE_X2_SCALAR;
        uint64_t outputOffset = start;
        const uint64_t end = start + length;
        while (outputOffset < end) {
            const uint64_t remain = end - outputOffset;
            const uint32_t count =
                static_cast<uint32_t>(remain < tileLength_ ? remain : tileLength_);
            RunSegment(outputOffset, x1Scalar ? 0 : outputOffset, x2Scalar ? 0 : outputOffset,
                       x1Scalar, x2Scalar, count);
            outputOffset += count;
        }
    }

    __aicore__ inline void FillRowBroadcast(TQue<QuePosition::VECIN, 1> &queue,
                                             GlobalTensor<T> &gm,
                                             uint32_t rows, uint32_t inner)
    {
        LocalTensor<T> rowLocal = rowBuf_.Get<T>();
        DataCopy(rowLocal, gm[0], inner);
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        LocalTensor<T> local = queue.AllocTensor<T>();
        const uint32_t dstShape[2] = {rows, inner};
        const uint32_t srcShape[2] = {1, inner};
        Broadcast<T, 2, 0>(local, rowLocal, dstShape, srcShape);
        queue.EnQue(local);
    }

    __aicore__ inline void ProcessRowBroadcast()
    {
        const uint32_t inner = static_cast<uint32_t>(outputDims_[rank_ - 1]);
        const uint64_t totalRows = totalLength_ / inner;
        const uint64_t rowsPerCore = (totalRows + blockNum_ - 1) / blockNum_;
        uint64_t row = static_cast<uint64_t>(GetBlockIdx()) * rowsPerCore;
        if (row >= totalRows) {
            return;
        }
        const uint64_t rowEnd =
            row + rowsPerCore < totalRows ? row + rowsPerCore : totalRows;
        const uint32_t maxRowsPerTile = tileLength_ / inner;
        while (row < rowEnd) {
            const uint32_t rows = static_cast<uint32_t>(
                rowEnd - row < maxRowsPerTile ? rowEnd - row : maxRowsPerTile);
            const uint32_t count = rows * inner;
            const uint64_t outputOffset = row * inner;
            const uint32_t computeCount = AlignComputeCount(count);
            if (mode_ == LESS_EQUAL_MODE_X1_ROW) {
                FillRowBroadcast(x1Queue_, x1Gm_, rows, inner);
                FillInput(x2Queue_, x2Gm_, outputOffset, false, count, computeCount);
            } else {
                FillInput(x1Queue_, x1Gm_, outputOffset, false, count, computeCount);
                FillRowBroadcast(x2Queue_, x2Gm_, rows, inner);
            }
            ComputeVector(computeCount);
            CopyOutput(outputOffset, count);
            row += rows;
        }
    }

    __aicore__ inline void ProcessRepeatRows(uint64_t start, uint64_t length)
    {
        const uint64_t inner = outputDims_[rank_ - 1];
        const bool x1Repeat = mode_ == LESS_EQUAL_MODE_X1_REPEAT;
        uint64_t outputOffset = start;
        const uint64_t end = start + length;
        while (outputOffset < end) {
            const uint64_t repeatedOffset = outputOffset % inner;
            const uint64_t rowRemain = inner - repeatedOffset;
            uint64_t segment = end - outputOffset;
            segment = segment < rowRemain ? segment : rowRemain;
            segment = segment < tileLength_ ? segment : tileLength_;
            RunSegment(outputOffset,
                       x1Repeat ? repeatedOffset : outputOffset,
                       x1Repeat ? outputOffset : repeatedOffset,
                       false, false, static_cast<uint32_t>(segment));
            outputOffset += segment;
        }
    }

    __aicore__ inline void ProcessGeneral(uint64_t start, uint64_t length)
    {
        const uint64_t inner = rank_ == 0 ? 1 : outputDims_[rank_ - 1];
        const bool x1ScalarLast = rank_ != 0 && x1Strides_[rank_ - 1] == 0;
        const bool x2ScalarLast = rank_ != 0 && x2Strides_[rank_ - 1] == 0;

        uint64_t outputOffset = start;
        const uint64_t end = start + length;
        while (outputOffset < end) {
            const uint64_t rowRemain = inner - outputOffset % inner;
            uint64_t segment = end - outputOffset;
            segment = segment < rowRemain ? segment : rowRemain;
            segment = segment < tileLength_ ? segment : tileLength_;
            RunSegment(outputOffset, InputOffset(outputOffset, x1Strides_),
                       InputOffset(outputOffset, x2Strides_),
                       x1ScalarLast, x2ScalarLast, static_cast<uint32_t>(segment));
            outputOffset += segment;
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECIN, 1> x2Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> maskBuf_;
    TBuf<QuePosition::VECCALC> onesBuf_;
    TBuf<QuePosition::VECCALC> selectBuf_;
    TBuf<QuePosition::VECCALC> castX1Buf_;
    TBuf<QuePosition::VECCALC> castX2Buf_;
    TBuf<QuePosition::VECCALC> minBuf_;
    TBuf<QuePosition::VECCALC> rowBuf_;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<uint8_t> yGm_;
    uint64_t outputDims_[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Strides_[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Strides_[LESS_EQUAL_MAX_DIMS];
    uint64_t totalLength_;
    uint32_t rank_;
    uint32_t blockNum_;
    uint32_t tileLength_;
    uint32_t mode_;
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
