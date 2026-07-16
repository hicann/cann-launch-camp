#include "kernel_operator.h"

#include <type_traits>

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {
constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t TILE_LENGTH = 1024;
constexpr uint32_t COMPARE_MASK_BYTES = TILE_LENGTH / 8;

template <typename T>
__aicore__ inline bool LessEqualValue(T x1, T x2)
{
    return x1 <= x2;
}

template <>
__aicore__ inline bool LessEqualValue<half>(half x1, half x2)
{

    return static_cast<float>(x1) <= static_cast<float>(x2);
}
}

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        blockStart_ = AscendC::GetBlockIdx() * tiling.blockLength;
        if (blockStart_ >= tiling.length) {
            length_ = 0;
        } else {
            length_ = tiling.length - blockStart_;
            if (length_ > tiling.blockLength) {
                length_ = tiling.blockLength;
            }
        }

        isBroadcast_ = tiling.isBroadcast;
        scalarBroadcast_ = tiling.scalarBroadcast;
        rank_ = tiling.rank;
        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            outDims_[i] = tiling.outDims[i];
            x1Strides_[i] = tiling.x1Strides[i];
            x2Strides_[i] = tiling.x2Strides[i];
        }

        x1Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y + blockStart_, length_);
        if (scalarBroadcast_ == 1) {
            scalar_ = x1Gm_.GetValue(0);
        } else if (scalarBroadcast_ == 2) {
            scalar_ = x2Gm_.GetValue(0);
        }

        pipe_.InitBuffer(x1Queue_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X1));
        pipe_.InitBuffer(x2Queue_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X1));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuf_, COMPARE_MASK_BYTES);
        pipe_.InitBuffer(work1Buf_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(work2Buf_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(work3Buf_, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(work4Buf_, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t tileCount = static_cast<uint32_t>(
            (static_cast<uint64_t>(length_) + TILE_LENGTH - 1) / TILE_LENGTH);
        for (uint32_t tile = 0; tile < tileCount; ++tile) {
            uint32_t offset = tile * TILE_LENGTH;
            uint32_t count = length_ - offset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }
            if (isBroadcast_ == 0) {
                CopyIn(offset, count);
                ComputeContiguous(count);
            } else if (scalarBroadcast_ != 0 && count == TILE_LENGTH) {
                CopyInScalarBroadcast(offset, count);
                ComputeScalarBroadcast(count);
            } else {
                ComputeBroadcast(offset, count);
            }
            CopyOut(offset, count);
        }
    }

    __aicore__ inline void CopyInScalarBroadcast(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X1> inputLocal = x1Queue_.AllocTensor<DT_X1>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(DT_X1)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X1> padParams{true, 0, 0, 0};
        if (scalarBroadcast_ == 1) {
            AscendC::DataCopyPad(inputLocal, x2Gm_[blockStart_ + offset],
                                 copyParams, padParams);
        } else {
            AscendC::DataCopyPad(inputLocal, x1Gm_[blockStart_ + offset],
                                 copyParams, padParams);
        }
        x1Queue_.EnQue(inputLocal);
    }

    __aicore__ inline void ComputeScalarBroadcast(uint32_t count)
    {
        AscendC::LocalTensor<DT_X1> inputLocal = x1Queue_.DeQue<DT_X1>();
        AscendC::LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> cmpMask = maskBuf_.Get<uint8_t>();
        const bool inputIsLeft = scalarBroadcast_ == 2;

        if constexpr (std::is_same<DT_X1, half>::value ||
                      std::is_same<DT_X1, float>::value) {
            AscendC::CompareScalar(cmpMask, inputLocal, scalar_,
                inputIsLeft ? AscendC::CMPMODE::LE : AscendC::CMPMODE::GE,
                count);
        } else if constexpr (std::is_same<DT_X1, int32_t>::value) {
            AscendC::LocalTensor<int32_t> minLocal = work1Buf_.Get<int32_t>();
            AscendC::Mins(minLocal, inputLocal, scalar_, count);
            AscendC::PipeBarrier<PIPE_V>();
            if (inputIsLeft) {
                AscendC::Compare(cmpMask, minLocal, inputLocal,
                                 AscendC::CMPMODE::EQ, count);
            } else {
                AscendC::CompareScalar(cmpMask, minLocal, scalar_,
                                       AscendC::CMPMODE::EQ, count);
            }
        } else {
            AscendC::LocalTensor<half> inputHalf = work1Buf_.Get<half>();
            AscendC::Cast(inputHalf, inputLocal, AscendC::RoundMode::CAST_NONE,
                          count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(cmpMask, inputHalf,
                static_cast<half>(scalar_),
                inputIsLeft ? AscendC::CMPMODE::LE : AscendC::CMPMODE::GE,
                count);
        }

        ExpandMask(yLocal, cmpMask, count);
        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(inputLocal);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue_.AllocTensor<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue_.AllocTensor<DT_X1>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(DT_X1)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X1> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(x1Local, x1Gm_[blockStart_ + offset], copyParams, padParams);
        AscendC::DataCopyPad(x2Local, x2Gm_[blockStart_ + offset], copyParams, padParams);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void ComputeContiguous(uint32_t count)
    {
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue_.DeQue<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue_.DeQue<DT_X1>();
        AscendC::LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();

        if (count == TILE_LENGTH) {
            ComputeVector(yLocal, x1Local, x2Local);
        } else {

            for (uint32_t i = 0; i < count; ++i) {
                yLocal.SetValue(i, static_cast<uint8_t>(LessEqualValue<DT_X1>(
                    x1Local.GetValue(i), x2Local.GetValue(i))));
            }
        }

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void ComputeVector(
        const AscendC::LocalTensor<uint8_t> &yLocal,
        const AscendC::LocalTensor<DT_X1> &x1Local,
        const AscendC::LocalTensor<DT_X1> &x2Local)
    {
        AscendC::LocalTensor<uint8_t> cmpMask = maskBuf_.Get<uint8_t>();

        if constexpr (std::is_same<DT_X1, half>::value ||
                      std::is_same<DT_X1, float>::value) {
            AscendC::Compare(cmpMask, x1Local, x2Local,
                             AscendC::CMPMODE::LE, TILE_LENGTH);
        } else if constexpr (std::is_same<DT_X1, int32_t>::value) {

            AscendC::LocalTensor<int32_t> minLocal = work1Buf_.Get<int32_t>();
            AscendC::Min(minLocal, x1Local, x2Local, TILE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(cmpMask, minLocal, x1Local,
                             AscendC::CMPMODE::EQ, TILE_LENGTH);
        } else {
            // Every int8 value is represented exactly by half. This enables
            // the hardware LE compare without changing comparison semantics.
            AscendC::LocalTensor<half> x1Half = work1Buf_.Get<half>();
            AscendC::LocalTensor<half> x2Half = work2Buf_.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE,
                          TILE_LENGTH);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE,
                          TILE_LENGTH);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(cmpMask, x1Half, x2Half,
                             AscendC::CMPMODE::LE, TILE_LENGTH);
        }

        ExpandMask(yLocal, cmpMask, TILE_LENGTH);
    }

    __aicore__ inline void ExpandMask(
        const AscendC::LocalTensor<uint8_t> &yLocal,
        const AscendC::LocalTensor<uint8_t> &cmpMask,
        uint32_t count)
    {
        AscendC::LocalTensor<half> ones = work3Buf_.Get<half>();
        AscendC::LocalTensor<half> selected = work4Buf_.Get<half>();
        AscendC::Duplicate(ones, static_cast<half>(1.0f), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(selected, cmpMask, ones, static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal, selected, AscendC::RoundMode::CAST_NONE,
                      count);
    }

    __aicore__ inline void ComputeBroadcast(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();


        uint32_t coordinates[LESS_EQUAL_MAX_DIMS];
        uint32_t remaining = blockStart_ + offset;
        uint32_t x1Index = 0;
        uint32_t x2Index = 0;
        for (uint32_t reverse = 0; reverse < rank_; ++reverse) {
            uint32_t axis = rank_ - 1 - reverse;
            uint32_t coordinate = remaining % outDims_[axis];
            remaining /= outDims_[axis];
            coordinates[axis] = coordinate;
            x1Index += coordinate * x1Strides_[axis];
            x2Index += coordinate * x2Strides_[axis];
        }

        for (uint32_t i = 0; i < count; ++i) {
            yLocal.SetValue(i, static_cast<uint8_t>(
                LessEqualValue<DT_X1>(x1Gm_.GetValue(x1Index), x2Gm_.GetValue(x2Index))));

            for (uint32_t reverse = 0; reverse < rank_; ++reverse) {
                uint32_t axis = rank_ - 1 - reverse;
                ++coordinates[axis];
                x1Index += x1Strides_[axis];
                x2Index += x2Strides_[axis];
                if (coordinates[axis] < outDims_[axis]) {
                    break;
                }
                coordinates[axis] = 0;
                x1Index -= outDims_[axis] * x1Strides_[axis];
                x2Index -= outDims_[axis] * x2Strides_[axis];
            }
        }
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        AscendC::LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(uint8_t)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x1Queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x2Queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work3Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> work4Buf_;
    AscendC::GlobalTensor<DT_X1> x1Gm_;
    AscendC::GlobalTensor<DT_X1> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint32_t length_ = 0;
    uint32_t blockStart_ = 0;
    uint32_t rank_ = 0;
    uint32_t isBroadcast_ = 0;
    uint32_t scalarBroadcast_ = 0;
    DT_X1 scalar_ = static_cast<DT_X1>(0);
    uint32_t outDims_[LESS_EQUAL_MAX_DIMS];
    uint32_t x1Strides_[LESS_EQUAL_MAX_DIMS];
    uint32_t x2Strides_[LESS_EQUAL_MAX_DIMS];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}
