#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr uint32_t BUFFER_NUM = 1U;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR clipValueMin,
        GM_ADDR clipValueMax,
        GM_ADDR y,
        uint32_t length,
        uint32_t tileLength,
        uint32_t minIsScalar,
        uint32_t maxIsScalar) {
        length_ = length;
        tileLength_ = tileLength;
        minIsScalar_ = (minIsScalar != 0U);
        maxIsScalar_ = (maxIsScalar != 0U);

        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X *>(x), length_);
        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X *>(y), length_);
        minGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X *>(clipValueMin),
            minIsScalar_ ? 1U : length_);
        maxGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X *>(clipValueMax),
            maxIsScalar_ ? 1U : length_);

        if (minIsScalar_) {
            minScalarValue_ = minGm_.GetValue(0);
        }
        if (maxIsScalar_) {
            maxScalarValue_ = maxGm_.GetValue(0);
        }

        const uint32_t tileBytes =
            tileLength_ * static_cast<uint32_t>(sizeof(DT_X));
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileBytes);
        pipe_.InitBuffer(tmpBuffer_, tileBytes);

        if (!minIsScalar_) {
            pipe_.InitBuffer(minQueue_, BUFFER_NUM, tileBytes);
        }
        if (!maxIsScalar_) {
            pipe_.InitBuffer(maxQueue_, BUFFER_NUM, tileBytes);
        }
    }

    __aicore__ inline void Process() {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t baseLength = length_ / blockNum;
        const uint32_t remainder = length_ % blockNum;

        uint32_t blockLength;
        uint32_t globalOffset;
        if (blockIdx < remainder) {
            blockLength = baseLength + 1U;
            globalOffset = blockIdx * blockLength;
        } else {
            blockLength = baseLength;
            globalOffset = remainder * (baseLength + 1U) +
                           (blockIdx - remainder) * baseLength;
        }

        uint32_t processed = 0U;
        while (processed < blockLength) {
            uint32_t currentLength = blockLength - processed;
            if (currentLength > tileLength_) {
                currentLength = tileLength_;
            }

            CopyIn(globalOffset + processed, currentLength);
            Compute(currentLength);
            CopyOut(globalOffset + processed, currentLength);
            processed += currentLength;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        const AscendC::DataCopyExtParams copyParams{
            1,
            count * static_cast<uint32_t>(sizeof(DT_X)),
            0,
            0,
            0};
        const AscendC::DataCopyPadExtParams<DT_X> padParams{
            false, 0, 0, 0};

        AscendC::LocalTensor<DT_X> xLocal =
            xQueue_.template AllocTensor<DT_X>();
        AscendC::DataCopyPad(
            xLocal, xGm_[offset], copyParams, padParams);
        xQueue_.EnQue(xLocal);

        if (!minIsScalar_) {
            AscendC::LocalTensor<DT_X> minLocal =
                minQueue_.template AllocTensor<DT_X>();
            AscendC::DataCopyPad(
                minLocal, minGm_[offset], copyParams, padParams);
            minQueue_.EnQue(minLocal);
        }

        if (!maxIsScalar_) {
            AscendC::LocalTensor<DT_X> maxLocal =
                maxQueue_.template AllocTensor<DT_X>();
            AscendC::DataCopyPad(
                maxLocal, maxGm_[offset], copyParams, padParams);
            maxQueue_.EnQue(maxLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t count) {
        const int32_t computeCount = static_cast<int32_t>(count);
        AscendC::LocalTensor<DT_X> xLocal =
            xQueue_.template DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> tmpLocal =
            tmpBuffer_.template Get<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal =
            yQueue_.template AllocTensor<DT_X>();

        if (minIsScalar_) {
            AscendC::Maxs(
                tmpLocal, xLocal, minScalarValue_, computeCount);
        } else {
            AscendC::LocalTensor<DT_X> minLocal =
                minQueue_.template DeQue<DT_X>();
            AscendC::Max(
                tmpLocal, xLocal, minLocal, computeCount);
            minQueue_.FreeTensor(minLocal);
        }

        if (maxIsScalar_) {
            AscendC::Mins(
                yLocal, tmpLocal, maxScalarValue_, computeCount);
        } else {
            AscendC::LocalTensor<DT_X> maxLocal =
                maxQueue_.template DeQue<DT_X>();
            AscendC::Min(
                yLocal, tmpLocal, maxLocal, computeCount);
            maxQueue_.FreeTensor(maxLocal);
        }

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        const AscendC::DataCopyExtParams copyParams{
            1,
            count * static_cast<uint32_t>(sizeof(DT_X)),
            0,
            0,
            0};
        AscendC::LocalTensor<DT_X> yLocal =
            yQueue_.template DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        yQueue_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> minQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> maxQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer_;

    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> minGm_;
    AscendC::GlobalTensor<DT_X> maxGm_;
    AscendC::GlobalTensor<DT_X> yGm_;

    uint32_t length_;
    uint32_t tileLength_;
    bool minIsScalar_;
    bool maxIsScalar_;
    DT_X minScalarValue_;
    DT_X maxScalarValue_;
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(
    GM_ADDR x,
    GM_ADDR clip_value_min,
    GM_ADDR clip_value_max,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        ClipByValueTilingData, tilingData, tiling);

    KernelClipByValue<DT_X> op;
    op.Init(
        x,
        clip_value_min,
        clip_value_max,
        y,
        tilingData.length,
        tilingData.tileLength,
        tilingData.minIsScalar,
        tilingData.maxIsScalar);
    op.Process();
}
