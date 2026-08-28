// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

constexpr uint32_t BUFFER_NUM = 2;

template <typename T>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR clipValueMin,
        GM_ADDR clipValueMax,
        GM_ADDR y,
        const ClipByValueTilingData &tiling)
    {
        tileLength_ = tiling.tileLength == 0 ? 32U : tiling.tileLength;
        alignElems_ = tiling.alignElems == 0 ? (32U / sizeof(T)) : tiling.alignElems;
        minIsScalar_ = tiling.minIsScalar != 0;
        maxIsScalar_ = tiling.maxIsScalar != 0;

        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t perCore =
            tiling.lengthPerCore == 0 ? tiling.totalLength : tiling.lengthPerCore;
        startOffset_ = blockIdx * perCore;
        if (startOffset_ >= tiling.totalLength) {
            coreLength_ = 0;
            return;
        }
        coreLength_ = tiling.totalLength - startOffset_;
        if (coreLength_ > perCore) {
            coreLength_ = perCore;
        }

        xGm_.SetGlobalBuffer((__gm__ T *)x + startOffset_, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ T *)y + startOffset_, coreLength_);

        if (minIsScalar_) {
            minGm_.SetGlobalBuffer((__gm__ T *)clipValueMin, 1);
            minScalar_ = minGm_.GetValue(0);
        } else {
            minGm_.SetGlobalBuffer(
                (__gm__ T *)clipValueMin + startOffset_, coreLength_);
        }

        if (maxIsScalar_) {
            maxGm_.SetGlobalBuffer((__gm__ T *)clipValueMax, 1);
            maxScalar_ = maxGm_.GetValue(0);
        } else {
            maxGm_.SetGlobalBuffer(
                (__gm__ T *)clipValueMax + startOffset_, coreLength_);
        }

        pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        if (!minIsScalar_) {
            pipe_.InitBuffer(minQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        }
        if (!maxIsScalar_) {
            pipe_.InitBuffer(maxQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if (coreLength_ == 0) {
            return;
        }

        for (uint64_t offset = 0; offset < coreLength_; offset += tileLength_) {
            const uint64_t remain = coreLength_ - offset;
            processLength_ = static_cast<uint32_t>(
                remain < tileLength_ ? remain : tileLength_);
            CopyIn(offset);
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyGmToLocal(
        const AscendC::LocalTensor<T> &local,
        const AscendC::GlobalTensor<T> &global,
        uint64_t offset)
    {
        if ((processLength_ % alignElems_) == 0) {
            AscendC::DataCopy(local, global[offset], processLength_);
        } else {
            AscendC::DataCopyExtParams params{
                1,
                static_cast<uint32_t>(processLength_ * sizeof(T)),
                0,
                0,
                0
            };
            AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
            AscendC::DataCopyPad(local, global[offset], params, pad);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset)
    {
        AscendC::LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        CopyGmToLocal(xLocal, xGm_, offset);
        xQueue_.EnQue(xLocal);

        if (!minIsScalar_) {
            AscendC::LocalTensor<T> minLocal = minQueue_.AllocTensor<T>();
            CopyGmToLocal(minLocal, minGm_, offset);
            minQueue_.EnQue(minLocal);
        }

        if (!maxIsScalar_) {
            AscendC::LocalTensor<T> maxLocal = maxQueue_.AllocTensor<T>();
            CopyGmToLocal(maxLocal, maxGm_, offset);
            maxQueue_.EnQue(maxLocal);
        }
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> xLocal = xQueue_.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();

        if (minIsScalar_) {
            AscendC::Maxs(yLocal, xLocal, minScalar_, processLength_);
        } else {
            AscendC::LocalTensor<T> minLocal = minQueue_.DeQue<T>();
            AscendC::Max(yLocal, xLocal, minLocal, processLength_);
            minQueue_.FreeTensor(minLocal);
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (maxIsScalar_) {
            AscendC::Mins(yLocal, yLocal, maxScalar_, processLength_);
        } else {
            AscendC::LocalTensor<T> maxLocal = maxQueue_.DeQue<T>();
            AscendC::Min(yLocal, yLocal, maxLocal, processLength_);
            maxQueue_.FreeTensor(maxLocal);
        }

        yQueue_.EnQue(yLocal);
        xQueue_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset)
    {
        AscendC::LocalTensor<T> yLocal = yQueue_.DeQue<T>();
        if ((processLength_ % alignElems_) == 0) {
            AscendC::DataCopy(yGm_[offset], yLocal, processLength_);
        } else {
            AscendC::DataCopyExtParams params{
                1,
                static_cast<uint32_t>(processLength_ * sizeof(T)),
                0,
                0,
                0
            };
            AscendC::DataCopyPad(yGm_[offset], yLocal, params);
        }
        yQueue_.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> xQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> minQueue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> maxQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue_;

    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> minGm_;
    AscendC::GlobalTensor<T> maxGm_;
    AscendC::GlobalTensor<T> yGm_;

    uint64_t startOffset_ = 0;
    uint64_t coreLength_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t processLength_ = 0;
    uint32_t alignElems_ = 0;
    bool minIsScalar_ = true;
    bool maxIsScalar_ = true;
    T minScalar_{};
    T maxScalar_{};
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(
    GM_ADDR x,
    GM_ADDR clipValueMin,
    GM_ADDR clipValueMax,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tilingData, tiling);

    KernelClipByValue<DT_X> op;
    op.Init(x, clipValueMin, clipValueMax, y, tilingData);
    op.Process();
}
