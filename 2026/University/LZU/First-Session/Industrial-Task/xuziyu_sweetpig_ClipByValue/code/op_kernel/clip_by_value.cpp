// Kernel侧核函数实现
#include "kernel_operator.h"

#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_BYTES = 32;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, ClipByValueTilingData tiling) {
        totalLength_  = tiling.length;
        blockLength_  = tiling.blockLength;
        tileLength_   = tiling.tileLength;
        isScalarMin_  = static_cast<bool>(tiling.isScalarMin);
        isScalarMax_  = static_cast<bool>(tiling.isScalarMax);

        elementsPerBlock_ = BLOCK_BYTES / sizeof(DT_X);

        uint32_t blockId = GetBlockIdx();
        offset_ = blockId * blockLength_;
        if (offset_ >= totalLength_) {
            myLength_ = 0;
            return;
        }
        myLength_ = blockLength_;
        if (offset_ + myLength_ > totalLength_) {
            myLength_ = totalLength_ - offset_;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X*)x + offset_, myLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X*)y + offset_, myLength_);

        if (isScalarMin_) {
            GlobalTensor<DT_X> scalarMinGm;
            scalarMinGm.SetGlobalBuffer((__gm__ DT_X*)clip_value_min);
            clipMinScalar_ = scalarMinGm.GetValue(0);
        } else {
            clipMinGm_.SetGlobalBuffer((__gm__ DT_X*)clip_value_min + offset_, myLength_);
        }

        if (isScalarMax_) {
            GlobalTensor<DT_X> scalarMaxGm;
            scalarMaxGm.SetGlobalBuffer((__gm__ DT_X*)clip_value_max);
            clipMaxScalar_ = scalarMaxGm.GetValue(0);
        } else {
            clipMaxGm_.SetGlobalBuffer((__gm__ DT_X*)clip_value_max + offset_, myLength_);
        }

        uint32_t alignTileLen = AlignUp(tileLength_, elementsPerBlock_);
        uint32_t tileSizeBytes = alignTileLen * sizeof(DT_X);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileSizeBytes);
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileSizeBytes);
        if (!isScalarMin_) {
            pipe_.InitBuffer(inQueueMin_, BUFFER_NUM, tileSizeBytes);
        }
        if (!isScalarMax_) {
            pipe_.InitBuffer(inQueueMax_, BUFFER_NUM, tileSizeBytes);
        }
    }

    __aicore__ inline void Process() {
        if (myLength_ == 0) {
            return;
        }
        uint32_t myTileNum = (myLength_ + tileLength_ - 1) / tileLength_;
        for (uint32_t i = 0; i < myTileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t num, uint32_t align) {
        return (num + align - 1) / align * align;
    }

    __aicore__ inline uint32_t GetActualTileSize(uint32_t tileIdx) {
        uint32_t start = tileIdx * tileLength_;
        uint32_t remaining = myLength_ - start;
        return (remaining < tileLength_) ? remaining : tileLength_;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx) {
        uint32_t actualSize = GetActualTileSize(tileIdx);
        uint32_t alignSize  = AlignUp(actualSize, elementsPerBlock_);
        uint32_t gmOffset   = tileIdx * tileLength_;

        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm_[gmOffset], alignSize);
        inQueueX_.EnQue<DT_X>(xLocal);

        if (!isScalarMin_) {
            LocalTensor<DT_X> minLocal = inQueueMin_.AllocTensor<DT_X>();
            DataCopy(minLocal, clipMinGm_[gmOffset], alignSize);
            inQueueMin_.EnQue<DT_X>(minLocal);
        }

        if (!isScalarMax_) {
            LocalTensor<DT_X> maxLocal = inQueueMax_.AllocTensor<DT_X>();
            DataCopy(maxLocal, clipMaxGm_[gmOffset], alignSize);
            inQueueMax_.EnQue<DT_X>(maxLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t tileIdx) {
        uint32_t actualSize = GetActualTileSize(tileIdx);
        uint32_t alignSize  = AlignUp(actualSize, elementsPerBlock_);

        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        // y = max(x, clip_value_min)
        if (isScalarMin_) {
            Maxs<DT_X>(yLocal, xLocal, clipMinScalar_, alignSize);
        } else {
            LocalTensor<DT_X> minLocal = inQueueMin_.DeQue<DT_X>();
            Max<DT_X>(yLocal, xLocal, minLocal, alignSize);
            inQueueMin_.FreeTensor<DT_X>(minLocal);
        }

        // y = min(y, clip_value_max)
        if (isScalarMax_) {
            Mins<DT_X>(yLocal, yLocal, clipMaxScalar_, alignSize);
        } else {
            LocalTensor<DT_X> maxLocal = inQueueMax_.DeQue<DT_X>();
            Min<DT_X>(yLocal, yLocal, maxLocal, alignSize);
            inQueueMax_.FreeTensor<DT_X>(maxLocal);
        }

        outQueueY_.EnQue<DT_X>(yLocal);
        inQueueX_.FreeTensor<DT_X>(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        uint32_t actualSize = GetActualTileSize(tileIdx);
        uint32_t alignSize  = AlignUp(actualSize, elementsPerBlock_);
        uint32_t gmOffset   = tileIdx * tileLength_;

        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        DataCopy(yGm_[gmOffset], yLocal, alignSize);
        outQueueY_.FreeTensor<DT_X>(yLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM>  inQueueX_;
    TQue<TPosition::VECIN, BUFFER_NUM>  inQueueMin_;
    TQue<TPosition::VECIN, BUFFER_NUM>  inQueueMax_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueY_;

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> clipMinGm_;
    GlobalTensor<DT_X> clipMaxGm_;
    GlobalTensor<DT_X> yGm_;

    DT_X clipMinScalar_{};
    DT_X clipMaxScalar_{};

    uint32_t totalLength_      = 0;
    uint32_t blockLength_      = 0;
    uint32_t tileLength_       = 0;
    uint32_t elementsPerBlock_ = 0;
    uint32_t offset_           = 0;
    uint32_t myLength_         = 0;
    bool     isScalarMin_      = false;
    bool     isScalarMax_      = false;
};

template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;
    op.Init(x, clip_value_min, clip_value_max, y, tiling_data);
    op.Process();
}
