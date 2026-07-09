// Kernel-side FastGelu implementation.
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_ELEM_NUM = 8192;

template <class DT_X>
class KernelFastGeluSmall {
public:
    __aicore__ inline KernelFastGeluSmall() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileDataNum) {
        totalLength_ = totalLength;
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x, totalLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y, totalLength_);

        pipe_.InitBuffer(xBuf_, tileDataNum * sizeof(DT_X));
        pipe_.InitBuffer(yBuf_, tileDataNum * sizeof(DT_X));
        pipe_.InitBuffer(sigmoidBuf_, tileDataNum * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (totalLength_ == 0) {
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = totalLength_ * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};

        AscendC::LocalTensor<DT_X> xLocal = xBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = yBuf_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> sigmoidLocal = sigmoidBuf_.Get<DT_X>();

        AscendC::DataCopyPad(xLocal, xGm_[0], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(sigmoidLocal, xLocal, static_cast<DT_X>(1.702f), totalLength_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(sigmoidLocal, sigmoidLocal, totalLength_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal, xLocal, sigmoidLocal, totalLength_);
        AscendC::DataCopyPad(yGm_[0], yLocal, copyParams);
    }

private:
    uint32_t totalLength_ = 0;

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigmoidBuf_;
};

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum) {
        totalLength_ = totalLength;
        tileDataNum_ = tileDataNum;

        uint32_t core_index = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * core_index;
        if (core_index < tailBlockNum) {
            coreDataNum_ = bigCoreDataNum;
            tileNum_ = finalBigTileNum;
            tailDataNum_ = bigTailDataNum;
        } else {
            coreDataNum_ = smallCoreDataNum;
            tileNum_ = finalSmallTileNum;
            tailDataNum_ = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (core_index - tailBlockNum);
        }

        coreOffset_ = globalBufferIndex;
        realCoreDataNum_ = totalLength_ > coreOffset_ ? totalLength_ - coreOffset_ : 0;
        if (realCoreDataNum_ > coreDataNum_) {
            realCoreDataNum_ = coreDataNum_;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset_, realCoreDataNum_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset_, realCoreDataNum_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileDataNum_ * sizeof(DT_X));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, tileDataNum_ * sizeof(DT_X));
        pipe_.InitBuffer(sigmoidBuf_, tileDataNum_ * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (totalLength_ == 0 || realCoreDataNum_ == 0 || tileNum_ == 0) {
            return;
        }

        for (uint32_t i = 0; i < tileNum_; ++i) {
            uint32_t offset = i * tileDataNum_;
            if (offset >= realCoreDataNum_) {
                return;
            }

            uint32_t processDataNum = (i == tileNum_ - 1) ? tailDataNum_ : tileDataNum_;
            uint32_t remainingDataNum = realCoreDataNum_ - offset;
            if (processDataNum > remainingDataNum) {
                processDataNum = remainingDataNum;
            }
            ProcessTile(offset, processDataNum);
        }
    }
private:
    __aicore__ inline void ProcessTile(uint32_t offset, uint32_t count) {
        CopyIn(offset, count);
        Compute(count);
        CopyOut(offset, count);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};

        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        AscendC::DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> sigmoidLocal = sigmoidBuf_.Get<DT_X>();

        AscendC::Muls(sigmoidLocal, xLocal, static_cast<DT_X>(1.702f), count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid(sigmoidLocal, sigmoidLocal, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal, xLocal, sigmoidLocal, count);

        outQueueY_.EnQue(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(DT_X);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        AscendC::LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        AscendC::DataCopyPad(yGm_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    uint32_t totalLength_ = 0;
    uint32_t coreDataNum_ = 0;
    uint32_t tileNum_ = 0;
    uint32_t tileDataNum_ = TILE_ELEM_NUM;
    uint32_t tailDataNum_ = 0;
    uint32_t coreOffset_ = 0;
    uint32_t realCoreDataNum_ = 0;

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigmoidBuf_;
};

template <typename DT_X, int IS_SMALL_SHAPE>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    if (IS_SMALL_SHAPE == 1) {
        KernelFastGeluSmall<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.tileDataNum);
        op.Process();
        return;
    }

    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, tiling_data.tailBlockNum);
    op.Process();
}
