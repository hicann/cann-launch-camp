// Kernel侧核函数实现
#include "kernel_operator.h"
#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max,
                                GM_ADDR y, const ClipByValueTilingData &tiling_data) {
        tiling = tiling_data;

        uint32_t coreIdx = GetBlockIdx();
        blockNum = (coreIdx < tiling.remBlocks) ? (tiling.avgBlocksPerCore + 1) : tiling.avgBlocksPerCore;
        startBlockIdx = coreIdx * tiling.avgBlocksPerCore +
                        (coreIdx < tiling.remBlocks ? coreIdx : tiling.remBlocks);

        lastBlockLength = 0;
        if (blockNum > 0) {
            uint32_t lastBlockIdx = startBlockIdx + blockNum - 1;
            uint32_t lastBlockEnd = (lastBlockIdx + 1) * tiling.blockSize;
            if (lastBlockEnd > tiling.totalLength) {
                lastBlockLength = tiling.totalLength - lastBlockIdx * tiling.blockSize;
            } else {
                lastBlockLength = tiling.blockSize;
            }
        }

        // 小数据场景(blockNum==1)用BUFFER_NUM=1节省UB，大数据场景用BUFFER_NUM=3做流水
        uint32_t bufNum = (blockNum <= 1) ? 1 : 3;

        uint32_t gmLen = tiling.totalLength;
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, gmLen);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, gmLen);
        if (tiling.minIsScalar) {
            minGm.SetGlobalBuffer((__gm__ DT_X *)clip_value_min, 1);
            minScalarVal = minGm.GetValue(0);
        } else {
            minGm.SetGlobalBuffer((__gm__ DT_X *)clip_value_min, gmLen);
        }
        if (tiling.maxIsScalar) {
            maxGm.SetGlobalBuffer((__gm__ DT_X *)clip_value_max, 1);
            maxScalarVal = maxGm.GetValue(0);
        } else {
            maxGm.SetGlobalBuffer((__gm__ DT_X *)clip_value_max, gmLen);
        }

        uint32_t blockBytes = tiling.blockSize * sizeof(DT_X);
        pipe.InitBuffer(inQueueX, bufNum, blockBytes);
        pipe.InitBuffer(outQueueY, bufNum, blockBytes);
        if (!tiling.minIsScalar) {
            pipe.InitBuffer(inQueueMin, bufNum, blockBytes);
        }
        if (!tiling.maxIsScalar) {
            pipe.InitBuffer(inQueueMax, bufNum, blockBytes);
        }
    }

    __aicore__ inline void Process() {
        if (blockNum == 0 || lastBlockLength == 0) {
            return;
        }

        // 所有块统一走三阶段流水（MTE 与 Vector 并行）
        CopyIn(0);
        for (uint32_t bi = 0; bi < blockNum - 1; bi++) {
            CopyIn(bi + 1);
            Compute(bi);
            CopyOut(bi);
        }
        Compute(blockNum - 1);
        CopyOut(blockNum - 1);
    }

private:
    ClipByValueTilingData tiling;

    uint32_t blockNum;
    uint32_t startBlockIdx;
    uint32_t lastBlockLength;

    DT_X minScalarVal;
    DT_X maxScalarVal;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> minGm;
    GlobalTensor<DT_X> maxGm;
    GlobalTensor<DT_X> yGm;

    TPipe pipe;
    static constexpr uint32_t MAX_BUF_NUM = 3;
    TQue<QuePosition::VECIN, MAX_BUF_NUM> inQueueX;
    TQue<QuePosition::VECIN, MAX_BUF_NUM> inQueueMin;
    TQue<QuePosition::VECIN, MAX_BUF_NUM> inQueueMax;
    TQue<QuePosition::VECOUT, MAX_BUF_NUM> outQueueY;

    // 第bi块的有效元素数
    __aicore__ inline uint32_t GetValidLen(uint32_t bi) {
        bool isLast = (bi == blockNum - 1);
        return isLast ? lastBlockLength : tiling.blockSize;
    }

    // 第bi块的对齐元素数（向上对齐到alignNum，用于搬入和计算）
    __aicore__ inline uint32_t GetPaddedLen(uint32_t bi) {
        uint32_t valid = GetValidLen(bi);
        return ((valid + tiling.alignNum - 1) / tiling.alignNum) * tiling.alignNum;
    }

    // 第bi块的GM偏移
    __aicore__ inline uint32_t GetGmOffset(uint32_t bi) {
        return (startBlockIdx + bi) * tiling.blockSize;
    }

    // 流水阶段1：搬入数据到UB
    __aicore__ inline void CopyIn(uint32_t bi) {
        uint32_t len = GetPaddedLen(bi);
        if (len == 0) return;

        uint32_t gmOffset = GetGmOffset(bi);

        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[gmOffset], len);
        inQueueX.EnQue(xLocal);

        if (!tiling.minIsScalar) {
            LocalTensor<DT_X> minLocal = inQueueMin.AllocTensor<DT_X>();
            DataCopy(minLocal, minGm[gmOffset], len);
            inQueueMin.EnQue(minLocal);
        }
        if (!tiling.maxIsScalar) {
            LocalTensor<DT_X> maxLocal = inQueueMax.AllocTensor<DT_X>();
            DataCopy(maxLocal, maxGm[gmOffset], len);
            inQueueMax.EnQue(maxLocal);
        }
    }

    // 流水阶段2：矢量计算
    __aicore__ inline void Compute(uint32_t bi) {
        uint32_t len = GetPaddedLen(bi);
        if (len == 0) return;

        int32_t calCount = static_cast<int32_t>(len);

        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        if (tiling.minIsScalar && tiling.maxIsScalar) {
            Maxs(yLocal, xLocal, minScalarVal, calCount);
            Mins(yLocal, yLocal, maxScalarVal, calCount);
        } else if (tiling.minIsScalar && !tiling.maxIsScalar) {
            LocalTensor<DT_X> maxLocal = inQueueMax.DeQue<DT_X>();
            Maxs(yLocal, xLocal, minScalarVal, calCount);
            Min(yLocal, yLocal, maxLocal, calCount);
            inQueueMax.FreeTensor(maxLocal);
        } else if (!tiling.minIsScalar && tiling.maxIsScalar) {
            LocalTensor<DT_X> minLocal = inQueueMin.DeQue<DT_X>();
            Max(yLocal, xLocal, minLocal, calCount);
            Mins(yLocal, yLocal, maxScalarVal, calCount);
            inQueueMin.FreeTensor(minLocal);
        } else {
            LocalTensor<DT_X> minLocal = inQueueMin.DeQue<DT_X>();
            LocalTensor<DT_X> maxLocal = inQueueMax.DeQue<DT_X>();
            Max(yLocal, xLocal, minLocal, calCount);
            Min(yLocal, yLocal, maxLocal, calCount);
            inQueueMin.FreeTensor(minLocal);
            inQueueMax.FreeTensor(maxLocal);
        }

        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
    }

    // 流水阶段3：搬出结果到GM
    __aicore__ inline void CopyOut(uint32_t bi) {
        uint32_t validLen = GetValidLen(bi);
        uint32_t paddedLen = GetPaddedLen(bi);
        if (validLen == 0) return;

        uint32_t gmOffset = GetGmOffset(bi);
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if (validLen == paddedLen) {
            DataCopy(yGm[gmOffset], yLocal, validLen);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = validLen * sizeof(DT_X);
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(yGm[gmOffset], yLocal, outParams);
        }

        outQueueY.FreeTensor(yLocal);
    }
};

template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max,
                                          GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);

    KernelClipByValue<DT_X> op;
    op.Init(x, clip_value_min, clip_value_max, y, tiling_data);
    op.Process();
}
