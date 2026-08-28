// Kernel侧核函数实现
#include "kernel_operator.h"
#include "clip_by_value_tiling.h"

// TilingKey 模板特化声明
#include "tiling_key_clip_by_value.h"

using namespace AscendC;

// SCALAR_MODE 编码已废弃，改为运行时从 tiling 数据获取 minIsScalar / maxIsScalar
template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max,
                                 GM_ADDR y, const ClipByValueTilingData &td) {
        totalLength_ = td.totalLength;
        blockLength_ = td.blockLength;
        tileLength_  = td.tileLength;
        bufferDepth_ = td.bufferDepth;

        // 从 tiling 数据中读取标量/张量标志
        minIsScalar_ = td.minIsScalar;
        maxIsScalar_ = td.maxIsScalar;

        // 计算当前核的偏移和工作量
        uint32_t blockId = GetBlockIdx();
        coreOffset_ = blockId * blockLength_;
        if (coreOffset_ >= totalLength_) {
            coreLength_ = 0;
            return;
        }
        coreLength_ = totalLength_ - coreOffset_;
        if (coreLength_ > blockLength_) {
            coreLength_ = blockLength_;
        }

        // 设置 GM 张量
        gmX_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset_);
        gmY_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset_);
        gmMin_.SetGlobalBuffer((__gm__ DT_X *)clip_value_min);
        gmMax_.SetGlobalBuffer((__gm__ DT_X *)clip_value_max);

        // 初始化队列
        if (bufferDepth_ == 0) {
            // 直通模式：TBuf 替代 TQue，省掉 EnQue/DeQue 开销
            pipe.InitBuffer(bufPassX_, tileLength_ * sizeof(DT_X));
            pipe.InitBuffer(bufPassY_, tileLength_ * sizeof(DT_X));
            if (!minIsScalar_) {
                pipe.InitBuffer(bufPassMin_, tileLength_ * sizeof(DT_X));
            }
            if (!maxIsScalar_) {
                pipe.InitBuffer(bufPassMax_, tileLength_ * sizeof(DT_X));
            }
            // 标量 buffer（直通路径）
            if (minIsScalar_) {
                pipe.InitBuffer(bufScalarMin_, 32);
            }
            if (maxIsScalar_) {
                pipe.InitBuffer(bufScalarMax_, 32);
            }
            return;
        }

        // 常规流水线：双缓冲
        uint32_t bufNum = 2;
        pipe.InitBuffer(inQueueX_, bufNum, tileLength_ * sizeof(DT_X));
        pipe.InitBuffer(outQueueY_, bufNum, tileLength_ * sizeof(DT_X));
        if (!minIsScalar_) {
            pipe.InitBuffer(inQueueMin_, bufNum, tileLength_ * sizeof(DT_X));
        }
        if (!maxIsScalar_) {
            pipe.InitBuffer(inQueueMax_, bufNum, tileLength_ * sizeof(DT_X));
        }
        // 标量 buffer（独立 buffer，避免 barrier 冲突）
        if (minIsScalar_) {
            pipe.InitBuffer(bufScalarMin_, 32);
        }
        if (maxIsScalar_) {
            pipe.InitBuffer(bufScalarMax_, 32);
        }
    }

    __aicore__ inline void LoadScalarsAsync() {
        DataCopyExtParams scalarCopyParams{1, sizeof(DT_X), 0, 0, 0};
        DataCopyPadExtParams<DT_X> scalarPadParams{false, 0, 0, static_cast<DT_X>(0)};
        if (minIsScalar_) {
            DataCopyPad(bufScalarMin_.Get<DT_X>(), gmMin_, scalarCopyParams, scalarPadParams);
        }
        if (maxIsScalar_) {
            DataCopyPad(bufScalarMax_.Get<DT_X>(), gmMax_, scalarCopyParams, scalarPadParams);
        }
    }

    __aicore__ inline void SyncScalars() {
        if (minIsScalar_) {
            scalarMin_ = bufScalarMin_.Get<DT_X>().GetValue(0);
        }
        if (maxIsScalar_) {
            scalarMax_ = bufScalarMax_.Get<DT_X>().GetValue(0);
        }
    }

    __aicore__ inline void Process() __attribute__((always_inline)) {
        if (coreLength_ == 0) return;

        // 直通路径
        if (bufferDepth_ == 0) {
            ProcessPassThrough();
            return;
        }

        uint32_t fullTiles = coreLength_ / tileLength_;
        uint32_t tailLength = coreLength_ - fullTiles * tileLength_;
        uint32_t totalTiles = fullTiles + (tailLength > 0 ? 1 : 0);
        uint32_t lastTileLen = (tailLength > 0) ? tailLength : tileLength_;

        if (totalTiles == 1) {
            CopyInTile(0, lastTileLen);
            LoadScalarsAsync();
            pipe_barrier(PIPE_MTE2);
            SyncScalars();
            ComputeTile(0, lastTileLen);
            CopyOutTile(0, lastTileLen);
            return;
        }

        if (totalTiles == 2) {
            CopyInTile(0, tileLength_);
            CopyInTile(1, lastTileLen);
            LoadScalarsAsync();
            pipe_barrier(PIPE_MTE2);
            SyncScalars();
            ComputeTile(0, tileLength_);
            CopyOutTile(0, tileLength_);
            ComputeTile(1, lastTileLen);
            CopyOutTile(1, lastTileLen);
            return;
        }

        // 统一双缓冲流水线
        CopyInTile(0, tileLength_);
        LoadScalarsAsync();
        pipe_barrier(PIPE_MTE2);
        SyncScalars();
        ComputeTile(0, tileLength_);
        CopyInTile(1, tileLength_);

        for (uint32_t i = 1; i < totalTiles - 1; i++) {
            uint32_t nextLen = (i + 1 == totalTiles - 1) ? lastTileLen : tileLength_;
            CopyOutTile(i - 1, tileLength_);
            ComputeTile(i, tileLength_);
            CopyInTile(i + 1, nextLen);
        }

        CopyOutTile(totalTiles - 2, tileLength_);
        ComputeTile(totalTiles - 1, lastTileLen);
        CopyOutTile(totalTiles - 1, lastTileLen);
    }

    // 直通路径
    __aicore__ inline void ProcessPassThrough() __attribute__((always_inline)) {
        LocalTensor<DT_X> xLocal = bufPassX_.Get<DT_X>();
        LocalTensor<DT_X> yLocal = bufPassY_.Get<DT_X>();
        uint32_t len = coreLength_;

        DataCopyExtParams params{1, (uint32_t)(len * sizeof(DT_X)), 0, 0, 0};
        DataCopyExtParams scalarParams{1, sizeof(DT_X), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        DataCopyPad(xLocal, gmX_[0], params, padParams);

        if (!minIsScalar_) {
            LocalTensor<DT_X> minLocal = bufPassMin_.Get<DT_X>();
            DataCopyPad(minLocal, gmMin_[coreOffset_], params, padParams);
        } else {
            DataCopyPad(bufScalarMin_.Get<DT_X>(), gmMin_, scalarParams, padParams);
        }

        if (!maxIsScalar_) {
            LocalTensor<DT_X> maxLocal = bufPassMax_.Get<DT_X>();
            DataCopyPad(maxLocal, gmMax_[coreOffset_], params, padParams);
        } else {
            DataCopyPad(bufScalarMax_.Get<DT_X>(), gmMax_, scalarParams, padParams);
        }

        pipe_barrier(PIPE_MTE2);

        if (minIsScalar_) scalarMin_ = bufScalarMin_.Get<DT_X>().GetValue(0);
        if (maxIsScalar_) scalarMax_ = bufScalarMax_.Get<DT_X>().GetValue(0);

        // y = max(min(x, max), min)
        if (maxIsScalar_) {
            Mins(yLocal, xLocal, scalarMax_, (int32_t)len);
        } else {
            LocalTensor<DT_X> maxLocal = bufPassMax_.Get<DT_X>();
            Min(yLocal, xLocal, maxLocal, (int32_t)len);
        }
        if (minIsScalar_) {
            Maxs(yLocal, yLocal, scalarMin_, (int32_t)len);
        } else {
            LocalTensor<DT_X> minLocal = bufPassMin_.Get<DT_X>();
            Max(yLocal, yLocal, minLocal, (int32_t)len);
        }

        DataCopyPad(gmY_[0], yLocal, params);
        pipe_barrier(PIPE_MTE3);
    }

private:
    __aicore__ inline void CopyInTile(uint32_t tileIdx, uint32_t length) {
        uint32_t offset = tileIdx * tileLength_;
        bool aligned = (length * sizeof(DT_X)) % 32 == 0;
        DataCopyExtParams copyParams{1, (uint32_t)(length * sizeof(DT_X)), 0, 0, 0};
        DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};

        LocalTensor<DT_X> xLocal = inQueueX_.AllocTensor<DT_X>();
        if (aligned) DataCopy(xLocal, gmX_[offset], length);
        else DataCopyPad(xLocal, gmX_[offset], copyParams, padParams);
        inQueueX_.EnQue(xLocal);

        if (!minIsScalar_) {
            LocalTensor<DT_X> minLocal = inQueueMin_.AllocTensor<DT_X>();
            if (aligned) DataCopy(minLocal, gmMin_[coreOffset_ + offset], length);
            else DataCopyPad(minLocal, gmMin_[coreOffset_ + offset], copyParams, padParams);
            inQueueMin_.EnQue(minLocal);
        }
        if (!maxIsScalar_) {
            LocalTensor<DT_X> maxLocal = inQueueMax_.AllocTensor<DT_X>();
            if (aligned) DataCopy(maxLocal, gmMax_[coreOffset_ + offset], length);
            else DataCopyPad(maxLocal, gmMax_[coreOffset_ + offset], copyParams, padParams);
            inQueueMax_.EnQue(maxLocal);
        }
    }

    __aicore__ inline void ComputeTile(uint32_t tileIdx, uint32_t length) {
        LocalTensor<DT_X> xLocal = inQueueX_.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY_.AllocTensor<DT_X>();

        // step 1: y = min(x, max)
        if (maxIsScalar_) {
            Mins(yLocal, xLocal, scalarMax_, (int32_t)length);
        } else {
            LocalTensor<DT_X> maxLocal = inQueueMax_.DeQue<DT_X>();
            Min(yLocal, xLocal, maxLocal, (int32_t)length);
            inQueueMax_.FreeTensor(maxLocal);
        }

        // step 2: y = max(y, min)
        if (minIsScalar_) {
            Maxs(yLocal, yLocal, scalarMin_, (int32_t)length);
        } else {
            LocalTensor<DT_X> minLocal = inQueueMin_.DeQue<DT_X>();
            Max(yLocal, yLocal, minLocal, (int32_t)length);
            inQueueMin_.FreeTensor(minLocal);
        }

        outQueueY_.EnQue<DT_X>(yLocal);
        inQueueX_.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOutTile(uint32_t tileIdx, uint32_t length) {
        uint32_t offset = tileIdx * tileLength_;
        bool aligned = (length * sizeof(DT_X)) % 32 == 0;
        DataCopyExtParams copyParams{1, (uint32_t)(length * sizeof(DT_X)), 0, 0, 0};

        LocalTensor<DT_X> yLocal = outQueueY_.DeQue<DT_X>();
        if (aligned) DataCopy(gmY_[offset], yLocal, length);
        else DataCopyPad(gmY_[offset], yLocal, copyParams);
        outQueueY_.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 3> inQueueX_;
    TQue<TPosition::VECOUT, 3> outQueueY_;
    TQue<TPosition::VECIN, 3> inQueueMin_;
    TQue<TPosition::VECIN, 3> inQueueMax_;
    GlobalTensor<DT_X> gmX_;
    GlobalTensor<DT_X> gmY_;
    GlobalTensor<DT_X> gmMin_;
    GlobalTensor<DT_X> gmMax_;
    uint32_t totalLength_;
    uint32_t blockLength_;
    uint32_t tileLength_;
    uint32_t coreOffset_;
    uint32_t coreLength_;
    uint32_t bufferDepth_;
    DT_X scalarMin_;
    DT_X scalarMax_;

    // 运行时标量标志（替代模板参数 SCALAR_MODE）
    bool minIsScalar_;
    bool maxIsScalar_;

    // 直通路径 buffer
    TBuf<TPosition::VECCALC> bufPassX_;
    TBuf<TPosition::VECCALC> bufPassY_;
    TBuf<TPosition::VECCALC> bufPassMin_;
    TBuf<TPosition::VECCALC> bufPassMax_;
    TBuf<TPosition::VECCALC> bufScalarMin_;
    TBuf<TPosition::VECCALC> bufScalarMax_;
};

// kernel 入口函数：去掉 SCALAR_MODE 模板参数
template <typename DT_X>
__global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min,
                                         GM_ADDR clip_value_max, GM_ADDR y,
                                         GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    KernelClipByValue<DT_X> op;          // 只需数据类型
    op.Init(x, clip_value_min, clip_value_max, y, tiling_data);
    op.Process();
}