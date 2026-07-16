// Kernel 侧：多核切分、非对齐搬运、分段广播与全向量比较。
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {
template <typename T>
struct QueueBufferCount {
    static constexpr uint32_t VALUE = 1U;
};

// float32 是测试点 2 的主要吞吐路径。只对该类型启用双缓冲，使 GM 搬运
// 能与 Vector 比较/布尔展开流水重叠；其余类型（尤其 int32）保持 v3 行为。
template <>
struct QueueBufferCount<float> {
    static constexpr uint32_t VALUE = 2U;
};

template <typename T>
struct CalcBufferCount {
    static constexpr uint32_t VALUE = 0U;
};

template <>
struct CalcBufferCount<int8_t> {
    static constexpr uint32_t VALUE = 2U;
};

__aicore__ inline uint32_t AlignUp32(uint32_t value) {
    return (value + 31U) / 32U * 32U;
}

__aicore__ inline void SelectMaskToBool(const AscendC::LocalTensor<int8_t> &yLocal,
                                        const AscendC::LocalTensor<uint8_t> &maskLocal,
                                        const AscendC::LocalTensor<half> &selectLocal,
                                        uint32_t count) {
    // Compare 后复用已不再读取的输入/计算 UB，不再单独申请选择缓冲。
    AscendC::Duplicate(selectLocal, static_cast<half>(1), count);
    AscendC::Select(selectLocal, maskLocal, selectLocal, static_cast<half>(0),
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(yLocal, selectLocal, AscendC::RoundMode::CAST_NONE, count);
}

template <typename T>
struct LessEqualCompute;

template <>
struct LessEqualCompute<half> {
    __aicore__ static inline void Run(const AscendC::LocalTensor<int8_t> &yLocal,
                                     const AscendC::LocalTensor<half> &x1Local,
                                     const AscendC::LocalTensor<half> &x2Local,
                                     const AscendC::LocalTensor<uint8_t> &maskLocal,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<half> &selectLocal,
                                     uint32_t count, uint32_t compareCount,
                                     bool x1Scalar, bool x2Scalar) {
        if (x1Scalar && !x2Scalar) {
            // scalar <= tensor 等价于 tensor >= scalar，避免先扩展整块 scalar。
            AscendC::CompareScalar(maskLocal, x2Local, x1Local.GetValue(0U),
                                   AscendC::CMPMODE::GE, compareCount);
        } else if (x2Scalar && !x1Scalar) {
            AscendC::CompareScalar(maskLocal, x1Local, x2Local.GetValue(0U),
                                   AscendC::CMPMODE::LE, compareCount);
        } else {
            if (x1Scalar) {
                AscendC::Duplicate(x1Local, x1Local.GetValue(0U), compareCount);
            }
            if (x2Scalar) {
                AscendC::Duplicate(x2Local, x2Local.GetValue(0U), compareCount);
            }
            if (x1Scalar || x2Scalar) {
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Compare(maskLocal, x1Local, x2Local, AscendC::CMPMODE::LE, compareCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SelectMaskToBool(yLocal, maskLocal, selectLocal, count);
    }
};

template <>
struct LessEqualCompute<float> {
    __aicore__ static inline void Run(const AscendC::LocalTensor<int8_t> &yLocal,
                                     const AscendC::LocalTensor<float> &x1Local,
                                     const AscendC::LocalTensor<float> &x2Local,
                                     const AscendC::LocalTensor<uint8_t> &maskLocal,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<half> &selectLocal,
                                     uint32_t count, uint32_t compareCount,
                                     bool x1Scalar, bool x2Scalar) {
        if (x1Scalar && !x2Scalar) {
            AscendC::CompareScalar(maskLocal, x2Local, x1Local.GetValue(0U),
                                   AscendC::CMPMODE::GE, compareCount);
        } else if (x2Scalar && !x1Scalar) {
            AscendC::CompareScalar(maskLocal, x1Local, x2Local.GetValue(0U),
                                   AscendC::CMPMODE::LE, compareCount);
        } else {
            if (x1Scalar) {
                AscendC::Duplicate(x1Local, x1Local.GetValue(0U), compareCount);
            }
            if (x2Scalar) {
                AscendC::Duplicate(x2Local, x2Local.GetValue(0U), compareCount);
            }
            if (x1Scalar || x2Scalar) {
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Compare(maskLocal, x1Local, x2Local, AscendC::CMPMODE::LE, compareCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SelectMaskToBool(yLocal, maskLocal, selectLocal, count);
    }
};

template <>
struct LessEqualCompute<int32_t> {
    __aicore__ static inline void Run(const AscendC::LocalTensor<int8_t> &yLocal,
                                     const AscendC::LocalTensor<int32_t> &x1Local,
                                     const AscendC::LocalTensor<int32_t> &x2Local,
                                     const AscendC::LocalTensor<uint8_t> &maskLocal,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<uint8_t> &,
                                     const AscendC::LocalTensor<half> &selectLocal,
                                     uint32_t count, uint32_t compareCount,
                                     bool x1Scalar, bool x2Scalar) {
        if (x1Scalar) {
            AscendC::Duplicate(x1Local, x1Local.GetValue(0U), compareCount);
        }
        if (x2Scalar) {
            AscendC::Duplicate(x2Local, x2Local.GetValue(0U), compareCount);
        }
        if (x1Scalar || x2Scalar) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        // A2 的 Compare 对 int32 只支持 EQ。min(x1,x2)==x1 与 x1<=x2 等价，
        // 且全程保持 int32，对 INT_MIN/INT_MAX 也不丢精度。
        // Min 允许目的与源操作数完全重叠；原地写入 x2，省去一块 int32 UB。
        AscendC::Min(x2Local, x1Local, x2Local, compareCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Compare(maskLocal, x2Local, x1Local, AscendC::CMPMODE::EQ, compareCount);
        AscendC::PipeBarrier<PIPE_V>();
        SelectMaskToBool(yLocal, maskLocal, selectLocal, count);
    }
};

template <>
struct LessEqualCompute<int8_t> {
    __aicore__ static inline void Run(const AscendC::LocalTensor<int8_t> &yLocal,
                                     const AscendC::LocalTensor<int8_t> &x1Local,
                                     const AscendC::LocalTensor<int8_t> &x2Local,
                                     const AscendC::LocalTensor<uint8_t> &maskLocal,
                                     const AscendC::LocalTensor<uint8_t> &calc1Raw,
                                     const AscendC::LocalTensor<uint8_t> &calc2Raw,
                                     const AscendC::LocalTensor<half> &selectLocal,
                                     uint32_t count, uint32_t compareCount,
                                     bool x1Scalar, bool x2Scalar) {
        const AscendC::LocalTensor<half> x1Half = calc1Raw.ReinterpretCast<half>();
        const AscendC::LocalTensor<half> x2Half = calc2Raw.ReinterpretCast<half>();
        if (x1Scalar && !x2Scalar) {
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(maskLocal, x2Half,
                                   static_cast<half>(x1Local.GetValue(0U)),
                                   AscendC::CMPMODE::GE, compareCount);
        } else if (x2Scalar && !x1Scalar) {
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(maskLocal, x1Half,
                                   static_cast<half>(x2Local.GetValue(0U)),
                                   AscendC::CMPMODE::LE, compareCount);
        } else {
            if (x1Scalar) {
                AscendC::Duplicate(x1Half, static_cast<half>(x1Local.GetValue(0U)), compareCount);
            } else {
                AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, count);
            }
            if (x2Scalar) {
                AscendC::Duplicate(x2Half, static_cast<half>(x2Local.GetValue(0U)), compareCount);
            } else {
                AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            // int8 -> half 为完全精确转换，可直接使用 half 的 LE 向量比较。
            AscendC::Compare(maskLocal, x1Half, x2Half, AscendC::CMPMODE::LE, compareCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        SelectMaskToBool(yLocal, maskLocal, selectLocal, count);
    }
};
}  // namespace

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData *tilingData,
                                AscendC::TPipe *pipe) {
        pipe_ = pipe;
        x1Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);

        totalLength_ = tilingData->totalLength;
        blockLength_ = tilingData->blockLength;
        rank_ = tilingData->rank;
        tileLength_ = tilingData->tileLength;
        noBroadcast_ = tilingData->noBroadcast;
        segmentLength_ = tilingData->segmentLength;
        x1SegmentScalar_ = tilingData->x1SegmentScalar;
        x2SegmentScalar_ = tilingData->x2SegmentScalar;
        segmentOuterRank_ = tilingData->segmentOuterRank;
        for (uint32_t i = 0U; i < rank_; ++i) {
            outputShape_[i] = tilingData->outputShape[i];
            x1Strides_[i] = tilingData->x1Strides[i];
            x2Strides_[i] = tilingData->x2Strides[i];
        }

        const uint32_t inputBufferBytes = AlignUp32(tileLength_ * sizeof(DT_X1));
        const uint32_t outputBufferBytes = AlignUp32(tileLength_ * sizeof(int8_t));
        const uint32_t maskBufferBytes = AlignUp32((tileLength_ + 7U) / 8U);
        const uint32_t calcTypeBytes = sizeof(DT_X1) > sizeof(half) ? sizeof(DT_X1) : sizeof(half);
        const uint32_t calcBufferBytes = AlignUp32(tileLength_ * calcTypeBytes);
        pipe_->InitBuffer(x1Queue_, QueueBufferCount<DT_X1>::VALUE, inputBufferBytes);
        pipe_->InitBuffer(x2Queue_, QueueBufferCount<DT_X1>::VALUE, inputBufferBytes);
        pipe_->InitBuffer(yQueue_, QueueBufferCount<DT_X1>::VALUE, outputBufferBytes);
        pipe_->InitBuffer(maskBuffer_, maskBufferBytes);
        if (CalcBufferCount<DT_X1>::VALUE >= 1U) {
            pipe_->InitBuffer(calc1Buffer_, calcBufferBytes);
        }
        if (CalcBufferCount<DT_X1>::VALUE >= 2U) {
            pipe_->InitBuffer(calc2Buffer_, calcBufferBytes);
        }
    }

    __aicore__ inline void Process() {
        if (totalLength_ == 0U || blockLength_ == 0U) {
            return;
        }
        const uint64_t coreStart = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockLength_;
        if (coreStart >= totalLength_) {
            return;
        }
        const uint64_t remain = totalLength_ - coreStart;
        const uint64_t coreLength = remain < blockLength_ ? remain : blockLength_;

        uint64_t processed = 0U;
        while (processed < coreLength) {
            const uint64_t tileRemain = coreLength - processed;
            uint32_t count = static_cast<uint32_t>(
                tileRemain < static_cast<uint64_t>(tileLength_) ? tileRemain : tileLength_);
            const uint64_t globalOffset = coreStart + processed;
            uint64_t segmentIndex = 0U;
            uint64_t innerOffset = 0U;
            if (noBroadcast_ == 0U) {
                segmentIndex = globalOffset / segmentLength_;
                innerOffset = globalOffset - segmentIndex * segmentLength_;
                const uint64_t segmentRemain = segmentLength_ - innerOffset;
                if (segmentRemain < static_cast<uint64_t>(count)) {
                    count = static_cast<uint32_t>(segmentRemain);
                }
            }
            CopyIn(globalOffset, segmentIndex, innerOffset, count);
            Compute(count);
            CopyOut(globalOffset, count);
            processed += count;
        }
    }

private:
    __aicore__ inline uint32_t GetCompareCount(uint32_t count) const {
        const uint32_t compareTypeBytes = sizeof(DT_X1) == sizeof(int8_t) ? sizeof(half)
                                                                          : sizeof(DT_X1);
        const uint32_t elementsPerVector = 256U / compareTypeBytes;
        return (count + elementsPerVector - 1U) / elementsPerVector * elementsPerVector;
    }

    __aicore__ inline void GetBroadcastOffsets(uint64_t segmentIndex,
                                               uint64_t innerOffset,
                                               uint64_t &offsetX1,
                                               uint64_t &offsetX2) const {
        offsetX1 = 0U;
        offsetX2 = 0U;
        uint64_t remainder = segmentIndex;
        for (int32_t dim = static_cast<int32_t>(segmentOuterRank_) - 1; dim >= 0; --dim) {
            const uint64_t coordinate = remainder % outputShape_[dim];
            remainder /= outputShape_[dim];
            offsetX1 += coordinate * x1Strides_[dim];
            offsetX2 += coordinate * x2Strides_[dim];
        }
        if (x1SegmentScalar_ == 0U) {
            offsetX1 += innerOffset;
        }
        if (x2SegmentScalar_ == 0U) {
            offsetX2 += innerOffset;
        }
    }

    __aicore__ inline void CopyIn(uint64_t globalOffset,
                                  uint64_t segmentIndex,
                                  uint64_t innerOffset,
                                  uint32_t count) {
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue_.template AllocTensor<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue_.template AllocTensor<DT_X1>();

        if (noBroadcast_ != 0U) {
            CopyGmToLocal(x1Local, x1Gm_[globalOffset], count);
            CopyGmToLocal(x2Local, x2Gm_[globalOffset], count);
        } else {
            uint64_t offsetX1 = 0U;
            uint64_t offsetX2 = 0U;
            GetBroadcastOffsets(segmentIndex, innerOffset, offsetX1, offsetX2);
            if (x1SegmentScalar_ != 0U) {
                x1Local.SetValue(0U, x1Gm_.GetValue(offsetX1));
            } else {
                CopyGmToLocal(x1Local, x1Gm_[offsetX1], count);
            }
            if (x2SegmentScalar_ != 0U) {
                x2Local.SetValue(0U, x2Gm_.GetValue(offsetX2));
            } else {
                CopyGmToLocal(x2Local, x2Gm_[offsetX2], count);
            }
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t count) {
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue_.template DeQue<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue_.template DeQue<DT_X1>();
        AscendC::LocalTensor<int8_t> yLocal = yQueue_.template AllocTensor<int8_t>();
        AscendC::LocalTensor<uint8_t> maskLocal = maskBuffer_.Get<uint8_t>();
        const bool x1Scalar = noBroadcast_ == 0U && x1SegmentScalar_ != 0U;
        const bool x2Scalar = noBroadcast_ == 0U && x2SegmentScalar_ != 0U;
        // half/float 不需要计算缓冲，int32 只需要一个，int8 需要两个。
        // 未使用的入参以 maskLocal 作占位，避免无效的 InitBuffer/Get 开销。
        AscendC::LocalTensor<uint8_t> calc1Raw = maskLocal;
        AscendC::LocalTensor<uint8_t> calc2Raw = maskLocal;
        if (CalcBufferCount<DT_X1>::VALUE >= 1U) {
            calc1Raw = calc1Buffer_.Get<uint8_t>();
        }
        if (CalcBufferCount<DT_X1>::VALUE >= 2U) {
            calc2Raw = calc2Buffer_.Get<uint8_t>();
        }
        // half/float/int32 的输入缓冲均足以复用为 half 临时结果；int8 使用
        // 已完成 Compare 的 x1Half 计算缓冲。
        AscendC::LocalTensor<half> selectLocal = x1Local.template ReinterpretCast<half>();
        if (sizeof(DT_X1) == sizeof(int8_t)) {
            selectLocal = calc1Raw.ReinterpretCast<half>();
        }
        LessEqualCompute<DT_X1>::Run(yLocal, x1Local, x2Local, maskLocal,
                                     calc1Raw, calc2Raw, selectLocal,
                                     count, GetCompareCount(count), x1Scalar, x2Scalar);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t globalOffset, uint32_t count) {
        AscendC::LocalTensor<int8_t> yLocal = yQueue_.template DeQue<int8_t>();
        if ((count & 31U) == 0U) {
            AscendC::DataCopy(yGm_[globalOffset], yLocal, count);
        } else {
            const AscendC::DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
            AscendC::DataCopyPad(yGm_[globalOffset], yLocal, copyParams);
        }
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyGmToLocal(const AscendC::LocalTensor<DT_X1> &dst,
                                         const AscendC::GlobalTensor<DT_X1> &src,
                                         uint32_t count) {
        const uint32_t copyBytes = count * static_cast<uint32_t>(sizeof(DT_X1));
        if ((copyBytes & 31U) == 0U) {
            AscendC::DataCopy(dst, src, count);
        } else {
            const AscendC::DataCopyExtParams copyParams{1U, copyBytes, 0U, 0U, 0U};
            const AscendC::DataCopyPadExtParams<DT_X1> padParams{
                true, 0U, 0U, static_cast<DT_X1>(0)};
            AscendC::DataCopyPad(dst, src, copyParams, padParams);
        }
    }

    AscendC::TPipe *pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, QueueBufferCount<DT_X1>::VALUE> x1Queue_;
    AscendC::TQue<AscendC::QuePosition::VECIN, QueueBufferCount<DT_X1>::VALUE> x2Queue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QueueBufferCount<DT_X1>::VALUE> yQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc1Buffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc2Buffer_;

    AscendC::GlobalTensor<DT_X1> x1Gm_;
    AscendC::GlobalTensor<DT_X1> x2Gm_;
    AscendC::GlobalTensor<int8_t> yGm_;

    uint64_t totalLength_;
    uint64_t blockLength_;
    uint32_t rank_;
    uint32_t tileLength_;
    uint32_t noBroadcast_;
    uint64_t segmentLength_;
    uint32_t x1SegmentScalar_;
    uint32_t x2SegmentScalar_;
    uint32_t segmentOuterRank_;
    uint64_t outputShape_[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Strides_[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Strides_[LESS_EQUAL_MAX_DIMS];

};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    (void)tiling;
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    AscendC::TPipe pipe;
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, &tiling_data, &pipe);
    op.Process();
}
