#include <type_traits>

#include "kernel_operator.h"
#include "less_equal_custom_tiling.h"
#include "tiling_key_less_equal_custom.h"

namespace {
constexpr uint32_t ALIGN_BYTES = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr float NEGATIVE_ONE = -1.0F;
constexpr float POSITIVE_ONE = 1.0F;
constexpr float FP16_MIN_NORMAL = 0.00000005960464477539063F;
constexpr float FP16_SCALE = 4096.0F;
constexpr float FP32_MIN_NORMAL = 1.1754943508222875e-38F;
constexpr float FP32_SCALE_1 = 1125899906842624.0F;
constexpr float FP32_SCALE_2 = 67108864.0F;
}

template <typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData *tiling)
    {
        totalLength_ = tiling->totalLength;
        dims_ = tiling->dims;
        blockDim_ = tiling->blockDim == 0 ? 1 : tiling->blockDim;
        chunkSize_ = tiling->ubChunkSize == 0 ? 1 : tiling->ubChunkSize;
        const uint32_t maxCopyElements = 65535U / sizeof(T);
        if (chunkSize_ > maxCopyElements) {
            chunkSize_ = maxCopyElements;
        }
        blockIdx_ = AscendC::GetBlockIdx();

        for (int32_t i = 0; i < dims_; ++i) {
            x1Shape_[i] = tiling->x1Shape[i];
            x2Shape_[i] = tiling->x2Shape[i];
            outShape_[i] = tiling->outShape[i];
            x1Stride_[i] = tiling->x1Stride[i];
            x2Stride_[i] = tiling->x2Stride[i];
        }

        x1Gm_.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2);
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)y, totalLength_);

        const uint32_t inputAlign = ALIGN_BYTES / sizeof(T);
        const uint32_t inputBufferElements = ((chunkSize_ + inputAlign - 1) / inputAlign) * inputAlign;
        const uint32_t outputBufferBytes = ((chunkSize_ + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        pipe_.InitBuffer(x1Queue_, BUFFER_NUM, inputBufferElements * sizeof(T));
        pipe_.InitBuffer(x2Queue_, BUFFER_NUM, inputBufferElements * sizeof(T));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, outputBufferBytes);
        pipe_.InitBuffer(workT_, inputBufferElements * sizeof(T));
        pipe_.InitBuffer(workHalf1_, inputBufferElements * sizeof(half));
        pipe_.InitBuffer(workHalf2_, inputBufferElements * sizeof(half));
        pipe_.InitBuffer(workFloat_, inputBufferElements * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (totalLength_ == 0 || dims_ <= 0) {
            return;
        }

        bool noBroadcast = true;
        for (int32_t i = 0; i < dims_; ++i) {
            if (x1Shape_[i] != outShape_[i] || x2Shape_[i] != outShape_[i]) {
                noBroadcast = false;
                break;
            }
        }
        if (noBroadcast) {
            ProcessLinear();
        } else {
            ProcessBroadcast();
        }
    }

private:
    __aicore__ inline void ProcessLinear()
    {
        const uint64_t elementsPerCore = totalLength_ / blockDim_ +
            (totalLength_ % blockDim_ != 0);
        const uint64_t start = static_cast<uint64_t>(blockIdx_) * elementsPerCore;
        uint64_t end = start + elementsPerCore;
        if (end > totalLength_) {
            end = totalLength_;
        }
        for (uint64_t offset = start; offset < end; offset += chunkSize_) {
            const uint32_t count = static_cast<uint32_t>(chunkSize_ < end - offset ? chunkSize_ : end - offset);
            CopyIn(offset, offset, count, false, false);
            Compute(count);
            CopyOut(offset, count);
        }
    }

    __aicore__ inline void ProcessBroadcast()
    {
        const uint32_t inner = outShape_[dims_ - 1];
        if (inner == 0) {
            return;
        }
        const uint64_t rows = totalLength_ / inner;
        const uint64_t rowsPerCore = rows / blockDim_ + (rows % blockDim_ != 0);
        const uint64_t firstRow = static_cast<uint64_t>(blockIdx_) * rowsPerCore;
        uint64_t lastRow = firstRow + rowsPerCore;
        if (lastRow > rows) {
            lastRow = rows;
        }
        if (firstRow >= lastRow) {
            return;
        }

        const bool x1InnerBroadcast = x1Shape_[dims_ - 1] == 1;
        const bool x2InnerBroadcast = x2Shape_[dims_ - 1] == 1;
        for (uint64_t row = firstRow; row < lastRow; ++row) {
            uint64_t linear = row;
            uint64_t x1Base = 0;
            uint64_t x2Base = 0;
            for (int32_t dim = dims_ - 2; dim >= 0; --dim) {
                const uint64_t coordinate = linear % outShape_[dim];
                linear /= outShape_[dim];
                if (x1Shape_[dim] != 1) {
                    x1Base += coordinate * x1Stride_[dim];
                }
                if (x2Shape_[dim] != 1) {
                    x2Base += coordinate * x2Stride_[dim];
                }
            }
            for (uint32_t offset = 0; offset < inner; offset += chunkSize_) {
                const uint32_t count = chunkSize_ < inner - offset ? chunkSize_ : inner - offset;
                CopyIn(x1InnerBroadcast ? x1Base : x1Base + offset,
                       x2InnerBroadcast ? x2Base : x2Base + offset,
                       count, x1InnerBroadcast, x2InnerBroadcast);
                Compute(count);
                CopyOut(row * inner + offset, count);
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t x1Offset, uint64_t x2Offset, uint32_t count,
                                  bool x1Broadcast, bool x2Broadcast)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
        if (x1Broadcast) {
            AscendC::DataCopyParams params{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
            AscendC::DataCopyPad(x1Local, x1Gm_[x1Offset], params, padParams);
            // 建立 MTE2 -> Vector 依赖，确保读取广播标量前搬运已经完成。
            x1Queue_.EnQue(x1Local);
            x1Local = x1Queue_.DeQue<T>();
            if constexpr (std::is_same_v<T, int8_t>) {
                // Duplicate 不支持 int8：借助 half 工作区完成标量扩展。
                AscendC::LocalTensor<half> broadcast = workHalf1_.Get<half>();
                AscendC::Cast(broadcast, x1Local, AscendC::RoundMode::CAST_NONE, 1);
                AscendC::Duplicate(broadcast, broadcast.GetValue(0), count);
                AscendC::Cast(x1Local, broadcast, AscendC::RoundMode::CAST_NONE, count);
            } else {
                AscendC::Duplicate(x1Local, x1Local.GetValue(0), count);
            }
        } else {
            AscendC::DataCopyParams params{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
            AscendC::DataCopyPad(x1Local, x1Gm_[x1Offset], params, padParams);
        }
        if (x2Broadcast) {
            AscendC::DataCopyParams params{1, static_cast<uint16_t>(sizeof(T)), 0, 0};
            AscendC::DataCopyPad(x2Local, x2Gm_[x2Offset], params, padParams);
            x2Queue_.EnQue(x2Local);
            x2Local = x2Queue_.DeQue<T>();
            if constexpr (std::is_same_v<T, int8_t>) {
                AscendC::LocalTensor<half> broadcast = workHalf2_.Get<half>();
                AscendC::Cast(broadcast, x2Local, AscendC::RoundMode::CAST_NONE, 1);
                AscendC::Duplicate(broadcast, broadcast.GetValue(0), count);
                AscendC::Cast(x2Local, broadcast, AscendC::RoundMode::CAST_NONE, count);
            } else {
                AscendC::Duplicate(x2Local, x2Local.GetValue(0), count);
            }
        } else {
            AscendC::DataCopyParams params{1, static_cast<uint16_t>(count * sizeof(T)), 0, 0};
            AscendC::DataCopyPad(x2Local, x2Gm_[x2Offset], params, padParams);
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> x1 = x1Queue_.DeQue<T>();
        AscendC::LocalTensor<T> x2 = x2Queue_.DeQue<T>();
        AscendC::LocalTensor<uint8_t> y = yQueue_.AllocTensor<uint8_t>();
        AscendC::LocalTensor<T> tmp = workT_.Get<T>();

        if constexpr (std::is_same_v<T, half>) {
            AscendC::Max(tmp, x1, x2, count);
            AscendC::Sub(tmp, x2, tmp, count);
            AscendC::Abs(tmp, tmp, count);
            AscendC::Mins(tmp, tmp, static_cast<half>(FP16_MIN_NORMAL), count);
            AscendC::Muls(tmp, tmp, static_cast<half>(FP16_SCALE), count);
            AscendC::Muls(tmp, tmp, static_cast<half>(FP16_SCALE), count);
            AscendC::Adds(tmp, tmp, static_cast<half>(NEGATIVE_ONE), count);
            AscendC::Abs(tmp, tmp, count);
            AscendC::Cast(y, tmp, AscendC::RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<T, float>) {
            AscendC::LocalTensor<half> fp16 = workHalf1_.Get<half>();
            AscendC::Max(tmp, x1, x2, count);
            AscendC::Sub(tmp, x2, tmp, count);
            AscendC::Abs(tmp, tmp, count);
            AscendC::Mins(tmp, tmp, FP32_MIN_NORMAL, count);
            AscendC::Muls(tmp, tmp, FP32_SCALE_1, count);
            AscendC::Muls(tmp, tmp, FP32_SCALE_1, count);
            AscendC::Muls(tmp, tmp, FP32_SCALE_2, count);
            AscendC::Adds(tmp, tmp, NEGATIVE_ONE, count);
            AscendC::Abs(tmp, tmp, count);
            AscendC::Cast(fp16, tmp, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(y, fp16, AscendC::RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            AscendC::LocalTensor<half> x1Half = workHalf1_.Get<half>();
            AscendC::LocalTensor<half> x2Half = workHalf2_.Get<half>();
            AscendC::LocalTensor<half> yHalf = workFloat_.Get<half>();
            AscendC::Cast(x1Half, x1, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(x2Half, x2, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Min(yHalf, x1Half, x2Half, count);
            AscendC::Sub(yHalf, x2Half, yHalf, count);
            AscendC::Mins(yHalf, yHalf, static_cast<half>(POSITIVE_ONE), count);
            AscendC::Sub(x1Half, x1Half, x2Half, count);
            AscendC::Abs(x1Half, x1Half, count);
            AscendC::Mins(x1Half, x1Half, static_cast<half>(POSITIVE_ONE), count);
            AscendC::Duplicate(x2Half, static_cast<half>(POSITIVE_ONE), count);
            AscendC::Sub(x1Half, x2Half, x1Half, count);
            AscendC::Add(yHalf, yHalf, x1Half, count);
            AscendC::Cast(y, yHalf, AscendC::RoundMode::CAST_NONE, count);
        } else {
            AscendC::LocalTensor<half> fp16 = workHalf1_.Get<half>();
            AscendC::LocalTensor<int32_t> aux = workFloat_.Get<int32_t>();

            // neg1/neg2: 输入为负时为 1，否则为 0。先钳位再平方，
            // 避免对 INT_MIN 直接 Abs/Neg 产生溢出。
            AscendC::Mins(tmp, x1, static_cast<int32_t>(0), count);
            AscendC::Maxs(tmp, tmp, static_cast<int32_t>(-1), count);
            AscendC::Mul(tmp, tmp, tmp, count);  // neg1
            AscendC::Mins(aux, x2, static_cast<int32_t>(0), count);
            AscendC::Maxs(aux, aux, static_cast<int32_t>(-1), count);
            AscendC::Mul(aux, aux, aux, count);  // neg2

            // 同号时 x2-x1 一定落在 int32 范围内。异号 lane 即使回绕，
            // 后续也会被 sameSign 掩码清零，不参与最终结果。
            AscendC::Sub(x1, x2, x1, count);
            AscendC::Mins(x1, x1, static_cast<int32_t>(0), count);
            AscendC::Maxs(x1, x1, static_cast<int32_t>(-1), count);
            AscendC::Adds(x1, x1, static_cast<int32_t>(1), count);  // same-sign result

            AscendC::Sub(x2, tmp, aux, count);
            // x2 此时只可能为 -1/0/1；int32 Abs 在 CANN 9.0/dav-c220
            // 不受支持，平方可等价得到 1/0/1。
            AscendC::Mul(x2, x2, x2, count);  // differentSign = neg1 xor neg2
            AscendC::Duplicate(aux, static_cast<int32_t>(1), count);
            AscendC::Sub(aux, aux, x2, count);  // sameSign
            AscendC::Mul(x1, x1, aux, count);
            AscendC::Mul(tmp, tmp, x2, count);  // 异号时仅 x1<0 为 true
            AscendC::Add(tmp, tmp, x1, count);

            AscendC::LocalTensor<float> fp32 = workFloat_.Get<float>();
            AscendC::Cast(fp32, tmp, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(fp16, fp32, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Cast(y, fp16, AscendC::RoundMode::CAST_NONE, count);
        }

        yQueue_.EnQue(y);
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<uint8_t> y = yQueue_.DeQue<uint8_t>();
        AscendC::DataCopyParams params{1, static_cast<uint16_t>(count), 0, 0};
        AscendC::DataCopyPad(yGm_[offset], y, params);
        yQueue_.FreeTensor(y);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x1Queue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> x2Queue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workT_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workHalf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workHalf2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workFloat_;
    AscendC::GlobalTensor<T> x1Gm_;
    AscendC::GlobalTensor<T> x2Gm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint64_t totalLength_ = 0;
    int32_t dims_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t chunkSize_ = 1;
    uint32_t blockIdx_ = 0;
    int32_t x1Shape_[MAX_DIMS];
    int32_t x2Shape_[MAX_DIMS];
    int32_t outShape_[MAX_DIMS];
    uint64_t x1Stride_[MAX_DIMS];
    uint64_t x2Stride_[MAX_DIMS];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}
