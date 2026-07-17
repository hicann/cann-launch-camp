// Kernel侧核函数实现
#include <type_traits>
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

namespace {
constexpr int32_t LE_BUFFER_NUM = 2;      // 双缓冲, 流水并行
constexpr uint32_t LE_TILE = 4096;        // 单次计算的元素上限

// 以下比较公式来自 CANN 官方 less_equal 算子实现(纯算术, 不依赖 Compare/Select),
// 已在生产环境验证正确, 避免自行猜测 Compare/Select 位掩码语义带来的风险.
constexpr float NEGATIVE_ONE_FP32 = -1.0F;
constexpr float POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t NEGATIVE_ONE_I32 = -1;
constexpr int32_t POSITIVE_ONE_I32 = 1;
constexpr float MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float MAX_MUL_FP16 = 4096;
constexpr float MIN_ACCURACY_FP32 = 1.1754943508222875e-38F;
constexpr float MAX_MUL_1_FP32 = 1125899906842624.0F;
constexpr float MAX_MUL_2_FP32 = 67108864.0F;
}  // namespace

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &t, TPipe *pipe) {
        tiling_ = t;
        pipe_ = pipe;

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));

        // 逐元素模式使用的队列(双缓冲)
        pipe_->InitBuffer(inQueueX1_, LE_BUFFER_NUM, LE_TILE * sizeof(DT_X1));
        pipe_->InitBuffer(inQueueX2_, LE_BUFFER_NUM, LE_TILE * sizeof(DT_X1));
        pipe_->InitBuffer(outQueueY_, LE_BUFFER_NUM, LE_TILE * sizeof(int8_t));

        // 计算用临时缓冲(与官方实现的 calc_buf_1..4 对应)
        pipe_->InitBuffer(calcBuf1_, LE_TILE * sizeof(DT_X1));
        pipe_->InitBuffer(calcBuf2_, LE_TILE * sizeof(half));
        pipe_->InitBuffer(calcBuf3_, LE_TILE * sizeof(half));
        pipe_->InitBuffer(calcBuf4_, LE_TILE * sizeof(float));

        // 广播模式下加载单个标量(用于最内层广播维)的小缓冲
        pipe_->InitBuffer(scalarBuf_, 32);
    }

    __aicore__ inline void Process() {
        if (tiling_.totalLength == 0) {
            return;
        }
        if (tiling_.mode == 0) {
            ProcessElementwise();
        } else {
            ProcessBroadcast();
        }
    }

private:
    __aicore__ inline void ComputeHalf(const LocalTensor<half> &x1L, const LocalTensor<half> &x2L,
                                       const LocalTensor<half> &y, uint32_t count) {
        // relu(x1-x2) = x1 - min(x1,x2): 0 表示 x1<=x2, 正数表示 x1>x2(Min 已在官方 int8 路径验证支持 half)
        Min(y, x1L, x2L, count);
        Sub(y, x1L, y, count);
        Mins(y, y, static_cast<half>(MIN_ACCURACY_FP16), count);
        Muls(y, y, static_cast<half>(MAX_MUL_FP16), count);
        Muls(y, y, static_cast<half>(MAX_MUL_FP16), count);
        Adds(y, y, static_cast<half>(NEGATIVE_ONE_FP32), count);
        Abs(y, y, count);
    }

    __aicore__ inline void ComputeFloat(const LocalTensor<float> &x1L, const LocalTensor<float> &x2L,
                                        const LocalTensor<float> &y, uint32_t count) {
        Max(y, x1L, x2L, count);
        Sub(y, x2L, y, count);
        Abs(y, y, count);
        Mins(y, y, static_cast<float>(MIN_ACCURACY_FP32), count);
        Muls(y, y, static_cast<float>(MAX_MUL_1_FP32), count);
        Muls(y, y, static_cast<float>(MAX_MUL_1_FP32), count);
        Muls(y, y, static_cast<float>(MAX_MUL_2_FP32), count);
        Adds(y, y, static_cast<float>(NEGATIVE_ONE_FP32), count);
        Abs(y, y, count);
    }

    // x2h 会被就地修改为常量 1, 调用方不得复用其内容
    __aicore__ inline void ComputeInt8(const LocalTensor<half> &x1h, const LocalTensor<half> &x2h,
                                       const LocalTensor<half> &y, uint32_t count) {
        // relu(x1-x2) = x1 - min(x1,x2): int8 差值必为整数, 故直接 clamp 到 1 再翻转即可,
        // 无需官方模板里"两段分别算<和==再相加"的写法.
        Min(y, x1h, x2h, count);
        Sub(y, x1h, y, count);
        Mins(y, y, static_cast<half>(POSITIVE_ONE_FP32), count);      // y=0(true) / 1(false)
        Duplicate(x2h, static_cast<half>(POSITIVE_ONE_FP32), count);   // x2h 复用为常量 1
        Sub(y, x2h, y, count);                                          // y=1-y: 1(true) / 0(false)
    }

    // x2L 会被就地修改为常量 1, 调用方不得复用其内容
    __aicore__ inline void ComputeInt32(const LocalTensor<int32_t> &x1L, const LocalTensor<int32_t> &x2L,
                                        const LocalTensor<int32_t> &y, uint32_t count) {
        Min(y, x1L, x2L, count);
        Sub(y, x1L, y, count);
        Mins(y, y, static_cast<int32_t>(POSITIVE_ONE_I32), count);
        Duplicate(x2L, static_cast<int32_t>(POSITIVE_ONE_I32), count);
        Sub(y, x2L, y, count);
    }

    // x1<=x2 -> bool(int8), 结果写入 yL. x1L/x2L 内容计算后不再可用.
    __aicore__ inline void ComputeLE(const LocalTensor<DT_X1> &x1L,
                                     const LocalTensor<DT_X1> &x2L,
                                     const LocalTensor<int8_t> &yL, uint32_t count) {
        if constexpr (std::is_same_v<DT_X1, half>) {
            LocalTensor<half> y = calcBuf1_.Get<half>();
            ComputeHalf(x1L, x2L, y, count);
            Cast(yL, y, RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<DT_X1, float>) {
            LocalTensor<float> y = calcBuf1_.Get<float>();
            LocalTensor<half> yFp16 = calcBuf2_.Get<half>();
            ComputeFloat(x1L, x2L, y, count);
            Cast(yFp16, y, RoundMode::CAST_NONE, count);
            Cast(yL, yFp16, RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            LocalTensor<half> x1h = calcBuf2_.Get<half>();
            LocalTensor<half> x2h = calcBuf3_.Get<half>();
            LocalTensor<half> yh = calcBuf4_.Get<half>();
            Cast(x1h, x1L, RoundMode::CAST_NONE, count);
            Cast(x2h, x2L, RoundMode::CAST_NONE, count);
            ComputeInt8(x1h, x2h, yh, count);
            Cast(yL, yh, RoundMode::CAST_NONE, count);
        } else {  // int32_t
            LocalTensor<int32_t> y = calcBuf1_.Get<int32_t>();
            LocalTensor<float> yFp32 = calcBuf4_.Get<float>();
            LocalTensor<half> yFp16 = calcBuf3_.Get<half>();
            ComputeInt32(x1L, x2L, y, count);
            Cast(yFp32, y, RoundMode::CAST_NONE, count);
            Cast(yFp16, yFp32, RoundMode::CAST_NONE, count);
            Cast(yL, yFp16, RoundMode::CAST_NONE, count);
        }
    }

    __aicore__ inline void ProcessElementwise() {
        uint32_t total = tiling_.totalLength;
        uint32_t bd = tiling_.blockDim;
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t perCore = (total + bd - 1) / bd;
        uint32_t start = idx * perCore;
        if (start >= total) {
            return;
        }
        uint32_t remain = perCore;
        if (start + remain > total) {
            remain = total - start;
        }

        uint32_t off = start;
        while (remain > 0) {
            uint32_t cur = remain < LE_TILE ? remain : LE_TILE;

            LocalTensor<DT_X1> x1L = inQueueX1_.AllocTensor<DT_X1>();
            LocalTensor<DT_X1> x2L = inQueueX2_.AllocTensor<DT_X1>();
            CopyInPad(x1L, x1Gm_, off, cur);
            CopyInPad(x2L, x2Gm_, off, cur);
            inQueueX1_.EnQue(x1L);
            inQueueX2_.EnQue(x2L);

            LocalTensor<DT_X1> x1In = inQueueX1_.DeQue<DT_X1>();
            LocalTensor<DT_X1> x2In = inQueueX2_.DeQue<DT_X1>();
            LocalTensor<int8_t> yL = outQueueY_.AllocTensor<int8_t>();
            ComputeLE(x1In, x2In, yL, cur);
            outQueueY_.EnQue(yL);
            inQueueX1_.FreeTensor(x1In);
            inQueueX2_.FreeTensor(x2In);

            LocalTensor<int8_t> yOut = outQueueY_.DeQue<int8_t>();
            CopyOutPad(yGm_, off, yOut, cur);
            outQueueY_.FreeTensor(yOut);

            off += cur;
            remain -= cur;
        }
    }

    __aicore__ inline void ProcessBroadcast() {
        uint32_t numRows = tiling_.numRows;
        uint32_t lastDim = tiling_.lastDim;
        uint32_t rank = tiling_.rank;
        uint32_t bd = tiling_.blockDim;
        uint32_t idx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t rpc = (numRows + bd - 1) / bd;
        uint32_t sr = idx * rpc;
        if (sr >= numRows) {
            return;
        }
        uint32_t er = sr + rpc;
        if (er > numRows) {
            er = numRows;
        }

        bool x1Bcast = tiling_.x1LastBroad != 0;
        bool x2Bcast = tiling_.x2LastBroad != 0;

        for (uint32_t rrow = sr; rrow < er; rrow++) {
            // 由行号还原各外层维度坐标, 计算 x1/x2 行基址
            uint32_t x1Base = 0;
            uint32_t x2Base = 0;
            uint32_t tmp = rrow;
            for (int d = static_cast<int>(rank) - 2; d >= 0; d--) {
                uint32_t dimSize = tiling_.outShape[d];
                uint32_t coord = tmp % dimSize;
                tmp /= dimSize;
                x1Base += coord * tiling_.x1Stride[d];
                x2Base += coord * tiling_.x2Stride[d];
            }

            // 若该操作数最内层维度是广播, 整行共用同一个标量, 每行只需读取一次.
            DT_X1 x1Scalar{};
            DT_X1 x2Scalar{};
            if (x1Bcast) {
                x1Scalar = LoadScalar(x1Gm_, x1Base);
            }
            if (x2Bcast) {
                x2Scalar = LoadScalar(x2Gm_, x2Base);
            }

            uint32_t off = 0;
            while (off < lastDim) {
                uint32_t cur = (lastDim - off) < LE_TILE ? (lastDim - off) : LE_TILE;

                LocalTensor<DT_X1> x1L = inQueueX1_.AllocTensor<DT_X1>();
                LocalTensor<DT_X1> x2L = inQueueX2_.AllocTensor<DT_X1>();
                if (x1Bcast) {
                    FillScalar(x1L, x1Scalar, cur);
                } else {
                    DataCopyExtParams cp{1, static_cast<uint32_t>(cur * sizeof(DT_X1)), 0, 0, 0};
                    DataCopyPadExtParams<DT_X1> pad{false, 0, 0, 0};
                    DataCopyPad(x1L, x1Gm_[x1Base + off], cp, pad);
                }
                if (x2Bcast) {
                    FillScalar(x2L, x2Scalar, cur);
                } else {
                    DataCopyExtParams cp{1, static_cast<uint32_t>(cur * sizeof(DT_X1)), 0, 0, 0};
                    DataCopyPadExtParams<DT_X1> pad{false, 0, 0, 0};
                    DataCopyPad(x2L, x2Gm_[x2Base + off], cp, pad);
                }
                inQueueX1_.EnQue(x1L);
                inQueueX2_.EnQue(x2L);

                LocalTensor<DT_X1> x1In = inQueueX1_.DeQue<DT_X1>();
                LocalTensor<DT_X1> x2In = inQueueX2_.DeQue<DT_X1>();
                LocalTensor<int8_t> yL = outQueueY_.AllocTensor<int8_t>();
                ComputeLE(x1In, x2In, yL, cur);
                outQueueY_.EnQue(yL);
                inQueueX1_.FreeTensor(x1In);
                inQueueX2_.FreeTensor(x2In);

                LocalTensor<int8_t> yOut = outQueueY_.DeQue<int8_t>();
                DataCopyExtParams cpOut{1, static_cast<uint32_t>(cur * sizeof(int8_t)), 0, 0, 0};
                DataCopyPad(yGm_[static_cast<uint64_t>(rrow) * lastDim + off], yOut, cpOut);
                outQueueY_.FreeTensor(yOut);

                off += cur;
            }
        }
    }

    // 读取一个标量(该行内广播操作数的公共值). DMA 写入 UB 后需同步才能标量读取.
    __aicore__ inline DT_X1 LoadScalar(const GlobalTensor<DT_X1> &gm, uint32_t base) {
        LocalTensor<DT_X1> tmp = scalarBuf_.Get<DT_X1>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0};
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, 0};
        DataCopyPad(tmp, gm[base], cp, pad);
        pipe_barrier(PIPE_ALL);
        return tmp.GetValue(0);
    }

    // 用一个已在寄存器中的标量填满 local 的前 cur 个元素(纯向量运算, 无需额外同步).
    __aicore__ inline void FillScalar(const LocalTensor<DT_X1> &local, DT_X1 val, uint32_t cur) {
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            // int8 不支持向量 Duplicate: 经 half 广播后回转
            LocalTensor<half> tmp = calcBuf2_.Get<half>();
            Duplicate(tmp, static_cast<half>(val), cur);
            Cast(local, tmp, RoundMode::CAST_NONE, cur);
        } else {
            Duplicate(local, val, cur);
        }
    }

    __aicore__ inline void CopyInPad(const LocalTensor<DT_X1> &local,
                                     const GlobalTensor<DT_X1> &gm, uint32_t off,
                                     uint32_t cur) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur * sizeof(DT_X1)), 0, 0, 0};
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, 0};
        DataCopyPad(local, gm[off], cp, pad);
    }

    __aicore__ inline void CopyOutPad(const GlobalTensor<int8_t> &gm, uint32_t off,
                                      const LocalTensor<int8_t> &local, uint32_t cur) {
        DataCopyExtParams cp{1, static_cast<uint32_t>(cur * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(gm[off], local, cp);
    }

private:
    TPipe *pipe_;
    LessEqualTilingData tiling_;

    GlobalTensor<DT_X1> x1Gm_;
    GlobalTensor<DT_X1> x2Gm_;
    GlobalTensor<int8_t> yGm_;

    TQue<TPosition::VECIN, LE_BUFFER_NUM> inQueueX1_;
    TQue<TPosition::VECIN, LE_BUFFER_NUM> inQueueX2_;
    TQue<TPosition::VECOUT, LE_BUFFER_NUM> outQueueY_;

    // calcBuf1_: DT_X1 类型的 y_compute(half/float/int32 路径使用)
    // calcBuf2_/calcBuf3_: half 类型临时张量
    // calcBuf4_: float 类型临时张量(int8 路径下按 half 复用)
    TBuf<TPosition::VECCALC> calcBuf1_;
    TBuf<TPosition::VECCALC> calcBuf2_;
    TBuf<TPosition::VECCALC> calcBuf3_;
    TBuf<TPosition::VECCALC> calcBuf4_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    TPipe pipe;
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data, &pipe);
    op.Process();
}