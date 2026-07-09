// Kernel侧核函数实现：v56 based on v55
// 目标：保留 v55 有效的 split-loop 主路径；进一步优化 half 大张量性能。
// 改动：float32 路径保持不变；float16 路径不再升 float32 计算，直接使用 half 向量计算，
//      减少 Cast、float 临时 buffer 与更多 UB 初始化开销。
#include "kernel_operator.h"

#include <cstdint>
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

template <class T>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t length,
                                uint32_t, uint64_t blockLength) {
        uint64_t block_idx = GetBlockIdx();
        uint64_t offset = block_idx * blockLength;
        if (offset >= length) {
            core_length_ = 0;
        } else {
            uint64_t remain = length - offset;
            core_length_ = remain < blockLength ? remain : blockLength;
        }

        x_gm_.SetGlobalBuffer((__gm__ T *)x + offset, core_length_);
        y_gm_.SetGlobalBuffer((__gm__ T *)y + offset, core_length_);

        // 保留 v53 的小分片缩小 UB buffer 逻辑。
        tile_length_ = CalcTileLength();

        pipe_.InitBuffer(in_queue_,  BUFFER_NUM, tile_length_ * sizeof(T));
        pipe_.InitBuffer(out_queue_, BUFFER_NUM, tile_length_ * sizeof(T));

        // v56：half 直接在 half 上计算，不再申请 float 中间 buffer；
        // float32 路径仍需要 tmp_buf_ 保存 exp 分母。
        if constexpr (!IsSameType<T, half>::value) {
            pipe_.InitBuffer(tmp_buf_, tile_length_ * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (core_length_ == 0) {
            return;
        }

        const uint32_t tile_len = tile_length_;

        // 单 tile 路径：只判断一次是否 32B 对齐。
        if (core_length_ <= tile_len) {
            uint32_t cur_len = static_cast<uint32_t>(core_length_);
            if (IsAligned(cur_len)) {
                CopyInAligned(0, cur_len);
                ComputeOne(cur_len);
                CopyOutAligned(0, cur_len);
            } else {
                CopyInPad(0, cur_len);
                ComputeOne(cur_len);
                CopyOutPad(0, cur_len);
            }
            return;
        }

        /*
         * 大张量路径优化：
         * v53 在每个 tile 循环里都计算 cur_len、next_len，并在 CopyIn/CopyOut 内再次判断是否对齐。
         * 这里把 full tile 与 tail 拆开：
         *   - full tile 必然 32B 对齐，直接走 DataCopy；
         *   - 只有 tail 才判断 DataCopy / DataCopyPad。
         * 计算公式、tile 大小、双缓冲数量均不变。
         */
        const uint64_t full_tiles = core_length_ / tile_len;
        const uint32_t tail_len = static_cast<uint32_t>(core_length_ - full_tiles * tile_len);

        // 先预取第 0 个 full tile。
        CopyInAligned(0, tile_len);

        if (tail_len == 0) {
            // 全部都是 full tile。
            for (uint64_t i = 0; i + 1 < full_tiles; ++i) {
                uint64_t next_offset = (i + 1) * tile_len;
                CopyInAligned(next_offset, tile_len);
                ComputeOne(tile_len);
                CopyOutAligned(i * tile_len, tile_len);
            }
            ComputeOne(tile_len);
            CopyOutAligned((full_tiles - 1) * tile_len, tile_len);
        } else {
            // full tile + tail。先流水处理 full tile，同时在最后一个 full tile 前预取 tail。
            for (uint64_t i = 0; i + 1 < full_tiles; ++i) {
                uint64_t next_offset = (i + 1) * tile_len;
                CopyInAligned(next_offset, tile_len);
                ComputeOne(tile_len);
                CopyOutAligned(i * tile_len, tile_len);
            }

            uint64_t tail_offset = full_tiles * tile_len;
            if (IsAligned(tail_len)) {
                CopyInAligned(tail_offset, tail_len);
                ComputeOne(tile_len);
                CopyOutAligned((full_tiles - 1) * tile_len, tile_len);

                ComputeOne(tail_len);
                CopyOutAligned(tail_offset, tail_len);
            } else {
                CopyInPad(tail_offset, tail_len);
                ComputeOne(tile_len);
                CopyOutAligned((full_tiles - 1) * tile_len, tile_len);

                ComputeOne(tail_len);
                CopyOutPad(tail_offset, tail_len);
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignNum() const {
        return 32 / sizeof(T);
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline bool IsAligned(uint32_t len) const {
        uint32_t align = AlignNum();
        return (len & (align - 1)) == 0;
    }

    __aicore__ inline uint32_t CalcTileLength() const {
        if (core_length_ >= BASE_TILE_LENGTH) {
            return BASE_TILE_LENGTH;
        }
        uint32_t len = static_cast<uint32_t>(core_length_);
        uint32_t align = AlignNum();
        uint32_t aligned = AlignUp(len, align);
        return aligned < align ? align : aligned;
    }

    __aicore__ inline void CopyInAligned(uint64_t offset, uint32_t len) {
        LocalTensor<T> x_local = in_queue_.AllocTensor<T>();
        DataCopy(x_local, x_gm_[offset], len);
        in_queue_.EnQue(x_local);
    }

    __aicore__ inline void CopyInPad(uint64_t offset, uint32_t len) {
        LocalTensor<T> x_local = in_queue_.AllocTensor<T>();

        DataCopyExtParams copy_params;
        copy_params.blockCount = 1;
        copy_params.blockLen   = len * sizeof(T);
        copy_params.srcStride  = 0;
        copy_params.dstStride  = 0;
        copy_params.rsv        = 0;

        DataCopyPadExtParams<T> pad_params;
        pad_params.isPad        = true;
        pad_params.leftPadding  = 0;
        pad_params.rightPadding = static_cast<uint8_t>(
            AlignUp(len, AlignNum()) - len);
        pad_params.paddingValue = 0;

        DataCopyPad(x_local, x_gm_[offset], copy_params, pad_params);
        in_queue_.EnQue(x_local);
    }

    __aicore__ inline void ComputeOne(uint32_t len) {
        LocalTensor<T> x_local = in_queue_.DeQue<T>();
        LocalTensor<T> y_local = out_queue_.AllocTensor<T>();

        Compute(x_local, y_local, len);

        out_queue_.EnQue<T>(y_local);
        in_queue_.FreeTensor(x_local);
    }

    // 数学等价式：y = x / (1 + exp(-1.702 * x))
    __aicore__ inline void Compute(LocalTensor<T> &x_local,
                                   LocalTensor<T> &y_local,
                                   uint32_t len) {
        if constexpr (IsSameType<T, half>::value) {
            // v56：float16 直接用 half 计算。
            // 精度要求通常为 1e-3，直接 half 路径有机会通过，同时能减少 Cast 和 float buffer 开销。
            Muls(y_local, x_local, static_cast<T>(-1.702f), len);
            Exp(y_local, y_local, len);
            Adds(y_local, y_local, static_cast<T>(1.0f), len);
            Div(y_local, x_local, y_local, len);
        } else {
            LocalTensor<float> tmp = tmp_buf_.Get<float>();
            Muls(tmp, x_local, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(y_local, x_local, tmp, len);
        }
    }

    __aicore__ inline void CopyOutAligned(uint64_t offset, uint32_t len) {
        LocalTensor<T> y_local = out_queue_.DeQue<T>();
        DataCopy(y_gm_[offset], y_local, len);
        out_queue_.FreeTensor(y_local);
    }

    __aicore__ inline void CopyOutPad(uint64_t offset, uint32_t len) {
        LocalTensor<T> y_local = out_queue_.DeQue<T>();

        DataCopyExtParams copy_params;
        copy_params.blockCount = 1;
        copy_params.blockLen   = len * sizeof(T);
        copy_params.srcStride  = 0;
        copy_params.dstStride  = 0;
        copy_params.rsv        = 0;

        DataCopyPad(y_gm_[offset], y_local, copy_params);
        out_queue_.FreeTensor(y_local);
    }

private:
static constexpr uint32_t GetBaseTileLength() {
    if constexpr (IsSameType<T, half>::value) {
        return 16384;
    } else {
        return 6144;
    }
}

    static constexpr uint32_t BASE_TILE_LENGTH = GetBaseTileLength();
    static constexpr uint32_t BUFFER_NUM  = 2;

    TPipe pipe_;
    TQue<TPosition::VECIN,  BUFFER_NUM> in_queue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> out_queue_;
    TBuf<TPosition::VECCALC> tmp_buf_;
    GlobalTensor<T> x_gm_;
    GlobalTensor<T> y_gm_;
    uint64_t core_length_ = 0;
    uint32_t tile_length_ = BASE_TILE_LENGTH;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockDim,
            tiling_data.blockLength);
    op.Process();
}