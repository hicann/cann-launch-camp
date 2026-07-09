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

        // Double-buffering: pipeline DMA with compute
        pipe_.InitBuffer(in_queue_,  BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(out_queue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        // f16: 2 temp buffers (xf + tmp);  f32: 1 temp buffer (tmp only)
        if constexpr (IsSameType<T, half>::value) {
            pipe_.InitBuffer(x_float_buf_, TILE_LENGTH * sizeof(float));
        }
        pipe_.InitBuffer(tmp_buf_, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (core_length_ == 0) {
            return;
        }

        const uint32_t tile_len = TILE_LENGTH;
        const uint64_t total_tiles = (core_length_ + tile_len - 1) / tile_len;

        // === Prefetch first tile ===
        uint32_t first_len = (core_length_ >= tile_len)
            ? tile_len : static_cast<uint32_t>(core_length_);
        CopyIn(0, first_len);

        // === Double-buffered main loop ===
        // Pattern: Prefetch(N+1) || Compute(N) || Store(N)
        for (uint64_t i = 0; i < total_tiles; i++) {
            uint64_t offset = i * tile_len;
            uint32_t cur_len = (i == total_tiles - 1)
                ? static_cast<uint32_t>(core_length_ - offset) : tile_len;

            // Start DMA for next tile (non-blocking, overlaps with compute)
            if (i < total_tiles - 1) {
                uint64_t next_offset = offset + tile_len;
                uint32_t next_len = (next_offset + tile_len > core_length_)
                    ? static_cast<uint32_t>(core_length_ - next_offset) : tile_len;
                CopyIn(next_offset, next_len);
            }

            // Deque current input (waits for its DMA to complete)
            LocalTensor<T> x_local = in_queue_.DeQue<T>();
            LocalTensor<T> y_local = out_queue_.AllocTensor<T>();

            Compute(x_local, y_local, cur_len);

            out_queue_.EnQue<T>(y_local);
            in_queue_.FreeTensor(x_local);

            // Store result (DeQue syncs on output buffer ready)
            LocalTensor<T> y_out = out_queue_.DeQue<T>();
            CopyOut(offset, cur_len, y_out);
        }
    }

private:
    __aicore__ inline uint32_t AlignNum() const {
        return 32 / sizeof(T);
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const {
        return (value + align - 1) / align * align;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t len) {
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

    // ── Optimized Compute (simplified formula, f16/f32 specialized) ──
    // y = x / (1 + e^(-1.702*x)) — mathematically identical to original.
    // f16: Cast→f32, compute, Cast back.   f32: compute in-place on x_local.
    __aicore__ inline void Compute(LocalTensor<T> &x_local,
                                   LocalTensor<T> &y_local,
                                   uint32_t len) {
        LocalTensor<float> tmp = tmp_buf_.Get<float>();

        if constexpr (IsSameType<T, half>::value) {
            // ── f16 path ──────────────────────────────────────────
            LocalTensor<float> xf = x_float_buf_.Get<float>();
            Cast(xf, x_local, RoundMode::CAST_NONE, len);

            Muls(tmp, xf, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(xf, xf, tmp, len);

            Cast(y_local, xf, RoundMode::CAST_RINT, len);
        } else {
            // ── f32 path: no copy, compute directly on x_local ────
            Muls(tmp, x_local, -1.702f, len);
            Exp(tmp, tmp, len);
            Adds(tmp, tmp, 1.0f, len);
            Div(y_local, x_local, tmp, len);
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t len,
                                    LocalTensor<T> &y_local) {
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
    // ── Compile-time tile sizing ──────────────────────────────────
    // f16 (2 B/elem): larger tile → fewer DMA launches
    // f32 (4 B/elem): standard tile to stay within UB budget
    static constexpr uint32_t GetTileLength() {
        if constexpr (IsSameType<T, half>::value) return 8192;
        else return 5632;
    }
    static constexpr uint32_t TILE_LENGTH = GetTileLength();
    static constexpr uint32_t BUFFER_NUM  = 2;  // double buffering

    TPipe pipe_;
    TQue<QuePosition::VECIN,  BUFFER_NUM> in_queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> out_queue_;
    // tmp_buf_ reused: -1.702*x → exp → den.  x_float_buf_ only for f16.
    TBuf<QuePosition::VECCALC> x_float_buf_;
    TBuf<QuePosition::VECCALC> tmp_buf_;
    GlobalTensor<T> x_gm_;
    GlobalTensor<T> y_gm_;
    uint64_t core_length_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data.length, tiling_data.blockDim,
            tiling_data.blockLength);
    op.Process();
}
