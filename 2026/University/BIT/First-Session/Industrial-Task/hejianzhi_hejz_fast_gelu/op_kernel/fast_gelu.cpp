#include "kernel_operator.h"

#include <cstdint>

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

template <class T>
class FastGeluWorker {
public:
    __aicore__ inline FastGeluWorker() = default;

    __aicore__ inline void Prepare(GM_ADDR x, GM_ADDR y,
                                   uint64_t total_length,
                                   uint32_t /*unused*/, uint64_t tile_length) {
        uint64_t block_id = GetBlockIdx();
        uint64_t start_pos = block_id * tile_length;

        if (start_pos >= total_length) {
            active_length_ = 0;
        } else {
            uint64_t tail = total_length - start_pos;
            active_length_ = tail < tile_length ? tail : tile_length;
        }

        x_gm_.SetGlobalBuffer((__gm__ T *)x + start_pos, active_length_);
        y_gm_.SetGlobalBuffer((__gm__ T *)y + start_pos, active_length_);

        pipe_.InitBuffer(in_queue_, BUFFER_NUM, TILE_SIZE * sizeof(T));
        pipe_.InitBuffer(out_queue_, BUFFER_NUM, TILE_SIZE * sizeof(T));
        if constexpr (IsSameType<T, half>::value) {
            pipe_.InitBuffer(x_float_buf_, TILE_SIZE * sizeof(float));
        }
        pipe_.InitBuffer(tmp_buf_, TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Execute() {
        if (active_length_ == 0) {
            return;
        }

        const uint32_t tile_size = TILE_SIZE;
        const uint64_t tile_count = (active_length_ + tile_size - 1) / tile_size;

        uint32_t first_chunk = static_cast<uint32_t>(
            active_length_ < tile_size ? active_length_ : tile_size);
        LoadBlock(0, first_chunk);

        for (uint64_t idx = 0; idx < tile_count; ++idx) {
            uint64_t base = idx * tile_size;
            uint32_t chunk_len = static_cast<uint32_t>(
                idx == tile_count - 1 ? active_length_ - base : tile_size);

            if (idx + 1 < tile_count) {
                uint64_t next_base = base + tile_size;
                uint32_t next_chunk = static_cast<uint32_t>(
                    next_base + tile_size > active_length_
                        ? active_length_ - next_base
                        : tile_size);
                LoadBlock(next_base, next_chunk);
            }

            LocalTensor<T> input = in_queue_.DeQue<T>();
            LocalTensor<T> output = out_queue_.AllocTensor<T>();

            ComputeBlock(input, output, chunk_len);

            out_queue_.EnQue<T>(output);
            in_queue_.FreeTensor(input);

            LocalTensor<T> ready_out = out_queue_.DeQue<T>();
            StoreBlock(base, chunk_len, ready_out);
        }
    }

private:
    __aicore__ inline uint32_t ElementsPerAlign() const {
        return 32 / sizeof(T);
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t alignment) const {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline void LoadBlock(uint64_t offset, uint32_t length) {
        LocalTensor<T> buffer = in_queue_.AllocTensor<T>();

        DataCopyExtParams params{};
        params.blockCount = 1;
        params.blockLen = length * sizeof(T);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        DataCopyPadExtParams<T> pad{};
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = static_cast<uint8_t>(
            CeilDiv(length, ElementsPerAlign()) - length);
        pad.paddingValue = 0;

        DataCopyPad(buffer, x_gm_[offset], params, pad);
        in_queue_.EnQue(buffer);
    }

    __aicore__ inline void ComputeBlock(LocalTensor<T> &input,
                                        LocalTensor<T> &output,
                                        uint32_t length) {
        LocalTensor<float> temp = tmp_buf_.Get<float>();

        if constexpr (IsSameType<T, half>::value) {
            LocalTensor<float> tmp_fp = x_float_buf_.Get<float>();
            Cast(tmp_fp, input, RoundMode::CAST_NONE, length);

            Muls(temp, tmp_fp, -1.702f, length);
            Exp(temp, temp, length);
            Adds(temp, temp, 1.0f, length);
            Div(tmp_fp, tmp_fp, temp, length);

            Cast(output, tmp_fp, RoundMode::CAST_RINT, length);
        } else {
            Muls(temp, input, -1.702f, length);
            Exp(temp, temp, length);
            Adds(temp, temp, 1.0f, length);
            Div(output, input, temp, length);
        }
    }

    __aicore__ inline void StoreBlock(uint64_t offset, uint32_t length,
                                      LocalTensor<T> &output) {
        DataCopyExtParams params{};
        params.blockCount = 1;
        params.blockLen = length * sizeof(T);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;

        DataCopyPad(y_gm_[offset], output, params);
        out_queue_.FreeTensor(output);
    }

private:
    static constexpr uint32_t GetTileSize() {
        if constexpr (IsSameType<T, half>::value) {
            return 10240;
        }
        return 6144;
    }

    static constexpr uint32_t TILE_SIZE = GetTileSize();
    static constexpr uint32_t BUFFER_NUM = 2;

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> in_queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> out_queue_;
    TBuf<QuePosition::VECCALC> x_float_buf_;
    TBuf<QuePosition::VECCALC> tmp_buf_;
    GlobalTensor<T> x_gm_;
    GlobalTensor<T> y_gm_;
    uint64_t active_length_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    FastGeluWorker<DT_X> worker;
    worker.Prepare(x, y, tiling_data.length,
                   tiling_data.blockDim, tiling_data.blockLength);
    worker.Execute();
}
