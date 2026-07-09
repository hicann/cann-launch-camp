// Kernel侧核函数实现
#include "kernel_operator.h"
#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr float ATTR = 1.702f;
constexpr float ATTR_HALF = 0.851f;
constexpr float EXP_MIN = -88.0f;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData& tiling) {
        uint32_t block_idx = AscendC::GetBlockIdx();

        // ========== 单核模式：所有数据给核0 ==========
        if (block_idx == 0) {
            core_data_len_ = tiling.length;
            data_offset_ = 0;
        } else {
            core_data_len_ = 0;
            data_offset_ = 0;
            return;
        }
        // ==========================================

        if (core_data_len_ == 0) return;

        tile_len_ = tiling.tile_len;
        tile_num_ = tiling.tile_num;
        tail_len_ = tiling.tail_len;

        __gm__ DT_X* x_base = (__gm__ DT_X*)((__gm__ uint8_t*)x + data_offset_ * sizeof(DT_X));
        __gm__ DT_X* y_base = (__gm__ DT_X*)((__gm__ uint8_t*)y + data_offset_ * sizeof(DT_X));
        x_gm_.SetGlobalBuffer(x_base, core_data_len_);
        y_gm_.SetGlobalBuffer(y_base, core_data_len_);

        pipe_.InitBuffer(in_queue_, BUFFER_NUM, tile_len_ * sizeof(DT_X));
        pipe_.InitBuffer(out_queue_, BUFFER_NUM, tile_len_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp_buf_, tile_len_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp_buf2_, tile_len_ * sizeof(DT_X));
        pipe_.InitBuffer(tmp_buf3_, tile_len_ * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (core_data_len_ == 0) return;

        for (int32_t i = 0; i < tile_num_; i++) {
            process_len_ = (i == tile_num_ - 1) ? tail_len_ : tile_len_;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        LocalTensor<DT_X> in_local = in_queue_.AllocTensor<DT_X>();
        DataCopy(in_local, x_gm_[progress * tile_len_], process_len_);
        in_queue_.EnQue(in_local);
    }

    __aicore__ inline void Compute(int32_t progress) {
        LocalTensor<DT_X> in_local = in_queue_.DeQue<DT_X>();
        LocalTensor<DT_X> out_local = out_queue_.AllocTensor<DT_X>();
        LocalTensor<DT_X> tmp1 = tmp_buf_.Get<DT_X>();
        LocalTensor<DT_X> tmp2 = tmp_buf2_.Get<DT_X>();
        LocalTensor<DT_X> tmp3 = tmp_buf3_.Get<DT_X>();

        // y = x * exp(0.851*(x-|x|)) / (1 + exp(-1.702*|x|))

        // tmp1 = |x|
        Abs(tmp1, in_local, process_len_);

        // tmp2 = x - |x|
        Sub(tmp2, in_local, tmp1, process_len_);

        // tmp2 = 0.851 * (x - |x|)
        Muls(tmp2, tmp2, static_cast<DT_X>(ATTR_HALF), process_len_);

        // tmp2 = exp(0.851 * (x - |x|))
        Exp(tmp2, tmp2, process_len_);

        // tmp3 = x * exp(0.851 * (x - |x|))  ← 分子
        Mul(tmp3, in_local, tmp2, process_len_);

        // tmp1 = -1.702 * |x|
        Muls(tmp1, tmp1, static_cast<DT_X>(-ATTR), process_len_);

        // tmp1 = exp(-1.702 * |x|)
        Exp(tmp1, tmp1, process_len_);

        // tmp1 = 1 + exp(-1.702 * |x|)  ← 分母
        Adds(tmp1, tmp1, static_cast<DT_X>(1.0f), process_len_);

        // out_local = 分子 / 分母
        Div(out_local, tmp3, tmp1, process_len_);

        out_queue_.EnQue(out_local);
        in_queue_.FreeTensor(in_local);
        tmp_buf_.FreeTensor(tmp1);
        tmp_buf2_.FreeTensor(tmp2);
        tmp_buf3_.FreeTensor(tmp3);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        LocalTensor<DT_X> out_local = out_queue_.DeQue<DT_X>();
        DataCopy(y_gm_[progress * tile_len_], out_local, process_len_);
        out_queue_.FreeTensor(out_local);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> in_queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> out_queue_;
    TBuf<QuePosition::VECCALC> tmp_buf_;
    TBuf<QuePosition::VECCALC> tmp_buf2_;
    TBuf<QuePosition::VECCALC> tmp_buf3_;
    GlobalTensor<DT_X> x_gm_;
    GlobalTensor<DT_X> y_gm_;
    uint32_t core_data_len_ = 0;
    uint32_t data_offset_ = 0;
    uint32_t tile_len_ = 0;
    uint32_t tile_num_ = 0;
    uint32_t tail_len_ = 0;
    uint32_t process_len_ = 0;
};

template <typename T>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<T> op;
    op.Init(x, y, tiling_data);
    op.Process();
}