
#include "kernel_operator.h"
#include "clip_by_value_tiling.h"
#include "tiling_key_clip_by_value.h"
using namespace AscendC;

// 娴佹按绾垮弻缂撳啿鏁伴噺
constexpr int32_t PIPE_BUFFER_CNT = 2;
// 鍚戦噺鎿嶄綔鍥哄畾32瀛楄妭瀵归綈
constexpr uint32_t VEC_ALIGN_BYTE = 32;

template <class DT_X>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

  
    __aicore__ inline void Init(GM_ADDR x_gm, GM_ADDR min_gm, GM_ADDR max_gm, GM_ADDR y_gm, ClipByValueTilingData tile_cfg) {
       
        total_elem_ = tile_cfg.length;
        core_block_len_ = tile_cfg.blockLength;
        tile_chunk_len_ = tile_cfg.tileLength;
        min_scalar_flag_ = static_cast<bool>(tile_cfg.isScalarMin);
        max_scalar_flag_ = static_cast<bool>(tile_cfg.isScalarMax);
        elem_per_vec_ = VEC_ALIGN_BYTE / sizeof(DT_X);

      
        uint32_t core_id = GetBlockIdx();
        gm_data_offset_ = core_id * core_block_len_;

       
        if (gm_data_offset_ >= total_elem_) {
            core_process_elem_ = 0;
            return;
        }

        
        core_process_elem_ = core_block_len_;
        if (gm_data_offset_ + core_process_elem_ > total_elem_) {
            core_process_elem_ = total_elem_ - gm_data_offset_;
        }

        
        x_global_buf_.SetGlobalBuffer((__gm__ DT_X*)x_gm + gm_data_offset_, core_process_elem_);
        y_global_buf_.SetGlobalBuffer((__gm__ DT_X*)y_gm + gm_data_offset_, core_process_elem_);

        if (min_scalar_flag_) {
            GlobalTensor<DT_X> min_scalar_gm;
            min_scalar_gm.SetGlobalBuffer((__gm__ DT_X*)min_gm);
            clip_min_val_ = min_scalar_gm.GetValue(0);
        } else {
            min_global_buf_.SetGlobalBuffer((__gm__ DT_X*)min_gm + gm_data_offset_, core_process_elem_);
        }

       
        if (max_scalar_flag_) {
            GlobalTensor<DT_X> max_scalar_gm;
            max_scalar_gm.SetGlobalBuffer((__gm__ DT_X*)max_gm);
            clip_max_val_ = max_scalar_gm.GetValue(0);
        } else {
            max_global_buf_.SetGlobalBuffer((__gm__ DT_X*)max_gm + gm_data_offset_, core_process_elem_);
        }

       
        uint32_t align_tile_size = AlignUp(tile_chunk_len_, elem_per_vec_);
        uint32_t tile_byte_size = align_tile_size * sizeof(DT_X);
        pipe_stream_.InitBuffer(in_x_queue_, PIPE_BUFFER_CNT, tile_byte_size);
        pipe_stream_.InitBuffer(out_y_queue_, PIPE_BUFFER_CNT, tile_byte_size);

        if (!min_scalar_flag_) {
            pipe_stream_.InitBuffer(in_min_queue_, PIPE_BUFFER_CNT, tile_byte_size);
        }
        if (!max_scalar_flag_) {
            pipe_stream_.InitBuffer(in_max_queue_, PIPE_BUFFER_CNT, tile_byte_size);
        }
    }

    
    __aicore__ inline void Process() {
        if (core_process_elem_ == 0) {
            return;
        }
        uint32_t tile_total_num = (core_process_elem_ + tile_chunk_len_ - 1) / tile_chunk_len_;
        for (uint32_t tile_idx = 0; tile_idx < tile_total_num; tile_idx++) {
            CopyDataIn(tile_idx);
            ComputeClip(tile_idx);
            CopyDataOut(tile_idx);
        }
    }

private:
  
    __aicore__ inline uint32_t AlignUp(uint32_t num, uint32_t align) {
        return ((num + align - 1) / align) * align;
    }

    
    __aicore__ inline uint32_t GetTileRealElem(uint32_t tile_idx) {
        uint32_t tile_start = tile_idx * tile_chunk_len_;
        uint32_t remain_elem = core_process_elem_ - tile_start;
        return (remain_elem < tile_chunk_len_) ? remain_elem : tile_chunk_len_;
    }

   
    __aicore__ inline void CopyDataIn(uint32_t tile_idx) {
        uint32_t real_elem = GetTileRealElem(tile_idx);
        uint32_t align_elem = AlignUp(real_elem, elem_per_vec_);
        uint32_t tile_gm_off = tile_idx * tile_chunk_len_;

        LocalTensor<DT_X> x_local = in_x_queue_.AllocTensor<DT_X>();
        DataCopy(x_local, x_global_buf_[tile_gm_off], align_elem);
        in_x_queue_.EnQue<DT_X>(x_local);

        if (!min_scalar_flag_) {
            LocalTensor<DT_X> min_local = in_min_queue_.AllocTensor<DT_X>();
            DataCopy(min_local, min_global_buf_[tile_gm_off], align_elem);
            in_min_queue_.EnQue<DT_X>(min_local);
        }
        if (!max_scalar_flag_) {
            LocalTensor<DT_X> max_local = in_max_queue_.AllocTensor<DT_X>();
            DataCopy(max_local, max_global_buf_[tile_gm_off], align_elem);
            in_max_queue_.EnQue<DT_X>(max_local);
        }
    }

    
    __aicore__ inline void ComputeClip(uint32_t tile_idx) {
        uint32_t real_elem = GetTileRealElem(tile_idx);
        uint32_t align_elem = AlignUp(real_elem, elem_per_vec_);
        LocalTensor<DT_X> x_local = in_x_queue_.DeQue<DT_X>();
        LocalTensor<DT_X> y_local = out_y_queue_.AllocTensor<DT_X>();

       
        if (min_scalar_flag_) {
            Maxs<DT_X>(y_local, x_local, clip_min_val_, align_elem);
        } else {
            LocalTensor<DT_X> min_local = in_min_queue_.DeQue<DT_X>();
            Max<DT_X>(y_local, x_local, min_local, align_elem);
            in_min_queue_.FreeTensor<DT_X>(min_local);
        }

       
        if (max_scalar_flag_) {
            Mins<DT_X>(y_local, y_local, clip_max_val_, align_elem);
        } else {
            LocalTensor<DT_X> max_local = in_max_queue_.DeQue<DT_X>();
            Min<DT_X>(y_local, y_local, max_local, align_elem);
            in_max_queue_.FreeTensor<DT_X>(max_local);
        }

        out_y_queue_.EnQue<DT_X>(y_local);
        in_x_queue_.FreeTensor<DT_X>(x_local);
    }

    __aicore__ inline void CopyDataOut(uint32_t tile_idx) {
        uint32_t real_elem = GetTileRealElem(tile_idx);
        uint32_t align_elem = AlignUp(real_elem, elem_per_vec_);
        uint32_t tile_gm_off = tile_idx * tile_chunk_len_;
        LocalTensor<DT_X> y_local = out_y_queue_.DeQue<DT_X>();
        DataCopy(y_global_buf_[tile_gm_off], y_local, align_elem);
        out_y_queue_.FreeTensor<DT_X>(y_local);
    }

private:
    TPipe pipe_stream_;
    TQue<TPosition::VECIN, PIPE_BUFFER_CNT>  in_x_queue_;
    TQue<TPosition::VECIN, PIPE_BUFFER_CNT>  in_min_queue_;
    TQue<TPosition::VECIN, PIPE_BUFFER_CNT>  in_max_queue_;
    TQue<TPosition::VECOUT, PIPE_BUFFER_CNT> out_y_queue_;

    GlobalTensor<DT_X> x_global_buf_;
    GlobalTensor<DT_X> min_global_buf_;
    GlobalTensor<DT_X> max_global_buf_;
    GlobalTensor<DT_X> y_global_buf_;

    DT_X clip_min_val_{};
    DT_X clip_max_val_{};

    uint32_t total_elem_ = 0;
    uint32_t core_block_len_ = 0;
    uint32_t tile_chunk_len_ = 0;
    uint32_t elem_per_vec_ = 0;
    uint32_t gm_data_offset_ = 0;
    uint32_t core_process_elem_ = 0;
    bool min_scalar_flag_ = false;
    bool max_scalar_flag_ = false;
};


template <typename DT_X>
 __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClipByValueTilingData);
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tile_info, tiling);
    KernelClipByValue<DT_X> clip_op;
    clip_op.Init(x, clip_value_min, clip_value_max, y, tile_info);
    clip_op.Process();
}