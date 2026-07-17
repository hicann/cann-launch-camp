// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_INPUT_X>
class KernelGelu {
public:
    using TYPE_X = DT_INPUT_X;
    using TYPE_Y = DT_INPUT_X;

    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x,
                                GM_ADDR output,
                                uint32_t small_core_data_num,
                                uint32_t big_core_data_num,
                                uint32_t final_big_tile_num,
                                uint32_t final_small_tile_num,
                                uint32_t tile_data_num,
                                uint32_t small_tail_data_num,
                                uint32_t big_tail_data_num,
                                uint32_t tail_block_num) {
        const uint32_t core_num = AscendC::GetBlockIdx();
        uint32_t global_buffer_index = big_core_data_num * core_num;

        this->tileDataNum = tile_data_num;
        this->coreDataNum = (core_num < tail_block_num) ? big_core_data_num : small_core_data_num;
        this->tileNum = (core_num < tail_block_num) ? final_big_tile_num : final_small_tile_num;
        this->tailDataNum = (core_num < tail_block_num) ? big_tail_data_num : small_tail_data_num;

        if (core_num >= tail_block_num) {
            global_buffer_index -= (big_core_data_num - small_core_data_num) * (core_num - tail_block_num);
        }

        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X *>(input_x) + global_buffer_index, this->coreDataNum);
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_Y *>(output) + global_buffer_index, this->coreDataNum);

        this->pipe.InitBuffer(this->inQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        this->pipe.InitBuffer(this->outQueue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            const uint32_t current_tile_size = (i + 1 == this->tileNum) ? this->tailDataNum : this->tileDataNum;
            this->CopyIn(i, current_tile_size);
            this->Compute(current_tile_size);
            this->CopyOut(i, current_tile_size);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;

    const TYPE_X inv_sqrt2 = static_cast<TYPE_X>(0.70710678);

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_tile_size) {
        AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, this->xGm[progress * this->tileDataNum], current_tile_size);
        this->inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t current_tile_size) {
        AscendC::LocalTensor<TYPE_X> xLocal = this->inQueue.template DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template AllocTensor<TYPE_Y>();
        AscendC::Muls(yLocal, xLocal, this->inv_sqrt2, current_tile_size);
        AscendC::Erf(yLocal, yLocal, current_tile_size);
        AscendC::Adds(yLocal, yLocal, static_cast<TYPE_Y>(1.0f), current_tile_size);
        AscendC::Muls(yLocal, yLocal, static_cast<TYPE_Y>(0.5f), current_tile_size);
        AscendC::Mul(yLocal, yLocal, xLocal, current_tile_size);

        this->outQueue.EnQue(yLocal);
        this->inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_tile_size) {
        AscendC::LocalTensor<TYPE_Y> yLocal = this->outQueue.template DeQue<TYPE_Y>();
        AscendC::DataCopy(this->yGm[progress * this->tileDataNum], yLocal, current_tile_size);
        this->outQueue.FreeTensor(yLocal);
    }
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x,
            output,
            tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,
            tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}