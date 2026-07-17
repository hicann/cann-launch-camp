// Kernel侧核函数实现
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

constexpr int32_t BUFFER_NUM = 2;

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                 const LessEqualTilingData &tiling)
    {
        uint32_t blockIdx   = AscendC::GetBlockIdx();
        uint32_t blockDim   = AscendC::GetBlockNum();
        this->totalLength   = tiling.totalLength;
        this->tileNum       = tiling.tileNum;
        this->blockLength   = this->totalLength / blockDim;
        this->tileLength    = this->blockLength / this->tileNum / BUFFER_NUM;

        uint32_t offset = this->blockLength * blockIdx;

        x1Gm.SetGlobalBuffer((__gm__ DT_X1*)x1 + offset, this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1*)x2 + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + offset, this->blockLength);

        pipe.InitBuffer(inQ1, BUFFER_NUM, this->tileLength * sizeof(DT_X1));
        pipe.InitBuffer(inQ2, BUFFER_NUM, this->tileLength * sizeof(DT_X1));
        pipe.InitBuffer(outQ, BUFFER_NUM, this->tileLength * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        int32_t loop = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loop; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        auto x1L = inQ1.template AllocTensor<DT_X1>();
        auto x2L = inQ2.template AllocTensor<DT_X1>();
        AscendC::DataCopy(x1L, x1Gm[progress * this->tileLength],
                           this->tileLength);
        AscendC::DataCopy(x2L, x2Gm[progress * this->tileLength],
                           this->tileLength);
        inQ1.EnQue(x1L);
        inQ2.EnQue(x2L);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        auto x1L = inQ1.template DeQue<DT_X1>();
        auto x2L = inQ2.template DeQue<DT_X1>();
        auto yL  = outQ.template AllocTensor<uint8_t>();
        AscendC::Compare(yL, x1L, x2L, AscendC::CMPMODE::LE,
                          this->tileLength);
        outQ.EnQue<uint8_t>(yL);
        inQ1.FreeTensor(x1L);
        inQ2.FreeTensor(x2L);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        auto yL = outQ.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yL,
                           this->tileLength);
        outQ.FreeTensor(yL);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQ1, inQ2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQ;
    AscendC::GlobalTensor<DT_X1>     x1Gm, x2Gm;
    AscendC::GlobalTensor<uint8_t>   yGm;
    uint32_t totalLength, blockLength, tileNum, tileLength;
};

template <typename DT_X1>
 __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}
