// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tiling_data) {
        uint32_t blockIdx = AscendC::GetBlockIdx();  // 当前核的块索引(coreID)，非核总数
        this->tileDataNum = tiling_data.tileDataNum;
        if (blockIdx < tiling_data.tailBlockNum) {
            this->coreDataNum = tiling_data.bigCoreDataNum;
            this->tileNum = tiling_data.finalBigTileNum;
            this->tailDataNum = tiling_data.bigTailDataNum;
            offset = static_cast<uint64_t>(tiling_data.bigCoreDataNum) * blockIdx;
        } else {
            this->coreDataNum = tiling_data.smallCoreDataNum;
            this->tileNum = tiling_data.finalSmallTileNum;
            this->tailDataNum = tiling_data.smallTailDataNum;
            offset = static_cast<uint64_t>(tiling_data.bigCoreDataNum) * tiling_data.tailBlockNum + static_cast<uint64_t>(tiling_data.smallCoreDataNum) * (blockIdx - tiling_data.tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + offset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + offset, coreDataNum);
        pipe.InitBuffer(inQueueX, 1, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, 1, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(buf1, tileDataNum * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        uint32_t processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              processDataNum = this->tailDataNum;
            }
            this -> processDataNum = processDataNum;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
      AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
      AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
      AscendC::LocalTensor<DT_X> buf1Local = buf1.Get<DT_X>();

      AscendC::Muls(buf1Local, xLocal, (DT_X)-1.702f, this->processDataNum);  // -1.702x
      AscendC::Exp(buf1Local, buf1Local, this->processDataNum);               // e^{-1.702x}
      AscendC::Adds(buf1Local, buf1Local, (DT_X)1.0f, this->processDataNum);  // 1 + e^{-1.702x}
      AscendC::Div(yLocal, xLocal, buf1Local, this->processDataNum);          // x / (...)
      inQueueX.FreeTensor(xLocal);
      outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();  
      AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
      outQueueY.FreeTensor(yLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf1;  // 中间计算缓冲
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint64_t offset;  // GM偏移用64位，防大张量越界
    uint32_t tileDataNum, tileNum, tailDataNum, coreDataNum, processDataNum;
};

template <typename DT_X>
 __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
