// Kernel侧核函数实现：小输入单核单发 + 大输入多核流水（双 kernel）
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

// ===================== 大输入：多核 + tile 流水 =====================
// 与基线完全一致（多核负载均衡、DataCopy 32B 对齐、Exp/Div 四算子）。
// 大输入是带宽主导，这套已验证竞争力，原样保留，零回归。
template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tiling_data) {
        uint32_t coreNum = AscendC::GetBlockIdx();
        this->tileDataNum = tiling_data.tileDataNum;
        if (coreNum < tiling_data.tailBlockNum) {
            this->coreDataNum = tiling_data.bigCoreDataNum;
            this->tileNum = tiling_data.finalBigTileNum;
            this->tailDataNum = tiling_data.bigTailDataNum;
            offset = tiling_data.bigCoreDataNum * coreNum;
        } else {
            this->coreDataNum = tiling_data.smallCoreDataNum;
            this->tileNum = tiling_data.finalSmallTileNum;
            this->tailDataNum = tiling_data.smallTailDataNum;
            offset = tiling_data.bigCoreDataNum * tiling_data.tailBlockNum + tiling_data.smallCoreDataNum * (coreNum - tiling_data.tailBlockNum);
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
    uint32_t offset, tileDataNum, tileNum, tailDataNum, coreDataNum, processDataNum;
};

// ===================== 小输入：单核单发 =====================
// host 保证 block_dim=1。整块一次 DataCopyPad 进 → 算 → 一次出，无 tile 循环。
// 目的：把 launch/同步这些固定开销只付一次（小输入是固定开销主导，不是算力主导）。
// 用 TQue（而非裸 TBuf）承担 MTE2->V / V->MTE3 的跨 pipe 同步，正确性无需手动 PipeBarrier。
template <class DT_X>
class KernelFastGeluSmall {
public:
    __aicore__ inline KernelFastGeluSmall() {}
    // totalLength: 实际元素数（搬运/计算长度）; bufDataNum: 向上对齐到 32B 的元素数（buffer 尺寸）
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t bufDataNum) {
        this->totalLength = totalLength;
        xGm.SetGlobalBuffer((__gm__ DT_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y, totalLength);
        pipe.InitBuffer(inQueueX, 1, bufDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, 1, bufDataNum * sizeof(DT_X));
        pipe.InitBuffer(buf1, bufDataNum * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->totalLength == 0) {
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }
private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(this->totalLength * sizeof(DT_X));  // 精确字节数，可非 32B 对齐
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> buf1Local = buf1.Get<DT_X>();

        AscendC::Muls(buf1Local, xLocal, (DT_X)-1.702f, this->totalLength);  // -1.702x
        AscendC::Exp(buf1Local, buf1Local, this->totalLength);               // e^{-1.702x}
        AscendC::Adds(buf1Local, buf1Local, (DT_X)1.0f, this->totalLength);  // 1 + e^{-1.702x}
        AscendC::Div(yLocal, xLocal, buf1Local, this->totalLength);          // x / (...)
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(this->totalLength * sizeof(DT_X));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[0], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf1;  // 中间计算缓冲
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t totalLength;
};

template <typename DT_X>
 __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    if (tiling_data.isSmallShape == 1) {
        KernelFastGeluSmall<DT_X> op;
        op.Init(x, y, tiling_data.totalLength, tiling_data.tileDataNum);
        op.Process();
        return;
    }
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
