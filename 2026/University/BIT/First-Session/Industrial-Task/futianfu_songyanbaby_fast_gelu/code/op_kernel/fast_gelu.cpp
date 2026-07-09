// Kernel侧核函数实现 - 优化版
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

constexpr int32_t BUFFER_NUM = 2;    // 双缓冲

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t coreDataNum,
                                uint32_t tileNum, uint32_t tileDataNum,
                                uint32_t tailDataNum, uint32_t tmpSize,
                                uint32_t globalBufferOffset)
    {
        this->coreDataNum = coreDataNum;
        this->tileNum = tileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;

        // 设置GlobalTensor: 大/小核数据量不同, 偏移量由调用侧精确计算
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + globalBufferOffset, coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + globalBufferOffset, coreDataNum);

        // 分配Pipe缓冲区: 双缓冲输入/输出 + FasterGelu临时buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(tmpBuf, tmpSize);
    }

    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum;

        // 处理前 N-1 个完整tile (无分支)
        for (uint32_t i = 0; i < loopCount - 1; i++) {
            this->processLen = this->tileDataNum;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

        // 处理最后一个tile (可能是尾块)
        if (loopCount > 0) {
            uint32_t lastIdx = loopCount - 1;
            this->processLen = this->tailDataNum;
            CopyIn(lastIdx);
            Compute(lastIdx);
            CopyOut(lastIdx);
        }
    }

private:
    // 从GM搬入数据到Local (使用32B对齐块搬移)
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processLen);
        inQueueX.EnQue(xLocal);
    }

    // 调用FasterGelu计算
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

        // 使用AscendC高阶API FasterGelu完成激活函数计算
        AscendC::FasterGelu(yLocal, xLocal, tmpLocal, this->processLen);

        outQueueY.EnQue<DT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // 将结果从Local搬出到GM
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processLen;
};

template <typename DT_X>
 __global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelFastGelu<DT_X> op;
    uint32_t coreIdx = AscendC::GetBlockIdx();

    // 大/小核分发及全局偏移计算:
    // 前 tailBlockNum 个核为大核(多处理32B数据), 其余为小核
    uint32_t globalBufferOffset;
    if (coreIdx < tilingData.tailBlockNum) {
        globalBufferOffset = coreIdx * tilingData.bigCoreDataNum;
        op.Init(x, y,
                tilingData.bigCoreDataNum,
                tilingData.bigTileNum,
                tilingData.tileDataNum,
                tilingData.bigTailDataNum,
                tilingData.tmpSize,
                globalBufferOffset);
    } else {
        // 小核偏移 = 所有大核占用 + 之前的小核占用
        globalBufferOffset = tilingData.tailBlockNum * tilingData.bigCoreDataNum
                           + (coreIdx - tilingData.tailBlockNum) * tilingData.smallCoreDataNum;
        op.Init(x, y,
                tilingData.smallCoreDataNum,
                tilingData.smallTileNum,
                tilingData.tileDataNum,
                tilingData.smallTailDataNum,
                tilingData.tmpSize,
                globalBufferOffset);
    }
    op.Process();
}
