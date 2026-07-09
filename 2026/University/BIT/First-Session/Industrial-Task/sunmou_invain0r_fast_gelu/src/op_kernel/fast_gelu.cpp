// Kernel侧核函数实现
// 数学优化: y = x * exp(0.851*(x-|x|)) / (1+exp(-1.702*|x|))
//                = x * sigmoid(1.702 * x)   (恒等变换, 精确等价)
// 将 9 次运算(Abs+Sub+Muls+Exp+Mul+Muls+Exp+Adds+Div) 降为 3 次 (Muls+Sigmoid+Mul)
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;   // 双缓冲流水
constexpr uint32_t BLOCK_SIZE = 32; // 数据对齐块大小（字节）

template<typename DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tilingData)
    {
        uint32_t coreIdx = GetBlockIdx();
        uint32_t totalLength = tilingData.length;
        uint32_t coreNum = tilingData.blockNum;

        this->tileDataNum = tilingData.tileDataNum;

        // 恢复与原实现一致的块级划分，保证数据边界不变
        // 1) 计算 32 字节块总数
        uint32_t typeLength = sizeof(DT_X);
        uint32_t inputBytes = totalLength * typeLength;
        uint32_t alignedBytes = ((inputBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
        uint32_t totalBlocks = alignedBytes / BLOCK_SIZE;
        uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;

        // 2) 块级均匀分配：每个核得到 baseBlocks 或 baseBlocks+1 个块
        uint32_t baseBlocks = totalBlocks / coreNum;
        uint32_t remainBlocks = totalBlocks % coreNum;

        // 3) 计算当前核在块空间中的起始位置和块数
        //    前 remainBlocks 个核各多 1 个块
        uint32_t startBlock = coreIdx * baseBlocks + (coreIdx < remainBlocks ? coreIdx : remainBlocks);
        uint32_t numBlocks = baseBlocks + (coreIdx < remainBlocks ? 1 : 0);

        // 4) 将块索引转换为元素索引
        uint32_t start = startBlock * elementsPerBlock;
        this->coreDataNum = numBlocks * elementsPerBlock;

        // 5) 保护：确保不超过实际数据长度（最后一个核可能超出）
        if (start + this->coreDataNum > totalLength) {
            this->coreDataNum = totalLength - start;
        }

        // 计算 tile 数量和最后一个 tile 的大小
        this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
        this->tailDataNum = this->coreDataNum % this->tileDataNum;
        this->tailDataNum = (this->tailDataNum == 0) ? this->tileDataNum : this->tailDataNum;

        // 设置全局张量
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, this->coreDataNum);

        // 分配缓冲区: 输入双缓冲 + 输出双缓冲 = 4 slots
        // 优化: 复用 yLocal 作为 Muls/Sigmoid 的临时缓冲区，消除 tmpBuf
        // 从 5 slots 降到 4 slots，tile 大小提升 25%
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataNum == 0) {
            return;
        }

        // 预加载第一个tile
        this->processDataNum = this->tileDataNum;
        CopyIn(0);

        for (uint32_t i = 0; i < this->tileNum - 1; i++) {
            this->processDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            Compute(i);
            uint32_t savedLen = this->processDataNum;
            this->processDataNum = (i + 1 == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn(i + 1);
            CopyOut(i, savedLen);
        }

        // 最后一个tile
        this->processDataNum = this->tailDataNum;
        Compute(this->tileNum - 1);
        CopyOut(this->tileNum - 1, this->processDataNum);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        // 复用 yLocal 作为临时缓冲区，消除单独的 tmpBuf
        // 流程: yLocal = 1.702*x → sigmoid(yLocal) → Mul(x, yLocal) → yLocal
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        // y = x * sigmoid(1.702 * x)
        // 只需3步, 代替原来的9步: Abs→Sub→Muls→Exp→Mul→Muls→Exp→Adds→Div
        Muls(yLocal, xLocal, (DT_X)1.702f, this->processDataNum);   // yLocal = 1.702 * x (临时值)
        Sigmoid(yLocal, yLocal, this->processDataNum);               // yLocal = sigmoid(1.702*x)
        Mul(yLocal, xLocal, yLocal, this->processDataNum);           // yLocal = x * sigmoid(...)

        // xLocal已用完, 提前释放让下一个CopyIn尽早启动DMA
        inQueueX.FreeTensor(xLocal);

        outQueueY.EnQue<DT_X>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}