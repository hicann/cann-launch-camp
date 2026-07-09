// Kernel侧核函数实现
#include "kernel_operator.h"

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint64_t totalLength,
        uint32_t usedCoreNum,
        uint32_t alignElemNum,
        uint64_t baseBlockNum,
        uint32_t tailBlockNum,
        uint32_t tileLength,
        uint32_t smallMode
    ) {
        this->totalLength = totalLength;
        this->usedCoreNum = usedCoreNum;
        this->alignElemNum = alignElemNum;
        this->baseBlockNum = baseBlockNum;
        this->tailBlockNum = tailBlockNum;
        this->tileLength = tileLength;
        this->smallMode = smallMode;

        if (this->totalLength == 0 || this->usedCoreNum == 0) {
            this->coreStart = 0;
            this->coreLength = 0;
            return;
        }

        uint32_t coreId = GetBlockIdx();

        if (coreId >= this->usedCoreNum) {
            this->coreStart = 0;
            this->coreLength = 0;
            return;
        }

        if (this->smallMode == 1) {
            // 小 shape：只用一个 core 处理全部数据
            this->coreStart = 0;
            this->coreLength = this->totalLength;
        } else {
            // 大 shape：按 32B block 分配，保证各 core 起始地址尽量 32B 对齐
            uint64_t coreBlockStart = 0;
            uint64_t coreBlockNum = 0;

            if (coreId < this->tailBlockNum) {
                coreBlockNum = this->baseBlockNum + 1;
                coreBlockStart = static_cast<uint64_t>(coreId) * coreBlockNum;
            } else {
                coreBlockNum = this->baseBlockNum;
                coreBlockStart =
                    static_cast<uint64_t>(this->tailBlockNum) * (this->baseBlockNum + 1) +
                    static_cast<uint64_t>(coreId - this->tailBlockNum) * this->baseBlockNum;
            }

            this->coreStart = coreBlockStart * static_cast<uint64_t>(this->alignElemNum);

            uint64_t coverLength = coreBlockNum * static_cast<uint64_t>(this->alignElemNum);

            if (this->coreStart >= this->totalLength) {
                this->coreLength = 0;
            } else {
                uint64_t remainLength = this->totalLength - this->coreStart;
                this->coreLength = coverLength < remainLength ? coverLength : remainLength;
            }
        }

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));

        // 优化点：
        // 不再申请 denBuf。
        // Compute 中直接复用 yLocal 作为分母中间变量。
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0 || this->tileLength == 0) {
            return;
        }

        uint64_t offset = 0;

        while (offset < this->coreLength) {
            uint32_t currentTileLength =
                static_cast<uint32_t>(
                    ((this->coreLength - offset) > this->tileLength)
                        ? this->tileLength
                        : (this->coreLength - offset)
                );

            CopyIn(offset, currentTileLength);
            Compute(currentTileLength);
            CopyOut(offset, currentTileLength);

            offset += currentTileLength;
        }
    }

private:
    __aicore__ inline bool IsAlignedLength(uint32_t length) {
        return (length % this->alignElemNum) == 0;
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t currentTileLength) {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();

        if (IsAlignedLength(currentTileLength)) {
            DataCopy(xLocal, xGm[offset], currentTileLength);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentTileLength * sizeof(DT_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPadExtParams<DT_X> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_X>(0);

            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileLength) {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();

        // FastGelu 原公式：
        // y = x * exp(0.851 * (x - abs(x))) / (1 + exp(-1.702 * abs(x)))
        //
        // 等价化简：
        // y = x / (1 + exp(-1.702 * x))
        //
        // 本版本优化：
        // 直接复用 yLocal 作为分母中间变量，减少 denBuf。
        //
        // yLocal = -1.702 * x
        // yLocal = exp(yLocal)
        // yLocal = yLocal + 1
        // yLocal = x / yLocal

        Muls(yLocal, xLocal, static_cast<DT_X>(-1.702), currentTileLength);

        Exp(yLocal, yLocal, currentTileLength);

        Adds(yLocal, yLocal, static_cast<DT_X>(1.0), currentTileLength);

        Div(yLocal, xLocal, yLocal, currentTileLength);

        outQueueY.EnQue<DT_X>(yLocal);

        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t currentTileLength) {
        LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();

        if (IsAlignedLength(currentTileLength)) {
            DataCopy(yGm[offset], yLocal, currentTileLength);
        } else {
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentTileLength * sizeof(DT_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPad(yGm[offset], yLocal, copyParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;

    uint64_t totalLength = 0;
    uint32_t usedCoreNum = 1;
    uint32_t alignElemNum = 1;
    uint64_t baseBlockNum = 0;
    uint32_t tailBlockNum = 0;
    uint32_t tileLength = 16384;
    uint32_t smallMode = 0;

    uint64_t coreStart = 0;
    uint64_t coreLength = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);

    KernelFastGelu<DT_X> op;

    op.Init(
        x,
        y,
        tiling_data.totalLength,
        tiling_data.usedCoreNum,
        tiling_data.alignElemNum,
        tiling_data.baseBlockNum,
        tiling_data.tailBlockNum,
        tiling_data.tileLength,
        tiling_data.smallMode
    );

    op.Process();
}