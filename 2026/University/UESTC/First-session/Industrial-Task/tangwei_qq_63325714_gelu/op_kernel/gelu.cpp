#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_LENGTH = 4096;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR inputX,
                                GM_ADDR output,
                                uint32_t length,
                                uint32_t blockNum,
                                uint32_t tileLength) {
        if (length == 0) {
            length = 1;
        }
        if (blockNum == 0) {
            blockNum = 1;
        }
        if (tileLength == 0 || tileLength > MAX_TILE_LENGTH) {
            tileLength = MAX_TILE_LENGTH;
        }

        this->length = length;
        this->blockNum = blockNum;
        this->tileLength = tileLength;
        this->blockIdx = AscendC::GetBlockIdx();

        uint32_t alignElements = 32 / sizeof(DT_INPUT_X);
        uint32_t totalBlocks = (this->length + alignElements - 1) / alignElements;
        uint32_t blocksPerCore = totalBlocks / this->blockNum;
        uint32_t remBlocks = totalBlocks % this->blockNum;

        if (this->blockIdx < remBlocks) {
            this->coreLength = (blocksPerCore + 1) * alignElements;
            this->coreOffset = this->blockIdx * this->coreLength;
        } else {
            this->coreLength = blocksPerCore * alignElements;
            this->coreOffset = this->blockIdx * this->coreLength + remBlocks * alignElements;
        }

        if (this->coreOffset + this->coreLength > this->length) {
            this->coreLength = this->length - this->coreOffset;
        }

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)inputX + this->coreOffset, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + this->coreOffset, this->coreLength);

        this->tileNum = (this->coreLength + this->tileLength - 1) / this->tileLength;

        pipe.InitBuffer(inQueueX, BUFFER_NUM, MAX_TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, MAX_TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(tmpBuffer, BUFFER_NUM, MAX_TILE_LENGTH * sizeof(DT_INPUT_X));
        pipe.InitBuffer(bQueue, BUFFER_NUM, MAX_TILE_LENGTH * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }

        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t offset = i * this->tileLength;
            uint32_t remain = this->coreLength - offset;
            uint32_t processLength = remain > this->tileLength ? this->tileLength : remain;
            uint32_t calcLength = (processLength + (32 / sizeof(DT_INPUT_X)) - 1) / (32 / sizeof(DT_INPUT_X)) * (32 / sizeof(DT_INPUT_X));

            AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
            AscendC::DataCopy(xLocal, xGm[offset], calcLength);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<DT_INPUT_X> xCompute = inQueueX.DeQue<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> tmp = tmpBuffer.AllocTensor<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> xClamped = bQueue.AllocTensor<DT_INPUT_X>();

            DT_INPUT_X cScale = static_cast<DT_INPUT_X>(0.7071067811865476);
            DT_INPUT_X cOne = static_cast<DT_INPUT_X>(1.0);
            DT_INPUT_X cHalf = static_cast<DT_INPUT_X>(0.5);
            DT_INPUT_X cThreshPos = static_cast<DT_INPUT_X>(6.0);
            DT_INPUT_X cThreshNeg = static_cast<DT_INPUT_X>(-6.0);

            AscendC::Mins(xClamped, xCompute, cThreshPos, calcLength);
            AscendC::Maxs(xClamped, xClamped, cThreshNeg, calcLength);

            AscendC::Muls(tmp, xClamped, cScale, calcLength);
            AscendC::Erf(tmp, tmp, calcLength);
            AscendC::Adds(tmp, tmp, cOne, calcLength);
            AscendC::Muls(tmp, tmp, cHalf, calcLength);
            AscendC::Mul(yLocal, xCompute, tmp, calcLength);

            outQueueY.EnQue(yLocal);
            bQueue.FreeTensor(xClamped);
            tmpBuffer.FreeTensor(tmp);
            inQueueX.FreeTensor(xCompute);

            AscendC::LocalTensor<DT_INPUT_X> yOut = outQueueY.DeQue<DT_INPUT_X>();
            AscendC::DataCopy(yGm[offset], yOut, processLength);
            outQueueY.FreeTensor(yOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> tmpBuffer;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> bQueue;

    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> yGm;

    uint32_t length;
    uint32_t blockNum;
    uint32_t blockIdx;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t coreLength;
    uint32_t coreOffset;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR inputX, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(inputX, output, tilingData.length, tilingData.blockNum, tilingData.tileLength);
    op.Process();
}