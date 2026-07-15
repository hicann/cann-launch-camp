#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr uint32_t TILE_SIZE = 4096;
constexpr uint32_t QUEUE_DEPTH = 2;
constexpr float    SQRT2_OVER_PI = 0.7978845608028654f;
constexpr float    COEFF_C3      = 0.044714f;
constexpr float    COEFF_C5      = 0.000806f;

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t totalLen,
        uint32_t blkLen) {
        this->totalLength = totalLen;
        this->blockLen = blkLen;

        uint32_t coreIdx = GetBlockIdx();
        this->coreStart = coreIdx * blockLen;

        uint32_t coreEnd = this->coreStart + blockLen;
        if (coreEnd > totalLength) {
            coreEnd = totalLength;
        }

        if (this->coreStart >= totalLength) {
            this->dataLen = 0;
        } else {
            this->dataLen = coreEnd - this->coreStart;
        }

        inputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x, totalLength);
        outputGlobal.SetGlobalBuffer((__gm__ DT_INPUT_X*)output, totalLength);

        pipe.InitBuffer(inQ, QUEUE_DEPTH, TILE_SIZE * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQ, QUEUE_DEPTH, TILE_SIZE * sizeof(DT_INPUT_X));
        pipe.InitBuffer(bufA, TILE_SIZE * sizeof(DT_INPUT_X));
        pipe.InitBuffer(bufB, TILE_SIZE * sizeof(DT_INPUT_X));
    }

    __aicore__ inline void Process() {
        if (dataLen == 0) return;

        DT_INPUT_X half  = static_cast<DT_INPUT_X>(0.5);
        DT_INPUT_X one   = static_cast<DT_INPUT_X>(1.0);
        DT_INPUT_X c1    = static_cast<DT_INPUT_X>(SQRT2_OVER_PI);
        DT_INPUT_X c3    = static_cast<DT_INPUT_X>(COEFF_C3);
        DT_INPUT_X c5    = static_cast<DT_INPUT_X>(COEFF_C5);

        LocalTensor<DT_INPUT_X> t1 = bufA.Get<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> t2 = bufB.Get<DT_INPUT_X>();

        for (uint32_t off = 0; off < dataLen; off += TILE_SIZE) {
            uint32_t curLen = TILE_SIZE;
            if (off + TILE_SIZE > dataLen) {
                curLen = dataLen - off;
            }
            uint32_t globalOff = coreStart + off;

            LocalTensor<DT_INPUT_X> inLocal = inQ.AllocTensor<DT_INPUT_X>();
            DataCopy(inLocal, inputGlobal[globalOff], curLen);
            inQ.EnQue(inLocal);

            inLocal = inQ.DeQue<DT_INPUT_X>();
            LocalTensor<DT_INPUT_X> outLocal = outQ.AllocTensor<DT_INPUT_X>();

            Mul(t1, inLocal, inLocal, curLen);
            Mul(t2, t1, inLocal, curLen);
            Mul(t1, t1, t2, curLen);
            Muls(t1, t1, c5, curLen);
            Muls(t2, t2, c3, curLen);
            Add(t2, t2, t1, curLen);
            Add(t2, inLocal, t2, curLen);
            Muls(t2, t2, c1, curLen);
            Tanh(t2, t2, curLen);
            Adds(t2, t2, one, curLen);
            Muls(t1, inLocal, half, curLen);
            Mul(outLocal, t1, t2, curLen);

            outQ.EnQue(outLocal);
            inQ.FreeTensor(inLocal);

            outLocal = outQ.DeQue<DT_INPUT_X>();
            DataCopy(outputGlobal[globalOff], outLocal, curLen);
            outQ.FreeTensor(outLocal);
        }
    }

private:
    uint32_t totalLength;
    uint32_t blockLen;
    uint32_t coreStart;
    uint32_t dataLen;

    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQ;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outQ;
    TBuf<QuePosition::VECCALC> bufA;
    TBuf<QuePosition::VECCALC> bufB;

    GlobalTensor<DT_INPUT_X> inputGlobal;
    GlobalTensor<DT_INPUT_X> outputGlobal;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(
        input_x,
        output,
        tiling_data.length,
        tiling_data.blockLength
    );
    op.Process();
}