#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

template <class DT_INPUT_X>
class KernelGelu {
private:
    static constexpr uint32_t UB_SIZE = 192 * 1024;               // 910B UB 大小
    static constexpr uint32_t BUFFER_NUM = 2;                     // 双缓冲
    static constexpr uint32_t NUM_BUFS = BUFFER_NUM * 2 + 2;      // inputQ + outputQ + tmp1 + tmp2
    static constexpr uint32_t TILE_LENGTH = []() constexpr -> uint32_t {
        uint32_t elem_size = sizeof(DT_INPUT_X);
        uint32_t max_elem = (UB_SIZE * 9 / 10) / (NUM_BUFS * elem_size);
        max_elem = (max_elem / 32) * 32;   // 对齐到 32
        if (max_elem < 1) max_elem = 1;
        return max_elem;
        }();

    using T = DT_INPUT_X;

public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
        uint64_t length, uint64_t blockLength) {
        this->length = length;
        this->blockLength = blockLength;

        uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        this->start = blockIdx * blockLength;
        uint64_t end = this->start + blockLength;
        if (end > length) { end = length; }
        this->validLength = (this->start >= length) ? 0 : (end - this->start);

        inputGm.SetGlobalBuffer((__gm__ T*)input_x, static_cast<uint32_t>(length));
        outputGm.SetGlobalBuffer((__gm__ T*)output, static_cast<uint32_t>(length));

        pipe.InitBuffer(inputQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmpBuf1, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(tmpBuf2, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (validLength == 0) return;

        uint32_t nTiles = static_cast<uint32_t>((validLength + TILE_LENGTH - 1) / TILE_LENGTH);

        if (nTiles == 1) {
            CopyIn(static_cast<uint32_t>(start), static_cast<uint32_t>(validLength));
            Compute(static_cast<uint32_t>(validLength));
            CopyOut(static_cast<uint32_t>(start), static_cast<uint32_t>(validLength));
            return;
        }

        CopyIn(static_cast<uint32_t>(start), TILE_LENGTH);

        for (uint32_t i = 0; i + 1 < nTiles; ++i) {
            uint64_t curOff = start + static_cast<uint64_t>(i) * TILE_LENGTH;
            uint64_t nextOff = start + static_cast<uint64_t>(i + 1) * TILE_LENGTH;
            uint32_t nextLen = (i + 2 == nTiles) ?
                static_cast<uint32_t>(validLength - (static_cast<uint64_t>(i + 1) * TILE_LENGTH)) :
                TILE_LENGTH;

            CopyIn(static_cast<uint32_t>(nextOff), nextLen);
            Compute(TILE_LENGTH);
            CopyOut(static_cast<uint32_t>(curOff), TILE_LENGTH);
        }

        uint64_t lastOff = start + static_cast<uint64_t>(nTiles - 1) * TILE_LENGTH;
        uint32_t lastLen = static_cast<uint32_t>(validLength - static_cast<uint64_t>(nTiles - 1) * TILE_LENGTH);
        Compute(lastLen);
        CopyOut(static_cast<uint32_t>(lastOff), lastLen);
    }

private:
    __aicore__ inline void CopyIn(uint32_t off, uint32_t n) {
        LocalTensor<T> L = inputQueue.AllocTensor<T>();
        DataCopy(L, inputGm[off], n);
        inputQueue.EnQue(L);
    }

    __aicore__ inline void Compute(uint32_t n) {
        LocalTensor<T> in = inputQueue.DeQue<T>();
        LocalTensor<T> out = outputQueue.AllocTensor<T>();
        LocalTensor<T> t1 = tmpBuf1.Get<T>();
        LocalTensor<T> t2 = tmpBuf2.Get<T>();

        constexpr T inv_sqrt2 = static_cast<T>(0.7071067811865475f);
        Muls(t1, in, inv_sqrt2, n);            // t1 = x / sqrt(2)
        Erf(t2, t1, n);                        // t2 = erf(x / sqrt(2))
        Adds(t2, t2, static_cast<T>(1.0f), n); // t2 = 1 + erf(...)
        Mul(t1, in, t2, n);                    // t1 = x * (1 + erf)
        Muls(out, t1, static_cast<T>(0.5f), n);// out = 0.5 * ...

        outputQueue.EnQue(out);
        inputQueue.FreeTensor(in);
    }

    __aicore__ inline void CopyOut(uint32_t off, uint32_t n) {
        LocalTensor<T> L = outputQueue.DeQue<T>();
        DataCopy(outputGm[off], L, n);
        outputQueue.FreeTensor(L);
    }

private:
    uint64_t length, blockLength, start, validLength;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1, tmpBuf2;
    GlobalTensor<T> inputGm, outputGm;
};

// ---------- 算子入口 ----------
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, td, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, td.length, td.blockLength);
    op.Process();
}