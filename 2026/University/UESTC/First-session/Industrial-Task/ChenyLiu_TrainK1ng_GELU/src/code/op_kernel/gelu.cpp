#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 4096;

__aicore__ inline void computeFasterGeluHalf(
    AscendC::LocalTensor<half>& y,
    AscendC::LocalTensor<half>& x,
    uint32_t count)
{
    constexpr half NEG_CLIP = (half)(-6.0f);
    AscendC::Maxs(x, x, NEG_CLIP, count);
    
    constexpr half K0 = (half)0.044715f;
    constexpr half K1 = (half)0.7978845608f;
    constexpr half HALF = (half)0.5f;
    constexpr half ONE = (half)1.0f;
    
    AscendC::Mul(y, x, x, count);
    AscendC::Muls(y, y, K0, count);
    AscendC::Adds(y, y, ONE, count);
    AscendC::Mul(y, y, x, count);
    AscendC::Muls(y, y, K1, count);
    AscendC::Tanh(y, y, count);
    AscendC::Adds(y, y, ONE, count);
    AscendC::Muls(y, y, HALF, count);
    AscendC::Mul(y, y, x, count);
}


__aicore__ inline void computeErfFloat(
    AscendC::LocalTensor<float>& y,
    AscendC::LocalTensor<float>& x,
    uint32_t count)
{
    constexpr float RECIP_SQRT2 = 0.7071067811865475f;
    
    AscendC::Muls(y, x, RECIP_SQRT2, count);
    AscendC::Erf(y, y, count);
    AscendC::Adds(y, y, 1.0f, count);
    AscendC::Muls(y, y, 0.5f, count);
    AscendC::Mul(y, y, x, count);
}

// =============================================================================
// KernelGelu 主类模板
// =============================================================================
template<class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu(){}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;

    uint32_t blockLength = 0;
    uint32_t offset = 0;
};

template<class T>
__aicore__ inline void KernelGelu<T>::Init(GM_ADDR input_x, GM_ADDR output, uint32_t length)
{
    uint32_t coreNum = AscendC::GetBlockNum();
    uint32_t coreId = AscendC::GetBlockIdx();

    uint32_t avg = length / coreNum;
    uint32_t rem = length % coreNum;

    if (coreId < rem) {
        blockLength = avg + 1;
        offset = coreId * blockLength;
    } else {
        blockLength = avg;
        offset = rem * (avg + 1) + (coreId - rem) * avg;
    }

    xGm.SetGlobalBuffer((__gm__ T*)input_x + offset, blockLength);
    yGm.SetGlobalBuffer((__gm__ T*)output + offset, blockLength);

    pipe.InitBuffer(inQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
    pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(T));
}

template<class T>
__aicore__ inline void KernelGelu<T>::Process()
{
    if (!blockLength)
        return;

    uint32_t pos = 0;
    uint32_t remain = blockLength;

    while (remain >= TILE_LENGTH) {
        CopyIn(pos, TILE_LENGTH);
        Compute(TILE_LENGTH);
        CopyOut(pos, TILE_LENGTH);
        pos += TILE_LENGTH;
        remain -= TILE_LENGTH;
    }

    if (remain) {
        CopyIn(pos, remain);
        Compute(remain);
        CopyOut(pos, remain);
    }
}

template<class T>
__aicore__ inline void KernelGelu<T>::CopyIn(uint32_t offset, uint32_t count)
{
    auto x = inQueue.AllocTensor<T>();

    uint32_t alignCount = (sizeof(T) == 2) ? 16 : 8;
    if (count % alignCount == 0) {
        AscendC::DataCopy(x, xGm[offset], count);
    } else {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;

        AscendC::DataCopyPad(x, xGm[offset], copyParams, padParams);
    }

    inQueue.EnQue(x);
}

template<class T>
__aicore__ inline void KernelGelu<T>::CopyOut(uint32_t offset, uint32_t count)
{
    auto y = outQueue.DeQue<T>();

    uint32_t alignCount = (sizeof(T) == 2) ? 16 : 8;
    if (count % alignCount == 0) {
        AscendC::DataCopy(yGm[offset], y, count);
    } else {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = count * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        AscendC::DataCopyPad(yGm[offset], y, copyParams);
    }

    outQueue.FreeTensor(y);
}

template<>
__aicore__ inline void KernelGelu<half>::Compute(uint32_t count)
{
    auto x = inQueue.DeQue<half>();
    auto y = outQueue.AllocTensor<half>();

    computeFasterGeluHalf(y, x, count);

    outQueue.EnQue(y);
    inQueue.FreeTensor(x);
}

template<>
__aicore__ inline void KernelGelu<float>::Compute(uint32_t count)
{
    auto x = inQueue.DeQue<float>();
    auto y = outQueue.AllocTensor<float>();

    computeErfFloat(y, x, count);

    outQueue.EnQue(y);
    inQueue.FreeTensor(x);
}

template<typename DT_INPUT_X>
__global__ __aicore__
void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tilingData.length);
    op.Process();
}