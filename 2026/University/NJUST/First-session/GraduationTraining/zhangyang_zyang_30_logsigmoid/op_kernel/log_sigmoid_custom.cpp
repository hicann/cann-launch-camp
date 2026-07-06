#include "kernel_operator.h"
#include "kernel_log_sigmoid_custom_tiling.h"

using namespace AscendC;

template<typename T>
__aicore__ inline void LogSigmoidMath(
    const LocalTensor<T>& input,
    const LocalTensor<T>& output,
    LocalTensor<T>& buf1,
    LocalTensor<T>& buf2,
    uint32_t length)
{
    // buf1 = |x|
    Abs(buf1, input, length);

    // buf1 = -|x|
    Muls(buf1, buf1, (T)(-1.0), length);

    // buf1 = exp(-|x|)            
    Exp(buf1, buf1, length);

    // buf1 = 1 + exp(-|x|)
    Adds(buf1, buf1, (T)(1.0), length);

    // buf1 = log(1 + exp(-|x|))
    Log(buf1, buf1, length);

    // buf2 = min(x, 0)
    Mins(buf2, input, (T)(0.0), length);

    // output = min(x, 0) - log(1 + exp(-|x|))
    Sub(output, buf2, buf1, length);
}



__aicore__ inline void ProcessFloat32(
    GM_ADDR x,
    GM_ADDR y,
    uint32_t start,
    uint32_t curLen,
    uint32_t alignedLen)
{
    TPipe pipe;
    TQue<QuePosition::VECIN, 1>   inQue;
    TQue<QuePosition::VECOUT, 1>  outQue;
    TBuf<QuePosition::VECCALC>    calcBuf;

    LocalTensor<float> inLocal  = inQue.AllocTensor<float>(alignedLen);
    LocalTensor<float> outLocal = outQue.AllocTensor<float>(alignedLen);
    LocalTensor<float> buf1     = calcBuf.AllocTensor<float>(alignedLen);
    LocalTensor<float> buf2     = calcBuf.AllocTensor<float>(alignedLen);

    // GM → L1
    DataCopy(inLocal, x + start * sizeof(float), curLen);

    // Compute
    LogSigmoidMath<float>(inLocal, outLocal, buf1, buf2, alignedLen);

    // L1 → GM
    DataCopy(y + start * sizeof(float), outLocal, curLen);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
    calcBuf.FreeTensor(buf1);
    calcBuf.FreeTensor(buf2);
}


__aicore__ inline void ProcessFloat16(
    GM_ADDR x,
    GM_ADDR y,
    uint32_t start,
    uint32_t curLen,
    uint32_t alignedLen)
{
    TPipe pipe;
    TQue<QuePosition::VECIN, 1>   inQue;
    TQue<QuePosition::VECOUT, 1>  outQue;
    TBuf<QuePosition::VECCALC>    calcBuf;

    LocalTensor<half> inLocal  = inQue.AllocTensor<half>(alignedLen);
    LocalTensor<half> outLocal = outQue.AllocTensor<half>(alignedLen);
    LocalTensor<half> buf1     = calcBuf.AllocTensor<half>(alignedLen);
    LocalTensor<half> buf2     = calcBuf.AllocTensor<half>(alignedLen);

    DataCopy(inLocal, x + start * sizeof(half), curLen);

    LogSigmoidMath<half>(inLocal, outLocal, buf1, buf2, alignedLen);

    DataCopy(y + start * sizeof(half), outLocal, curLen);

    inQue.FreeTensor(inLocal);
    outQue.FreeTensor(outLocal);
    calcBuf.FreeTensor(buf1);
    calcBuf.FreeTensor(buf2);
}


__aicore__ inline void ProcessBfloat16(
    GM_ADDR x,
    GM_ADDR y,
    uint32_t start,
    uint32_t curLen,
    uint32_t alignedLen)
{
    TPipe pipe;
    TQue<QuePosition::VECIN, 1>   inQue;
    TQue<QuePosition::VECOUT, 1>  outQue;
    TBuf<QuePosition::VECCALC>    calcBuf;

    // --- bf16 buffers for I/O ---
    LocalTensor<bfloat16_t> inBf16  = inQue.AllocTensor<bfloat16_t>(alignedLen);
    LocalTensor<bfloat16_t> outBf16 = outQue.AllocTensor<bfloat16_t>(alignedLen);

    // --- float32 buffers for intermediate computation ---
    LocalTensor<float> inF32  = calcBuf.AllocTensor<float>(alignedLen);
    LocalTensor<float> outF32 = calcBuf.AllocTensor<float>(alignedLen);
    LocalTensor<float> buf1   = calcBuf.AllocTensor<float>(alignedLen);
    LocalTensor<float> buf2   = calcBuf.AllocTensor<float>(alignedLen);

    // 1. Copy bf16 input from global memory
    DataCopy(inBf16, x + start * sizeof(bfloat16_t), curLen);

    // 2. Upcast:  bf16 → float32   
    Cast(inF32, inBf16, curLen);

    // 3. Compute LogSigmoid in float32 precision
    LogSigmoidMath<float>(inF32, outF32, buf1, buf2, alignedLen);

    // 4. Downcast:  float32 → bf16   (truncate mantissa)
    Cast(outBf16, outF32, curLen);

    // 5. Copy bf16 result to global memory
    DataCopy(y + start * sizeof(bfloat16_t), outBf16, curLen);

    // --- release buffers ---
    inQue.FreeTensor(inBf16);
    outQue.FreeTensor(outBf16);
    calcBuf.FreeTensor(inF32);
    calcBuf.FreeTensor(outF32);
    calcBuf.FreeTensor(buf1);
    calcBuf.FreeTensor(buf2);
}


extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // --- 1. Parse tiling constants ---
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.totalLength;
    uint32_t blockDim    = tilingData.blockDim;
    uint32_t tileLength  = tilingData.tileLength;
    uint32_t dataType    = tilingData.dataType;

    // --- 2. Determine this core's slice ---
    uint32_t blockIdx = GetBlockIdx();
    uint32_t start = blockIdx * tileLength;
    if (start >= totalLength) {
        return;  // nothing to do on this core
    }
    uint32_t end = start + tileLength;
    if (end > totalLength) {
        end = totalLength;
    }
    uint32_t curLen = end - start;

   
    uint32_t alignedLen = (curLen + 31u) / 32u * 32u;

    // --- 3. Dispatch by data type ---
    switch (dataType) {
    case 0:   // float32
        ProcessFloat32(x, y, start, curLen, alignedLen);
        break;
    case 1:   // float16 (half)
        ProcessFloat16(x, y, start, curLen, alignedLen);
        break;
    default:  // bfloat16  (case 2)
        ProcessBfloat16(x, y, start, curLen, alignedLen);
        break;
    }
}
