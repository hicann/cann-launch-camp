#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
using namespace AscendC;

constexpr uint32_t QUEUE_STACK = 1;
constexpr uint32_t SLICE_SIZE = 1024;

template<typename DTYPE>
class LogSigmoidCore
{
public:
    __aicore__ inline LogSigmoidCore() = default;

    __aicore__ inline void InitMem(GM_ADDR xBase, GM_ADDR yBase, uint32_t allElem)
    {
        all_element = allElem;
        uint32_t coreCount = GetBlockNum();
        uint32_t coreId = GetBlockIdx();

        uint32_t singleCoreElem = (all_element + coreCount - 1) / coreCount;
        core_offset = coreId * singleCoreElem;

        if (core_offset >= all_element)
        {
            core_valid = 0;
        }
        else
        {
            uint32_t core_end = core_offset + singleCoreElem;
            core_end = core_end > all_element ? all_element : core_end;
            core_valid = core_end - core_offset;
        }

        inputGM.SetGlobalBuffer((__gm__ DTYPE*)xBase + core_offset, core_valid);
        outputGM.SetGlobalBuffer((__gm__ DTYPE*)yBase + core_offset, core_valid);

        pipe.InitBuffer(vecInQueue, QUEUE_STACK, SLICE_SIZE * sizeof(DTYPE));
        pipe.InitBuffer(vecOutQueue, QUEUE_STACK, SLICE_SIZE * sizeof(DTYPE));
        pipe.InitBuffer(floatBufA, SLICE_SIZE * sizeof(float));
        pipe.InitBuffer(floatBufB, SLICE_SIZE * sizeof(float));
    }

    __aicore__ inline void ComputeAllSlice()
    {
        if (core_valid == 0) return;
        uint32_t sliceOff = 0;
        while (sliceOff < core_valid)
        {
            uint32_t curCalcNum = SLICE_SIZE;
            if (sliceOff + SLICE_SIZE > core_valid)
            {
                curCalcNum = core_valid - sliceOff;
            }
            LoadGMData(sliceOff, curCalcNum);
            RunLogSigmoid(curCalcNum);
            WriteGMData(sliceOff, curCalcNum);
            sliceOff += SLICE_SIZE;
        }
    }

private:
    __aicore__ inline void LoadGMData(uint32_t off, uint32_t calcNum)
    {
        LocalTensor<DTYPE> xLocal = vecInQueue.AllocTensor<DTYPE>();
        DataCopyExtParams copyCfg{};
        copyCfg.blockCount = 1;
        copyCfg.blockLen = calcNum * sizeof(DTYPE);
        copyCfg.srcStride = 0;
        copyCfg.dstStride = 0;
        copyCfg.rsv = 0;

        DataCopyPadExtParams<DTYPE> padCfg{};
        padCfg.isPad = false;
        padCfg.leftPadding = 0;
        padCfg.rightPadding = 0;
        padCfg.paddingValue = 0;
        DataCopyPad(xLocal, inputGM[off], copyCfg, padCfg);
        vecInQueue.EnQue(xLocal);
    }

    __aicore__ inline void RunLogSigmoid(uint32_t calcNum)
    {
        LocalTensor<DTYPE> xLocal = vecInQueue.DeQue<DTYPE>();
        LocalTensor<DTYPE> yLocal = vecOutQueue.AllocTensor<DTYPE>();
        LocalTensor<float> fBuf1 = floatBufA.Get<float>();
        LocalTensor<float> fBuf2 = floatBufB.Get<float>();

        if constexpr(sizeof(DTYPE) == sizeof(float))
        {
            Muls(fBuf1, xLocal, -1.0f, calcNum);
            PipeBarrier<PIPE_V>();
            Exp(fBuf2, fBuf1, calcNum);
            PipeBarrier<PIPE_V>();
            Adds(fBuf2, fBuf2, 1.0f, calcNum);
            PipeBarrier<PIPE_V>();
            Log(fBuf1, fBuf2, calcNum);
            PipeBarrier<PIPE_V>();
            Muls(yLocal, fBuf1, -1.0f, calcNum);
        }
        else
        {
            Cast(fBuf1, xLocal, RoundMode::CAST_NONE, calcNum);
            PipeBarrier<PIPE_V>();
            Muls(fBuf1, fBuf1, -1.0f, calcNum);
            PipeBarrier<PIPE_V>();
            Exp(fBuf2, fBuf1, calcNum);
            PipeBarrier<PIPE_V>();
            Adds(fBuf2, fBuf2, 1.0f, calcNum);
            PipeBarrier<PIPE_V>();
            Log(fBuf1, fBuf2, calcNum);
            PipeBarrier<PIPE_V>();
            Muls(fBuf1, fBuf1, -1.0f, calcNum);
            PipeBarrier<PIPE_V>();
            Cast(yLocal, fBuf1, RoundMode::CAST_RINT, calcNum);
        }
        vecOutQueue.EnQue(yLocal);
        vecInQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void WriteGMData(uint32_t off, uint32_t calcNum)
    {
        LocalTensor<DTYPE> yLocal = vecOutQueue.DeQue<DTYPE>();
        DataCopyExtParams copyCfg{};
        copyCfg.blockCount = 1;
        copyCfg.blockLen = calcNum * sizeof(DTYPE);
        copyCfg.srcStride = 0;
        copyCfg.dstStride = 0;
        copyCfg.rsv = 0;
        DataCopyPad(outputGM[off], yLocal, copyCfg);
        vecOutQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_STACK> vecInQueue;
    TQue<QuePosition::VECOUT, QUEUE_STACK> vecOutQueue;
    TBuf<QuePosition::VECCALC> floatBufA;
    TBuf<QuePosition::VECCALC> floatBufB;

    GlobalTensor<DTYPE> inputGM;
    GlobalTensor<DTYPE> outputGM;

    uint32_t all_element;
    uint32_t core_offset;
    uint32_t core_valid;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tileParam, tiling);

    if (tileParam.dtype_code == 0)
    {
        LogSigmoidCore<half> op;
        op.InitMem(x, y, tileParam.elem_total);
        op.ComputeAllSlice();
    }
    else if (tileParam.dtype_code == 1)
    {
        LogSigmoidCore<float> op;
        op.InitMem(x, y, tileParam.elem_total);
        op.ComputeAllSlice();
    }
    else if (tileParam.dtype_code == 2)
    {
        LogSigmoidCore<bfloat16_t> op;
        op.InitMem(x, y, tileParam.elem_total);
        op.ComputeAllSlice();
    }
}
