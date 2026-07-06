#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t PIPELINE_STAGES = 2;

template<typename T_IN, typename T_OUT>
class LogSigmoidKernel {
public:
    __aicore__ inline LogSigmoidKernel() {}

    __aicore__ inline void Init(GM_ADDR gmIn, GM_ADDR gmOut,
                                uint32_t smallElemCnt, uint32_t bigElemCnt,
                                uint32_t bigTileTotal, uint32_t smallTileTotal,
                                uint32_t tileElemCap, uint32_t smallLastElem,
                                uint32_t bigLastElem, uint32_t extraBlkCnt)
    {
        uint32_t curCoreId = AscendC::GetBlockIdx();
        uint32_t gmOffset = bigElemCnt * curCoreId;
        this->tileCapacity = tileElemCap;

        if (curCoreId < extraBlkCnt) {
            this->coreTotalElems = bigElemCnt;
            this->totalTiles = bigTileTotal;
            this->lastTileElems = bigLastElem;
        } else {

            this->coreTotalElems = smallElemCnt;
            this->totalTiles = smallTileTotal;
            this->lastTileElems = smallLastElem;
            gmOffset -= (bigElemCnt - smallElemCnt) * (curCoreId - extraBlkCnt);
        }

        inGm.SetGlobalBuffer((__gm__ T_IN*)gmIn + gmOffset, this->coreTotalElems);
        outGm.SetGlobalBuffer((__gm__ T_OUT*)gmOut + gmOffset, this->coreTotalElems);

        pipe.InitBuffer(qIn, PIPELINE_STAGES, this->tileCapacity * sizeof(T_IN));
        pipe.InitBuffer(qOut, PIPELINE_STAGES, this->tileCapacity * sizeof(T_OUT));

        if constexpr (std::is_same<T_IN, __bf16>::value) {
            pipe.InitBuffer(bufTmpIn, this->tileCapacity * sizeof(float));
            pipe.InitBuffer(bufTmpOut, this->tileCapacity * sizeof(float));
        }
    }

    __aicore__ inline void Run()
    {
        int32_t numTiles = static_cast<int32_t>(this->totalTiles);
        this->curProcElems = this->tileCapacity;

        for (int32_t t = 0; t < numTiles; t++) {
            if (t == numTiles - 1) {
                this->curProcElems = this->lastTileElems;
            }
            LoadData(t);
            DoCompute(t);
            StoreData(t);
        }
    }

private:
    __aicore__ inline void LoadData(int32_t tileIdx)
    {
        auto localIn = qIn.AllocTensor<T_IN>();
        AscendC::DataCopy(localIn,
                          inGm[tileIdx * this->tileCapacity],
                          this->curProcElems);
        qIn.EnQue(localIn);
    }

    __aicore__ inline void DoCompute(int32_t tileIdx)
    {
        auto localIn = qIn.DeQue<T_IN>();
        auto localOut = qOut.AllocTensor<T_OUT>();

        if constexpr (std::is_same<T_IN, __bf16>::value) {

            auto fp32In = bufTmpIn.Get<float>();
            auto fp32Out = bufTmpOut.Get<float>();

            AscendC::Cast(fp32In, localIn,
                          AscendC::RoundMode::CAST_NONE, this->curProcElems);

            AscendC::Sigmoid(fp32Out, fp32In, this->curProcElems);

            AscendC::Log(fp32Out, fp32Out, this->curProcElems);

            AscendC::Cast(localOut, fp32Out,
                          AscendC::RoundMode::CAST_RINT, this->curProcElems);
        } else {

            AscendC::Sigmoid(localOut, localIn, this->curProcElems);
            AscendC::Log(localOut, localOut, this->curProcElems);
        }

        qOut.EnQue<T_OUT>(localOut);
        qIn.FreeTensor(localIn);
    }

    __aicore__ inline void StoreData(int32_t tileIdx)
    {
        auto localOut = qOut.DeQue<T_OUT>();
        AscendC::DataCopy(outGm[tileIdx * this->tileCapacity],
                          localOut, this->curProcElems);
        qOut.FreeTensor(localOut);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_STAGES> qIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_STAGES> qOut;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmpIn;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmpOut;

    AscendC::GlobalTensor<T_IN> inGm;
    AscendC::GlobalTensor<T_OUT> outGm;

    uint32_t coreTotalElems;
    uint32_t totalTiles;
    uint32_t tileCapacity;
    uint32_t lastTileElems;
    uint32_t curProcElems;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tlData, tiling);

    LogSigmoidKernel<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y,
            tlData.smallCoreElemCnt,
            tlData.bigCoreElemCnt,
            tlData.bigTotalTileCnt,
            tlData.smallTotalTileCnt,
            tlData.elemPerTile,
            tlData.smallLastTileElemCnt,
            tlData.bigLastTileElemCnt,
            tlData.extraBlockCnt);
    op.Run();
}
