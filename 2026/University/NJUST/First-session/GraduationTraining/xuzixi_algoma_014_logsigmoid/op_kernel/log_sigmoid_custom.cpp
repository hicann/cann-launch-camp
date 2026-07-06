#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y>
struct ComputeEngine {
    static __aicore__ inline void Init(AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& tmpBuf, uint32_t tileDataNum) {
        // float 和 half 算子原生支持，不需要申请额外的临时 TBuf
    }

    static __aicore__ inline void Execute(
        AscendC::LocalTensor<TYPE_X>& xLocal,
        AscendC::LocalTensor<TYPE_Y>& zLocal,
        AscendC::TBuf<AscendC::TPosition::VECCALC>& tmpBuf,
        uint32_t processDataNum
    ) {
        // 原生计算流水线
        AscendC::Muls(zLocal, xLocal, static_cast<TYPE_Y>(-1.0f), processDataNum);
        AscendC::Exp(zLocal, zLocal, processDataNum);
        AscendC::Adds(zLocal, zLocal, static_cast<TYPE_Y>(1.0f), processDataNum);
        AscendC::Ln(zLocal, zLocal, processDataNum);
        AscendC::Muls(zLocal, zLocal, static_cast<TYPE_Y>(-1.0f), processDataNum);
    }
};

template<>
struct ComputeEngine<bfloat16_t, bfloat16_t> {
    static __aicore__ inline void Init(AscendC::TPipe& pipe, AscendC::TBuf<AscendC::TPosition::VECCALC>& tmpBuf, uint32_t tileDataNum) {
        // 分配与当前 Tile 大小对齐后的 float 临时空间
        uint32_t alignDataNum = (tileDataNum + 15) & (~15); 
        pipe.InitBuffer(tmpBuf, alignDataNum * sizeof(float));
    }

    static __aicore__ inline void Execute(
        AscendC::LocalTensor<bfloat16_t>& xLocal,
        AscendC::LocalTensor<bfloat16_t>& zLocal,
        AscendC::TBuf<AscendC::TPosition::VECCALC>& tmpBuf,
        uint32_t processDataNum
    ) {
        AscendC::LocalTensor<float> tmpTensor = tmpBuf.Get<float>();
        
        // 必须按 16 的倍数（对应 32 字节）进行矢量计算，防止硬件非法访问
        uint32_t calcNum = (processDataNum + 15) & (~15);

        // 步骤 A: 将输入的 bfloat16_t 转换为 float (从低变高，使用 CAST_NONE)
        AscendC::Cast(tmpTensor, xLocal, AscendC::RoundMode::CAST_NONE, calcNum);
        
        // 步骤 B: 在 float 高精度下完成全部 LogSigmoid 矢量运算
        // LogSigmoid(x) = -ln(1 + exp(-x))
        AscendC::Muls(tmpTensor, tmpTensor, -1.0f, calcNum);
        AscendC::Exp(tmpTensor, tmpTensor, calcNum);
        AscendC::Adds(tmpTensor, tmpTensor, 1.0f, calcNum);
        AscendC::Ln(tmpTensor, tmpTensor, calcNum);
        AscendC::Muls(tmpTensor, tmpTensor, -1.0f, calcNum);
        
        // 步骤 C: 将最终计算结果由 float 转回 bfloat16_t
        AscendC::Cast(zLocal, tmpTensor, AscendC::RoundMode::CAST_RINT, calcNum);
    }
};

template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        
        // 根据不同类型分发初始化
        ComputeEngine<TYPE_X, TYPE_Y>::Init(pipe, tmpBuf, this->tileDataNum);
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        // 通过计算引擎分发执行计算
        ComputeEngine<TYPE_X, TYPE_Y>::Execute(xLocal, zLocal, tmpBuf, this->processDataNum);

        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;          
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;        
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf; // 临时中间变量 Buffer
    AscendC::GlobalTensor<TYPE_X> xGm;                                        
    AscendC::GlobalTensor<TYPE_Y> zGm;                                        
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.smallCoreDataNum, 
            tilingData.bigCoreDataNum, tilingData.finalBigTileNum, 
            tilingData.finalSmallTileNum, tilingData.tileDataNum, 
            tilingData.smallTailDataNum, tilingData.bigTailDataNum, 
            tilingData.tailBlockNum);
    op.Process();
}