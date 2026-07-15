#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;
constexpr float GELU_ONE = 1.0f;
constexpr float GELU_HALF = 0.5f;
constexpr float GELU_INV_SQRT2 = 0.7071067811865475f;
constexpr float GELU_FP16_CUBE_COEF = 0.0455399241f;
constexpr float GELU_FP16_NEG_COEF = -1.595769122f;

#define QUE_POS(suffix) AscendC::QuePosition::VEC##suffix

// USE_TBUF_PATH 为 1 时使用 TBuf 单 tile 路径；为 0 时使用 TQue 双 buffer 路径。
template <typename TYPE_X, int USE_TBUF_PATH>
class KernelGelu
{
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum, uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        // 前 tailBlockNum 个核处理大核分片，其余核处理小核分片。
        if (coreNum < tailBlockNum)
        {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        }
        else
        {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE_X *)input_x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_X *)output + globalBufferIndex, this->coreDataNum);

        // 小 tile 场景用 TBuf 减少 EnQue/DeQue 开销；普通场景用双 buffer 队列流水。
        if constexpr (USE_TBUF_PATH)
        {
            pipe.InitBuffer(inBufX, this->tileDataNum * sizeof(TYPE_X));
            pipe.InitBuffer(outBufY, this->tileDataNum * sizeof(TYPE_X));
        }
        else
        {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
            pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        }

        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if constexpr (USE_TBUF_PATH)
        {
            ProcessTbufTiles();
        }
        else
        {
            ProcessMultiTile();
        }
    }

private:
    __aicore__ inline void ProcessTbufTiles()
    {
        // TBuf 路径按 tile 顺序执行 CopyIn -> Compute -> CopyOut。
        AscendC::LocalTensor<TYPE_X> xLocal = inBufX.Get<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outBufY.Get<TYPE_X>();

        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (uint32_t i = 0; i < this->tileNum; i++)
        {
            if (i == loopCount - 1)
            {
                this->processDataNum = this->tailDataNum;
            }
            uint32_t offset = i * this->tileDataNum;
            AscendC::DataCopy(xLocal, xGm[offset], this->processDataNum);
            AscendC::PipeBarrier<PIPE_ALL>();
            ComputeGelu(xLocal, yLocal);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(yGm[offset], yLocal, this->processDataNum);
            if (i + 1 < this->tileNum)
            {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessMultiTile()
    {
        // 队列路径使用输入/输出队列组织多 tile 数据搬运与计算。
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (uint32_t i = 0; i < this->tileNum; i++)
        {
            if (i == loopCount - 1)
            {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.AllocTensor<TYPE_X>();
        ComputeGelu(xLocal, yLocal);
        outQueueY.EnQue<TYPE_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeGelu(AscendC::LocalTensor<TYPE_X> xLocal, AscendC::LocalTensor<TYPE_X> yLocal)
    {
        if constexpr (std::is_same_v<TYPE_X, float>)
        {
            // float32 使用精确形式：0.5 * x * (1 + erf(x / sqrt(2)))。
            AscendC::LocalTensor<float> tmp = tmpBuf1.Get<float>();
            AscendC::Muls(tmp, xLocal, GELU_INV_SQRT2, this->processDataNum);
            AscendC::Erf(tmp, tmp, this->processDataNum);
            AscendC::Adds(tmp, tmp, GELU_ONE, this->processDataNum);
            AscendC::Muls(tmp, tmp, GELU_HALF, this->processDataNum);
            AscendC::Mul(yLocal, xLocal, tmp, this->processDataNum);
        }
        else
        {
            // float16 使用 sigmoid/logistic 近似形式，减少半精度路径计算开销。
            AscendC::LocalTensor<TYPE_X> tmp = tmpBuf1.Get<TYPE_X>();
            AscendC::Mul(tmp, xLocal, xLocal, this->processDataNum);
            AscendC::Mul(tmp, tmp, xLocal, this->processDataNum);
            AscendC::Muls(tmp, tmp, static_cast<TYPE_X>(GELU_FP16_CUBE_COEF), this->processDataNum);
            AscendC::Add(tmp, tmp, xLocal, this->processDataNum);
            AscendC::Muls(tmp, tmp, static_cast<TYPE_X>(GELU_FP16_NEG_COEF), this->processDataNum);
            AscendC::Exp(tmp, tmp, this->processDataNum);
            AscendC::Adds(tmp, tmp, static_cast<TYPE_X>(GELU_ONE), this->processDataNum);
            AscendC::Div(yLocal, xLocal, tmp, this->processDataNum);
        }
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> yLocal = outQueueY.DeQue<TYPE_X>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<QUE_POS(OUT), BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inBufX, outBufY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_INPUT_X, int USE_TBUF_PATH>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    // 从 Host 侧 tiling 数据初始化 kernel，再执行当前 core 对应的数据分片。
    KernelGelu<DT_INPUT_X, USE_TBUF_PATH> op;
    op.Init(input_x, output, tiling_data.smallCoreDataNum,
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum,
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    op.Process();
}
