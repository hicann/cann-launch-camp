#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr float HALF = 0.5f;
constexpr float TANH_APPROX_COEF = 0.044715f;
constexpr float TANH_APPROX_SCALE = 0.7978845608028654f;
constexpr float GELU_FAST_C1 = 7.977558490e-01f;
constexpr float GELU_FAST_C3 = 3.681900816e-02f;
constexpr float GELU_FAST_C5 = -3.201727046e-04f;

constexpr auto GELU_VEC_OUT = static_cast<AscendC::TPosition>(
    static_cast<int>(AscendC::TPosition::VECIN) + 1);

template <typename T>
__aicore__ inline void ComputeGeluByDtype(AscendC::LocalTensor<T>& xLocal,
    AscendC::LocalTensor<T>& yLocal, AscendC::LocalTensor<uint8_t>& tanhTmpLocal, uint32_t count)
{
    AscendC::Mul(yLocal, xLocal, xLocal, count);
    AscendC::Mul(yLocal, yLocal, xLocal, count);
    AscendC::Muls(yLocal, yLocal, static_cast<T>(TANH_APPROX_COEF), count);
    AscendC::Add(yLocal, yLocal, xLocal, count);
    AscendC::Muls(yLocal, yLocal, static_cast<T>(TANH_APPROX_SCALE), count);
    AscendC::Tanh(yLocal, yLocal, tanhTmpLocal, count);
    AscendC::Adds(yLocal, yLocal, static_cast<T>(1.0f), count);
    AscendC::Mul(yLocal, yLocal, xLocal, count);
    AscendC::Muls(yLocal, yLocal, static_cast<T>(HALF), count);
}

__aicore__ inline void ComputeGeluFloatFast(AscendC::LocalTensor<float>& xLocal,
    AscendC::LocalTensor<float>& yLocal, AscendC::LocalTensor<float>& workLocal,
    AscendC::LocalTensor<uint8_t>& tanhTmpLocal, uint32_t count)
{
    AscendC::Mul(workLocal, xLocal, xLocal, count);
    AscendC::Muls(yLocal, workLocal, GELU_FAST_C5, count);
    AscendC::Adds(yLocal, yLocal, GELU_FAST_C3, count);
    AscendC::Mul(yLocal, yLocal, workLocal, count);
    AscendC::Adds(yLocal, yLocal, GELU_FAST_C1, count);
    AscendC::Mul(yLocal, yLocal, xLocal, count);
    AscendC::Tanh(workLocal, yLocal, tanhTmpLocal, count);
    AscendC::Adds(yLocal, workLocal, 1.0f, count);
    AscendC::Mul(yLocal, yLocal, xLocal, count);
    AscendC::Muls(yLocal, yLocal, HALF, count);
}

template <typename T>
struct GeluComputeDispatch {
    __aicore__ inline static void InitBuffer(AscendC::TPipe& pipe,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& x2Buf,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& auxBuf, uint32_t tanhTmpSize, uint32_t tileLen)
    {
        (void)x2Buf;
        (void)tileLen;
        if (tanhTmpSize > 0) {
            pipe.InitBuffer(auxBuf, tanhTmpSize);
        }
    }

    __aicore__ inline static void Compute(AscendC::LocalTensor<T>& xLocal, AscendC::LocalTensor<T>& yLocal,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& x2Buf,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& auxBuf, uint32_t tanhTmpSize,
        uint32_t tileLen, uint32_t count)
    {
        (void)x2Buf;
        (void)tileLen;
        if (tanhTmpSize > 0) {
            AscendC::LocalTensor<uint8_t> tanhTmpLocal = auxBuf.Get<uint8_t>(tanhTmpSize);
            ComputeGeluByDtype(xLocal, yLocal, tanhTmpLocal, count);
            return;
        }

        AscendC::Mul(yLocal, xLocal, xLocal, count);
        AscendC::Mul(yLocal, yLocal, xLocal, count);
        AscendC::Muls(yLocal, yLocal, static_cast<T>(TANH_APPROX_COEF), count);
        AscendC::Add(yLocal, yLocal, xLocal, count);
        AscendC::Muls(yLocal, yLocal, static_cast<T>(TANH_APPROX_SCALE), count);
        AscendC::Tanh(yLocal, yLocal, count);
        AscendC::Adds(yLocal, yLocal, static_cast<T>(1.0f), count);
        AscendC::Mul(yLocal, yLocal, xLocal, count);
        AscendC::Muls(yLocal, yLocal, static_cast<T>(HALF), count);
    }
};

template <>
struct GeluComputeDispatch<float> {
    __aicore__ inline static void InitBuffer(AscendC::TPipe& pipe,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& x2Buf,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& auxBuf, uint32_t tanhTmpSize, uint32_t tileLen)
    {
        pipe.InitBuffer(x2Buf, tileLen * sizeof(float));
        pipe.InitBuffer(auxBuf, tanhTmpSize);
    }

    __aicore__ inline static void Compute(AscendC::LocalTensor<float>& xLocal, AscendC::LocalTensor<float>& yLocal,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& x2Buf,
        AscendC::TBuf<AscendC::QuePosition::VECCALC>& auxBuf, uint32_t tanhTmpSize,
        uint32_t tileLen, uint32_t count)
    {
        AscendC::LocalTensor<float> workLocal = x2Buf.Get<float>(tileLen);
        AscendC::LocalTensor<uint8_t> tanhTmpLocal = auxBuf.Get<uint8_t>(tanhTmpSize);
        ComputeGeluFloatFast(xLocal, yLocal, workLocal, tanhTmpLocal, count);
    }
};

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const GeluTilingData* td)
    {
        alignUnit = td->alignUnit;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockStart = AlignDown(static_cast<uint32_t>(
            static_cast<uint64_t>(td->totalLength) * blockIdx / td->blockNum));
        uint32_t blockEnd = (blockIdx + 1 == td->blockNum)
            ? td->totalLength
            : AlignDown(static_cast<uint32_t>(
                static_cast<uint64_t>(td->totalLength) * (blockIdx + 1) / td->blockNum));
        total = blockEnd - blockStart;

        tileNum = (total + td->tileLength - 1) / td->tileLength;
        if (tileNum == 0) tileNum = 1;
        tailTileSize = total - td->tileLength * (tileNum - 1);

        tileLen = td->tileLength;
        tanhTmpSize = td->tanhTmpSize;
        uint32_t gmLength = tileLen * (tileNum - 1) + AlignCount(tailTileSize);
        xGm.SetGlobalBuffer((__gm__ T*)x + blockStart, gmLength);
        yGm.SetGlobalBuffer((__gm__ T*)y + blockStart, gmLength);

        if (tileNum == 1) {
            pipe.InitBuffer(xBuf0, tileLen * sizeof(T));
            pipe.InitBuffer(yBuf0, tileLen * sizeof(T));
        } else {
            pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, tileLen * sizeof(T));
            pipe.InitBuffer(outQueueY, DOUBLE_BUFFER, tileLen * sizeof(T));
        }
        GeluComputeDispatch<T>::InitBuffer(pipe, x2Buf, tanhTmpBuf, tanhTmpSize, tileLen);
    }

    __aicore__ inline void Process()
    {
        if (total == 0) return;
        if (tileNum == 1) {
            ProcessDirect();
        } else {
            ProcessPipelined();
        }
    }

private:
    __aicore__ inline uint32_t AlignDown(uint32_t count)
    {
        return count & ~(alignUnit - 1);
    }

    __aicore__ inline uint32_t AlignCount(uint32_t count)
    {
        return (count + alignUnit - 1) & ~(alignUnit - 1);
    }

    __aicore__ inline void ComputeGelu(AscendC::LocalTensor<T>& xLocal, AscendC::LocalTensor<T>& yLocal, uint32_t count)
    {
        GeluComputeDispatch<T>::Compute(xLocal, yLocal, x2Buf, tanhTmpBuf, tanhTmpSize, tileLen, count);
    }

    __aicore__ inline void ProcessDirect()
    {
        uint32_t count = tailTileSize;
        uint32_t alignedCount = AlignCount(count);

        AscendC::LocalTensor<T> xLocal = xBuf0.Get<T>(tileLen);
        AscendC::DataCopy(xLocal, xGm[0], alignedCount);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::LocalTensor<T> yLocal = yBuf0.Get<T>(tileLen);
        ComputeGelu(xLocal, yLocal, count);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(yGm[0], yLocal, alignedCount);
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        uint32_t alignedCount = AlignCount(count);
        AscendC::DataCopy(xLocal, xGm[tileIdx * tileLen], alignedCount);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        ComputeGelu(xLocal, yLocal, count);
        outQueueY.EnQue<T>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t count)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        uint32_t alignedCount = AlignCount(count);
        AscendC::DataCopy(yGm[tileIdx * tileLen], yLocal, alignedCount);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessPipelined()
    {
        CopyIn(0, tileLen);
        uint32_t lastTileIdx = tileNum - 1;
        for (uint32_t i = 0; i < lastTileIdx; i++) {
            Compute(tileLen);
            uint32_t nextCount = (i + 1 == lastTileIdx) ? tailTileSize : tileLen;
            CopyIn(i + 1, nextCount);
            CopyOut(i, tileLen);
        }
        Compute(tailTileSize);
        CopyOut(lastTileIdx, tailTileSize);
    }

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> xBuf0;
    AscendC::TBuf<AscendC::TPosition::VECOUT> yBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> x2Buf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tanhTmpBuf;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<GELU_VEC_OUT, 1> outQueueY;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t tileLen;
    uint32_t total;
    uint32_t tileNum;
    uint32_t tailTileSize;
    uint32_t alignUnit;
    uint32_t tanhTmpSize;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, &tiling_data);
    op.Process();
}
