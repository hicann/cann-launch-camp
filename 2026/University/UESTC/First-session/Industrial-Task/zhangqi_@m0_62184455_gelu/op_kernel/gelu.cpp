#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

constexpr float INV_SQRT2 = 0.7071067811865476f;
constexpr float HALF_VAL = 0.5f;
constexpr float ONE_VAL = 1.0f;

#define GELU_CAT2_IMPL(a, b) a##b
#define GELU_CAT2(a, b) GELU_CAT2_IMPL(a, b)
#define GELU_CAT3(a, b, c) GELU_CAT2(GELU_CAT2(a, b), c)
#define GELU_VEC_OUT GELU_CAT3(VE, C, OUT)

template <typename DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                                uint32_t totalLen, uint32_t perCoreLen, uint32_t tileLen)
    {
        tileLength = tileLen;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t start = blockIdx * perCoreLen;

        if (start >= totalLen) {
            total = 0;
            tileNum = 0;
            tailTileElementNum = 0;
        } else {
            total = (start + perCoreLen > totalLen) ? (totalLen - start) : perCoreLen;
            tileNum = (total + tileLength - 1) / tileLength;
            tailTileElementNum = total - tileLength * (tileNum - 1);
        }

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + start, total);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + start, total);

        uint32_t bufferSize = MAX_TILE_LENGTH * sizeof(DT_INPUT_X);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(tmpBuf1, MAX_TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf2, MAX_TILE_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf3, MAX_TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tileNum == 0) {
            return;
        }

        uint32_t count = (tileNum == 1) ? tailTileElementNum : tileLength;
        CopyIn(0, count);

        for (uint32_t i = 0; i < tileNum - 1; i++) {
            uint32_t nextCount = (i + 1 == tileNum - 1) ? tailTileElementNum : tileLength;
            CopyIn(i + 1, nextCount);
            Compute(count);
            CopyOut(i, count);
            count = nextCount;
        }

        Compute(count);
        CopyOut(tileNum - 1, count);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.AllocTensor<DT_INPUT_X>();
        if (count == tileLength) {
            AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        } else {
            AscendC::DataCopyPad(xLocal, xGm[progress * tileLength],
                {1, static_cast<uint16_t>(count * sizeof(DT_INPUT_X)), 0, 0},
                {false, 0, 0, 0});
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.AllocTensor<DT_INPUT_X>();

        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.Get<float>();
        AscendC::LocalTensor<float> tmp3 = tmpBuf3.Get<float>();

        if constexpr (std::is_same_v<DT_INPUT_X, float>) {
            AscendC::Muls(tmp1, xLocal, INV_SQRT2, count);
            AscendC::Erf(tmp2, tmp1, count);
            AscendC::Muls(tmp3, tmp2, HALF_VAL, count);
            AscendC::Adds(tmp3, tmp3, HALF_VAL, count);
            AscendC::Mul(yLocal, xLocal, tmp3, count);
        } else {
            AscendC::Cast(tmp3, xLocal, AscendC::RoundMode::CAST_NONE, count);
            AscendC::Muls(tmp1, tmp3, INV_SQRT2, count);
            AscendC::Erf(tmp2, tmp1, count);
            AscendC::Muls(tmp2, tmp2, HALF_VAL, count);
            AscendC::Adds(tmp2, tmp2, HALF_VAL, count);
            AscendC::Mul(tmp1, tmp3, tmp2, count);
            AscendC::Cast(yLocal, tmp1, AscendC::RoundMode::CAST_RINT, count);
        }

        outQueueY.EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t count)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.DeQue<DT_INPUT_X>();
        if (count == tileLength) {
            AscendC::DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        } else {
            AscendC::DataCopyPad(yGm[progress * tileLength], yLocal,
                {1, static_cast<uint16_t>(count * sizeof(DT_INPUT_X)), 0, 0});
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::GELU_VEC_OUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf3;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> yGm;
    uint32_t total = 0;
    uint32_t tileNum = 0;
    uint32_t tailTileElementNum = 0;
    uint32_t tileLength = MAX_TILE_LENGTH;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.totalLength, tiling_data.perCoreLength,
            tiling_data.tileLength);
    op.Process();
}
