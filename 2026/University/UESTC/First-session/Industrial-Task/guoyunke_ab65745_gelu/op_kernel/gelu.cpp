// Kernel侧核函数实现 - half用Gelu API，float用Erf
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

// float Erf 常量
constexpr float INV_SQRT_2 = 0.7071067811865475f;
constexpr float COEF_HALF = 0.5f;

// ============================================================
// 计算辅助类模板（主模板）
// ============================================================
template<typename T, typename Enable = void>
struct GeluComputeHelper;

// ============================================================
// half 特化：直接调用 AscendC::Gelu 高阶 API
// ============================================================
template<typename T>
struct GeluComputeHelper<T, typename std::enable_if<std::is_same<T, half>::value>::type> {
    __aicore__ static inline void compute(const AscendC::LocalTensor<T>& dst,
                                          const AscendC::LocalTensor<T>& src,
                                          uint32_t dataSize,
                                          AscendC::LocalTensor<T>& tmp) {
        // half 直接使用硬件加速 API
        AscendC::Gelu<T, true>(dst, src, dataSize);
    }
};

// ============================================================
// float 特化：Erf + sharedTmpBuffer（高精度）
// ============================================================
template<typename T>
struct GeluComputeHelper<T, typename std::enable_if<std::is_same<T, float>::value>::type> {
    __aicore__ static inline void compute(const AscendC::LocalTensor<T>& dst,
                                          const AscendC::LocalTensor<T>& src,
                                          uint32_t dataSize,
                                          AscendC::LocalTensor<T>& tmp) {
        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        AscendC::Muls(tmp, src, static_cast<T>(INV_SQRT_2), dataSize);
        AscendC::Erf(tmp, tmp, dataSize);
        AscendC::Adds(dst, tmp, static_cast<T>(1.0f), dataSize);
        AscendC::Mul(dst, src, dst, dataSize);
        AscendC::Muls(dst, dst, static_cast<T>(COEF_HALF), dataSize);
    }
};

// ============================================================
// Kernel 主类 - 双缓冲流水线
// ============================================================
template<typename TYPE>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum,
                                uint32_t tmpSize)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = static_cast<uint64_t>(bigCoreDataNum) * blockIdx;
        this->tileDataNum = tileDataNum;

        if (blockIdx < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (blockIdx - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ TYPE*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE*)y + globalBufferIndex, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE));
        pipe.InitBuffer(tmpBuf, tmpSize);
    }

    __aicore__ inline void Process()
    {
        int32_t totalTiles = this->tileNum;

        CopyIn(0);

        for (int32_t i = 0; i < totalTiles; i++) {
            if (i < totalTiles - 1) {
                CopyIn(i + 1);
            }
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t copyLen = (progress == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        AscendC::LocalTensor<TYPE> xLocal = inQueueX.AllocTensor<TYPE>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], copyLen);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE> xLocal = inQueueX.DeQue<TYPE>();
        AscendC::LocalTensor<TYPE> zLocal = outQueueZ.AllocTensor<TYPE>();
        AscendC::LocalTensor<TYPE> tmp = tmpBuf.Get<TYPE>();
        uint32_t dataLen = (progress == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;

        GeluComputeHelper<TYPE>::compute(zLocal, xLocal, dataLen, tmp);

        inQueueX.FreeTensor(xLocal);
        outQueueZ.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t copyLen = (progress == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
        AscendC::LocalTensor<TYPE> zLocal = outQueueZ.DeQue<TYPE>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, copyLen);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<TYPE> xGm, zGm;
    AscendC::TBuf<> tmpBuf;

    uint32_t coreDataNum = 0, tileNum = 0, tileDataNum = 0, tailDataNum = 0;
};

template<typename D_T>
__global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelGelu<D_T> op;
    op.Init(x, y,
            tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
            tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
            tilingData.tileDataNum,
            tilingData.smallTailDataNum, tilingData.bigTailDataNum,
            tilingData.tailBlockNum,
            tilingData.tmpSize);
    op.Process();
}