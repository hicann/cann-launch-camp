// ==========================================================
//  Kernel侧核函数实现 - 近似GELU激活函数 (分级tile + 大核/小核)
// ==========================================================
//  公式: gelu(x) ≈ x / (1 + exp(-scale * (x + c1*x^3 + c3*x^5)))
//        其中 scale = 2 * sqrt(2/pi) = 1.59576912
//        由 tanh 形式 0.5*x*(1+tanh(sqrt(2/pi)*(x+c1*x^3+c3*x^5)))
//        化简而来，等价但改用 exp；保留 x^5 项以获得更高精度。
//
//  Tiling两层切分:
//    第1层 核间: 大核/小核模型
//      前tailBlockNum个核 = 大核 bigCoreDataNum 元素 (多1)
//      剩余核 = 小核 smallCoreDataNum 元素
//    第2层 核内: tileDataNum 按总数据量分级, 尾tile在Host做32B上对齐
//
//  计算步骤:
//    1. z = x^2, tmp = x^2   (保存x^2)
//    2. z = z * x = x^3
//    3. tmp = tmp * z = x^5  (x^2 * x^3, z未缩放前)
//    4. z = c1 * x^3
//    5. tmp = c3 * x^5
//    6. z = c1*x^3 + c3*x^5
//    7. z = x + z            (x + c1*x^3 + c3*x^5)
//    8. z = -scale * z       (-1.59576912 * z)
//    9. z = exp(z)
//   10. z = z + 1            (1 + exp(z))
//   11. z = x / z            (result)

#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;          // 双缓冲
constexpr float   C1              = 0.044715f;               // x^3系数
constexpr float   C3              = 0.001072f;               // x^5系数
constexpr float   SQRT_2_DIV_PI   = 0.7978845608028654f;    // √(2/π)
constexpr float   SCALE           = -2.0f * SQRT_2_DIV_PI;  // -1.59576912, exp内缩放因子

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum, uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum,
                                uint32_t bufferNum)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        // 大核 / 小核分发: 前tailBlockNum个大核, 其余为小核
        if (blockIdx < tailBlockNum) {
            // 大核
            this->coreDataNum = bigCoreDataNum;
            this->tileNum     = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
            this->gmOffset    = blockIdx * bigCoreDataNum;
        } else {
            // 小核
            this->coreDataNum = smallCoreDataNum;
            this->tileNum     = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            this->gmOffset    = tailBlockNum * bigCoreDataNum
                              + (blockIdx - tailBlockNum) * smallCoreDataNum;
        }

        // GM视图: 扩展一倍tileDataNum作对齐余量
        xGm.SetGlobalBuffer((__gm__ T *)x + this->gmOffset,
                            this->coreDataNum + this->tileDataNum);
        zGm.SetGlobalBuffer((__gm__ T *)z + this->gmOffset,
                            this->coreDataNum + this->tileDataNum);

        // 三缓冲队列: 单槽大小 = tileDataNum * sizeof(T)
        pipe.InitBuffer(inQueueX,  bufferNum, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueZ, bufferNum, this->tileDataNum * sizeof(T));
        // 临时buffer: 存x^2用于算x^5 (=x^2*x^3), 单槽无队列
        pipe.InitBuffer(tmpBuf, this->tileDataNum * sizeof(T));
    }

    // 主循环: 满块迭代 + 尾块迭代
    // tailDataNum已在Host侧完成32B对齐, 此处直接使用
    __aicore__ inline void Process()
    {
        constexpr T one  = static_cast<T>(1.0f);
this->processDataNum = this->tileDataNum;
        // 满块迭代
        for (uint32_t i = 0; i + 1 < this->tileNum; i++) {
            
            CopyIn(i);
            Compute(one);
            CopyOut(i);
        }
        // 尾块迭代 (tailDataNum已对齐)
        if (this->tileNum > 0) {
            this->processDataNum = this->tailDataNum;
            CopyIn(this->tileNum - 1);
            Compute(one);
            CopyOut(this->tileNum - 1);
        }
    }

private:
    // CopyIn: GM→UB, Alloc→DataCopy→EnQue
    __aicore__ inline void CopyIn(uint32_t round)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopy(xLocal, xGm[round * this->tileDataNum],
                          this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    // Compute: 五阶近似GELU (exp形式)
    //   gelu(x) ≈ x / (1 + exp(-scale * (x + c1*x^3 + c3*x^5)))
    //   其中 scale = 2 * sqrt(2/pi)
    //   步骤: x²→save→x³→x⁵(=x²*x³)→c1*x³|c3*x⁵→sum→x+sum→scale→exp→+1→x/(1+exp)
    __aicore__ inline void Compute(T one)
    {
        constexpr T c1         = static_cast<T>(C1);
        constexpr T c3         = static_cast<T>(C3);
        constexpr T scale      = static_cast<T>(SCALE);        // -1.59576912

        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::LocalTensor<T> tLocal = tmpBuf.Get<T>();

        // Step 1: z = x², 同时保存x²到tmp
        AscendC::Mul(zLocal, xLocal, xLocal, this->processDataNum);
        AscendC::Mul(tLocal, xLocal, xLocal, this->processDataNum);
        // Step 2: z = z * x = x³
        AscendC::Mul(zLocal, zLocal, xLocal, this->processDataNum);
        // Step 3: t = x² * x³ = x⁵  (此时z仍是x³, 未被缩放)
        AscendC::Mul(tLocal, tLocal, zLocal, this->processDataNum);
        // Step 4: z = c1 * x³
        AscendC::Muls(zLocal, zLocal, c1, this->processDataNum);
        // Step 5: t = c3 * x⁵
        AscendC::Muls(tLocal, tLocal, c3, this->processDataNum);
        // Step 6: z = c1*x³ + c3*x⁵
        AscendC::Add(zLocal, zLocal, tLocal, this->processDataNum);
        // Step 7: z = x + c1*x³ + c3*x⁵
        AscendC::Add(zLocal, xLocal, zLocal, this->processDataNum);
        // Step 8: z = -scale * z = -1.59576912 * z
        AscendC::Muls(zLocal, zLocal, scale, this->processDataNum);
        // Step 9: z = exp(z)
        AscendC::Exp(zLocal, zLocal, this->processDataNum);
        // Step 10: z = z + 1
        AscendC::Adds(zLocal, zLocal, one, this->processDataNum);
        // Step 11: z = x / (1 + exp(z))
        AscendC::Div(zLocal, xLocal, zLocal, this->processDataNum);

        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // CopyOut: UB→GM, DeQue→DataCopy→Free
    __aicore__ inline void CopyOut(uint32_t round)
    {
        AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopy(zGm[round * this->tileDataNum], zLocal,
                          this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<> tmpBuf;  // 临时buffer: 存x^2→算x^5
    AscendC::GlobalTensor<T> xGm, zGm;
    uint32_t coreDataNum, tileNum, tileDataNum;
    uint32_t tailDataNum, processDataNum;
    uint32_t gmOffset;
};

// 入口核函数: REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output,
                                GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output,
            tiling_data.smallCoreDataNum, tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum, tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum,
            tiling_data.bufferNum);
    op.Process();
}