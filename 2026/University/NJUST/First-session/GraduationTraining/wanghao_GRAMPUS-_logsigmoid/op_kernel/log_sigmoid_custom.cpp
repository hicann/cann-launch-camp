#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

// 模板类：自动适配不同数据类型
template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() = default;

    // 初始化：绑定内存、计算当前核的分片范围
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        const uint32_t coreId = AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;

        // 计算当前核在全局内存中的起始偏移
        uint32_t globalOffset = bigCoreDataNum * coreId;
        if (coreId < tailBlockNum) {
            // 大核：多处理1块数据
            this->coreTotalNum = bigCoreDataNum;
            this->tileTotalNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 普通核
            this->coreTotalNum = smallCoreDataNum;
            this->tileTotalNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalOffset -= (bigCoreDataNum - smallCoreDataNum) * (coreId - tailBlockNum);
        }

        // 绑定全局内存
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_X*>(x) + globalOffset, this->coreTotalNum);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ TYPE_Y*>(y) + globalOffset, this->coreTotalNum);

        // 初始化双缓冲队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // bf16场景：申请float32临时缓存
        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            pipe.InitBuffer(tmpFloatIn, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatOut, this->tileDataNum * sizeof(float));
        }
    }

    // 主循环：分片处理数据
    __aicore__ inline void Process()
    {
        if (coreTotalNum == 0 || tileTotalNum == 0) return;

        for (uint32_t i = 0; i < tileTotalNum; ++i) {
            // 最后一片用尾数据长度
            currentDataNum = (i == tileTotalNum - 1) ? tailDataNum : tileDataNum;
            CopyIn(i);   // 数据搬入片上
            Compute();   // 计算
            CopyOut(i);  // 结果写回全局内存
        }
    }

private:
    // 搬入数据
    __aicore__ inline void CopyIn(uint32_t tileIdx)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[tileIdx * tileDataNum], currentDataNum);
        inQueueX.EnQue(xLocal);
    }

    // 核心计算
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            // bf16分支：转float32计算，完了再转回来
            AscendC::LocalTensor<float> xFloat = tmpFloatIn.Get<float>();
            AscendC::LocalTensor<float> yFloat = tmpFloatOut.Get<float>();

            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, currentDataNum);
            AscendC::Sigmoid(yFloat, xFloat, currentDataNum);
            AscendC::Log(yFloat, yFloat, currentDataNum);
            AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, currentDataNum);
        } else {
            // float16/float32分支：直接计算
            AscendC::Sigmoid(yLocal, xLocal, currentDataNum);
            AscendC::Log(yLocal, yLocal, currentDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    // 写出结果
    __aicore__ inline void CopyOut(uint32_t tileIdx)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[tileIdx * tileDataNum], yLocal, currentDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    // 双缓冲队列
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // bf16专用临时缓存
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatIn;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatOut;

    // 全局内存句柄
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    // 运行参数
    uint32_t coreTotalNum = 0;
    uint32_t tileTotalNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t currentDataNum = 0;
};

// 算子入口函数
extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y,
           tilingData.smallCoreDataNum,
           tilingData.bigCoreDataNum,
           tilingData.finalBigTileNum,
           tilingData.finalSmallTileNum,
           tilingData.tileDataNum,
           tilingData.smallTailDataNum,
           tilingData.bigTailDataNum,
           tilingData.tailBlockNum);
    op.Process();
}
