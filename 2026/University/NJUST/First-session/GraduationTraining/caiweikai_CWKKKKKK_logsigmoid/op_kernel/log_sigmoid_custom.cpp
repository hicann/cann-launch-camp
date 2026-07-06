#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>   // 用于std::is_same判断类型

constexpr int32_t BUFFER_NUM = 2;   // 双缓冲流水

// 模板参数：TYPE_X 输入类型，TYPE_Y 输出类型（通常相同）
template<typename TYPE_X, typename TYPE_Y>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    __aicore__ inline void Init(GM_ADDR x,
                                GM_ADDR y,
                                uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,
                                uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum,
                                uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        uint32_t coreIdx = AscendC::GetBlockIdx();   // 当前核索引

        this->tileDataNum = tileDataNum;

        // 根据核索引判断是否属于尾核（分配更多数据）
        uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;   // 大核的起始偏移

        if (coreIdx < tailBlockNum) {
            // 尾核：多处理一个BLOCK_SIZE的数据
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 普通核
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            // 非尾核的起始偏移需要减去之前尾核多占的数据
            globalBufferIndex -=
                (bigCoreDataNum - smallCoreDataNum) * (coreIdx - tailBlockNum);
        }

        // 设置全局内存指针，指向当前核负责的数据段
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        // 初始化输入输出队列（双缓冲）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // 若为bfloat16，额外分配float临时缓冲区用于中转计算
        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatY, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        // 依次处理每个tile
        for (uint32_t i = 0; i < this->tileNum; i++) {
            // 最后一个tile使用tailDataNum，其余使用完整tileDataNum
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            } else {
                this->processDataNum = this->tileDataNum;
            }

            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        // 从全局内存拷贝到本地内存
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        // 针对bfloat16需要转float计算（因为部分数学函数不支持bf16）
        if constexpr (std::is_same<TYPE_X, bfloat16_t>::value) {
            AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
            AscendC::LocalTensor<float> yFloat = tmpFloatY.Get<float>();

            // bf16 -> float
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            // y = sigmoid(x)
            AscendC::Sigmoid(yFloat, xFloat, this->processDataNum);
            // y = ln(sigmoid(x)) 即 LogSigmoid
            AscendC::Ln(yFloat, yFloat, this->processDataNum);

            // float -> bf16
            AscendC::Cast(yLocal, yFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else {
            // half 和 float 直接计算
            AscendC::Sigmoid(yLocal, xLocal, this->processDataNum);
            AscendC::Ln(yLocal, yLocal, this->processDataNum);
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        // 将计算结果从本地拷贝回全局内存
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;

    // 输入输出队列，位置分别为VECIN和VECOUT
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // 用于bfloat16中转的临时float缓冲区
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatY;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;

    uint32_t coreDataNum;      // 当前核处理的元素总数
    uint32_t tileNum;          // 当前核的tile个数
    uint32_t tileDataNum;      // 每个完整tile的元素数
    uint32_t tailDataNum;      // 最后一个tile的元素数
    uint32_t processDataNum;   // 当前tile实际处理的元素数
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;   // DTYPE_X/Y由编译时自动推导

    op.Init(x,
            y,
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
