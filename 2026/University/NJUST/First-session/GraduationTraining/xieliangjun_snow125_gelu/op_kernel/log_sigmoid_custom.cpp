#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include <type_traits>

constexpr int32_t BUFFER_NUM = 2;

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
        // 获取当前 AI Core 的编号
        uint32_t coreIndex = AscendC::GetBlockIdx();

        // 先按大核长度估算当前核在 Global Memory 中的起始偏移。
        // 小核分支里会再把多估的部分扣掉
        uint32_t globalBufferIndex = bigCoreDataNum * coreIndex;
        this->tileDataNum = tileDataNum;

        if (coreIndex < tailBlockNum) {
            // 大核使用大核的数据量、tile 次数和最后一块大小。
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 小核使用基础数据量、tile 次数和最后一块大小。
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (coreIndex - tailBlockNum);
        }

        // 把当前核负责的输入、输出片段绑定到 GlobalTensor。
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        // 为输入队列、输出队列、float 临时空间分配 UB。
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // 每个核按 tileNum 循环处理自己的数据，最后一次可能是尾块。
        for (int32_t i = 0; i < this->tileNum; i++) {
            this->processDataNum = this->tileDataNum;
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
        // 从 Global Memory 搬一块输入数据到 UB。
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.AllocTensor<TYPE_Y>();

        AscendC::LocalTensor<float> tmp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.Get<float>();

        if constexpr (std::is_same_v<TYPE_X, float>) {
            AscendC::Muls(tmp0, xLocal, 1.0f, this->processDataNum);
        } else {
            AscendC::Cast(
                tmp0,
                xLocal,
                AscendC::RoundMode::CAST_NONE,
                this->processDataNum
            );
        }

        // 使用高阶 Sigmoid API 
        AscendC::Sigmoid(
            tmp1,
            tmp0,
            this->processDataNum
        );
        //使用高阶LOG API 
        AscendC::Log(
            tmp0,
            tmp1,
            this->processDataNum
        );

        if constexpr (std::is_same_v<TYPE_Y, float>) {
            AscendC::Muls(yLocal, tmp0, 1.0f, this->processDataNum);
        } else {
            AscendC::Cast(
                yLocal,
                tmp0,
                AscendC::RoundMode::CAST_RINT,
                this->processDataNum
            );
        }

        outQueueY.EnQue<TYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从 UB 把计算结果搬回 Global Memory。
        AscendC::LocalTensor<TYPE_Y> yLocal = outQueueY.DeQue<TYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 请完成Kernel侧代码实现
    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, tilingData.smallCoreDataNum,
            tilingData.bigCoreDataNum, tilingData.finalBigTileNum,
            tilingData.finalSmallTileNum, tilingData.tileDataNum,
            tilingData.smallTailDataNum, tilingData.bigTailDataNum,
            tilingData.tailBlockNum);
    op.Process();
}
