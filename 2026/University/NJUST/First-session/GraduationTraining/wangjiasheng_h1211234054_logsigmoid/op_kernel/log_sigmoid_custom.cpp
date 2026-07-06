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
        // 当前 AI Core 编号
        uint32_t coreNum = AscendC::GetBlockIdx();

        // 当前 Core 在全局内存 GM 中的起始偏移
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();

        // 每个 tile 处理的数据个数
        this->tileDataNum = tileDataNum;

        // 前 tailBlockNum 个 Core 是 big core
        if (coreNum < tailBlockNum) {
            this->coreDataNum = bigCoreDataNum;
            this->tileNum = finalBigTileNum;
            this->tailDataNum = bigTailDataNum;
        } else {
            // 剩余 Core 是 small core
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;

            // 修正 small core 的 GM 起始偏移
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) *
                                 (AscendC::GetBlockIdx() - tailBlockNum);
        }

        // 设置输入 GM Tensor
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);

        // 设置输出 GM Tensor
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        // 输入队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));

        // 输出队列
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // 为所有类型准备 float 临时 buffer
        pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmpFloatZ, this->tileDataNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;

        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
            // 最后一个 tile 可能不是完整 tile
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
        // 从输入队列申请 LocalTensor
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();

        // 从 GM 拷贝数据到 UB
        AscendC::DataCopy(xLocal,
                          xGm[progress * this->tileDataNum],
                          this->processDataNum);

        // 入队，供 Compute 使用
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 取出输入
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();

        // 申请输出
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        // 获取临时 float buffer
        AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
        AscendC::LocalTensor<float> zFloat = tmpFloatZ.Get<float>();

        // 根据输入类型进行不同的处理
        if constexpr (std::is_same<TYPE_X, float>::value) {
            // float 类型直接计算
            ComputeLogSigmoid(zFloat, xLocal, this->processDataNum);
            // 直接赋值给输出（float 类型）
            AscendC::DataCopy(zLocal, zFloat, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, half>::value) {
            // half 转 float
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            ComputeLogSigmoid(zFloat, xFloat, this->processDataNum);
            // float 转 half
            AscendC::Cast(zLocal, zFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        } else if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            // bf16 转 float
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            ComputeLogSigmoid(zFloat, xFloat, this->processDataNum);
            // float 转 bf16
            AscendC::Cast(zLocal, zFloat, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        }

        // 输出入队
        outQueueZ.EnQue<TYPE_Y>(zLocal);

        // 释放输入 Tensor
        inQueueX.FreeTensor(xLocal);
    }

 __aicore__ inline void ComputeLogSigmoid(AscendC::LocalTensor<float>& output, AscendC::LocalTensor<float>& input, uint32_t dataNum)
{
    // 1. 计算 -x
    AscendC::Muls(output, input, -1.0f, dataNum);
    // 2. 计算 exp(-x)
    AscendC::Exp(output, output, dataNum);
    // 3. output = 1 + exp(-x)
    AscendC::Adds(output, output, 1.0f, dataNum);
    // 4. output = log(1 + exp(-x))
    AscendC::Log(output, output, dataNum);
    // 5. output = -log(1 + exp(-x))
    AscendC::Muls(output, output, -1.0f, dataNum);
}

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出结果
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();

        // 写回 GM
        AscendC::DataCopy(zGm[progress * this->tileDataNum],
                          zLocal,
                          this->processDataNum);

        // 释放输出 Tensor
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    // 输入队列
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;

    // 输出队列
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    // 临时 float buffer
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatZ;

    // 输入全局 Tensor
    AscendC::GlobalTensor<TYPE_X> xGm;

    // 输出全局 Tensor
    AscendC::GlobalTensor<TYPE_Y> zGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                         GM_ADDR y,
                                                         GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    // 注册并读取 TilingData
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

switch (tilingData.dataType) {
        case 0: { // float32
            KernelLogSigmoid<float, float> op;
            op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                    tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                    tilingData.tileDataNum, tilingData.smallTailDataNum,
                    tilingData.bigTailDataNum, tilingData.tailBlockNum);
            op.Process();
            break;
        }
        case 1: { // float16
            KernelLogSigmoid<half, half> op;
            op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                    tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                    tilingData.tileDataNum, tilingData.smallTailDataNum,
                    tilingData.bigTailDataNum, tilingData.tailBlockNum);
            op.Process();
            break;
        }
        case 2: { // bfloat16
            KernelLogSigmoid<bfloat16_t, bfloat16_t> op;
            op.Init(x, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum,
                    tilingData.finalBigTileNum, tilingData.finalSmallTileNum,
                    tilingData.tileDataNum, tilingData.smallTailDataNum,
                    tilingData.bigTailDataNum, tilingData.tailBlockNum);
            op.Process();
            break;
        }
        default:
            // 不应出现
            break;
    }
}
