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
        this->tileDataNum = tileDataNum;

        // 前 tailBlockNum 个 Core 是 big core，多处理一个数据块
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

        // 绑定输入输出全局内存
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);

        // 初始化输入、输出双缓冲队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));

        // BF16 分支需要两个 float 临时 buffer，用于类型中转计算
        if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            pipe.InitBuffer(tmpFloatX, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpFloatZ, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;

        for (int32_t i = 0; i < loopCount; i++) {
            // 最后一个 tile 可能不是完整 tile，使用尾数据量
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
        // 从输入队列申请 UB 空间，从 GM 拷贝数据
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
        AscendC::DataCopy(xLocal,
                          xGm[progress * this->tileDataNum],
                          this->processDataNum);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 取出输入数据，申请输出空间
        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.AllocTensor<TYPE_Y>();

        if constexpr (std::is_same<TYPE_X, __bf16>::value) {
            // ==================================================
            // BF16 特殊分支：硬件不支持直接计算 LogSigmoid
            // 计算流程：BF16 -> float -> Sigmoid -> Log -> BF16
            // ==================================================
            AscendC::LocalTensor<float> xFloat = tmpFloatX.Get<float>();
            AscendC::LocalTensor<float> zFloat = tmpFloatZ.Get<float>();

            // BF16 转 float
            AscendC::Cast(xFloat,
                          xLocal,
                          AscendC::RoundMode::CAST_NONE,
                          this->processDataNum);

            // 第一步：计算 Sigmoid(x)
            AscendC::Sigmoid(zFloat,
                             xFloat,
                             this->processDataNum);

            // 第二步：计算 log(Sigmoid(x))，原地计算节省空间
            AscendC::Log(zFloat,
                         zFloat,
                         this->processDataNum);

            // float 转回 BF16
            AscendC::Cast(zLocal,
                          zFloat,
                          AscendC::RoundMode::CAST_RINT,
                          this->processDataNum);
        } else {
            // ==================================================
            // float16 / float32 通用分支
            // 两步原地计算：先 Sigmoid，再取自然对数
            // ==================================================
            // 第一步：计算 Sigmoid(x)
            AscendC::Sigmoid(zLocal,
                             xLocal,
                             this->processDataNum);
            // 第二步：计算 log(Sigmoid(x))，原地计算无需额外临时空间
            AscendC::Log(zLocal,
                         zLocal,
                         this->processDataNum);
        }

        // 输出入队，释放输入空间
        outQueueZ.EnQue<TYPE_Y>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出结果，写回全局内存
        AscendC::LocalTensor<TYPE_Y> zLocal = outQueueZ.DeQue<TYPE_Y>();
        AscendC::DataCopy(zGm[progress * this->tileDataNum],
                          zLocal,
                          this->processDataNum);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    // 输入、输出双缓冲队列
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    // BF16 分支使用的 float 临时中转 buffer
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpFloatZ;

    // 全局内存 Tensor
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> zGm;

    // 运行时参数
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

// 算子入口函数，名称需与 Host 侧注册名一致

extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
                                                       GM_ADDR y,
                                                       GM_ADDR workspace,
                                                       GM_ADDR tiling)
{
    // 注册并读取 Tiling 数据
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据输入数据类型自动实例化模板
    KernelLogSigmoid<DTYPE_X, DTYPE_Y> op;

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
