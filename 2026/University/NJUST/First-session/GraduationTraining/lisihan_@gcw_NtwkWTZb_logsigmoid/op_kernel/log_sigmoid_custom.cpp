#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
#include "tiling_key_log_sigmoid_custom.h"
#include "kernel_operator_dump_tensor_intf_impl.h"

constexpr int32_t BUFFER_NUM = 1;

// ==========================================================================
// KernelLogSigmoid 算子类（模板类）
//   dtypeX: 输入数据类型（half / float / bfloat16_t）
//   dtypeY: 输出数据类型（与输入相同）
// ==========================================================================
template<typename dtypeX, typename dtypeY>
class KernelLogSigmoid {
public:
    __aicore__ inline KernelLogSigmoid() {}

    // ---- Init：初始化函数 ----
    // 根据Host侧传入的Tiling参数确定当前核的数据范围、tile切分与内存分配
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t smallCoreDataNum, uint32_t bigCoreDataNum,
                                uint32_t finalBigTileNum,  uint32_t finalSmallTileNum,
                                uint32_t tileDataNum,
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum,
                                uint32_t tailBlockNum)
    {
        // 获取当前核的索引（0 ~ coreNum-1）
        uint32_t coreIdx = AscendC::GetBlockIdx();
        // 按大核数据量计算GM偏移（先假设都是大核）
        uint32_t globalBufferIndex = bigCoreDataNum * coreIdx;

        // 判断当前核是大核还是小核
        if (coreIdx < tailBlockNum) {
            // ---- 大核：前 tailBlockNum 个核 ----
            this->coreDataNum = bigCoreDataNum;      // 总元素数
            this->tileNum     = finalBigTileNum;     // tile循环次数
            this->tailDataNum = bigTailDataNum;      // 最后一个tile元素数
        } else {
            // ---- 小核：剩余的核心 ----
            this->coreDataNum = smallCoreDataNum;
            this->tileNum     = finalSmallTileNum;
            this->tailDataNum = smallTailDataNum;
            // 修正GM偏移：小核的起始地址在大核数据之后
            // 需要减去大核与小核的数据量差值
            globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum)
                                  * (coreIdx - tailBlockNum);
        }
        this->tileDataNum = tileDataNum;  // 每次tile处理的最大元素数

        // 设置GlobalTensor指向当前核的GM区域
        xGm.SetGlobalBuffer((__gm__ dtypeX*)x + globalBufferIndex,
                            this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ dtypeY*)y + globalBufferIndex,
                            this->coreDataNum);

        // 初始化队列内存（输入/输出各1个Buffer）
        pipe.InitBuffer(inQueueX,  BUFFER_NUM,
                        this->tileDataNum * sizeof(dtypeX));
        pipe.InitBuffer(outQueueY, BUFFER_NUM,
                        this->tileDataNum * sizeof(dtypeY));

        // 初始化TBuf临时变量内存
        // 对于bf16，TBuf分配float大小空间做高精度中间计算
        // 对于float/half，TBuf分配同类型大小空间
        if constexpr (std::is_same_v<dtypeX, bfloat16_t>) {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(dtypeX));
            pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(dtypeX));
        }
    }

    // ---- Process：核心处理函数 ----
    // 按tile循环依次执行 CopyIn → Compute → CopyOut
    // 最后一个tile使用尾块数据量tailDataNum
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            // 最后一个tile切换为尾块大小
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // ---- CopyIn：数据搬入任务 ----
    // 将输入数据从 Global Memory 搬入 Local Memory（VECIN队列）
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 分配输入 LocalTensor
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        // 从GM拷贝当前tile数据（跳过前progress个tile）
        AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum],
                          this->processDataNum);
        // 放入输入队列，通知Compute任务
        inQueueX.EnQue(xLocal);
    }

    // ---- Compute：核心计算任务 ----
    // LogSigmoid(x) = -ln(1 + exp(-x))
    // 步骤：-x → exp(-x) → 1+exp(-x) → ln() → -结果
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从输入队列取出数据
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        // 为输出结果分配内存
        AscendC::LocalTensor<dtypeY> yLocal = outQueueY.AllocTensor<dtypeY>();

        if constexpr (std::is_same_v<dtypeX, bfloat16_t>) {
            // ========== bfloat16 路径 ==========
            // bf16无法直接做数学运算，先Cast到float32，
            // 在float32精度下完成LogSigmoid计算，再Cast回bf16

            // 获取float精度的TBuf临时变量
            AscendC::LocalTensor<float> fBuf0 = tmpBuf0.Get<float>();
            AscendC::LocalTensor<float> fBuf1 = tmpBuf1.Get<float>();

            // Step 1: bf16 → float32 类型转换
            AscendC::Cast(fBuf0, xLocal, AscendC::RoundMode::CAST_NONE,
                          this->processDataNum);
            // Step 2: fBuf1 = -x
            AscendC::Muls(fBuf1, fBuf0, (float)-1.0, this->processDataNum);
            // Step 3: fBuf0 = exp(-x)
            AscendC::Exp(fBuf0, fBuf1, this->processDataNum);
            // Step 4: fBuf0 = 1 + exp(-x)
            AscendC::Adds(fBuf0, fBuf0, (float)1.0, this->processDataNum);
            // Step 5: fBuf1 = ln(1 + exp(-x))
            AscendC::Ln(fBuf1, fBuf0, this->processDataNum);
            // Step 6: fBuf0 = -ln(1 + exp(-x)) = LogSigmoid(x)
            AscendC::Muls(fBuf0, fBuf1, (float)-1.0, this->processDataNum);
            // Step 7: float32 → bf16 类型转换回输出
            AscendC::Cast(yLocal, fBuf0, AscendC::RoundMode::CAST_RINT,
                          this->processDataNum);

        } else {
            // ========== float16 / float32 路径 ==========
            // 使用与输入相同精度的TBuf直接计算

            AscendC::LocalTensor<dtypeX> tBuf0 = tmpBuf0.Get<dtypeX>();
            AscendC::LocalTensor<dtypeX> tBuf1 = tmpBuf1.Get<dtypeX>();

            // Step 1: tBuf0 = -x
            AscendC::Muls(tBuf0, xLocal, (dtypeX)-1.0, this->processDataNum);
            // Step 2: tBuf1 = exp(-x)
            AscendC::Exp(tBuf1, tBuf0, this->processDataNum);
            // Step 3: tBuf1 = 1 + exp(-x)
            AscendC::Adds(tBuf1, tBuf1, (dtypeX)1.0, this->processDataNum);
            // Step 4: tBuf0 = ln(1 + exp(-x))
            AscendC::Ln(tBuf0, tBuf1, this->processDataNum);
            // Step 5: yLocal = -ln(1 + exp(-x)) = LogSigmoid(x)
            AscendC::Muls(yLocal, tBuf0, (dtypeX)-1.0, this->processDataNum);
        }

        // 将计算结果放入输出队列，通知CopyOut任务
        outQueueY.EnQue<dtypeY>(yLocal);
        // 释放不再使用的输入数据，避免内存泄漏
        inQueueX.FreeTensor(xLocal);
    }

    // ---- CopyOut：结果搬出任务 ----
    // 将计算结果从 Local Memory 搬出到 Global Memory
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 从输出队列取出计算结果
        AscendC::LocalTensor<dtypeY> yLocal = outQueueY.DeQue<dtypeY>();
        // 将结果从Local Memory拷贝到GM指定位置
        AscendC::DataCopy(yGm[progress * this->tileDataNum], yLocal,
                          this->processDataNum);
        // 释放输出临时内存
        outQueueY.FreeTensor(yLocal);
    }

private:
    // ---- 资源管理成员变量 ----
    AscendC::TPipe pipe;                                    // 内存管理总管
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX;   // 输入队列
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;  // 输出队列
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;    // TBuf0: 中间计算暂存
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;    // TBuf1: 中间计算暂存
    AscendC::GlobalTensor<dtypeX> xGm;                     // 输入GM张量
    AscendC::GlobalTensor<dtypeY> yGm;                     // 输出GM张量
    // ---- Tiling计算参数 ----
    uint32_t coreDataNum;     // 当前核处理的总元素数
    uint32_t tileNum;         // tile循环总次数
    uint32_t tileDataNum;     // 每个tile最大元素数
    uint32_t tailDataNum;     // 尾块元素数
    uint32_t processDataNum;  // 当前tile实际处理的元素数
};

// ==========================================================================
// 核函数入口（设备侧，在AI Core上执行）
// 模板参数 D_T_X/D_T_Y 由Host侧 ASCENDC_TPL_SEL_PARAM 传入
// ==========================================================================
template <typename D_T_X, typename D_T_Y>
__global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册TilingData类型
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    // 从tiling参数中解析Host侧设置的切分数据
    GET_TILING_DATA_WITH_STRUCT(LogSigmoidCustomTilingData, tiling_data, tiling);

    // 实例化算子类并传入数据类型模板参数
    KernelLogSigmoid<D_T_X, D_T_Y> op;
    // 传入Tiling参数完成初始化
    op.Init(x, y,
            tiling_data.smallCoreDataNum,  tiling_data.bigCoreDataNum,
            tiling_data.finalBigTileNum,   tiling_data.finalSmallTileNum,
            tiling_data.tileDataNum,
            tiling_data.smallTailDataNum,  tiling_data.bigTailDataNum,
            tiling_data.tailBlockNum);
    // 执行三级流水线计算
    op.Process();
}
