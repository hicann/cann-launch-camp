#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

// 双缓冲队列数
constexpr uint32_t BUFFER_NUM = 2;
// GELU 近似计算中用到的常数
constexpr float GELU_INV_SQRT2 = 0.70710678118654752440f; // 1/sqrt(2)
constexpr float GELU_HALF = 0.5f;
constexpr float GELU_ONE = 1.0f;
constexpr float GELU_NEG_CLAMP = -6.0f; // 输入截断下限，防止exp/erf溢出

// 向上取整除法
__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return y == 0 ? 0 : (x + y - 1) / y;
}

// 向上对齐
__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t align)
{
    return CeilDiv(x, align) * align;
}

// 取较小值
__aicore__ inline uint32_t MinU32(uint32_t x, uint32_t y)
{
    return x < y ? x : y;
}

// 取较大值
__aicore__ inline uint32_t MaxU32(uint32_t x, uint32_t y)
{
    return x > y ? x : y;
}

// ---------- 通用模板类（用于 FP16 或其他非 float 类型） ----------
template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    // 初始化：划分核间数据、设置缓冲区
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length, uint32_t inputTileLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();  // 当前核总数
        const uint32_t blockIdx = AscendC::GetBlockIdx();  // 当前核索引
        // 按块大小均匀划分数据，保证各核负载均衡
        const uint32_t blockStart = static_cast<uint32_t>((static_cast<uint64_t>(length) * blockIdx) / blockNum);
        const uint32_t blockEnd = static_cast<uint32_t>((static_cast<uint64_t>(length) * (blockIdx + 1)) / blockNum);

        this->blockLength = blockEnd - blockStart;          // 当前核负责的元素数
        this->tileLength = MaxU32(1, inputTileLength);      // tile大小（至少1）
        // 根据数据类型计算向量化对齐粒度（通常与硬件向量指令宽度有关，此处32字节对齐）
        const uint32_t elementPerBlock = 32 / sizeof(DT_INPUT_X);
        this->alignedTileLength = MaxU32(elementPerBlock, AlignUp(this->tileLength, elementPerBlock));

        // 设置全局内存指针，偏移到当前核负责的起始位置
        inputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + blockStart, this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + blockStart, this->blockLength);

        // 初始化队列和缓冲区（双缓冲）
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->alignedTileLength * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, this->alignedTileLength * sizeof(DT_INPUT_X));
        // 计算缓冲区用于类型提升（例如 FP16 -> FP32 计算），需要额外空间
        pipe.InitBuffer(calcBuf, this->alignedTileLength * sizeof(float) * 2);
    }

    // 主处理流程：循环处理每个 tile
    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            const uint32_t curLength = MinU32(this->tileLength, this->blockLength - offset);
            CopyIn(offset, curLength);   // 将数据从 GM 搬入 UB（输入队列）
            Compute(curLength);          // 在 UB 上执行 GELU 计算
            CopyOut(offset, curLength);  // 将结果从 UB 搬回 GM（输出队列）
        }
    }

private:
    // 数据搬入：从全局内存拷贝到 LocalTensor（UB），支持 pad 填充（便于向量化）
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.template AllocTensor<DT_INPUT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{true, 0, 0, 0}; // 自动填充0以对齐
        AscendC::DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
        inputQueue.EnQue(inputLocal); // 入队，供计算使用
    }

    // 计算核心：实现 GELU 近似公式：0.5 * x * (1 + erf(x / sqrt(2)))
    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<DT_INPUT_X> inputLocal = inputQueue.template DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.template AllocTensor<DT_INPUT_X>();
        // 取计算缓冲区，分为两部分：xFloat 和 work（用于中间结果）
        AscendC::LocalTensor<float> xFloat = calcBuf.Get<float>();
        AscendC::LocalTensor<float> work = xFloat[this->alignedTileLength];

        // 将输入（可能为 FP16）转换为 FP32 进行计算
        AscendC::Cast(xFloat, inputLocal, AscendC::RoundMode::CAST_NONE, curLength);
        // 截断 x 至 [-6, +inf) ，避免 erf 溢出（实际上只做了下限截断）
        AscendC::Maxs(xFloat, xFloat, GELU_NEG_CLAMP, curLength);
        // t = x / sqrt(2)
        AscendC::Muls(work, xFloat, GELU_INV_SQRT2, curLength);
        // 计算 erf(t)
        AscendC::Erf(work, work, curLength);
        // 1 + erf(t)
        AscendC::Adds(work, work, GELU_ONE, curLength);
        // 0.5 * (1 + erf(t))
        AscendC::Muls(work, work, GELU_HALF, curLength);
        // x * 0.5 * (1 + erf(x/sqrt2))
        AscendC::Mul(work, xFloat, work, curLength);
        // 将结果转换回原始数据类型（如 FP16）
        AscendC::Cast(outputLocal, work, AscendC::RoundMode::CAST_NONE, curLength);

        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal); // 释放输入缓冲区
    }

    // 数据搬出：将结果从 UB 拷贝回 GM
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<DT_INPUT_X> outputLocal = outputQueue.template DeQue<DT_INPUT_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(DT_INPUT_X)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[offset], outputLocal, copyParams);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;                                    // 管道，管理UB内存
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueue;   // 输入队列
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueue; // 输出队列
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;     // 计算缓冲区
    AscendC::GlobalTensor<DT_INPUT_X> inputGm;
    AscendC::GlobalTensor<DT_INPUT_X> outputGm;
    uint32_t blockLength = 0;
    uint32_t tileLength = 1;
    uint32_t alignedTileLength = 1;
};

// ---------- 模板特化：针对 float 类型的高效实现 ----------
// 由于 float 无需类型提升，可直接在原始类型上计算，节省内存和转换开销
template <>
class KernelGelu<float> {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length, uint32_t inputTileLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockStart = static_cast<uint32_t>((static_cast<uint64_t>(length) * blockIdx) / blockNum);
        const uint32_t blockEnd = static_cast<uint32_t>((static_cast<uint64_t>(length) * (blockIdx + 1)) / blockNum);

        this->blockLength = blockEnd - blockStart;
        this->tileLength = MaxU32(1, inputTileLength);
        // float 向量化对齐粒度设为 8（32字节/4字节），确保向量指令高效
        this->alignedTileLength = MaxU32(8, AlignUp(this->tileLength, 8));

        inputGm.SetGlobalBuffer((__gm__ float *)input_x + blockStart, this->blockLength);
        outputGm.SetGlobalBuffer((__gm__ float *)output + blockStart, this->blockLength);

        // 只需输入输出队列，无需额外的计算缓冲区
        pipe.InitBuffer(inputQueue, BUFFER_NUM, this->alignedTileLength * sizeof(float));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, this->alignedTileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < this->blockLength; offset += this->tileLength) {
            const uint32_t curLength = MinU32(this->tileLength, this->blockLength - offset);
            CopyIn(offset, curLength);
            Compute(curLength);
            CopyOut(offset, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<float> inputLocal = inputQueue.template AllocTensor<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(inputLocal, inputGm[offset], copyParams, padParams);
        inputQueue.EnQue(inputLocal);
    }

    // 直接使用 float 计算，无需类型转换，运算更高效
    __aicore__ inline void Compute(uint32_t curLength)
    {
        AscendC::LocalTensor<float> inputLocal = inputQueue.template DeQue<float>();
        AscendC::LocalTensor<float> outputLocal = outputQueue.template AllocTensor<float>();

        // 注意：这里复用 inputLocal 和 outputLocal 来减少临时变量
        // 先对输入进行截断（结果存到 outputLocal）
        AscendC::Maxs(outputLocal, inputLocal, GELU_NEG_CLAMP, curLength);
        // t = x / sqrt(2)，结果存到 inputLocal（覆盖原输入）
        AscendC::Muls(inputLocal, outputLocal, GELU_INV_SQRT2, curLength);
        // erf(t)，结果仍存 inputLocal
        AscendC::Erf(inputLocal, inputLocal, curLength);
        // 1 + erf(t)
        AscendC::Adds(inputLocal, inputLocal, GELU_ONE, curLength);
        // 0.5 * (1 + erf(t))
        AscendC::Muls(inputLocal, inputLocal, GELU_HALF, curLength);
        // x * 0.5 * (1 + erf(...))，其中 x 保存在 outputLocal（被截断后的原始值）
        AscendC::Mul(outputLocal, outputLocal, inputLocal, curLength);

        outputQueue.EnQue(outputLocal);
        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t curLength)
    {
        AscendC::LocalTensor<float> outputLocal = outputQueue.template DeQue<float>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLength * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(outputGm[offset], outputLocal, copyParams);
        outputQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outputQueue;
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    uint32_t blockLength = 0;
    uint32_t tileLength = 1;
    uint32_t alignedTileLength = 1;
};

// ---------- 核函数入口 ----------
// 每个核执行此函数，根据模板参数 DT_INPUT_X 实例化不同的 KernelGelu
template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册 tiling 数据结构（必须与算子注册时一致）
    REGISTER_TILING_DEFAULT(GeluTilingData);
    // 从 tiling 内存中解析出 tiling_data 结构体
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    // 创建算子对象并执行
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length, tiling_data.tileLength);
    op.Process();
}
