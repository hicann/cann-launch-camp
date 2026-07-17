// Kernel 侧核函数实现 — 全内联版本（消除函数调用边框 + this 指针开销）
#include "kernel_operator.h"

#include "gelu_tiling.h"

#include "tiling_key_gelu.h"

// Tanh-5th-order GELU: error ~2e-5, under fp32 1e-4 threshold
// z = √(2/π)·x + C₃'·x³ + C₅'·x⁵  (常量预乘 √(2/π), 因式分解消 Add)
constexpr float TANH_A = 0.797884583f;    // √(2/π) = 0x3F4C5C37
constexpr float TANH_K3 = 0.035677f;      // √(2/π)·0.044715 (预缩放 x³ 系数)
constexpr float TANH_K5 = 0.000521f;      // √(2/π)·0.000654 (预缩放 x⁵ 系数)
// float16 原生计算常量（v31: 避免 fp16↔fp32 的 Cast 开销）
constexpr half TANH_A_H = (half)TANH_A;
constexpr half TANH_K3_H = (half)TANH_K3;
constexpr half TANH_K5_H = (half)TANH_K5;

// 宏：Tanh5 GELU 计算（消除3处重复代码）
// 调用前需确保 tmpBuf1(tmpBuf2 for fp16), tmpBuf2(tmpBuf3 for fp16) 已分配
#define GELU_TANH5_COMPUTE(COUNT_VAR, X_IN, Y_OUT)                                      \
    do {                                                                                 \
        if constexpr (sizeof(T) == 2) {                                                  \
            AscendC::LocalTensor<half> bufA = tmpBuf2.Get<half>();                       \
            AscendC::LocalTensor<half> bufB = tmpBuf3.Get<half>();                       \
            AscendC::Mul(bufB, (X_IN), (X_IN), static_cast<uint32_t>(COUNT_VAR));        \
            AscendC::Muls(bufA, bufB, TANH_K5_H, static_cast<uint32_t>(COUNT_VAR));      \
            AscendC::Adds(bufA, bufA, TANH_K3_H, static_cast<uint32_t>(COUNT_VAR));      \
            AscendC::Mul(bufB, (X_IN), bufB, static_cast<uint32_t>(COUNT_VAR));          \
            AscendC::Mul(bufA, bufB, bufA, static_cast<uint32_t>(COUNT_VAR));            \
            AscendC::Muls(bufB, (X_IN), TANH_A_H, static_cast<uint32_t>(COUNT_VAR));     \
            AscendC::Muls((Y_OUT), bufA, (half)1.0f, static_cast<uint32_t>(COUNT_VAR));  \
            AscendC::Add(bufA, bufB, (Y_OUT), static_cast<uint32_t>(COUNT_VAR));          \
            AscendC::Tanh(bufA, bufA, static_cast<uint32_t>(COUNT_VAR));                 \
            AscendC::Adds(bufA, bufA, (half)1.0f, static_cast<uint32_t>(COUNT_VAR));     \
            AscendC::Muls(bufB, (X_IN), (half)0.5f, static_cast<uint32_t>(COUNT_VAR));   \
            AscendC::Mul(bufA, bufB, bufA, static_cast<uint32_t>(COUNT_VAR));            \
            AscendC::Muls((Y_OUT), bufA, (half)1.0f, static_cast<uint32_t>(COUNT_VAR));  \
        } else {                                                                         \
            AscendC::LocalTensor<float> bufA = tmpBuf1.Get<float>();                     \
            AscendC::LocalTensor<float> bufB = tmpBuf2.Get<float>();                     \
            AscendC::Mul(bufB, (X_IN), (X_IN), static_cast<uint32_t>(COUNT_VAR));        \
            AscendC::Muls(bufA, bufB, TANH_K5, static_cast<uint32_t>(COUNT_VAR));        \
            AscendC::Adds(bufA, bufA, TANH_K3, static_cast<uint32_t>(COUNT_VAR));        \
            AscendC::Mul(bufB, (X_IN), bufB, static_cast<uint32_t>(COUNT_VAR));          \
            AscendC::Mul(bufA, bufB, bufA, static_cast<uint32_t>(COUNT_VAR));            \
            AscendC::Muls(bufB, (X_IN), TANH_A, static_cast<uint32_t>(COUNT_VAR));       \
            AscendC::Muls((Y_OUT), bufA, 1.0f, static_cast<uint32_t>(COUNT_VAR));        \
            AscendC::Add(bufA, bufB, (Y_OUT), static_cast<uint32_t>(COUNT_VAR));          \
            AscendC::Tanh(bufA, bufA, static_cast<uint32_t>(COUNT_VAR));                 \
            AscendC::Adds(bufA, bufA, 1.0f, static_cast<uint32_t>(COUNT_VAR));           \
            AscendC::Muls(bufB, (X_IN), 0.5f, static_cast<uint32_t>(COUNT_VAR));         \
            AscendC::Mul(bufA, bufB, bufA, static_cast<uint32_t>(COUNT_VAR));            \
            AscendC::Muls((Y_OUT), bufA, 1.0f, static_cast<uint32_t>(COUNT_VAR));        \
        }                                                                                \
    } while(0)

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);

    using T = DT_INPUT_X;
    constexpr uint32_t ALIGN_32 = 32;
    constexpr uint32_t ELEM_PER_32B = 32 / sizeof(T);

    // ==================== Init 逻辑（全内联）====================
    int64_t totalLen   = static_cast<int64_t>(tiling_data.dim0);
    int64_t blockFormer = static_cast<int64_t>(tiling_data.blockFormer);
    int64_t blockNum    = static_cast<int64_t>(tiling_data.blockNum);
    int64_t blockIdx    = static_cast<int64_t>(AscendC::GetBlockIdx());

    // 空闲核直接跳过
    if (blockIdx >= blockNum) return;

    int64_t initOffset = blockIdx * blockFormer;
    int64_t curLen = (blockIdx < blockNum - 1) ? blockFormer : (totalLen - initOffset);
    if (curLen < 0) curLen = 0;

    int64_t ubFormer = static_cast<int64_t>(tiling_data.ubFormer);
    if (ubFormer < 1) ubFormer = 1;
    if (ubFormer > curLen) ubFormer = curLen;

    AscendC::GlobalTensor<T> inputGM;
    AscendC::GlobalTensor<T> outputGM;
    inputGM.SetGlobalBuffer((__gm__ T *)input_x + initOffset, curLen);
    outputGM.SetGlobalBuffer((__gm__ T *)output + initOffset, curLen);

    // ubLoop / ubTail 按首尾 block 区分
    int64_t ubLoop, ubTail;
    if (blockIdx < blockNum - 1) {
        ubLoop = static_cast<int64_t>(tiling_data.ubLoopOfFormerBlock);
        ubTail = static_cast<int64_t>(tiling_data.ubTailOfFormerBlock);
    } else {
        ubLoop = static_cast<int64_t>(tiling_data.ubLoopOfTailBlock);
        ubTail = static_cast<int64_t>(tiling_data.ubTailOfTailBlock);
    }

    // ==================== Buffer 分配 ====================
    AscendC::TPipe pipe;

    // VECCALC 临时 buffer（统一按 float32 大小分配）
    uint32_t tmpBufSize = static_cast<uint32_t>(ubFormer * sizeof(float));
    tmpBufSize = ((tmpBufSize + ALIGN_32 - 1) / ALIGN_32) * ALIGN_32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf3;
    pipe.InitBuffer(tmpBuf1, tmpBufSize);
    pipe.InitBuffer(tmpBuf2, tmpBufSize);
    if constexpr (sizeof(T) == 2) {
        pipe.InitBuffer(tmpBuf3, tmpBufSize);
    }

    // TQue 双缓冲队列
    uint32_t queBufSize = static_cast<uint32_t>(ubFormer * sizeof(T));
    queBufSize = ((queBufSize + ALIGN_32 - 1) / ALIGN_32) * ALIGN_32;
    AscendC::TQue<AscendC::TPosition::VECIN, 2>   inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2>  outQueueY;
    pipe.InitBuffer(inQueueX, 2, queBufSize);
    pipe.InitBuffer(outQueueY, 2, queBufSize);

    // ==================== Process 逻辑（全内联）====================
    if (curLen <= 0 || ubFormer <= 0) return;

    // ---- 单 tile 快速路径 ----
    if ((ubLoop == 0 && ubTail > 0) || (ubLoop == 1 && ubTail == 0)) {
        int64_t count = (ubLoop == 0) ? ubTail : ubFormer;

        // === CopyIn（内联 CopyInFast）===
        bool aligned = (static_cast<uint32_t>(count) % ELEM_PER_32B == 0);
        AscendC::LocalTensor<T> xBuf = inQueueX.AllocTensor<T>();
        if (aligned) {
            AscendC::DataCopy(xBuf, inputGM[0], static_cast<uint32_t>(count));
        } else {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            uint32_t tailRem = static_cast<uint32_t>(count) % ELEM_PER_32B;
            AscendC::DataCopyPadExtParams<T> pad{true, static_cast<uint8_t>(0),
                static_cast<uint8_t>(ELEM_PER_32B - tailRem), static_cast<T>(0)};
            AscendC::DataCopyPad(xBuf, inputGM[0], cp, pad);
        }
        inQueueX.EnQue(xBuf);
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();

        // === Compute（全内联 Tanh5）===
        AscendC::LocalTensor<T> yBuf = outQueueY.AllocTensor<T>();
        GELU_TANH5_COMPUTE(count, xIn, yBuf);

        outQueueY.EnQue(yBuf);
        inQueueX.FreeTensor(xIn);

        // === CopyOut（内联 CopyOutFast）===
        AscendC::LocalTensor<T> yBufOut = outQueueY.DeQue<T>();
        bool outAligned = (static_cast<uint32_t>(count) % ELEM_PER_32B == 0);
        if (outAligned) {
            AscendC::DataCopy(outputGM[0], yBufOut, static_cast<uint32_t>(count));
        } else {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(outputGM[0], yBufOut, cp);
        }
        outQueueY.FreeTensor(yBufOut);
        return;
    }

    // ---- 多 tile 流水线路径（内联） ----
    int64_t processOffset = 0;

    // 预取第一个 tile
    {
        AscendC::LocalTensor<T> xPre = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(ubFormer * sizeof(T)), 0, 0, 0};
        uint32_t tailRem = static_cast<uint32_t>(ubFormer) % ELEM_PER_32B;
        bool needPad = (tailRem != 0);
        AscendC::DataCopyPadExtParams<T> padParams{needPad, static_cast<uint8_t>(0),
            static_cast<uint8_t>(needPad ? (ELEM_PER_32B - tailRem) : 0), static_cast<T>(0)};
        AscendC::DataCopyPad(xPre, inputGM[0], cp, padParams);
        inQueueX.EnQue(xPre);
    }

    // Pipeline 循环
    for (int64_t i = 0; i < ubLoop; i++) {
        int64_t curOff = processOffset;
        processOffset += ubFormer;

        // 预取下一个 tile
        bool hasNext = (i + 1 < ubLoop) || (ubTail > 0);
        if (hasNext) {
            AscendC::LocalTensor<T> xNext = inQueueX.AllocTensor<T>();
            int64_t nextCount = (i + 1 < ubLoop) ? ubFormer : ubTail;
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(nextCount * sizeof(T)), 0, 0, 0};
            uint32_t tailRemN = static_cast<uint32_t>(nextCount) % ELEM_PER_32B;
            bool needPadN = (tailRemN != 0);
            AscendC::DataCopyPadExtParams<T> padParamsN{needPadN, static_cast<uint8_t>(0),
                static_cast<uint8_t>(needPadN ? (ELEM_PER_32B - tailRemN) : 0), static_cast<T>(0)};
            AscendC::DataCopyPad(xNext, inputGM[curOff + ubFormer], cp, padParamsN);
            inQueueX.EnQue(xNext);
        }

        // 计算当前 tile
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yCur = outQueueY.AllocTensor<T>();
        GELU_TANH5_COMPUTE(ubFormer, xIn, yCur);

        outQueueY.EnQue(yCur);
        inQueueX.FreeTensor(xIn);

        // 写回
        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams cpOut{1, static_cast<uint32_t>(ubFormer * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGM[curOff], yOut, cpOut);
        outQueueY.FreeTensor(yOut);
    }

    // 尾部 tile
    if (ubTail > 0) {
        AscendC::LocalTensor<T> xIn = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yCur = outQueueY.AllocTensor<T>();
        GELU_TANH5_COMPUTE(ubTail, xIn, yCur);

        outQueueY.EnQue(yCur);
        inQueueX.FreeTensor(xIn);

        AscendC::LocalTensor<T> yOut = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams cpOut{1, static_cast<uint32_t>(ubTail * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outputGM[processOffset], yOut, cpOut);
        outQueueY.FreeTensor(yOut);
    }
}

#undef GELU_TANH5_COMPUTE
