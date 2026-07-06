#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"

using namespace AscendC;

namespace {
// 与host严格对齐，不改动并行块数、tile尺寸（保证分片完全一致）
constexpr uint32_t BLOCK_NUM    = 8;
constexpr uint32_t TILE_LEN     = 1024;
constexpr uint32_t QUEUE_DEPTH  = 1;

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return (a < b) ? a : b;
}

// 统一模板实现：float / half 通用
template <typename T>
class LogSigmoidImpl {
public:
    __aicore__ inline LogSigmoidImpl() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize)
    {
        // 块均分逻辑完全保留（和原版一致）
        uint32_t blockIdx = GetBlockIdx();
        uint32_t perCore = (totalSize + BLOCK_NUM - 1) / BLOCK_NUM;
        uint32_t start   = blockIdx * perCore;

        if (start >= totalSize) {
            validLen = 0;
            return;
        }
        validLen = MinU32(perCore, totalSize - start);

        // 绑定分片GM
        xGm.SetGlobalBuffer((__gm__ T*)x + start, validLen);
        yGm.SetGlobalBuffer((__gm__ T*)y + start, validLen);

        // 初始化本地缓存
        pipe.InitBuffer(inQueue,  QUEUE_DEPTH, TILE_LEN * sizeof(T));
        pipe.InitBuffer(outQueue, QUEUE_DEPTH, TILE_LEN * sizeof(T));
        pipe.InitBuffer(tmp1Buf, TILE_LEN * sizeof(T));
        pipe.InitBuffer(tmp2Buf, TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < validLen; offset += TILE_LEN) {
            uint32_t curCnt = MinU32(TILE_LEN, validLen - offset);
            CopyIn(offset, curCnt);
            Compute(curCnt);
            CopyOut(offset, curCnt);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(cnt * sizeof(T)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    // 核心计算：完全保留原版公式、计算顺序、参数，保证数值一致
    __aicore__ inline void Compute(uint32_t cnt)
    {
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<T> tmp1 = tmp1Buf.Get<T>();
        LocalTensor<T> tmp2 = tmp2Buf.Get<T>();

        // y = -log(1 + exp(-x)) 完全原版步骤
        Muls(tmp1, xLocal, static_cast<T>(-1.0f), cnt);
        Exp(tmp2, tmp1, cnt);
        Adds(tmp1, tmp2, static_cast<T>(1.0f), cnt);
        Log(tmp2, tmp1, cnt);
        Muls(yLocal, tmp2, static_cast<T>(-1.0f), cnt);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(cnt * sizeof(T)), 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN,  QUEUE_DEPTH> inQueue;
    TQue<TPosition::VECOUT, QUEUE_DEPTH> outQueue;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    uint32_t validLen = 0;
};

// BF16 单独实现（保留原版cast策略、舍入模式，保证结果对齐）
class LogSigmoidBf16Impl {
public:
    __aicore__ inline LogSigmoidBf16Impl() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalSize)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t perCore = (totalSize + BLOCK_NUM - 1) / BLOCK_NUM;
        uint32_t start   = blockIdx * perCore;

        if (start >= totalSize) {
            validLen = 0;
            return;
        }
        validLen = MinU32(perCore, totalSize - start);

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + start, validLen);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + start, validLen);

        pipe.InitBuffer(inQueue,  QUEUE_DEPTH, TILE_LEN * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueue, QUEUE_DEPTH, TILE_LEN * sizeof(bfloat16_t));
        pipe.InitBuffer(fpBuf,    TILE_LEN * sizeof(float));
        pipe.InitBuffer(tmp1Buf,  TILE_LEN * sizeof(float));
        pipe.InitBuffer(tmp2Buf,  TILE_LEN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < validLen; offset += TILE_LEN) {
            uint32_t curCnt = MinU32(TILE_LEN, validLen - offset);
            CopyIn(offset, curCnt);
            Compute(curCnt);
            CopyOut(offset, curCnt);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<bfloat16_t> xLocal = inQueue.AllocTensor<bfloat16_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(cnt * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t cnt)
    {
        LocalTensor<bfloat16_t> xLocal = inQueue.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> yLocal = outQueue.AllocTensor<bfloat16_t>();
        LocalTensor<float>  xFloat = fpBuf.Get<float>();
        LocalTensor<float>  tmp1   = tmp1Buf.Get<float>();
        LocalTensor<float>  tmp2   = tmp2Buf.Get<float>();

        // 完全保留原版转换、计算、舍入策略
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, cnt);
        Muls(tmp1, xFloat, -1.0f, cnt);
        Exp(tmp2, tmp1, cnt);
        Adds(tmp1, tmp2, 1.0f, cnt);
        Log(tmp2, tmp1, cnt);
        Muls(tmp1, tmp2, -1.0f, cnt);
        Cast(yLocal, tmp1, RoundMode::CAST_RINT, cnt);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t cnt)
    {
        LocalTensor<bfloat16_t> yLocal = outQueue.DeQue<bfloat16_t>();
        DataCopyParams copyParams{1, static_cast<uint16_t>(cnt * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN,  QUEUE_DEPTH> inQueue;
    TQue<TPosition::VECOUT, QUEUE_DEPTH> outQueue;
    TBuf<TPosition::VECCALC> fpBuf;
    TBuf<TPosition::VECCALC> tmp1Buf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
    uint32_t validLen = 0;
};
}

// 入口完全兼容原版，TILING_KEY逻辑不变
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {
        LogSigmoidImpl<float> op;
        op.Init(x, y, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        LogSigmoidImpl<half> op;
        op.Init(x, y, tilingData.size);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        LogSigmoidBf16Impl op;
        op.Init(x, y, tilingData.size);
        op.Process();
    }
}

