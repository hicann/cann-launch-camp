#include <type_traits>

#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

namespace {
constexpr uint32_t LESS_EQUAL_BUFFER_NUM = 2;
constexpr float LESS_EQUAL_NEGATIVE_ONE_FP32 = -1.0F;
constexpr float LESS_EQUAL_POSITIVE_ONE_FP32 = 1.0F;
constexpr int32_t LESS_EQUAL_NEGATIVE_ONE_I32 = -1;
constexpr int32_t LESS_EQUAL_POSITIVE_ONE_I32 = 1;
constexpr float LESS_EQUAL_MIN_ACCURACY_FP16 = 0.00000005960464477539063F;
constexpr float LESS_EQUAL_MAX_MUL_FP16 = 4096.0F;
constexpr float LESS_EQUAL_MIN_ACCURACY_FP32 = 1.1754943508222875e-38F;
constexpr float LESS_EQUAL_MAX_MUL_1_FP32 = 1125899906842624.0F;
constexpr float LESS_EQUAL_MAX_MUL_2_FP32 = 67108864.0F;

__aicore__ inline uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + 31U) & ~31U;
}
}  // namespace

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x1));
        x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x2));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));
        cfg = tiling;

        const uint32_t blockIdx = GetBlockIdx();
        blockLength = cfg.baseBlockLength + (blockIdx < cfg.tailBlockNum ? 1 : 0);
        blockOffset = blockIdx * cfg.baseBlockLength + (blockIdx < cfg.tailBlockNum ? blockIdx : cfg.tailBlockNum);
        tileLength = cfg.tileLength;
        tileNum = (blockLength == 0) ? 0 : static_cast<uint32_t>((blockLength + tileLength - 1) / tileLength);

        if (tileNum != 0) {
            InitBuffers();
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength == 0 || tileLength == 0) {
            return;
        }
        for (uint32_t tileIndex = 0; tileIndex < tileNum; ++tileIndex) {
            const uint32_t validLength = GetTileValidLength(tileIndex);
            CopyIn(tileIndex, validLength);
            Compute(validLength);
            CopyOut(tileIndex, validLength);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        pipe.InitBuffer(x1Queue, LESS_EQUAL_BUFFER_NUM, AlignUp32(tileLength * sizeof(DT_X1)));
        pipe.InitBuffer(x2Queue, LESS_EQUAL_BUFFER_NUM, AlignUp32(tileLength * sizeof(DT_X1)));
        pipe.InitBuffer(yQueue, LESS_EQUAL_BUFFER_NUM, AlignUp32(tileLength * sizeof(int8_t)));

        if constexpr (std::is_same_v<DT_X1, half>) {
            pipe.InitBuffer(calcBuf1, AlignUp32(tileLength * sizeof(half)));
        } else if constexpr (std::is_same_v<DT_X1, float>) {
            pipe.InitBuffer(calcBuf1, AlignUp32(tileLength * sizeof(float)));
            pipe.InitBuffer(calcBuf2, AlignUp32(tileLength * sizeof(half)));
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(calcBuf2, AlignUp32(tileLength * sizeof(half)));
            pipe.InitBuffer(calcBuf3, AlignUp32(tileLength * sizeof(half)));
            pipe.InitBuffer(calcBuf4, AlignUp32(tileLength * sizeof(half)));
        } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(calcBuf1, AlignUp32(tileLength * sizeof(int32_t)));
            pipe.InitBuffer(calcBuf3, AlignUp32(tileLength * sizeof(half)));
            pipe.InitBuffer(calcBuf4, AlignUp32(tileLength * sizeof(float)));
        }
    }

    __aicore__ inline uint32_t GetTileValidLength(uint32_t tileIndex) const
    {
        const uint64_t tileOffset = static_cast<uint64_t>(tileIndex) * tileLength;
        const uint64_t remain = blockLength - tileOffset;
        return static_cast<uint32_t>(remain < tileLength ? remain : tileLength);
    }

    __aicore__ inline void CopyIn(uint32_t tileIndex, uint32_t validLength)
    {
        const uint64_t tileOffset = blockOffset + static_cast<uint64_t>(tileIndex) * tileLength;
        LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
        LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();

        CopyInContiguous(tileOffset, validLength, x1Local, x2Local);

        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t validLength)
    {
        LocalTensor<DT_X1> x1Local = x1Queue.DeQue<DT_X1>();
        LocalTensor<DT_X1> x2Local = x2Queue.DeQue<DT_X1>();
        LocalTensor<int8_t> yLocal = yQueue.AllocTensor<int8_t>();

        ComputeContiguous(x1Local, x2Local, yLocal, GetComputeLength(validLength));

        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIndex, uint32_t validLength)
    {
        const uint64_t tileOffset = blockOffset + static_cast<uint64_t>(tileIndex) * tileLength;
        LocalTensor<int8_t> yLocal = yQueue.DeQue<int8_t>();

        CopyOutContiguous(tileOffset, validLength, yLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void SetCopyParams(DataCopyExtParams &copyParams, uint32_t bytes) const
    {
        copyParams.blockCount = 1;
        copyParams.blockLen = bytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
    }

    __aicore__ inline void CopyInContiguous(
        uint64_t tileOffset, uint32_t validLength, LocalTensor<DT_X1> &x1Local, LocalTensor<DT_X1> &x2Local)
    {
        if (validLength == tileLength) {
            DataCopy(x1Local, x1Gm[tileOffset], tileLength);
            DataCopy(x2Local, x2Gm[tileOffset], tileLength);
            return;
        }

        DataCopyExtParams copyParams;
        SetCopyParams(copyParams, validLength * sizeof(DT_X1));
        DataCopyPadExtParams<DT_X1> padParams{false, 0, 0, 0};
        DataCopyPad(x1Local, x1Gm[tileOffset], copyParams, padParams);
        DataCopyPad(x2Local, x2Gm[tileOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyOutContiguous(
        uint64_t tileOffset, uint32_t validLength, LocalTensor<int8_t> &yLocal)
    {
        if (validLength == tileLength) {
            DataCopy(yGm[tileOffset], yLocal, tileLength);
            return;
        }

        DataCopyExtParams copyParams;
        SetCopyParams(copyParams, validLength * sizeof(int8_t));
        DataCopyPad(yGm[tileOffset], yLocal, copyParams);
    }

    __aicore__ inline uint32_t GetComputeLength(uint32_t validLength) const
    {
        return AlignUp32(validLength * sizeof(DT_X1)) / sizeof(DT_X1);
    }

    __aicore__ inline void ComputeHalf(
        LocalTensor<half> x1Local, LocalTensor<half> x2Local, LocalTensor<half> yCompute, uint32_t computeLength)
    {
        Max(yCompute, x1Local, x2Local, computeLength);
        Sub(yCompute, x2Local, yCompute, computeLength);
        Abs(yCompute, yCompute, computeLength);
        Mins(yCompute, yCompute, static_cast<half>(LESS_EQUAL_MIN_ACCURACY_FP16), computeLength);
        Muls(yCompute, yCompute, static_cast<half>(LESS_EQUAL_MAX_MUL_FP16), computeLength);
        Muls(yCompute, yCompute, static_cast<half>(LESS_EQUAL_MAX_MUL_FP16), computeLength);
        Adds(yCompute, yCompute, static_cast<half>(LESS_EQUAL_NEGATIVE_ONE_FP32), computeLength);
        Abs(yCompute, yCompute, computeLength);
    }

    __aicore__ inline void ComputeFloat(
        LocalTensor<float> x1Local, LocalTensor<float> x2Local, LocalTensor<float> yCompute, uint32_t computeLength)
    {
        Max(yCompute, x1Local, x2Local, computeLength);
        Sub(yCompute, x2Local, yCompute, computeLength);
        Abs(yCompute, yCompute, computeLength);
        Mins(yCompute, yCompute, static_cast<float>(LESS_EQUAL_MIN_ACCURACY_FP32), computeLength);
        Muls(yCompute, yCompute, static_cast<float>(LESS_EQUAL_MAX_MUL_1_FP32), computeLength);
        Muls(yCompute, yCompute, static_cast<float>(LESS_EQUAL_MAX_MUL_1_FP32), computeLength);
        Muls(yCompute, yCompute, static_cast<float>(LESS_EQUAL_MAX_MUL_2_FP32), computeLength);
        Adds(yCompute, yCompute, static_cast<float>(LESS_EQUAL_NEGATIVE_ONE_FP32), computeLength);
        Abs(yCompute, yCompute, computeLength);
    }

    __aicore__ inline void ComputeInt8(LocalTensor<half> x1LocalFp16, LocalTensor<half> x2LocalFp16,
        LocalTensor<half> yLocalFp16, uint32_t computeLength)
    {
        Min(yLocalFp16, x1LocalFp16, x2LocalFp16, computeLength);
        Sub(yLocalFp16, x2LocalFp16, yLocalFp16, computeLength);
        Mins(yLocalFp16, yLocalFp16, static_cast<half>(LESS_EQUAL_POSITIVE_ONE_FP32), computeLength);

        Sub(x1LocalFp16, x1LocalFp16, x2LocalFp16, computeLength);
        Abs(x1LocalFp16, x1LocalFp16, computeLength);
        Mins(x1LocalFp16, x1LocalFp16, static_cast<half>(LESS_EQUAL_POSITIVE_ONE_FP32), computeLength);
        Duplicate(x2LocalFp16, static_cast<half>(LESS_EQUAL_POSITIVE_ONE_FP32), computeLength);
        Sub(x1LocalFp16, x2LocalFp16, x1LocalFp16, computeLength);

        Add(yLocalFp16, yLocalFp16, x1LocalFp16, computeLength);
    }

    __aicore__ inline void ComputeInt32(
        LocalTensor<int32_t> x1Local, LocalTensor<int32_t> x2Local, LocalTensor<int32_t> yCompute,
        uint32_t computeLength)
    {
        Min(yCompute, x1Local, x2Local, computeLength);
        Sub(yCompute, x2Local, yCompute, computeLength);
        Mins(yCompute, yCompute, static_cast<int32_t>(LESS_EQUAL_POSITIVE_ONE_I32), computeLength);

        Sub(x1Local, x1Local, x2Local, computeLength);
        Mins(x1Local, x1Local, static_cast<int32_t>(LESS_EQUAL_POSITIVE_ONE_I32), computeLength);
        Maxs(x1Local, x1Local, static_cast<int32_t>(LESS_EQUAL_NEGATIVE_ONE_I32), computeLength);
        Mul(x1Local, x1Local, x1Local, computeLength);
        Duplicate(x2Local, static_cast<int32_t>(LESS_EQUAL_POSITIVE_ONE_I32), computeLength);
        Sub(x1Local, x2Local, x1Local, computeLength);

        Add(yCompute, yCompute, x1Local, computeLength);
    }

    __aicore__ inline void ComputeContiguous(
        LocalTensor<DT_X1> x1Local, LocalTensor<DT_X1> x2Local, LocalTensor<int8_t> yLocal, uint32_t computeLength)
    {
        if constexpr (std::is_same_v<DT_X1, half>) {
            LocalTensor<half> yCompute = calcBuf1.Get<half>();
            ComputeHalf(x1Local, x2Local, yCompute, computeLength);
            Cast(yLocal, yCompute, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, float>) {
            LocalTensor<float> yCompute = calcBuf1.Get<float>();
            LocalTensor<half> yFp16 = calcBuf2.Get<half>();
            ComputeFloat(x1Local, x2Local, yCompute, computeLength);
            Cast(yFp16, yCompute, RoundMode::CAST_NONE, computeLength);
            Cast(yLocal, yFp16, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            LocalTensor<half> x1LocalFp16 = calcBuf2.Get<half>();
            LocalTensor<half> x2LocalFp16 = calcBuf3.Get<half>();
            LocalTensor<half> yLocalFp16 = calcBuf4.Get<half>();
            Cast(x1LocalFp16, x1Local, RoundMode::CAST_NONE, computeLength);
            Cast(x2LocalFp16, x2Local, RoundMode::CAST_NONE, computeLength);
            ComputeInt8(x1LocalFp16, x2LocalFp16, yLocalFp16, computeLength);
            Cast(yLocal, yLocalFp16, RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
            LocalTensor<int32_t> yCompute = calcBuf1.Get<int32_t>();
            LocalTensor<half> yFp16 = calcBuf3.Get<half>();
            LocalTensor<float> yFp32 = calcBuf4.Get<float>();
            ComputeInt32(x1Local, x2Local, yCompute, computeLength);
            Cast(yFp32, yCompute, RoundMode::CAST_NONE, computeLength);
            Cast(yFp16, yFp32, RoundMode::CAST_NONE, computeLength);
            Cast(yLocal, yFp16, RoundMode::CAST_NONE, computeLength);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBuf1;
    TBuf<TPosition::VECCALC> calcBuf2;
    TBuf<TPosition::VECCALC> calcBuf3;
    TBuf<TPosition::VECCALC> calcBuf4;
    TQue<TPosition::VECIN, LESS_EQUAL_BUFFER_NUM> x1Queue;
    TQue<TPosition::VECIN, LESS_EQUAL_BUFFER_NUM> x2Queue;
    TQue<TPosition::VECOUT, LESS_EQUAL_BUFFER_NUM> yQueue;

    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<int8_t> yGm;

    LessEqualTilingData cfg;
    uint64_t blockOffset = 0;
    uint64_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tilingData);
    op.Process();
}
