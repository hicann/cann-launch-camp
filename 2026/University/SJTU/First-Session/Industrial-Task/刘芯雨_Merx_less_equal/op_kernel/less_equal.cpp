#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

constexpr uint32_t TILE_LENGTH = 3840;
constexpr uint32_t TILE_BUFFER_LENGTH = 4096;

// Fixed UB layout for the low-overhead contiguous fast path.  Reserve the
// input areas for the widest supported type so all addresses stay 32B aligned.
constexpr uint32_t X1_LOCAL_ADDR = 0;
constexpr uint32_t X2_LOCAL_ADDR = 16384;
constexpr uint32_t Y_LOCAL_ADDR = 32768;
constexpr uint32_t MASK_LOCAL_ADDR = 36864;
constexpr uint32_t HALF_A_LOCAL_ADDR = 37376;
constexpr uint32_t HALF_B_LOCAL_ADDR = 45568;
constexpr uint32_t INT_LOCAL_ADDR = 53760;

__aicore__ inline void ExpandCompareMask(const LocalTensor<uint8_t> &mask,
                                          LocalTensor<half> &halfA,
                                          LocalTensor<half> &halfB,
                                          LocalTensor<uint8_t> &output,
                                          uint32_t count) {
    Duplicate(halfA, static_cast<half>(1.0f), count);
    Select(halfB, mask, halfA, static_cast<half>(0.0f),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    LocalTensor<int8_t> outputInt8 = output.ReinterpretCast<int8_t>();
    Cast(outputInt8, halfB, RoundMode::CAST_NONE, count);
}

template <typename T>
struct VectorLessEqual;

template <>
struct VectorLessEqual<half> {
    __aicore__ static inline void Compute(const LocalTensor<half> &lhs,
                                           const LocalTensor<half> &rhs,
                                           LocalTensor<uint8_t> &mask,
                                           LocalTensor<half> &halfA,
                                           LocalTensor<half> &halfB,
                                           LocalTensor<int32_t> &intTmp,
                                           LocalTensor<uint8_t> &output,
                                           uint32_t count) {
        Compare(mask, lhs, rhs, CMPMODE::LE, count);
        ExpandCompareMask(mask, halfA, halfB, output, count);
    }
};

template <>
struct VectorLessEqual<float> {
    __aicore__ static inline void Compute(const LocalTensor<float> &lhs,
                                           const LocalTensor<float> &rhs,
                                           LocalTensor<uint8_t> &mask,
                                           LocalTensor<half> &halfA,
                                           LocalTensor<half> &halfB,
                                           LocalTensor<int32_t> &intTmp,
                                           LocalTensor<uint8_t> &output,
                                           uint32_t count) {
        Compare(mask, lhs, rhs, CMPMODE::LE, count);
        ExpandCompareMask(mask, halfA, halfB, output, count);
    }
};

template <>
struct VectorLessEqual<int32_t> {
    __aicore__ static inline void Compute(const LocalTensor<int32_t> &lhs,
                                           const LocalTensor<int32_t> &rhs,
                                           LocalTensor<uint8_t> &mask,
                                           LocalTensor<half> &halfA,
                                           LocalTensor<half> &halfB,
                                           LocalTensor<int32_t> &intTmp,
                                           LocalTensor<uint8_t> &output,
                                           uint32_t count) {
        Max(intTmp, lhs, rhs, static_cast<int32_t>(count));
        Compare(mask, intTmp, rhs, CMPMODE::EQ, count);
        ExpandCompareMask(mask, halfA, halfB, output, count);
    }
};

template <>
struct VectorLessEqual<int8_t> {
    __aicore__ static inline void Compute(const LocalTensor<int8_t> &lhs,
                                           const LocalTensor<int8_t> &rhs,
                                           LocalTensor<uint8_t> &mask,
                                           LocalTensor<half> &halfA,
                                           LocalTensor<half> &halfB,
                                           LocalTensor<int32_t> &intTmp,
                                           LocalTensor<uint8_t> &output,
                                           uint32_t count) {
        Cast(halfA, lhs, RoundMode::CAST_NONE, count);
        Cast(halfB, rhs, RoundMode::CAST_NONE, count);
        Compare(mask, halfA, halfB, CMPMODE::LE, count);
        Duplicate(halfA, static_cast<half>(1.0f), count);
        Select(halfB, mask, halfA, static_cast<half>(0.0f),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
        LocalTensor<int8_t> outputInt8 = output.ReinterpretCast<int8_t>();
        Cast(outputInt8, halfB, RoundMode::CAST_NONE, count);
    }
};

// Queue-less implementation for the overwhelmingly common contiguous case.
// Constructing TPipe/TQue objects has a noticeable fixed cost on small inputs;
// direct LocalTensor construction plus explicit events avoids that overhead.
template <class DT_X1>
class KernelLessEqualContiguous {
public:
    __aicore__ inline KernelLessEqualContiguous() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling) {
        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        outputLength = tiling.outputLength;
        blockLength = tiling.blockLength;
    }

    __aicore__ inline void Process() {
        const uint32_t start = GetBlockIdx() * blockLength;
        if (start >= outputLength) {
            return;
        }
        uint32_t end = start + blockLength;
        if (end > outputLength) {
            end = outputLength;
        }

        LocalTensor<DT_X1> x1Local(
            TPosition::VECIN, X1_LOCAL_ADDR, TILE_BUFFER_LENGTH);
        LocalTensor<DT_X1> x2Local(
            TPosition::VECIN, X2_LOCAL_ADDR, TILE_BUFFER_LENGTH);
        LocalTensor<uint8_t> yLocal(
            TPosition::VECOUT, Y_LOCAL_ADDR, TILE_BUFFER_LENGTH);
        LocalTensor<uint8_t> mask(
            TPosition::VECCALC, MASK_LOCAL_ADDR, TILE_BUFFER_LENGTH / 8);
        LocalTensor<half> halfA(
            TPosition::VECCALC, HALF_A_LOCAL_ADDR, TILE_BUFFER_LENGTH);
        LocalTensor<half> halfB(
            TPosition::VECCALC, HALF_B_LOCAL_ADDR, TILE_BUFFER_LENGTH);
        LocalTensor<int32_t> intTmp(
            TPosition::VECCALC, INT_LOCAL_ADDR, TILE_BUFFER_LENGTH);

        const uint32_t loopCount =
            (end - start + TILE_LENGTH - 1) / TILE_LENGTH;
        uint32_t loopIndex = 0;
        for (uint32_t offset = start; offset < end;
             offset += TILE_LENGTH, ++loopIndex) {
            uint32_t count = end - offset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }

            if (loopIndex != 0) {
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            DataCopyExtParams inputCopy{
                1, static_cast<uint32_t>(count * sizeof(DT_X1)), 0, 0, 0};
            DataCopyPadExtParams<DT_X1> padParams{
                false, 0, 0, static_cast<DT_X1>(0)};
            DataCopyPad(x1Local, x1Gm[offset], inputCopy, padParams);
            DataCopyPad(x2Local, x2Gm[offset], inputCopy, padParams);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            if (loopIndex != 0) {
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            const uint32_t vectorCount = (count + 255) / 256 * 256;
            VectorLessEqual<DT_X1>::Compute(
                x1Local, x2Local, mask, halfA, halfB, intTmp, yLocal,
                vectorCount);
            if (loopIndex + 1 != loopCount) {
                SetFlag<HardEvent::V_MTE2>(0);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            DataCopyExtParams outputCopy{
                1, static_cast<uint32_t>(count * sizeof(uint8_t)), 0, 0, 0};
            DataCopyPad(yGm[offset], yLocal, outputCopy);
            if (loopIndex + 1 != loopCount) {
                SetFlag<HardEvent::MTE3_V>(0);
            }
        }
    }

private:
    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t outputLength;
    uint32_t blockLength;
};

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &tiling) {
        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);

        outputLength = tiling.outputLength;
        blockLength = tiling.blockLength;
        rank = tiling.rank;
        noBroadcast = tiling.noBroadcast;
        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            outShape[i] = tiling.outShape[i];
            outStride[i] = tiling.outStride[i];
            x1Stride[i] = tiling.x1Stride[i];
            x2Stride[i] = tiling.x2Stride[i];
        }

        pipe.InitBuffer(inQueueX1, 1, TILE_BUFFER_LENGTH * sizeof(DT_X1));
        pipe.InitBuffer(inQueueX2, 1, TILE_BUFFER_LENGTH * sizeof(DT_X1));
        pipe.InitBuffer(outQueueY, 1, TILE_BUFFER_LENGTH * sizeof(uint8_t));
        pipe.InitBuffer(maskBuffer, TILE_BUFFER_LENGTH / 8);
        pipe.InitBuffer(halfBufferA, TILE_BUFFER_LENGTH * sizeof(half));
        pipe.InitBuffer(halfBufferB, TILE_BUFFER_LENGTH * sizeof(half));
        pipe.InitBuffer(intBuffer, TILE_BUFFER_LENGTH * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        const uint32_t start = GetBlockIdx() * blockLength;
        if (start >= outputLength) {
            return;
        }
        uint32_t end = start + blockLength;
        if (end > outputLength) {
            end = outputLength;
        }

        for (uint32_t offset = start; offset < end; offset += TILE_LENGTH) {
            uint32_t count = end - offset;
            if (count > TILE_LENGTH) {
                count = TILE_LENGTH;
            }
            if (noBroadcast != 0) {
                ProcessContiguous(offset, count);
            } else {
                ProcessBroadcast(offset, count);
            }
        }
    }

private:
    __aicore__ inline uint32_t GetInputOffset(uint32_t outputIndex,
                                               const uint32_t *inputStride) {
        uint32_t inputOffset = 0;
        for (uint32_t dim = 0; dim < rank; ++dim) {
            const uint32_t coordinate = (outputIndex / outStride[dim]) % outShape[dim];
            inputOffset += coordinate * inputStride[dim];
        }
        return inputOffset;
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count,
                                    LocalTensor<uint8_t> &yLocal) {
        outQueueY.EnQue<uint8_t>(yLocal);
        LocalTensor<uint8_t> result = outQueueY.DeQue<uint8_t>();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(uint8_t)), 0, 0, 0};
        DataCopyPad(yGm[offset], result, copyParams);
        outQueueY.FreeTensor(result);
    }

    __aicore__ inline void ProcessContiguous(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X1> x1Local = inQueueX1.AllocTensor<DT_X1>();
        LocalTensor<DT_X1> x2Local = inQueueX2.AllocTensor<DT_X1>();
        DataCopyExtParams inputCopy{
            1, static_cast<uint32_t>(count * sizeof(DT_X1)), 0, 0, 0};
        DataCopyPadExtParams<DT_X1> padParams{
            false, 0, 0, static_cast<DT_X1>(0)};
        DataCopyPad(x1Local, x1Gm[offset], inputCopy, padParams);
        DataCopyPad(x2Local, x2Gm[offset], inputCopy, padParams);
        inQueueX1.EnQue<DT_X1>(x1Local);
        inQueueX2.EnQue<DT_X1>(x2Local);

        x1Local = inQueueX1.DeQue<DT_X1>();
        x2Local = inQueueX2.DeQue<DT_X1>();
        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();
        LocalTensor<half> halfA = halfBufferA.Get<half>();
        LocalTensor<half> halfB = halfBufferB.Get<half>();
        LocalTensor<int32_t> intTmp = intBuffer.Get<int32_t>();
        const uint32_t vectorCount = (count + 255) / 256 * 256;
        VectorLessEqual<DT_X1>::Compute(
            x1Local, x2Local, mask, halfA, halfB, intTmp, yLocal, vectorCount);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        CopyOut(offset, count, yLocal);
    }

    __aicore__ inline void ProcessBroadcast(uint32_t offset, uint32_t count) {
        LocalTensor<DT_X1> x1Local = inQueueX1.AllocTensor<DT_X1>();
        LocalTensor<DT_X1> x2Local = inQueueX2.AllocTensor<DT_X1>();
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t outputIndex = offset + i;
            const uint32_t x1Offset = GetInputOffset(outputIndex, x1Stride);
            const uint32_t x2Offset = GetInputOffset(outputIndex, x2Stride);
            x1Local.SetValue(i, x1Gm.GetValue(x1Offset));
            x2Local.SetValue(i, x2Gm.GetValue(x2Offset));
        }
        inQueueX1.EnQue<DT_X1>(x1Local);
        inQueueX2.EnQue<DT_X1>(x2Local);
        x1Local = inQueueX1.DeQue<DT_X1>();
        x2Local = inQueueX2.DeQue<DT_X1>();

        LocalTensor<uint8_t> yLocal = outQueueY.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();
        LocalTensor<half> halfA = halfBufferA.Get<half>();
        LocalTensor<half> halfB = halfBufferB.Get<half>();
        LocalTensor<int32_t> intTmp = intBuffer.Get<int32_t>();
        const uint32_t vectorCount = (count + 255) / 256 * 256;
        VectorLessEqual<DT_X1>::Compute(
            x1Local, x2Local, mask, halfA, halfB, intTmp, yLocal, vectorCount);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
        CopyOut(offset, count, yLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECOUT, 1> outQueueY;
    TBuf<QuePosition::VECCALC> maskBuffer;
    TBuf<QuePosition::VECCALC> halfBufferA;
    TBuf<QuePosition::VECCALC> halfBufferB;
    TBuf<QuePosition::VECCALC> intBuffer;
    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<uint8_t> yGm;
    uint32_t outputLength;
    uint32_t blockLength;
    uint32_t rank;
    uint32_t noBroadcast;
    uint32_t outShape[LESS_EQUAL_MAX_DIMS];
    uint32_t outStride[LESS_EQUAL_MAX_DIMS];
    uint32_t x1Stride[LESS_EQUAL_MAX_DIMS];
    uint32_t x2Stride[LESS_EQUAL_MAX_DIMS];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    if (tiling_data.noBroadcast != 0) {
        InitSocState();
        KernelLessEqualContiguous<DT_X1> op;
        op.Init(x1, x2, y, tiling_data);
        op.Process();
    } else {
        InitSocState();
        KernelLessEqual<DT_X1> op;
        op.Init(x1, x2, y, tiling_data);
        op.Process();
    }
}
