#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1,
        C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1,
            C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ),
);

constexpr uint32_t BUFFER_NUM = LESS_EQUAL_BUFFER_NUM;
constexpr uint32_t PIPELINE_BUFFER_NUM = 2;

template <typename T>
struct LessEqualValue {
    using StorageType = T;

    __aicore__ inline static uint8_t Run(T a, T b)
    {
        return a <= b ? 1 : 0;
    }
};

template <>
struct LessEqualValue<float> {
    using StorageType = uint32_t;

    __aicore__ inline static uint8_t Run(uint32_t a, uint32_t b)
    {
        constexpr uint32_t signMask = 0x80000000U;
        constexpr uint32_t absMask = 0x7FFFFFFFU;
        constexpr uint32_t infCode = 0x7F800000U;
        const uint32_t absA = a & absMask;
        const uint32_t absB = b & absMask;

        if (absA > infCode || absB > infCode) {
            return 0;
        }
        if ((absA | absB) == 0) {
            return 1;
        }

        const uint32_t keyA =
            (a & signMask) != 0 ? ~a : (a ^ signMask);
        const uint32_t keyB =
            (b & signMask) != 0 ? ~b : (b ^ signMask);
        return keyA <= keyB ? 1 : 0;
    }
};

template <>
struct LessEqualValue<half> {
    using StorageType = uint16_t;

    __aicore__ inline static uint8_t Run(uint16_t a, uint16_t b)
    {
        constexpr uint16_t signMask = 0x8000U;
        constexpr uint16_t absMask = 0x7FFFU;
        constexpr uint16_t infCode = 0x7C00U;
        const uint16_t absA = a & absMask;
        const uint16_t absB = b & absMask;

        if (absA > infCode || absB > infCode) {
            return 0;
        }
        if ((absA | absB) == 0) {
            return 1;
        }

        const uint16_t keyA = (a & signMask) != 0
            ? static_cast<uint16_t>(~a)
            : static_cast<uint16_t>(a ^ signMask);
        const uint16_t keyB = (b & signMask) != 0
            ? static_cast<uint16_t>(~b)
            : static_cast<uint16_t>(b ^ signMask);
        return keyA <= keyB ? 1 : 0;
    }
};

template <typename T>
struct LessEqualFloatVectorTraits;

template <>
struct LessEqualFloatVectorTraits<half> {
    static constexpr uint32_t VECTOR_ELEMENTS = 128;
};

template <>
struct LessEqualFloatVectorTraits<float> {
    static constexpr uint32_t VECTOR_ELEMENTS = 64;
};

struct LessEqualHotTilingData {
    uint64_t totalLength;
    uint64_t workUnitNum;
    uint64_t smallCoreWorkNum;
    uint64_t bigCoreWorkNum;
    uint32_t blockDim;
    uint32_t tailBlockNum;
    uint32_t tileDataNum;
    uint32_t pathMode;
    uint32_t workBlockElements;
};

__aicore__ inline void GetWorkRange(const LessEqualHotTilingData &tiling,
                                    uint64_t &start,
                                    uint64_t &length)
{
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (tiling.blockDim == 1) {
        start = 0;
        length = tiling.workUnitNum;
        return;
    }
    if (blockIdx < tiling.tailBlockNum) {
        length = tiling.bigCoreWorkNum;
        start = static_cast<uint64_t>(blockIdx) *
            tiling.bigCoreWorkNum;
    } else {
        length = tiling.smallCoreWorkNum;
        start = static_cast<uint64_t>(tiling.tailBlockNum) *
            tiling.bigCoreWorkNum +
            static_cast<uint64_t>(blockIdx - tiling.tailBlockNum) *
            tiling.smallCoreWorkNum;
    }
}

__aicore__ inline void GetWorkRange(const LessEqualTilingData &tiling,
                                    uint64_t &start,
                                    uint64_t &length)
{
    const uint32_t blockIdx = AscendC::GetBlockIdx();
    if (tiling.blockDim == 1) {
        start = 0;
        length = tiling.workUnitNum;
        return;
    }
    if (blockIdx < tiling.tailBlockNum) {
        length = tiling.bigCoreWorkNum;
        start = static_cast<uint64_t>(blockIdx) *
            tiling.bigCoreWorkNum;
    } else {
        length = tiling.smallCoreWorkNum;
        start = static_cast<uint64_t>(tiling.tailBlockNum) *
            tiling.bigCoreWorkNum +
            static_cast<uint64_t>(blockIdx - tiling.tailBlockNum) *
            tiling.smallCoreWorkNum;
    }
}


__aicore__ inline void LoadSameShapeHotTiling(
    const __gm__ LessEqualTilingData *src,
    LessEqualHotTilingData &dst)
{
    dst.totalLength = src->totalLength;
    dst.workUnitNum = src->workUnitNum;
    dst.smallCoreWorkNum = src->smallCoreWorkNum;
    dst.bigCoreWorkNum = src->bigCoreWorkNum;
    dst.blockDim = src->blockDim;
    dst.tailBlockNum = src->tailBlockNum;
    dst.tileDataNum = src->tileDataNum;
    dst.pathMode = src->pathMode;
    dst.workBlockElements = src->workBlockElements;
}


constexpr uint32_t LessEqualStaticAlign32(uint32_t value)
{
    return (value + 31U) & ~31U;
}


class KernelLessEqualStaticHalfSameShape {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        x1Gm.SetGlobalBuffer((__gm__ half *)x1);
        x2Gm.SetGlobalBuffer((__gm__ half *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Raw = (__gm__ uint16_t *)x1;
        x2Raw = (__gm__ uint16_t *)x2;
        yRaw = (__gm__ uint8_t *)y;
    }

    __aicore__ inline void Process(
        const __gm__ LessEqualTilingData *tiling)
    {
        const uint64_t totalLength = tiling->totalLength;
        const uint64_t workUnitNum = tiling->workUnitNum;
        const uint64_t smallCoreWorkNum = tiling->smallCoreWorkNum;
        const uint64_t bigCoreWorkNum = tiling->bigCoreWorkNum;
        const uint32_t blockDim = tiling->blockDim;
        const uint32_t tailBlockNum = tiling->tailBlockNum;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (totalLength == 0 || blockIdx >= blockDim) {
            return;
        }

        uint64_t workStart = 0;
        uint64_t workLength = 0;
        if (blockDim == 1) {
            workLength = workUnitNum;
        } else if (blockIdx < tailBlockNum) {
            workLength = bigCoreWorkNum;
            workStart = static_cast<uint64_t>(blockIdx) *
                bigCoreWorkNum;
        } else {
            workLength = smallCoreWorkNum;
            workStart = static_cast<uint64_t>(tailBlockNum) *
                bigCoreWorkNum +
                static_cast<uint64_t>(blockIdx - tailBlockNum) *
                smallCoreWorkNum;
        }
        if (workLength == 0) {
            return;
        }

        const uint64_t start = workStart *
            LESS_EQUAL_STATIC_WORK_BLOCK_ELEMENTS;
        uint64_t length = workLength *
            LESS_EQUAL_STATIC_WORK_BLOCK_ELEMENTS;
        if (start + length > totalLength) {
            length = totalLength - start;
        }
    
        if (length == 0 || (start & 31ULL) != 0) {
            return;
        }

        AscendC::LocalTensor<half> x1Local(
            AscendC::TPosition::VECCALC, X1_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<half> x2Local(
            AscendC::TPosition::VECCALC, X2_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> yLocal(
            AscendC::TPosition::VECCALC, Y_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> maskLocal(
            AscendC::TPosition::VECCALC, MASK_ADDR, MASK_BYTES);
        AscendC::LocalTensor<half> oneHalfLocal(
            AscendC::TPosition::VECCALC, ONE_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<half> boolHalfLocal(
            AscendC::TPosition::VECCALC, BOOL_ADDR, TILE_ELEMENTS);

        const uint64_t vectorLength = (length / 32ULL) * 32ULL;
        if (vectorLength != 0) {
            uint32_t constantNum = static_cast<uint32_t>(
                vectorLength < TILE_ELEMENTS ? vectorLength : TILE_ELEMENTS);
            constantNum = AlignVector(constantNum);
            AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                               static_cast<int32_t>(constantNum));
        }

        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t validNum = static_cast<uint32_t>(
                (vectorLength - progress) > TILE_ELEMENTS
                    ? TILE_ELEMENTS : (vectorLength - progress));
            const uint32_t computeNum = AlignVector(validNum);
            AscendC::DataCopy(x1Local, x1Gm[start + progress], validNum);
            AscendC::DataCopy(x2Local, x2Gm[start + progress], validNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);

            AscendC::Compare(maskLocal, x1Local, x2Local,
                             AscendC::CMPMODE::LE, computeNum);
            AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                            static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            computeNum);
            AscendC::Cast(yLocal, boolHalfLocal,
                          AscendC::RoundMode::CAST_NONE, computeNum);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::DataCopy(yGm[start + progress], yLocal, validNum);
            progress += validNum;
            if (progress < vectorLength) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            }
        }

        
        for (uint64_t i = vectorLength; i < length; ++i) {
            yRaw[start + i] = LessEqualValue<half>::Run(
                x1Raw[start + i], x2Raw[start + i]);
        }
    }

private:
    __aicore__ inline static uint32_t AlignVector(uint32_t value)
    {
        constexpr uint32_t VECTOR_ELEMENTS = 128;
        return (value + VECTOR_ELEMENTS - 1U) /
            VECTOR_ELEMENTS * VECTOR_ELEMENTS;
    }

private:
    static constexpr uint32_t TILE_ELEMENTS =
        LESS_EQUAL_STATIC_HALF_TILE_ELEMENTS;
    static constexpr int32_t EVENT_ID = 0;
    static constexpr uint32_t X1_ADDR = 0;
    static constexpr uint32_t X2_ADDR = LessEqualStaticAlign32(
        X1_ADDR + TILE_ELEMENTS * sizeof(half));
    static constexpr uint32_t Y_ADDR = LessEqualStaticAlign32(
        X2_ADDR + TILE_ELEMENTS * sizeof(half));
    static constexpr uint32_t MASK_BYTES = LessEqualStaticAlign32(
        (TILE_ELEMENTS + 7U) / 8U);
    static constexpr uint32_t MASK_ADDR = LessEqualStaticAlign32(
        Y_ADDR + TILE_ELEMENTS * sizeof(uint8_t));
    static constexpr uint32_t ONE_ADDR = LessEqualStaticAlign32(
        MASK_ADDR + MASK_BYTES);
    static constexpr uint32_t BOOL_ADDR = LessEqualStaticAlign32(
        ONE_ADDR + TILE_ELEMENTS * sizeof(half));

    AscendC::GlobalTensor<half> x1Gm;
    AscendC::GlobalTensor<half> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ uint16_t *x1Raw;
    __gm__ uint16_t *x2Raw;
    __gm__ uint8_t *yRaw;
};


class KernelLessEqualStaticFloatSameShape {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        x1Gm.SetGlobalBuffer((__gm__ float *)x1);
        x2Gm.SetGlobalBuffer((__gm__ float *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Raw = (__gm__ uint32_t *)x1;
        x2Raw = (__gm__ uint32_t *)x2;
        yRaw = (__gm__ uint8_t *)y;
    }

    __aicore__ inline void Process(
        const __gm__ LessEqualTilingData *tiling)
    {
        const uint64_t totalLength = tiling->totalLength;
        const uint64_t workUnitNum = tiling->workUnitNum;
        const uint64_t smallCoreWorkNum = tiling->smallCoreWorkNum;
        const uint64_t bigCoreWorkNum = tiling->bigCoreWorkNum;
        const uint32_t blockDim = tiling->blockDim;
        const uint32_t tailBlockNum = tiling->tailBlockNum;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        if (totalLength == 0 || blockIdx >= blockDim) {
            return;
        }

        uint64_t workStart = 0;
        uint64_t workLength = 0;
        if (blockDim == 1) {
            workLength = workUnitNum;
        } else if (blockIdx < tailBlockNum) {
            workLength = bigCoreWorkNum;
            workStart = static_cast<uint64_t>(blockIdx) *
                bigCoreWorkNum;
        } else {
            workLength = smallCoreWorkNum;
            workStart = static_cast<uint64_t>(tailBlockNum) *
                bigCoreWorkNum +
                static_cast<uint64_t>(blockIdx - tailBlockNum) *
                smallCoreWorkNum;
        }
        if (workLength == 0) {
            return;
        }

        const uint64_t start = workStart *
            LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS;
        uint64_t length = workLength *
            LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS;
        if (start + length > totalLength) {
            length = totalLength - start;
        }
        if (length == 0 ||
            (start % LESS_EQUAL_STATIC_FLOAT_WORK_BLOCK_ELEMENTS) != 0) {
            return;
        }

        AscendC::LocalTensor<float> x1Local(
            AscendC::TPosition::VECCALC, X1_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<float> x2Local(
            AscendC::TPosition::VECCALC, X2_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> yLocal(
            AscendC::TPosition::VECCALC, Y_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<uint8_t> maskLocal(
            AscendC::TPosition::VECCALC, MASK_ADDR, MASK_BYTES);
        AscendC::LocalTensor<half> oneHalfLocal(
            AscendC::TPosition::VECCALC, ONE_ADDR, TILE_ELEMENTS);
        AscendC::LocalTensor<half> boolHalfLocal(
            AscendC::TPosition::VECCALC, BOOL_ADDR, TILE_ELEMENTS);

        const uint64_t vectorLength =
            (length / VECTOR_ELEMENTS) * VECTOR_ELEMENTS;
        if (vectorLength != 0) {
            const uint32_t constantNum = static_cast<uint32_t>(
                vectorLength < TILE_ELEMENTS
                    ? vectorLength : TILE_ELEMENTS);
            AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                               static_cast<int32_t>(constantNum));
        }

        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t validNum = TILE_ELEMENTS;
            const uint64_t left = vectorLength - progress;
            if (left < TILE_ELEMENTS) {
                validNum = static_cast<uint32_t>(left);
            }

            AscendC::DataCopy(x1Local, x1Gm[start + progress],
                              validNum);
            AscendC::DataCopy(x2Local, x2Gm[start + progress],
                              validNum);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID);

            AscendC::Compare(maskLocal, x1Local, x2Local,
                             AscendC::CMPMODE::LE, validNum);
            AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                            static_cast<half>(0.0f),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            validNum);
            AscendC::Cast(yLocal, boolHalfLocal,
                          AscendC::RoundMode::CAST_NONE, validNum);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID);
            AscendC::DataCopy(yGm[start + progress], yLocal,
                              validNum);
            progress += validNum;
            if (progress < vectorLength) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                    EVENT_ID);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                    EVENT_ID);
            }
        }

        
        for (uint64_t i = vectorLength; i < length; ++i) {
            yRaw[start + i] = LessEqualValue<float>::Run(
                x1Raw[start + i], x2Raw[start + i]);
        }
    }

private:
    static constexpr uint32_t VECTOR_ELEMENTS = 64;
    static constexpr uint32_t TILE_ELEMENTS =
        LESS_EQUAL_STATIC_FLOAT_TILE_ELEMENTS;
    static constexpr int32_t EVENT_ID = 0;
    static constexpr uint32_t X1_ADDR = 0;
    static constexpr uint32_t X2_ADDR = LessEqualStaticAlign32(
        X1_ADDR + TILE_ELEMENTS * sizeof(float));
    static constexpr uint32_t Y_ADDR = LessEqualStaticAlign32(
        X2_ADDR + TILE_ELEMENTS * sizeof(float));
    static constexpr uint32_t MASK_BYTES = LessEqualStaticAlign32(
        (TILE_ELEMENTS + 7U) / 8U);
    static constexpr uint32_t MASK_ADDR = LessEqualStaticAlign32(
        Y_ADDR + TILE_ELEMENTS * sizeof(uint8_t));
    static constexpr uint32_t ONE_ADDR = LessEqualStaticAlign32(
        MASK_ADDR + MASK_BYTES);
    static constexpr uint32_t BOOL_ADDR = LessEqualStaticAlign32(
        ONE_ADDR + TILE_ELEMENTS * sizeof(half));

    AscendC::GlobalTensor<float> x1Gm;
    AscendC::GlobalTensor<float> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ uint32_t *x1Raw;
    __gm__ uint32_t *x2Raw;
    __gm__ uint8_t *yRaw;
};

template <typename T>
class KernelLessEqualDirect {
public:
    using StorageType = typename LessEqualValue<T>::StorageType;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        x1Gm = (__gm__ StorageType *)x1;
        x2Gm = (__gm__ StorageType *)x2;
        yGm = (__gm__ uint8_t *)y;
    }


    __aicore__ inline void ProcessMicro(uint64_t totalLength,
                                        uint32_t mode)
    {
        if (totalLength == 0 || AscendC::GetBlockIdx() != 0) {
            return;
        }
        if (mode == LESS_EQUAL_MODE_X1_SCALAR) {
            DoX1Scalar(0, totalLength);
        } else if (mode == LESS_EQUAL_MODE_X2_SCALAR) {
            DoX2Scalar(0, totalLength);
        } else {
            DoSameShape(0, totalLength);
        }
    }

    __aicore__ inline void Process(const LessEqualTilingData &tiling)
    {
        if (tiling.totalLength == 0 ||
            AscendC::GetBlockIdx() >= tiling.blockDim) {
            return;
        }

        uint64_t start = 0;
        uint64_t length = 0;
        GetWorkRange(tiling, start, length);
        if (length == 0) {
            return;
        }

        if (tiling.pathMode == LESS_EQUAL_PATH_BROADCAST_GENERIC) {
            DoBroadcastGeneric(start, length, tiling);
        } else if (tiling.pathMode ==
                   LESS_EQUAL_PATH_BROADCAST_INNER_DIRECT) {
            DoBroadcastInner(start, length, tiling);
        } else if (tiling.mode == LESS_EQUAL_MODE_X1_SCALAR) {
            DoX1Scalar(start, length);
        } else if (tiling.mode == LESS_EQUAL_MODE_X2_SCALAR) {
            DoX2Scalar(start, length);
        } else {
            DoSameShape(start, length);
        }
    }

private:
    __aicore__ inline void DoSameShape(uint64_t start,
                                       uint64_t length)
    {
        DoContiguous(start, start, start, length);
    }

    __aicore__ inline void DoX1Scalar(uint64_t start,
                                      uint64_t length)
    {
        DoLeftScalar(start, x1Gm[0], start, length);
    }

    __aicore__ inline void DoX2Scalar(uint64_t start,
                                      uint64_t length)
    {
        DoRightScalar(start, start, x2Gm[0], length);
    }

    __aicore__ inline void DoContiguous(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t length)
    {
        uint64_t i = 0;
        for (; i + 15 < length; i += 16) {
#define LEQ_CONTIG(k) yGm[outputStart + i + (k)] = \
    LessEqualValue<T>::Run(x1Gm[x1Start + i + (k)], \
                           x2Gm[x2Start + i + (k)])
            LEQ_CONTIG(0);  LEQ_CONTIG(1);  LEQ_CONTIG(2);  LEQ_CONTIG(3);
            LEQ_CONTIG(4);  LEQ_CONTIG(5);  LEQ_CONTIG(6);  LEQ_CONTIG(7);
            LEQ_CONTIG(8);  LEQ_CONTIG(9);  LEQ_CONTIG(10); LEQ_CONTIG(11);
            LEQ_CONTIG(12); LEQ_CONTIG(13); LEQ_CONTIG(14); LEQ_CONTIG(15);
#undef LEQ_CONTIG
        }
        for (; i < length; ++i) {
            yGm[outputStart + i] = LessEqualValue<T>::Run(
                x1Gm[x1Start + i], x2Gm[x2Start + i]);
        }
    }

    __aicore__ inline void DoLeftScalar(
        uint64_t outputStart,
        StorageType value,
        uint64_t x2Start,
        uint64_t length)
    {
        uint64_t i = 0;
        for (; i + 15 < length; i += 16) {
#define LEQ_LEFT(k) yGm[outputStart + i + (k)] = \
    LessEqualValue<T>::Run(value, x2Gm[x2Start + i + (k)])
            LEQ_LEFT(0);  LEQ_LEFT(1);  LEQ_LEFT(2);  LEQ_LEFT(3);
            LEQ_LEFT(4);  LEQ_LEFT(5);  LEQ_LEFT(6);  LEQ_LEFT(7);
            LEQ_LEFT(8);  LEQ_LEFT(9);  LEQ_LEFT(10); LEQ_LEFT(11);
            LEQ_LEFT(12); LEQ_LEFT(13); LEQ_LEFT(14); LEQ_LEFT(15);
#undef LEQ_LEFT
        }
        for (; i < length; ++i) {
            yGm[outputStart + i] = LessEqualValue<T>::Run(
                value, x2Gm[x2Start + i]);
        }
    }

    __aicore__ inline void DoRightScalar(
        uint64_t outputStart,
        uint64_t x1Start,
        StorageType value,
        uint64_t length)
    {
        uint64_t i = 0;
        for (; i + 15 < length; i += 16) {
#define LEQ_RIGHT(k) yGm[outputStart + i + (k)] = \
    LessEqualValue<T>::Run(x1Gm[x1Start + i + (k)], value)
            LEQ_RIGHT(0);  LEQ_RIGHT(1);  LEQ_RIGHT(2);  LEQ_RIGHT(3);
            LEQ_RIGHT(4);  LEQ_RIGHT(5);  LEQ_RIGHT(6);  LEQ_RIGHT(7);
            LEQ_RIGHT(8);  LEQ_RIGHT(9);  LEQ_RIGHT(10); LEQ_RIGHT(11);
            LEQ_RIGHT(12); LEQ_RIGHT(13); LEQ_RIGHT(14); LEQ_RIGHT(15);
#undef LEQ_RIGHT
        }
        for (; i < length; ++i) {
            yGm[outputStart + i] = LessEqualValue<T>::Run(
                x1Gm[x1Start + i], value);
        }
    }

    __aicore__ inline void FillResult(
        uint64_t outputStart,
        uint8_t value,
        uint64_t length)
    {
        uint64_t i = 0;
        for (; i + 15 < length; i += 16) {
#define LEQ_FILL(k) yGm[outputStart + i + (k)] = value
            LEQ_FILL(0);  LEQ_FILL(1);  LEQ_FILL(2);  LEQ_FILL(3);
            LEQ_FILL(4);  LEQ_FILL(5);  LEQ_FILL(6);  LEQ_FILL(7);
            LEQ_FILL(8);  LEQ_FILL(9);  LEQ_FILL(10); LEQ_FILL(11);
            LEQ_FILL(12); LEQ_FILL(13); LEQ_FILL(14); LEQ_FILL(15);
#undef LEQ_FILL
        }
        for (; i < length; ++i) {
            yGm[outputStart + i] = value;
        }
    }

    __aicore__ inline void GetOuterOffsets(
        uint64_t outputStart,
        const LessEqualTilingData &tiling,
        uint64_t &x1Start,
        uint64_t &x2Start)
    {
        x1Start = 0;
        x2Start = 0;
        uint64_t remain = outputStart;
        for (uint32_t d = 0; d < tiling.innerStartDim; ++d) {
            const uint64_t coord = remain / tiling.outputStride[d];
            remain -= coord * tiling.outputStride[d];
            x1Start += coord * tiling.x1Stride[d];
            x2Start += coord * tiling.x2Stride[d];
        }
    }

    __aicore__ inline void DoBroadcastInner(
        uint64_t outerStart,
        uint64_t outerLength,
        const LessEqualTilingData &tiling)
    {
        const uint64_t outerEnd = outerStart + outerLength;
        for (uint64_t outer = outerStart; outer < outerEnd; ++outer) {
            const uint64_t outputStart =
                outer * tiling.innerDataNum;
            uint64_t x1Start = 0;
            uint64_t x2Start = 0;
            if (tiling.outerLinear != 0) {
                x1Start = outer * tiling.x1OuterStep;
                x2Start = outer * tiling.x2OuterStep;
            } else {
                GetOuterOffsets(outputStart, tiling, x1Start, x2Start);
            }
            DoInnerSegment(outputStart, x1Start, x2Start,
                           tiling.innerDataNum,
                           tiling.x1InnerMode,
                           tiling.x2InnerMode);
        }
    }

    __aicore__ inline void DoInnerSegment(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t length,
        uint32_t x1Mode,
        uint32_t x2Mode)
    {
        if (x1Mode == LESS_EQUAL_INNER_SCALAR) {
            const StorageType x1Value = x1Gm[x1Start];
            if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
                FillResult(outputStart,
                           LessEqualValue<T>::Run(
                               x1Value, x2Gm[x2Start]),
                           length);
            } else {
                DoLeftScalar(outputStart, x1Value, x2Start, length);
            }
            return;
        }
        if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
            DoRightScalar(outputStart, x1Start,
                          x2Gm[x2Start], length);
            return;
        }
        DoContiguous(outputStart, x1Start, x2Start, length);
    }

    __aicore__ inline void DoBroadcastGeneric(
        uint64_t start,
        uint64_t length,
        const LessEqualTilingData &tiling)
    {
        uint64_t coord[LESS_EQUAL_MAX_DIM] = {0};
        uint64_t x1Offset = 0;
        uint64_t x2Offset = 0;
        uint64_t remain = start;

        for (uint32_t d = 0; d < tiling.dimNum; ++d) {
            const uint64_t c = remain / tiling.outputStride[d];
            remain -= c * tiling.outputStride[d];
            coord[d] = c;
            x1Offset += c * tiling.x1Stride[d];
            x2Offset += c * tiling.x2Stride[d];
        }

        const uint64_t end = start + length;
        for (uint64_t out = start; out < end; ++out) {
            yGm[out] = LessEqualValue<T>::Run(
                x1Gm[x1Offset], x2Gm[x2Offset]);
            if (out + 1 == end) {
                break;
            }
            for (int32_t d = static_cast<int32_t>(tiling.dimNum) - 1;
                 d >= 0; --d) {
                ++coord[d];
                if (coord[d] < tiling.outputShape[d]) {
                    x1Offset += tiling.x1Stride[d];
                    x2Offset += tiling.x2Stride[d];
                    break;
                }
                coord[d] = 0;
                x1Offset -= (tiling.outputShape[d] - 1) *
                    tiling.x1Stride[d];
                x2Offset -= (tiling.outputShape[d] - 1) *
                    tiling.x2Stride[d];
            }
        }
    }

private:
    __gm__ StorageType *x1Gm;
    __gm__ StorageType *x2Gm;
    __gm__ uint8_t *yGm;
};

template <typename T, uint32_t QUEUE_DEPTH>
class KernelLessEqualFloatSameShapeVector {
public:
    using StorageType = typename LessEqualValue<T>::StorageType;
    using Traits = LessEqualFloatVectorTraits<T>;

    template <typename TilingType>
    __aicore__ inline void Init(GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const TilingType &tiling,
                                AscendC::TPipe &pipe)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Raw = (__gm__ StorageType *)x1;
        x2Raw = (__gm__ StorageType *)x2;
        yRaw = (__gm__ uint8_t *)y;
        tileDataNum = tiling.tileDataNum;
        maskBytes = ((tileDataNum + 255U) / 256U) * 32U;

        pipe.InitBuffer(x1Queue, QUEUE_DEPTH,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(x2Queue, QUEUE_DEPTH,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(yQueue, QUEUE_DEPTH,
                        tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(oneHalfBuf, tileDataNum * sizeof(half));
        pipe.InitBuffer(boolHalfBuf, tileDataNum * sizeof(half));
    }

    template <typename TilingType>
    __aicore__ inline void Process(const TilingType &tiling)
    {
        if (tiling.totalLength == 0 ||
            AscendC::GetBlockIdx() >= tiling.blockDim) {
            return;
        }

        uint64_t workStart = 0;
        uint64_t workLength = 0;
        GetWorkRange(tiling, workStart, workLength);
        if (workLength == 0) {
            return;
        }

        const uint64_t start =
            workStart * static_cast<uint64_t>(tiling.workBlockElements);
        uint64_t end = (workStart + workLength) *
            static_cast<uint64_t>(tiling.workBlockElements);
        if (end > tiling.totalLength) {
            end = tiling.totalLength;
        }

        const uint64_t length = end - start;
        const uint64_t vectorLength =
            (length / Traits::VECTOR_ELEMENTS) * Traits::VECTOR_ELEMENTS;
        const uint32_t constantNum = static_cast<uint32_t>(
            vectorLength < tileDataNum ? vectorLength : tileDataNum);
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                           static_cast<int32_t>(constantNum));
        ProcessSegment(start, length);
    }

private:
    __aicore__ inline void ProcessSegment(uint64_t start, uint64_t length)
    {
        const uint64_t vectorLength =
            (length / Traits::VECTOR_ELEMENTS) * Traits::VECTOR_ELEMENTS;
        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t dataNum = tileDataNum;
            const uint64_t left = vectorLength - progress;
            if (left < tileDataNum) {
                dataNum = static_cast<uint32_t>(left);
            }
            CopyIn(start + progress, dataNum);
            Compute(dataNum);
            CopyOut(start + progress, dataNum);
            progress += dataNum;
        }

        for (uint64_t i = vectorLength; i < length; ++i) {
            yRaw[start + i] = LessEqualValue<T>::Run(
                x1Raw[start + i], x2Raw[start + i]);
        }
    }

    __aicore__ inline void CopyIn(uint64_t start, uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();
        AscendC::DataCopy(x1Local, x1Gm[start], dataNum);
        AscendC::DataCopy(x2Local, x2Gm[start], dataNum);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> maskLocal =
            maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::LocalTensor<half> boolHalfLocal =
            boolHalfBuf.Get<half>();

        AscendC::Compare(maskLocal, x1Local, x2Local,
                         AscendC::CMPMODE::LE, dataNum);
        AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                        static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        dataNum);
        AscendC::Cast(yLocal, boolHalfLocal,
                      AscendC::RoundMode::CAST_NONE, dataNum);

        yQueue.EnQue(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t start, uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.template DeQue<uint8_t>();
        AscendC::DataCopy(yGm[start], yLocal, dataNum);
        yQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> x1Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> x2Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> boolHalfBuf;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ StorageType *x1Raw;
    __gm__ StorageType *x2Raw;
    __gm__ uint8_t *yRaw;
    uint32_t tileDataNum;
    uint32_t maskBytes;
};

template <typename T>
class KernelLessEqualFloatBroadcastVector {
public:
    using StorageType = typename LessEqualValue<T>::StorageType;
    using Traits = LessEqualFloatVectorTraits<T>;

    __aicore__ inline void Init(GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const LessEqualTilingData &tiling,
                                AscendC::TPipe &pipe)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Typed = (__gm__ T *)x1;
        x2Typed = (__gm__ T *)x2;
        x1Raw = (__gm__ StorageType *)x1;
        x2Raw = (__gm__ StorageType *)x2;
        yRaw = (__gm__ uint8_t *)y;
        tileDataNum = tiling.tileDataNum;
        maskBytes = ((tileDataNum + 255U) / 256U) * 32U;

        pipe.InitBuffer(x1Queue, BUFFER_NUM,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM,
                        tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(oneHalfBuf, tileDataNum * sizeof(half));
        pipe.InitBuffer(boolHalfBuf, tileDataNum * sizeof(half));
    }

    __aicore__ inline void Process(const LessEqualTilingData &tiling)
    {
        if (tiling.totalLength == 0 ||
            AscendC::GetBlockIdx() >= tiling.blockDim) {
            return;
        }

        uint64_t workStart = 0;
        uint64_t workLength = 0;
        GetWorkRange(tiling, workStart, workLength);
        if (workLength == 0) {
            return;
        }

        InitOneConstant(tiling, workStart, workLength);

        if (tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            ProcessGlobalScalar(tiling, workStart, workLength);
            return;
        }
        ProcessBroadcast(tiling, workStart, workLength);
    }

private:
    __aicore__ inline void InitOneConstant(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        uint64_t maxLength = tiling.innerSegmentDataNum;
        if (tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            const uint64_t start =
                workStart * static_cast<uint64_t>(tiling.workBlockElements);
            uint64_t end = (workStart + workLength) *
                static_cast<uint64_t>(tiling.workBlockElements);
            if (end > tiling.totalLength) {
                end = tiling.totalLength;
            }
            maxLength = end - start;
        } else if (maxLength > tiling.innerDataNum) {
            maxLength = tiling.innerDataNum;
        }

        const uint64_t vectorLength =
            (maxLength / Traits::VECTOR_ELEMENTS) *
            Traits::VECTOR_ELEMENTS;
        const uint32_t constantNum = static_cast<uint32_t>(
            vectorLength < tileDataNum ? vectorLength : tileDataNum);
        if (constantNum == 0) {
            return;
        }
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                           static_cast<int32_t>(constantNum));
    }

    __aicore__ inline void ProcessGlobalScalar(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        const uint64_t start =
            workStart * static_cast<uint64_t>(tiling.workBlockElements);
        uint64_t end = (workStart + workLength) *
            static_cast<uint64_t>(tiling.workBlockElements);
        if (end > tiling.totalLength) {
            end = tiling.totalLength;
        }
        const uint64_t length = end - start;

        if (tiling.mode == LESS_EQUAL_MODE_X1_SCALAR) {
            ProcessInnerSegment(start, 0, start, length,
                                LESS_EQUAL_INNER_SCALAR,
                                LESS_EQUAL_INNER_CONTIGUOUS);
        } else {
            ProcessInnerSegment(start, start, 0, length,
                                LESS_EQUAL_INNER_CONTIGUOUS,
                                LESS_EQUAL_INNER_SCALAR);
        }
    }

    __aicore__ inline void ProcessBroadcast(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        const uint64_t workEnd = workStart + workLength;
        for (uint64_t work = workStart; work < workEnd; ++work) {
            const uint64_t outer = work / tiling.innerSegmentNum;
            const uint64_t segmentIndex =
                work - outer * tiling.innerSegmentNum;
            uint64_t segmentOffset = 0;
            uint64_t segmentLength = 0;
            GetSegmentInfo(segmentIndex, tiling,
                           segmentOffset, segmentLength);
            if (segmentLength != 0) {
                ProcessOneOuter(outer, segmentOffset,
                                segmentLength, tiling);
            }
        }
    }

    __aicore__ inline void GetSegmentInfo(
        uint64_t segmentIndex,
        const LessEqualTilingData &tiling,
        uint64_t &segmentOffset,
        uint64_t &segmentLength)
    {
        segmentOffset = segmentIndex * tiling.innerSegmentDataNum;
        if (segmentOffset >= tiling.innerDataNum) {
            segmentLength = 0;
            return;
        }
        segmentLength = tiling.innerDataNum - segmentOffset;
        if (segmentLength > tiling.innerSegmentDataNum) {
            segmentLength = tiling.innerSegmentDataNum;
        }
    }

    __aicore__ inline void GetOuterOffsets(
        uint64_t outputStart,
        const LessEqualTilingData &tiling,
        uint64_t &x1Start,
        uint64_t &x2Start)
    {
        x1Start = 0;
        x2Start = 0;
        uint64_t remain = outputStart;
        for (uint32_t d = 0; d < tiling.innerStartDim; ++d) {
            const uint64_t coord = remain / tiling.outputStride[d];
            remain -= coord * tiling.outputStride[d];
            x1Start += coord * tiling.x1Stride[d];
            x2Start += coord * tiling.x2Stride[d];
        }
    }

    __aicore__ inline void GetRowStarts(
        uint64_t outer,
        const LessEqualTilingData &tiling,
        uint64_t &x1RowStart,
        uint64_t &x2RowStart)
    {
        if (tiling.outerLinear != 0) {
            x1RowStart = outer * tiling.x1OuterStep;
            x2RowStart = outer * tiling.x2OuterStep;
            return;
        }
        GetOuterOffsets(outer * tiling.innerDataNum, tiling,
                        x1RowStart, x2RowStart);
    }

    __aicore__ inline void ProcessOneOuter(
        uint64_t outer,
        uint64_t segmentOffset,
        uint64_t segmentLength,
        const LessEqualTilingData &tiling)
    {
        uint64_t x1RowStart = 0;
        uint64_t x2RowStart = 0;
        GetRowStarts(outer, tiling, x1RowStart, x2RowStart);
        const uint64_t x1Start =
            tiling.x1InnerMode == LESS_EQUAL_INNER_CONTIGUOUS
            ? x1RowStart + segmentOffset : x1RowStart;
        const uint64_t x2Start =
            tiling.x2InnerMode == LESS_EQUAL_INNER_CONTIGUOUS
            ? x2RowStart + segmentOffset : x2RowStart;
        const uint64_t outputStart =
            outer * tiling.innerDataNum + segmentOffset;
        ProcessInnerSegment(outputStart, x1Start, x2Start,
                            segmentLength,
                            tiling.x1InnerMode,
                            tiling.x2InnerMode);
    }

    __aicore__ inline void ProcessInnerSegment(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t length,
        uint32_t x1Mode,
        uint32_t x2Mode)
    {
       
        T x1ScalarValue = static_cast<T>(0);
        T x2ScalarValue = static_cast<T>(0);
        if (x1Mode == LESS_EQUAL_INNER_SCALAR) {
            x1ScalarValue = x1Typed[x1Start];
        }
        if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
            x2ScalarValue = x2Typed[x2Start];
        }

        const uint64_t vectorLength =
            (length / Traits::VECTOR_ELEMENTS) *
            Traits::VECTOR_ELEMENTS;
        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t dataNum = tileDataNum;
            const uint64_t left = vectorLength - progress;
            if (left < tileDataNum) {
                dataNum = static_cast<uint32_t>(left);
            }
            CopyIn(x1Start, x2Start, progress, dataNum,
                   x1Mode, x2Mode,
                   x1ScalarValue, x2ScalarValue);
            Compute(dataNum);
            CopyOut(outputStart + progress, dataNum);
            progress += dataNum;
        }

        ProcessScalarTail(outputStart, x1Start, x2Start,
                          vectorLength, length,
                          x1Mode, x2Mode);
    }

    __aicore__ inline void ProcessScalarTail(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t begin,
        uint64_t end,
        uint32_t x1Mode,
        uint32_t x2Mode)
    {
        if (begin >= end) {
            return;
        }
        if (x1Mode == LESS_EQUAL_INNER_SCALAR) {
            const auto x1Value = x1Raw[x1Start];
            if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
                const uint8_t value = LessEqualValue<T>::Run(
                    x1Value, x2Raw[x2Start]);
                for (uint64_t i = begin; i < end; ++i) {
                    yRaw[outputStart + i] = value;
                }
            } else {
                for (uint64_t i = begin; i < end; ++i) {
                    yRaw[outputStart + i] = LessEqualValue<T>::Run(
                        x1Value, x2Raw[x2Start + i]);
                }
            }
            return;
        }
        if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
            const auto x2Value = x2Raw[x2Start];
            for (uint64_t i = begin; i < end; ++i) {
                yRaw[outputStart + i] = LessEqualValue<T>::Run(
                    x1Raw[x1Start + i], x2Value);
            }
            return;
        }
        for (uint64_t i = begin; i < end; ++i) {
            yRaw[outputStart + i] = LessEqualValue<T>::Run(
                x1Raw[x1Start + i], x2Raw[x2Start + i]);
        }
    }

    __aicore__ inline void FillX1Local(
        const AscendC::LocalTensor<T> &local,
        uint64_t start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t mode,
        T scalarValue)
    {
        if (mode == LESS_EQUAL_INNER_CONTIGUOUS) {
            AscendC::DataCopy(local, x1Gm[start + progress], dataNum);
        } else {
            AscendC::Duplicate(local, scalarValue,
                               static_cast<int32_t>(dataNum));
        }
    }

    __aicore__ inline void FillX2Local(
        const AscendC::LocalTensor<T> &local,
        uint64_t start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t mode,
        T scalarValue)
    {
        if (mode == LESS_EQUAL_INNER_CONTIGUOUS) {
            AscendC::DataCopy(local, x2Gm[start + progress], dataNum);
        } else {
            AscendC::Duplicate(local, scalarValue,
                               static_cast<int32_t>(dataNum));
        }
    }

    __aicore__ inline void CopyIn(
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t x1Mode,
        uint32_t x2Mode,
        T x1ScalarValue,
        T x2ScalarValue)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
        FillX1Local(x1Local, x1Start, progress, dataNum,
                    x1Mode, x1ScalarValue);
        FillX2Local(x2Local, x2Start, progress, dataNum,
                    x2Mode, x2ScalarValue);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeFromLocal(
        const AscendC::LocalTensor<T> &x1Local,
        const AscendC::LocalTensor<T> &x2Local,
        uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> maskLocal =
            maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::LocalTensor<half> boolHalfLocal =
            boolHalfBuf.Get<half>();

        AscendC::Compare(maskLocal, x1Local, x2Local,
                         AscendC::CMPMODE::LE, dataNum);
        AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                        static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        dataNum);
        AscendC::Cast(yLocal, boolHalfLocal,
                      AscendC::RoundMode::CAST_NONE, dataNum);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.DeQue<T>();
        ComputeFromLocal(x1Local, x2Local, dataNum);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t outputStart,
                                   uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[outputStart], yLocal, dataNum);
        yQueue.FreeTensor(yLocal);
    }

    // Dead code removed: ProcessReusableOuterRange, ProcessReusableX1Tile, ProcessReusableX2Tile

private:
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x1Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x2Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> boolHalfBuf;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ T *x1Typed;
    __gm__ T *x2Typed;
    __gm__ StorageType *x1Raw;
    __gm__ StorageType *x2Raw;
    __gm__ uint8_t *yRaw;
    uint32_t tileDataNum;
    uint32_t maskBytes;
};

template <typename T>
struct LessEqualIntegralVectorOps;

template <>
struct LessEqualIntegralVectorOps<int32_t> {
    static constexpr uint32_t VECTOR_ELEMENTS = 64;

    __aicore__ inline static void InitTempBuffers(
        AscendC::TPipe &pipe,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempABuf,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &,
        uint32_t tileDataNum)
    {
        pipe.InitBuffer(tempABuf, tileDataNum * sizeof(int32_t));
    }

    __aicore__ inline static void Compute(
        const AscendC::LocalTensor<int32_t> &x1Local,
        const AscendC::LocalTensor<int32_t> &x2Local,
        const AscendC::LocalTensor<uint8_t> &yLocal,
        const AscendC::LocalTensor<uint8_t> &maskLocal,
        const AscendC::LocalTensor<half> &oneHalfLocal,
        const AscendC::LocalTensor<half> &boolHalfLocal,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempABuf,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &,
        uint32_t dataNum)
    {
        AscendC::LocalTensor<int32_t> minLocal =
            tempABuf.Get<int32_t>();
        AscendC::Min(minLocal, x1Local, x2Local,
                     static_cast<int32_t>(dataNum));
        // min(x1,x2)==x1 等价于 x1<=x2。
        AscendC::Compare(maskLocal, minLocal, x1Local,
                         AscendC::CMPMODE::EQ, dataNum);
        AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                        static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        dataNum);
        AscendC::Cast(yLocal, boolHalfLocal,
                      AscendC::RoundMode::CAST_NONE, dataNum);
    }
};

template <>
struct LessEqualIntegralVectorOps<int8_t> {
    static constexpr uint32_t VECTOR_ELEMENTS = 128;

    __aicore__ inline static void InitTempBuffers(
        AscendC::TPipe &pipe,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempABuf,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempBBuf,
        uint32_t tileDataNum)
    {
        pipe.InitBuffer(tempABuf, tileDataNum * sizeof(half));
        pipe.InitBuffer(tempBBuf, tileDataNum * sizeof(half));
    }

    __aicore__ inline static void Compute(
        const AscendC::LocalTensor<int8_t> &x1Local,
        const AscendC::LocalTensor<int8_t> &x2Local,
        const AscendC::LocalTensor<uint8_t> &yLocal,
        const AscendC::LocalTensor<uint8_t> &maskLocal,
        const AscendC::LocalTensor<half> &oneHalfLocal,
        const AscendC::LocalTensor<half> &boolHalfLocal,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempABuf,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &tempBBuf,
        uint32_t dataNum)
    {
        AscendC::LocalTensor<half> x1HalfLocal = tempABuf.Get<half>();
        AscendC::LocalTensor<half> x2HalfLocal = tempBBuf.Get<half>();
        AscendC::Cast(x1HalfLocal, x1Local,
                      AscendC::RoundMode::CAST_NONE, dataNum);
        AscendC::Cast(x2HalfLocal, x2Local,
                      AscendC::RoundMode::CAST_NONE, dataNum);
        AscendC::Compare(maskLocal, x1HalfLocal, x2HalfLocal,
                         AscendC::CMPMODE::LE, dataNum);
        AscendC::Select(boolHalfLocal, maskLocal, oneHalfLocal,
                        static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        dataNum);
        AscendC::Cast(yLocal, boolHalfLocal,
                      AscendC::RoundMode::CAST_NONE, dataNum);
    }
};


template <typename T>
struct LessEqualIntegralBroadcastFill {
    __aicore__ inline static void Run(
        const AscendC::LocalTensor<T> &dst,
        T scalarValue,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &,
        uint32_t dataNum)
    {
        AscendC::Duplicate(dst, scalarValue,
                           static_cast<int32_t>(dataNum));
    }
};

template <>
struct LessEqualIntegralBroadcastFill<int8_t> {
    __aicore__ inline static void Run(
        const AscendC::LocalTensor<int8_t> &dst,
        int8_t scalarValue,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &,
        uint32_t dataNum)
    {
        __ubuf__ int8_t *ptr =
            (__ubuf__ int8_t *)dst.GetPhyAddr();
        uint32_t i = 0;
        for (; i + 15 < dataNum; i += 16) {
            ptr[i + 0] = scalarValue;  ptr[i + 1] = scalarValue;
            ptr[i + 2] = scalarValue;  ptr[i + 3] = scalarValue;
            ptr[i + 4] = scalarValue;  ptr[i + 5] = scalarValue;
            ptr[i + 6] = scalarValue;  ptr[i + 7] = scalarValue;
            ptr[i + 8] = scalarValue;  ptr[i + 9] = scalarValue;
            ptr[i + 10] = scalarValue; ptr[i + 11] = scalarValue;
            ptr[i + 12] = scalarValue; ptr[i + 13] = scalarValue;
            ptr[i + 14] = scalarValue; ptr[i + 15] = scalarValue;
        }
        for (; i < dataNum; ++i) {
            ptr[i] = scalarValue;
        }
    }
};

template <typename T, uint32_t QUEUE_DEPTH>
class KernelLessEqualIntegralSameShapeVector {
public:
    using Ops = LessEqualIntegralVectorOps<T>;

    template <typename TilingType>
    __aicore__ inline void Init(GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const TilingType &tiling,
                                AscendC::TPipe &pipe)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Raw = (__gm__ T *)x1;
        x2Raw = (__gm__ T *)x2;
        yRaw = (__gm__ uint8_t *)y;
        tileDataNum = tiling.tileDataNum;
        maskBytes = ((tileDataNum + 255U) / 256U) * 32U;

        pipe.InitBuffer(x1Queue, QUEUE_DEPTH,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(x2Queue, QUEUE_DEPTH,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(yQueue, QUEUE_DEPTH,
                        tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(oneHalfBuf, tileDataNum * sizeof(half));
        pipe.InitBuffer(boolHalfBuf, tileDataNum * sizeof(half));
        Ops::InitTempBuffers(pipe, tempABuf, tempBBuf, tileDataNum);
    }

    template <typename TilingType>
    __aicore__ inline void Process(const TilingType &tiling)
    {
        if (tiling.totalLength == 0 ||
            AscendC::GetBlockIdx() >= tiling.blockDim) {
            return;
        }

        uint64_t workStart = 0;
        uint64_t workLength = 0;
        GetWorkRange(tiling, workStart, workLength);
        if (workLength == 0) {
            return;
        }

        const uint64_t start =
            workStart * static_cast<uint64_t>(tiling.workBlockElements);
        uint64_t end = (workStart + workLength) *
            static_cast<uint64_t>(tiling.workBlockElements);
        if (end > tiling.totalLength) {
            end = tiling.totalLength;
        }
        const uint64_t length = end - start;
        const uint64_t vectorLength =
            (length / Ops::VECTOR_ELEMENTS) * Ops::VECTOR_ELEMENTS;
        const uint32_t constantNum = static_cast<uint32_t>(
            vectorLength < tileDataNum ? vectorLength : tileDataNum);
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                           static_cast<int32_t>(constantNum));
        ProcessSegment(start, length);
    }

private:
    __aicore__ inline void ProcessSegment(uint64_t start, uint64_t length)
    {
        const uint64_t vectorLength =
            (length / Ops::VECTOR_ELEMENTS) * Ops::VECTOR_ELEMENTS;
        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t dataNum = tileDataNum;
            const uint64_t left = vectorLength - progress;
            if (left < tileDataNum) {
                dataNum = static_cast<uint32_t>(left);
            }
            CopyIn(start + progress, dataNum);
            Compute(dataNum);
            CopyOut(start + progress, dataNum);
            progress += dataNum;
        }

        for (uint64_t i = vectorLength; i < length; ++i) {
            yRaw[start + i] = LessEqualValue<T>::Run(
                x1Raw[start + i], x2Raw[start + i]);
        }
    }

    __aicore__ inline void CopyIn(uint64_t start, uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.template AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.template AllocTensor<T>();
        AscendC::DataCopy(x1Local, x1Gm[start], dataNum);
        AscendC::DataCopy(x2Local, x2Gm[start], dataNum);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.template DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.template DeQue<T>();
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.template AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> maskLocal =
            maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::LocalTensor<half> boolHalfLocal =
            boolHalfBuf.Get<half>();

        Ops::Compute(x1Local, x2Local, yLocal, maskLocal,
                     oneHalfLocal, boolHalfLocal,
                     tempABuf, tempBBuf, dataNum);

        yQueue.EnQue(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t start, uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.template DeQue<uint8_t>();
        AscendC::DataCopy(yGm[start], yLocal, dataNum);
        yQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> x1Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> x2Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> boolHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempABuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBBuf;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ T *x1Raw;
    __gm__ T *x2Raw;
    __gm__ uint8_t *yRaw;
    uint32_t tileDataNum;
    uint32_t maskBytes;
};

template <typename T>
class KernelLessEqualIntegralBroadcastVector {
public:
    using Ops = LessEqualIntegralVectorOps<T>;

    __aicore__ inline void Init(GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const LessEqualTilingData &tiling,
                                AscendC::TPipe &pipe)
    {
        x1Gm.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm.SetGlobalBuffer((__gm__ T *)x2);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y);
        x1Raw = (__gm__ T *)x1;
        x2Raw = (__gm__ T *)x2;
        yRaw = (__gm__ uint8_t *)y;
        tileDataNum = tiling.tileDataNum;
        maskBytes = ((tileDataNum + 255U) / 256U) * 32U;

        pipe.InitBuffer(x1Queue, BUFFER_NUM,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM,
                        tileDataNum * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM,
                        tileDataNum * sizeof(uint8_t));
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(oneHalfBuf, tileDataNum * sizeof(half));
        pipe.InitBuffer(boolHalfBuf, tileDataNum * sizeof(half));
        Ops::InitTempBuffers(pipe, tempABuf, tempBBuf, tileDataNum);
    }

    __aicore__ inline void Process(const LessEqualTilingData &tiling)
    {
        if (tiling.totalLength == 0 ||
            AscendC::GetBlockIdx() >= tiling.blockDim) {
            return;
        }
        uint64_t workStart = 0;
        uint64_t workLength = 0;
        GetWorkRange(tiling, workStart, workLength);
        if (workLength == 0) {
            return;
        }

        InitOneConstant(tiling, workStart, workLength);
        if (tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            ProcessGlobalScalar(tiling, workStart, workLength);
            return;
        }
        ProcessBroadcast(tiling, workStart, workLength);
    }

private:
    __aicore__ inline void InitOneConstant(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        uint64_t maxLength = tiling.innerSegmentDataNum;
        if (tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            const uint64_t start = workStart *
                static_cast<uint64_t>(tiling.workBlockElements);
            uint64_t end = (workStart + workLength) *
                static_cast<uint64_t>(tiling.workBlockElements);
            if (end > tiling.totalLength) {
                end = tiling.totalLength;
            }
            maxLength = end - start;
        } else if (maxLength > tiling.innerDataNum) {
            maxLength = tiling.innerDataNum;
        }
        const uint64_t vectorLength =
            (maxLength / Ops::VECTOR_ELEMENTS) * Ops::VECTOR_ELEMENTS;
        const uint32_t constantNum = static_cast<uint32_t>(
            vectorLength < tileDataNum ? vectorLength : tileDataNum);
        if (constantNum == 0) {
            return;
        }
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::Duplicate(oneHalfLocal, static_cast<half>(1.0f),
                           static_cast<int32_t>(constantNum));
    }

    __aicore__ inline void ProcessGlobalScalar(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        const uint64_t start = workStart *
            static_cast<uint64_t>(tiling.workBlockElements);
        uint64_t end = (workStart + workLength) *
            static_cast<uint64_t>(tiling.workBlockElements);
        if (end > tiling.totalLength) {
            end = tiling.totalLength;
        }
        const uint64_t length = end - start;
        if (tiling.mode == LESS_EQUAL_MODE_X1_SCALAR) {
            ProcessInnerSegment(start, 0, start, length,
                                LESS_EQUAL_INNER_SCALAR,
                                LESS_EQUAL_INNER_CONTIGUOUS);
        } else {
            ProcessInnerSegment(start, start, 0, length,
                                LESS_EQUAL_INNER_CONTIGUOUS,
                                LESS_EQUAL_INNER_SCALAR);
        }
    }

    __aicore__ inline void ProcessBroadcast(
        const LessEqualTilingData &tiling,
        uint64_t workStart,
        uint64_t workLength)
    {
        
        const uint64_t workEnd = workStart + workLength;
        for (uint64_t work = workStart; work < workEnd; ++work) {
            const uint64_t outer = work / tiling.innerSegmentNum;
            const uint64_t segmentIndex =
                work - outer * tiling.innerSegmentNum;
            uint64_t segmentOffset = 0;
            uint64_t segmentLength = 0;
            GetSegmentInfo(segmentIndex, tiling,
                           segmentOffset, segmentLength);
            if (segmentLength != 0) {
                ProcessOneOuter(outer, segmentOffset,
                                segmentLength, tiling);
            }
        }
    }

    __aicore__ inline void GetSegmentInfo(
        uint64_t segmentIndex,
        const LessEqualTilingData &tiling,
        uint64_t &segmentOffset,
        uint64_t &segmentLength)
    {
        segmentOffset = segmentIndex * tiling.innerSegmentDataNum;
        if (segmentOffset >= tiling.innerDataNum) {
            segmentLength = 0;
            return;
        }
        segmentLength = tiling.innerDataNum - segmentOffset;
        if (segmentLength > tiling.innerSegmentDataNum) {
            segmentLength = tiling.innerSegmentDataNum;
        }
    }

    __aicore__ inline void GetOuterOffsets(
        uint64_t outputStart,
        const LessEqualTilingData &tiling,
        uint64_t &x1Start,
        uint64_t &x2Start)
    {
        x1Start = 0;
        x2Start = 0;
        uint64_t remain = outputStart;
        for (uint32_t d = 0; d < tiling.innerStartDim; ++d) {
            const uint64_t coord = remain / tiling.outputStride[d];
            remain -= coord * tiling.outputStride[d];
            x1Start += coord * tiling.x1Stride[d];
            x2Start += coord * tiling.x2Stride[d];
        }
    }

    __aicore__ inline void GetRowStarts(
        uint64_t outer,
        const LessEqualTilingData &tiling,
        uint64_t &x1RowStart,
        uint64_t &x2RowStart)
    {
        if (tiling.outerLinear != 0) {
            x1RowStart = outer * tiling.x1OuterStep;
            x2RowStart = outer * tiling.x2OuterStep;
            return;
        }
        GetOuterOffsets(outer * tiling.innerDataNum, tiling,
                        x1RowStart, x2RowStart);
    }

    __aicore__ inline void ProcessOneOuter(
        uint64_t outer,
        uint64_t segmentOffset,
        uint64_t segmentLength,
        const LessEqualTilingData &tiling)
    {
        uint64_t x1RowStart = 0;
        uint64_t x2RowStart = 0;
        GetRowStarts(outer, tiling, x1RowStart, x2RowStart);
        const uint64_t x1Start =
            tiling.x1InnerMode == LESS_EQUAL_INNER_CONTIGUOUS
            ? x1RowStart + segmentOffset : x1RowStart;
        const uint64_t x2Start =
            tiling.x2InnerMode == LESS_EQUAL_INNER_CONTIGUOUS
            ? x2RowStart + segmentOffset : x2RowStart;
        ProcessInnerSegment(
            outer * tiling.innerDataNum + segmentOffset,
            x1Start, x2Start, segmentLength,
            tiling.x1InnerMode, tiling.x2InnerMode);
    }

    __aicore__ inline void ProcessInnerSegment(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t length,
        uint32_t x1Mode,
        uint32_t x2Mode)
    {
        T x1ScalarValue = static_cast<T>(0);
        T x2ScalarValue = static_cast<T>(0);
        if (x1Mode == LESS_EQUAL_INNER_SCALAR) {
            x1ScalarValue = x1Raw[x1Start];
        }
        if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
            x2ScalarValue = x2Raw[x2Start];
        }

        const uint64_t vectorLength =
            (length / Ops::VECTOR_ELEMENTS) * Ops::VECTOR_ELEMENTS;
        uint64_t progress = 0;
        while (progress < vectorLength) {
            uint32_t dataNum = tileDataNum;
            const uint64_t left = vectorLength - progress;
            if (left < tileDataNum) {
                dataNum = static_cast<uint32_t>(left);
            }
            CopyIn(x1Start, x2Start, progress, dataNum,
                   x1Mode, x2Mode,
                   x1ScalarValue, x2ScalarValue);
            Compute(dataNum);
            CopyOut(outputStart + progress, dataNum);
            progress += dataNum;
        }
        ProcessScalarTail(outputStart, x1Start, x2Start,
                          vectorLength, length,
                          x1Mode, x2Mode);
    }

    __aicore__ inline void ProcessScalarTail(
        uint64_t outputStart,
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t begin,
        uint64_t end,
        uint32_t x1Mode,
        uint32_t x2Mode)
    {
        if (begin >= end) {
            return;
        }
        if (x1Mode == LESS_EQUAL_INNER_SCALAR) {
            const auto x1Value = x1Raw[x1Start];
            if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
                const uint8_t value = LessEqualValue<T>::Run(
                    x1Value, x2Raw[x2Start]);
                for (uint64_t i = begin; i < end; ++i) {
                    yRaw[outputStart + i] = value;
                }
            } else {
                for (uint64_t i = begin; i < end; ++i) {
                    yRaw[outputStart + i] = LessEqualValue<T>::Run(
                        x1Value, x2Raw[x2Start + i]);
                }
            }
            return;
        }
        if (x2Mode == LESS_EQUAL_INNER_SCALAR) {
            const auto x2Value = x2Raw[x2Start];
            for (uint64_t i = begin; i < end; ++i) {
                yRaw[outputStart + i] = LessEqualValue<T>::Run(
                    x1Raw[x1Start + i], x2Value);
            }
            return;
        }
        for (uint64_t i = begin; i < end; ++i) {
            yRaw[outputStart + i] = LessEqualValue<T>::Run(
                x1Raw[x1Start + i], x2Raw[x2Start + i]);
        }
    }

    __aicore__ inline void FillX1Local(
        const AscendC::LocalTensor<T> &local,
        uint64_t start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t mode,
        T scalarValue)
    {
        if (mode == LESS_EQUAL_INNER_CONTIGUOUS) {
            AscendC::DataCopy(local, x1Gm[start + progress], dataNum);
        } else {
            LessEqualIntegralBroadcastFill<T>::Run(
                local, scalarValue, tempABuf, dataNum);
        }
    }

    __aicore__ inline void FillX2Local(
        const AscendC::LocalTensor<T> &local,
        uint64_t start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t mode,
        T scalarValue)
    {
        if (mode == LESS_EQUAL_INNER_CONTIGUOUS) {
            AscendC::DataCopy(local, x2Gm[start + progress], dataNum);
        } else {
            LessEqualIntegralBroadcastFill<T>::Run(
                local, scalarValue, tempBBuf, dataNum);
        }
    }

    __aicore__ inline void CopyIn(
        uint64_t x1Start,
        uint64_t x2Start,
        uint64_t progress,
        uint32_t dataNum,
        uint32_t x1Mode,
        uint32_t x2Mode,
        T x1ScalarValue,
        T x2ScalarValue)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
        FillX1Local(x1Local, x1Start, progress, dataNum,
                    x1Mode, x1ScalarValue);
        FillX2Local(x2Local, x2Start, progress, dataNum,
                    x2Mode, x2ScalarValue);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void ComputeFromLocal(
        const AscendC::LocalTensor<T> &x1Local,
        const AscendC::LocalTensor<T> &x2Local,
        uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.AllocTensor<uint8_t>();
        AscendC::LocalTensor<uint8_t> maskLocal =
            maskBuf.Get<uint8_t>();
        AscendC::LocalTensor<half> oneHalfLocal =
            oneHalfBuf.Get<half>();
        AscendC::LocalTensor<half> boolHalfLocal =
            boolHalfBuf.Get<half>();
        Ops::Compute(x1Local, x2Local, yLocal, maskLocal,
                     oneHalfLocal, boolHalfLocal,
                     tempABuf, tempBBuf, dataNum);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t dataNum)
    {
        AscendC::LocalTensor<T> x1Local = x1Queue.DeQue<T>();
        AscendC::LocalTensor<T> x2Local = x2Queue.DeQue<T>();
        ComputeFromLocal(x1Local, x2Local, dataNum);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t outputStart,
                                   uint32_t dataNum)
    {
        AscendC::LocalTensor<uint8_t> yLocal =
            yQueue.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[outputStart], yLocal, dataNum);
        yQueue.FreeTensor(yLocal);
    }

    // Dead code removed: ProcessReusableOuterRange, ProcessReusableX1Tile, ProcessReusableX2Tile

private:
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x1Queue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> x2Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> oneHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> boolHalfBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempABuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBBuf;
    AscendC::GlobalTensor<T> x1Gm;
    AscendC::GlobalTensor<T> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    __gm__ T *x1Raw;
    __gm__ T *x2Raw;
    __gm__ uint8_t *yRaw;
    uint32_t tileDataNum;
    uint32_t maskBytes;
};


template <typename T>
struct LessEqualRunner {
    __aicore__ inline static void Run(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        KernelLessEqualDirect<T> op;
        op.Init(x1, x2, y);
        op.Process(tiling);
    }
};

template <>
struct LessEqualRunner<int32_t> {
    __aicore__ inline static void Run(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualIntegralSameShapeVector<int32_t, PIPELINE_BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR) {
            KernelLessEqualIntegralSameShapeVector<int32_t, BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR ||
            tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            KernelLessEqualIntegralBroadcastVector<int32_t> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualDirect<int32_t> op;
        op.Init(x1, x2, y);
        op.Process(tiling);
    }

    __aicore__ inline static void RunHot(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualHotTilingData &tiling)
    {
        AscendC::TPipe pipe;
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualIntegralSameShapeVector<int32_t, PIPELINE_BUFFER_NUM> op;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualIntegralSameShapeVector<int32_t, BUFFER_NUM> op;
        op.Init(x1, x2, y, tiling, pipe);
        op.Process(tiling);
    }
};

template <>
struct LessEqualRunner<int8_t> {
    __aicore__ inline static void Run(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualIntegralSameShapeVector<int8_t, PIPELINE_BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR) {
            KernelLessEqualIntegralSameShapeVector<int8_t, BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR ||
            tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            KernelLessEqualIntegralBroadcastVector<int8_t> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualDirect<int8_t> op;
        op.Init(x1, x2, y);
        op.Process(tiling);
    }

    __aicore__ inline static void RunHot(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualHotTilingData &tiling)
    {
        AscendC::TPipe pipe;
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualIntegralSameShapeVector<int8_t, PIPELINE_BUFFER_NUM> op;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualIntegralSameShapeVector<int8_t, BUFFER_NUM> op;
        op.Init(x1, x2, y, tiling, pipe);
        op.Process(tiling);
    }
};

template <>
struct LessEqualRunner<half> {
    __aicore__ inline static void Run(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualFloatSameShapeVector<half, PIPELINE_BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR) {
            KernelLessEqualFloatSameShapeVector<half, BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR ||
            tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            KernelLessEqualFloatBroadcastVector<half> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualDirect<half> op;
        op.Init(x1, x2, y);
        op.Process(tiling);
    }

    __aicore__ inline static void RunHot(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualHotTilingData &tiling)
    {
        AscendC::TPipe pipe;
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualFloatSameShapeVector<half, PIPELINE_BUFFER_NUM> op;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualFloatSameShapeVector<half, BUFFER_NUM> op;
        op.Init(x1, x2, y, tiling, pipe);
        op.Process(tiling);
    }
};

template <>
struct LessEqualRunner<float> {
    __aicore__ inline static void Run(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualFloatSameShapeVector<float, PIPELINE_BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR) {
            KernelLessEqualFloatSameShapeVector<float, BUFFER_NUM> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        if (tiling.pathMode == LESS_EQUAL_PATH_BROADCAST_INNER_VECTOR ||
            tiling.pathMode == LESS_EQUAL_PATH_GLOBAL_SCALAR_VECTOR) {
            KernelLessEqualFloatBroadcastVector<float> op;
            AscendC::TPipe pipe;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualDirect<float> op;
        op.Init(x1, x2, y);
        op.Process(tiling);
    }

    __aicore__ inline static void RunHot(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualHotTilingData &tiling)
    {
        AscendC::TPipe pipe;
        if (tiling.pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE) {
            KernelLessEqualFloatSameShapeVector<float, PIPELINE_BUFFER_NUM> op;
            op.Init(x1, x2, y, tiling, pipe);
            op.Process(tiling);
            return;
        }
        KernelLessEqualFloatSameShapeVector<float, BUFFER_NUM> op;
        op.Init(x1, x2, y, tiling, pipe);
        op.Process(tiling);
    }
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1,
                                      GM_ADDR x2,
                                      GM_ADDR y,
                                      GM_ADDR workspace,
                                      GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LessEqualTilingData);

    const __gm__ LessEqualTilingData *tilingGm =
        reinterpret_cast<const __gm__ LessEqualTilingData *>(tiling);
    const uint32_t pathMode = tilingGm->pathMode;

   
    if (pathMode == LESS_EQUAL_PATH_STATIC_HALF_SAME_SHAPE) {
        AscendC::InitSocState();
        KernelLessEqualStaticHalfSameShape op;
        op.Init(x1, x2, y);
        op.Process(tilingGm);
        return;
    }
    if (pathMode == LESS_EQUAL_PATH_STATIC_FLOAT_SAME_SHAPE) {
        AscendC::InitSocState();
        KernelLessEqualStaticFloatSameShape op;
        op.Init(x1, x2, y);
        op.Process(tilingGm);
        return;
    }

    const uint32_t mode = tilingGm->mode;

    
    if (pathMode == LESS_EQUAL_PATH_DIRECT_SCALAR &&
        (mode == LESS_EQUAL_MODE_SAME_SHAPE ||
         mode == LESS_EQUAL_MODE_X1_SCALAR ||
         mode == LESS_EQUAL_MODE_X2_SCALAR)) {
        KernelLessEqualDirect<DT_X1> op;
        op.Init(x1, x2, y);
        op.ProcessMicro(tilingGm->totalLength, mode);
        return;
    }

    
    if (mode == LESS_EQUAL_MODE_SAME_SHAPE &&
        (pathMode == LESS_EQUAL_PATH_CONTIGUOUS_VECTOR ||
         pathMode == LESS_EQUAL_PATH_CONTIGUOUS_PIPELINE)) {
        LessEqualHotTilingData hotTiling;
        LoadSameShapeHotTiling(tilingGm, hotTiling);
        LessEqualRunner<DT_X1>::RunHot(x1, x2, y, hotTiling);
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(
        LessEqualTilingData, tilingData, tiling);
    LessEqualRunner<DT_X1>::Run(x1, x2, y, tilingData);
}