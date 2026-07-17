#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        const LessEqualTilingData& td)
    {
        mode = td.mode;
        totalLen = td.totalLen;
        tileLen = td.tileLen;
        blockDim = td.blockDim;
        perCore = td.perCore;
        ndim = td.ndim;
        lastDimLen = td.lastDimLen;
        totalRows = td.totalRows;

        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            outShape[i] = td.outShape[i];
            x1StrideArr[i] = td.x1Stride[i];
            x2StrideArr[i] = td.x2Stride[i];
        }

        x1Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X1*>(x1));

        x2Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X1*>(x2));

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(y));

        if (totalLen == 0) {
            return;
        }

        pipe.InitBuffer(
            qX1,
            2,
            tileLen * static_cast<uint32_t>(sizeof(DT_X1)));

        pipe.InitBuffer(
            qX2,
            2,
            tileLen * static_cast<uint32_t>(sizeof(DT_X1)));

        pipe.InitBuffer(
            qOut,
            2,
            tileLen * static_cast<uint32_t>(sizeof(int8_t)));

        pipe.InitBuffer(
            bufMask,
            ((tileLen / 8) + 31) / 32 * 32);

        pipe.InitBuffer(
            bufOnes,
            tileLen * static_cast<uint32_t>(sizeof(half)));

        pipe.InitBuffer(
            bufOutHalf,
            tileLen * static_cast<uint32_t>(sizeof(half)));

        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(
                bufAHalf,
                tileLen * static_cast<uint32_t>(sizeof(half)));

            pipe.InitBuffer(
                bufBHalf,
                tileLen * static_cast<uint32_t>(sizeof(half)));
        }

        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(
                bufI32,
                tileLen * static_cast<uint32_t>(sizeof(int32_t)));
        }

        LocalTensor<half> ones =
            bufOnes.Get<half>();

        Duplicate(
            ones,
            static_cast<half>(1.0),
            tileLen);
    }

    __aicore__ inline void Process()
    {
        if (totalLen == 0) {
            return;
        }

        if (mode == 0) {
            ProcessFast();
        } else {
            ProcessBcast();
        }
    }

private:
    __aicore__ inline uint32_t RoundUp256(
        uint32_t value)
    {
        return
            (value + 255U) /
            256U *
            256U;
    }

    __aicore__ inline void PostProcessMask(
        const LocalTensor<uint8_t>& mask,
        const LocalTensor<int8_t>& output,
        uint32_t alignedLen)
    {
        LocalTensor<half> ones =
            bufOnes.Get<half>();

        LocalTensor<half> outputHalf =
            bufOutHalf.Get<half>();

        Select(
            outputHalf,
            mask,
            ones,
            static_cast<half>(0.0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            alignedLen);

        Cast(
            output,
            outputHalf,
            RoundMode::CAST_RINT,
            alignedLen);
    }

    __aicore__ inline void ComputeTile(
        const LocalTensor<DT_X1>& inputA,
        const LocalTensor<DT_X1>& inputB,
        const LocalTensor<int8_t>& output,
        uint32_t alignedLen,
        bool inputAScalar,
        bool inputBScalar)
    {
        LocalTensor<uint8_t> mask =
            bufMask.Get<uint8_t>();

        if constexpr (
            std::is_same_v<DT_X1, int32_t>) {
            LocalTensor<int32_t> minimum =
                bufI32.Get<int32_t>();

            Min(
                minimum,
                inputA,
                inputB,
                static_cast<int32_t>(alignedLen));

            Compare(
                mask,
                minimum,
                inputA,
                CMPMODE::EQ,
                alignedLen);
        } else if constexpr (
            std::is_same_v<DT_X1, int8_t>) {
            LocalTensor<half> inputAHalf =
                bufAHalf.Get<half>();

            LocalTensor<half> inputBHalf =
                bufBHalf.Get<half>();

            if (inputAScalar) {
                int8_t value =
                    inputA.GetValue(0);

                Duplicate(
                    inputAHalf,
                    static_cast<half>(
                        static_cast<float>(
                            static_cast<int32_t>(value))),
                    alignedLen);
            } else {
                Cast(
                    inputAHalf,
                    inputA,
                    RoundMode::CAST_NONE,
                    alignedLen);
            }

            if (inputBScalar) {
                int8_t value =
                    inputB.GetValue(0);

                Duplicate(
                    inputBHalf,
                    static_cast<half>(
                        static_cast<float>(
                            static_cast<int32_t>(value))),
                    alignedLen);
            } else {
                Cast(
                    inputBHalf,
                    inputB,
                    RoundMode::CAST_NONE,
                    alignedLen);
            }

            Compare(
                mask,
                inputAHalf,
                inputBHalf,
                CMPMODE::LE,
                alignedLen);
        } else {
            Compare(
                mask,
                inputA,
                inputB,
                CMPMODE::LE,
                alignedLen);
        }

        PostProcessMask(
            mask,
            output,
            alignedLen);
    }

    __aicore__ inline void ProcessFast()
    {
        uint32_t blockIndex =
            GetBlockIdx();

        uint64_t offset =
            static_cast<uint64_t>(blockIndex) *
            perCore;

        if (offset >= totalLen) {
            return;
        }

        uint32_t currentCoreLen =
            perCore;

        if (offset + currentCoreLen > totalLen) {
            currentCoreLen =
                static_cast<uint32_t>(
                    totalLen - offset);
        }

        DataCopyPadExtParams<DT_X1> pad {
            false,
            0,
            0,
            static_cast<DT_X1>(0)
        };

        for (uint32_t processed = 0;
             processed < currentCoreLen;
             processed += tileLen) {
            uint32_t currentLen =
                tileLen;

            if (processed + currentLen >
                currentCoreLen) {
                currentLen =
                    currentCoreLen - processed;
            }

            uint32_t alignedLen =
                RoundUp256(currentLen);

            uint64_t base =
                offset + processed;

            LocalTensor<DT_X1> inputA =
                qX1.AllocTensor<DT_X1>();

            DataCopyPad(
                inputA,
                x1Gm[base],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            qX1.EnQue(inputA);

            LocalTensor<DT_X1> inputB =
                qX2.AllocTensor<DT_X1>();

            DataCopyPad(
                inputB,
                x2Gm[base],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            qX2.EnQue(inputB);

            inputA =
                qX1.DeQue<DT_X1>();

            inputB =
                qX2.DeQue<DT_X1>();

            LocalTensor<int8_t> output =
                qOut.AllocTensor<int8_t>();

            ComputeTile(
                inputA,
                inputB,
                output,
                alignedLen,
                false,
                false);

            qX1.FreeTensor(inputA);
            qX2.FreeTensor(inputB);
            qOut.EnQue(output);

            output =
                qOut.DeQue<int8_t>();

            DataCopyPad(
                yGm[base],
                output,
                DataCopyExtParams {
                    1,
                    currentLen,
                    0,
                    0,
                    0
                });

            qOut.FreeTensor(output);
        }
    }

    __aicore__ inline void InitRowState(
        uint32_t firstRow)
    {
        x1Base = 0;
        x2Base = 0;

        uint32_t remaining =
            firstRow;

        for (uint32_t dim = 0;
             dim + 1 < ndim;
             ++dim) {
            uint32_t multiplier = 1;

            for (uint32_t inner = dim + 1;
                 inner + 1 < ndim;
                 ++inner) {
                multiplier *=
                    outShape[inner];
            }

            uint32_t coordinate =
                multiplier == 0
                    ? 0
                    : remaining / multiplier;

            remaining =
                multiplier == 0
                    ? remaining
                    : remaining % multiplier;

            indices[dim] =
                coordinate;

            x1Base +=
                static_cast<int32_t>(coordinate) *
                x1StrideArr[dim];

            x2Base +=
                static_cast<int32_t>(coordinate) *
                x2StrideArr[dim];
        }
    }

    __aicore__ inline void AdvanceRowState()
    {
        if (ndim < 2) {
            return;
        }

        int32_t dim =
            static_cast<int32_t>(ndim) - 2;

        indices[dim]++;

        x1Base +=
            x1StrideArr[dim];

        x2Base +=
            x2StrideArr[dim];

        while (dim > 0 &&
               indices[dim] >=
                   outShape[dim]) {
            x1Base -=
                static_cast<int32_t>(
                    outShape[dim]) *
                x1StrideArr[dim];

            x2Base -=
                static_cast<int32_t>(
                    outShape[dim]) *
                x2StrideArr[dim];

            indices[dim] = 0;
            --dim;
            indices[dim]++;

            x1Base +=
                x1StrideArr[dim];

            x2Base +=
                x2StrideArr[dim];
        }
    }

    __aicore__ inline void LoadOperand(
        TQue<TPosition::VECIN, 2>& queue,
        const GlobalTensor<DT_X1>& global,
        int32_t baseIndex,
        int32_t lastStride,
        uint32_t column,
        uint32_t currentLen,
        uint32_t alignedLen,
        LocalTensor<DT_X1>& output)
    {
        DataCopyPadExtParams<DT_X1> pad {
            false,
            0,
            0,
            static_cast<DT_X1>(0)
        };

        LocalTensor<DT_X1> local =
            queue.AllocTensor<DT_X1>();

        if (lastStride == 0) {
            DataCopyPad(
                local,
                global[
                    static_cast<uint32_t>(
                        baseIndex)],
                DataCopyExtParams {
                    1,
                    static_cast<uint32_t>(
                        sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            queue.EnQue(local);

            local =
                queue.DeQue<DT_X1>();

            if constexpr (
                std::is_same_v<DT_X1, int8_t>) {
                int8_t value =
                    local.GetValue(0);

                const uint16_t byteValue =
                    static_cast<uint16_t>(
                        static_cast<uint8_t>(
                            value));

                const uint16_t packedValue =
                    byteValue |
                    static_cast<uint16_t>(
                        byteValue << 8U);

                LocalTensor<uint16_t> packed =
                    local
                        .template ReinterpretCast<
                            uint16_t>();

                Duplicate<uint16_t>(
                    packed,
                    packedValue,
                    static_cast<int32_t>(
                        (alignedLen + 1U) /
                        2U));
            } else {
                DT_X1 value =
                    local.GetValue(0);

                Duplicate(
                    local,
                    value,
                    alignedLen);
            }
        } else {
            DataCopyPad(
                local,
                global[
                    static_cast<uint32_t>(
                        baseIndex) +
                    column],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            queue.EnQue(local);

            local =
                queue.DeQue<DT_X1>();
        }

        output = local;
    }

    __aicore__ inline void ProcessBcast()
    {
        uint32_t blockIndex =
            GetBlockIdx();

        uint32_t firstRow =
            blockIndex * perCore;

        if (firstRow >= totalRows) {
            return;
        }

        uint32_t currentCoreRows =
            perCore;

        if (firstRow + currentCoreRows >
            totalRows) {
            currentCoreRows =
                totalRows - firstRow;
        }

        uint32_t rowLength =
            lastDimLen;

        int32_t x1LastStride =
            x1StrideArr[ndim - 1];

        int32_t x2LastStride =
            x2StrideArr[ndim - 1];

        InitRowState(firstRow);

        for (uint32_t row = 0;
             row < currentCoreRows;
             ++row) {
            uint32_t outputRowBase =
                (firstRow + row) *
                rowLength;

            for (uint32_t column = 0;
                 column < rowLength;
                 column += tileLen) {
                uint32_t currentLen =
                    tileLen;

                if (column + currentLen >
                    rowLength) {
                    currentLen =
                        rowLength - column;
                }

                uint32_t alignedLen =
                    RoundUp256(currentLen);

                LocalTensor<DT_X1> inputA;
                LocalTensor<DT_X1> inputB;

                LoadOperand(
                    qX1,
                    x1Gm,
                    x1Base,
                    x1LastStride,
                    column,
                    currentLen,
                    alignedLen,
                    inputA);

                LoadOperand(
                    qX2,
                    x2Gm,
                    x2Base,
                    x2LastStride,
                    column,
                    currentLen,
                    alignedLen,
                    inputB);

                bool inputAScalar =
                    x1LastStride == 0;

                bool inputBScalar =
                    x2LastStride == 0;

                LocalTensor<int8_t> output =
                    qOut.AllocTensor<int8_t>();

                ComputeTile(
                    inputA,
                    inputB,
                    output,
                    alignedLen,
                    inputAScalar,
                    inputBScalar);

                qX1.FreeTensor(inputA);
                qX2.FreeTensor(inputB);
                qOut.EnQue(output);

                output =
                    qOut.DeQue<int8_t>();

                DataCopyPad(
                    yGm[
                        outputRowBase +
                        column],
                    output,
                    DataCopyExtParams {
                        1,
                        currentLen,
                        0,
                        0,
                        0
                    });

                qOut.FreeTensor(output);
            }

            if (row + 1 <
                currentCoreRows) {
                AdvanceRowState();
            }
        }
    }

private:
    TPipe pipe;

    TQue<TPosition::VECIN, 2> qX1;
    TQue<TPosition::VECIN, 2> qX2;
    TQue<TPosition::VECOUT, 2> qOut;

    TBuf<TPosition::VECCALC> bufMask;
    TBuf<TPosition::VECCALC> bufOnes;
    TBuf<TPosition::VECCALC> bufOutHalf;
    TBuf<TPosition::VECCALC> bufAHalf;
    TBuf<TPosition::VECCALC> bufBHalf;
    TBuf<TPosition::VECCALC> bufI32;

    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<int8_t> yGm;

    uint32_t mode;
    uint64_t totalLen;
    uint32_t tileLen;
    uint32_t blockDim;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;

    uint32_t outShape[LE_MAX_DIM];
    int32_t x1StrideArr[LE_MAX_DIM];
    int32_t x2StrideArr[LE_MAX_DIM];

    int32_t x1Base;
    int32_t x2Base;
    uint32_t indices[LE_MAX_DIM];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        LessEqualTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        LessEqualTilingData,
        tilingData,
        tiling);

    KernelLessEqual<DT_X1> op;

    op.Init(
        x1,
        x2,
        y,
        tilingData);

    op.Process();
}