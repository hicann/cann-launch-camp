#include <type_traits>

#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

namespace {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t FP16_VECTOR_ALIGN_ELEMENTS = 128;
constexpr uint32_t FP32_VECTOR_ALIGN_ELEMENTS = 64;
constexpr uint32_t INT8_VECTOR_ALIGN_ELEMENTS = 256;
constexpr uint32_t BUFFER_PADDING_ELEMENTS = 128;
constexpr uint32_t INT8_BUFFER_PADDING_ELEMENTS = 256;
constexpr uint32_t OUTPUT_BLOCK_ELEMENTS = 32;
}

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const LessEqualTilingData &tilingData) {
        outputLength = tilingData.outputLength;
        rank = tilingData.rank;
        mode = tilingData.mode;
        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            outputShape[i] = tilingData.outputShape[i];
            x1Stride[i] = tilingData.x1Stride[i];
            x2Stride[i] = tilingData.x2Stride[i];
        }

        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1, tilingData.x1Length);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2, tilingData.x2Length);
        yGm.SetGlobalBuffer((__gm__ int8_t *)y, outputLength);

        const uint64_t blockNum = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t alignedBlocks = outputLength / OUTPUT_BLOCK_ELEMENTS;
        const uint64_t tailElements = outputLength % OUTPUT_BLOCK_ELEMENTS;
        const uint64_t baseBlocks = alignedBlocks / blockNum;
        const uint64_t remainderBlocks = alignedBlocks % blockNum;
        const uint64_t currentBlocks = baseBlocks + (blockIdx < remainderBlocks ? 1 : 0);
        const uint64_t startBlocks = blockIdx * baseBlocks +
                                     (blockIdx < remainderBlocks ? blockIdx : remainderBlocks);
        blockStart = startBlocks * OUTPUT_BLOCK_ELEMENTS;
        blockLength = currentBlocks * OUTPUT_BLOCK_ELEMENTS;
        if (blockIdx == blockNum - 1) {
            blockLength += tailElements;
        }

        constexpr uint32_t bufferLength = TILE_LENGTH +
            (std::is_same_v<DT_X1, int8_t> ? INT8_BUFFER_PADDING_ELEMENTS : BUFFER_PADDING_ELEMENTS);
        pipe.InitBuffer(x1Queue, BUFFER_NUM, bufferLength * sizeof(DT_X1));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, bufferLength * sizeof(DT_X1));
        pipe.InitBuffer(yQueue, BUFFER_NUM, bufferLength * sizeof(int8_t));
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(tmp1, bufferLength * sizeof(half));
            pipe.InitBuffer(tmp2, bufferLength * sizeof(half));
            pipe.InitBuffer(tmp3, bufferLength * sizeof(half));
        } else if constexpr (std::is_same_v<DT_X1, float> || std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(tmp1, bufferLength * sizeof(half));
        }
    }

    __aicore__ inline void Process() {
        if (blockLength == 0) {
            return;
        }
        if (mode == LESS_EQUAL_NO_BROADCAST) {
            ProcessFlat(false, false);
        } else if (mode == LESS_EQUAL_X1_SCALAR) {
            ProcessFlat(true, false);
        } else if (mode == LESS_EQUAL_X2_SCALAR) {
            ProcessFlat(false, true);
        } else {
            ProcessGeneralBroadcast();
        }
    }

private:
    __aicore__ inline uint32_t AlignComputeLength(uint32_t length) const {
        if constexpr (std::is_same_v<DT_X1, float> || std::is_same_v<DT_X1, int32_t>) {
            return (length + FP32_VECTOR_ALIGN_ELEMENTS - 1) /
                   FP32_VECTOR_ALIGN_ELEMENTS * FP32_VECTOR_ALIGN_ELEMENTS;
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            return (length + INT8_VECTOR_ALIGN_ELEMENTS - 1) /
                   INT8_VECTOR_ALIGN_ELEMENTS * INT8_VECTOR_ALIGN_ELEMENTS;
        } else {
            return (length + FP16_VECTOR_ALIGN_ELEMENTS - 1) /
                   FP16_VECTOR_ALIGN_ELEMENTS * FP16_VECTOR_ALIGN_ELEMENTS;
        }
    }

    __aicore__ inline void CopyInput(AscendC::LocalTensor<DT_X1> local,
                                     AscendC::GlobalTensor<DT_X1> &global, uint64_t offset,
                                     uint32_t length, uint32_t computeLength, bool scalar) {
        if (scalar) {
            if constexpr (std::is_same_v<DT_X1, int8_t>) {
                AscendC::LocalTensor<half> scalarHalf = tmp1.Get<half>();
                AscendC::Duplicate(scalarHalf, static_cast<half>(global.GetValue(offset)), computeLength);
                AscendC::Cast(local, scalarHalf, AscendC::RoundMode::CAST_NONE, computeLength);
            } else {
                AscendC::Duplicate(local, global.GetValue(offset), computeLength);
            }
            return;
        }
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = length * sizeof(DT_X1);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<DT_X1> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, global[offset], copyParams, padParams);
    }

    __aicore__ inline void Compute(uint32_t computeLength) {
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue.DeQue<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue.DeQue<DT_X1>();
        AscendC::LocalTensor<int8_t> yLocal = yQueue.AllocTensor<int8_t>();

        if constexpr (std::is_same_v<DT_X1, half>) {
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::LE, computeLength);
            AscendC::Duplicate(x1Local, static_cast<half>(1), computeLength);
            AscendC::Select(x1Local, yLocal, x1Local, static_cast<half>(0),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            AscendC::Cast(yLocal, x1Local, AscendC::RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, float>) {
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::LE, computeLength);
            AscendC::Duplicate(x1Local, 1.0F, computeLength);
            AscendC::Select(x1Local, yLocal, x1Local, 0.0F,
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            AscendC::LocalTensor<half> resultHalf = tmp1.Get<half>();
            AscendC::Cast(resultHalf, x1Local, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::Cast(yLocal, resultHalf, AscendC::RoundMode::CAST_NONE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
            AscendC::Min(x2Local, x1Local, x2Local, computeLength);
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::EQ, computeLength);
            AscendC::LocalTensor<half> resultHalf = tmp1.Get<half>();
            AscendC::Duplicate(resultHalf, static_cast<half>(1), computeLength);
            AscendC::Select(resultHalf, yLocal, resultHalf, static_cast<half>(0),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            AscendC::Cast(yLocal, resultHalf, AscendC::RoundMode::CAST_NONE, computeLength);
        } else {
            AscendC::LocalTensor<half> x1Half = tmp1.Get<half>();
            AscendC::LocalTensor<half> x2Half = tmp2.Get<half>();
            AscendC::LocalTensor<half> resultHalf = tmp3.Get<half>();
            AscendC::Cast(x1Half, x1Local, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::Cast(x2Half, x2Local, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::Compare(yLocal, x1Half, x2Half, AscendC::CMPMODE::LE, computeLength);
            AscendC::Duplicate(resultHalf, static_cast<half>(1), computeLength);
            AscendC::Select(resultHalf, yLocal, resultHalf, static_cast<half>(0),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, computeLength);
            AscendC::Cast(yLocal, resultHalf, AscendC::RoundMode::CAST_NONE, computeLength);
        }

        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
        yQueue.EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void ProcessTile(uint64_t outputOffset, uint64_t x1Offset,
                                       uint64_t x2Offset, uint32_t length,
                                       bool x1Scalar, bool x2Scalar) {
        const uint32_t computeLength = AlignComputeLength(length);
        AscendC::LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();
        CopyInput(x1Local, x1Gm, x1Offset, length, computeLength, x1Scalar);
        CopyInput(x2Local, x2Gm, x2Offset, length, computeLength, x2Scalar);
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);

        Compute(computeLength);

        AscendC::LocalTensor<int8_t> yLocal = yQueue.DeQue<int8_t>();
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = length * sizeof(int8_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void ProcessFlat(bool x1Scalar, bool x2Scalar) {
        uint64_t processed = 0;
        while (processed < blockLength) {
            const uint64_t remaining = blockLength - processed;
            const uint32_t length = remaining > TILE_LENGTH ? TILE_LENGTH : static_cast<uint32_t>(remaining);
            const uint64_t outputOffset = blockStart + processed;
            ProcessTile(outputOffset, x1Scalar ? 0 : outputOffset,
                        x2Scalar ? 0 : outputOffset, length, x1Scalar, x2Scalar);
            processed += length;
        }
    }

    __aicore__ inline void MapInputIndices(uint64_t outputIndex, uint64_t &x1Index,
                                           uint64_t &x2Index) const {
        x1Index = 0;
        x2Index = 0;
        uint64_t remaining = outputIndex;
        for (int32_t dim = static_cast<int32_t>(rank) - 1; dim >= 0; --dim) {
            const uint64_t coordinate = remaining % outputShape[dim];
            remaining /= outputShape[dim];
            x1Index += coordinate * x1Stride[dim];
            x2Index += coordinate * x2Stride[dim];
        }
    }

    __aicore__ inline void ProcessGeneralBroadcast() {
        const uint64_t innerLength = rank == 0 ? 1 : outputShape[rank - 1];
        const bool x1InnerScalar = rank == 0 || x1Stride[rank - 1] == 0;
        const bool x2InnerScalar = rank == 0 || x2Stride[rank - 1] == 0;
        uint64_t processed = 0;
        while (processed < blockLength) {
            const uint64_t outputOffset = blockStart + processed;
            const uint64_t rowRemaining = innerLength - outputOffset % innerLength;
            const uint64_t blockRemaining = blockLength - processed;
            uint64_t segmentLength = rowRemaining < blockRemaining ? rowRemaining : blockRemaining;
            if (segmentLength > TILE_LENGTH) {
                segmentLength = TILE_LENGTH;
            }

            uint64_t x1Offset;
            uint64_t x2Offset;
            MapInputIndices(outputOffset, x1Offset, x2Offset);
            ProcessTile(outputOffset, x1Offset, x2Offset, static_cast<uint32_t>(segmentLength),
                        x1InnerScalar, x2InnerScalar);
            processed += segmentLength;
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x1Queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x2Queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp3;
    AscendC::GlobalTensor<DT_X1> x1Gm;
    AscendC::GlobalTensor<DT_X1> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t outputShape[LESS_EQUAL_MAX_DIMS];
    uint64_t x1Stride[LESS_EQUAL_MAX_DIMS];
    uint64_t x2Stride[LESS_EQUAL_MAX_DIMS];
    uint64_t outputLength;
    uint64_t blockStart;
    uint64_t blockLength;
    uint32_t rank;
    uint32_t mode;
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                     GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}
