// LessEqual Kernel侧实现：Init -> Process -> CopyIn -> Compute -> CopyOut。
#include "kernel_operator.h"
#include <cstdint>
#include <type_traits>

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

// Process 当前按 CopyIn -> Compute -> CopyOut 串行处理，每个队列只需要一个 buffer。
// 释放的 UB 用于结果 mask 向量展开和整数类型的中间计算。
constexpr uint32_t LESS_EQUAL_BUFFER_NUM = 1;
constexpr uint32_t LESS_EQUAL_VECTOR_ELEMENTS = 128;

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    // Init：绑定 GM 地址、解析 tiling、计算当前核负责的输出区间，并初始化 UB 队列。
    __aicore__ inline void Init(GM_ADDR x1,
                                GM_ADDR x2,
                                GM_ADDR y,
                                const LessEqualTilingData &tiling_data) {
        tilingData = tiling_data;
        blockDim = tilingData.blockDim == 0 ? 1 : tilingData.blockDim;
        tileLength = tilingData.tileLength == 0 ? LESS_EQUAL_VECTOR_ELEMENTS : tilingData.tileLength;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t lengthPerCore =
            tilingData.length == 0 ? 0 : (tilingData.length + blockDim - 1) / blockDim;

        startOffset = blockIdx * lengthPerCore;
        if (startOffset < tilingData.length) {
            coreLength = tilingData.length - startOffset;
            if (coreLength > lengthPerCore) {
                coreLength = lengthPerCore;
            }
        } else {
            coreLength = 0;
        }

        x1Gm.SetGlobalBuffer((__gm__ DT_X1 *)x1, tilingData.x1Numel);
        x2Gm.SetGlobalBuffer((__gm__ DT_X1 *)x2, tilingData.x2Numel);
        yGm.SetGlobalBuffer((__gm__ uint8_t *)y, tilingData.length);

        x1IsScalar = tilingData.x1Numel == 1;
        x2IsScalar = tilingData.x2Numel == 1;
        x1IsLinear = tilingData.x1Numel == tilingData.length;
        x2IsLinear = tilingData.x2Numel == tilingData.length;

        if (x1IsScalar) {
            x1ScalarValue = x1Gm.GetValue(0);
        }
        if (x2IsScalar) {
            x2ScalarValue = x2Gm.GetValue(0);
        }

        pipe.InitBuffer(x1Queue, LESS_EQUAL_BUFFER_NUM, tileLength * sizeof(DT_X1));
        pipe.InitBuffer(x2Queue, LESS_EQUAL_BUFFER_NUM, tileLength * sizeof(DT_X1));
        pipe.InitBuffer(yQueue, LESS_EQUAL_BUFFER_NUM, tileLength * sizeof(uint8_t));

        // 两个计算 buffer 同时服务于 int8/int32 中间结果和 half 类型的 0/1 展开。
        // scratchElementBytes 至少为 half 大小，且对 float/int32 保留原类型容量。
        constexpr uint32_t scratchElementBytes =
            sizeof(DT_X1) > sizeof(half) ? sizeof(DT_X1) : sizeof(half);
        pipe.InitBuffer(calcBuffer1, tileLength * scratchElementBytes);
        pipe.InitBuffer(calcBuffer2, tileLength * scratchElementBytes);
    }

    // Process：按 tileLength 分块循环，每块严格执行 CopyIn -> Compute -> CopyOut。
    __aicore__ inline void Process() {
        if (coreLength == 0) {
            return;
        }

        for (uint32_t localOffset = 0; localOffset < coreLength; localOffset += tileLength) {
            processLength = coreLength - localOffset;
            if (processLength > tileLength) {
                processLength = tileLength;
            }

            const uint32_t globalOffset = startOffset + localOffset;
            CopyIn(globalOffset);
            Compute();
            CopyOut(globalOffset);
        }
    }

private:
    // 复杂广播快路径：每个tile只做一次线性下标到多维坐标的反解，后续逐元素用carry递增。
    __aicore__ inline void InitBroadcastState(uint32_t outputLinear,
                                              uint32_t *coord,
                                              uint32_t &x1Offset,
                                              uint32_t &x2Offset) const {
        uint32_t residual = outputLinear;
        x1Offset = 0;
        x2Offset = 0;

        for (int32_t dim = static_cast<int32_t>(tilingData.rank) - 1; dim >= 0; --dim) {
            const uint32_t dimSize = tilingData.outputShape[dim];
            const uint32_t coordinate = residual % dimSize;
            residual /= dimSize;
            coord[dim] = coordinate;
            x1Offset += coordinate * tilingData.x1Stride[dim];
            x2Offset += coordinate * tilingData.x2Stride[dim];
        }
    }

    __aicore__ inline void AdvanceBroadcastState(uint32_t *coord,
                                                 uint32_t &x1Offset,
                                                 uint32_t &x2Offset) const {
        for (int32_t dim = static_cast<int32_t>(tilingData.rank) - 1; dim >= 0; --dim) {
            const uint32_t dimSize = tilingData.outputShape[dim];
            const uint32_t x1Stride = tilingData.x1Stride[dim];
            const uint32_t x2Stride = tilingData.x2Stride[dim];

            coord[dim] += 1;
            x1Offset += x1Stride;
            x2Offset += x2Stride;

            if (coord[dim] < dimSize) {
                return;
            }

            coord[dim] = 0;
            x1Offset -= dimSize * x1Stride;
            x2Offset -= dimSize * x2Stride;
        }
    }

    __aicore__ inline bool ScalarLessEqual(DT_X1 lhs, DT_X1 rhs) const {
        if constexpr (std::is_same_v<DT_X1, half>) {
            return static_cast<float>(lhs) <= static_cast<float>(rhs);
        } else {
            return lhs <= rhs;
        }
    }

    __aicore__ inline void CopyLinearToLocal(AscendC::LocalTensor<DT_X1> local,
                                             AscendC::GlobalTensor<DT_X1> &gm,
                                             uint32_t globalOffset) {
        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(processLength * sizeof(DT_X1)),
            0,
            0,
            0};
        AscendC::DataCopyPadExtParams<DT_X1> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad(local, gm[globalOffset], copyParams, padParams);
    }

    // 将广播输入收集到连续 UB。若广播后存在足够长的连续尾部 span，则对每个 span
    // 的32B对齐主体使用DMA，仅短头尾使用标量；完全离散时回退到carry逐元素寻址。
    __aicore__ inline void CopyBroadcastToLocal(AscendC::LocalTensor<DT_X1> local,
                                                AscendC::GlobalTensor<DT_X1> &gm,
                                                uint32_t globalOffset,
                                                uint32_t contiguousSpan,
                                                bool useX1Offset) {
        constexpr uint32_t alignElements = 32U / sizeof(DT_X1);
        if (contiguousSpan < alignElements) {
            uint32_t coord[LESS_EQUAL_MAX_RANK] = {0};
            uint32_t x1Offset = 0;
            uint32_t x2Offset = 0;
            InitBroadcastState(globalOffset, coord, x1Offset, x2Offset);
            for (uint32_t i = 0; i < processLength; ++i) {
                const uint32_t inputOffset = useX1Offset ? x1Offset : x2Offset;
                local.SetValue(i, gm.GetValue(inputOffset));
                AdvanceBroadcastState(coord, x1Offset, x2Offset);
            }
            return;
        }

        uint32_t localIndex = 0;
        while (localIndex < processLength) {
            const uint32_t outputLinear = globalOffset + localIndex;
            const uint32_t offsetInSpan = outputLinear % contiguousSpan;
            uint32_t runLength = contiguousSpan - offsetInSpan;
            if (runLength > processLength - localIndex) {
                runLength = processLength - localIndex;
            }

            uint32_t coord[LESS_EQUAL_MAX_RANK] = {0};
            uint32_t x1Offset = 0;
            uint32_t x2Offset = 0;
            InitBroadcastState(outputLinear, coord, x1Offset, x2Offset);
            uint32_t inputOffset = useX1Offset ? x1Offset : x2Offset;

            uint32_t headCount = 0;
            const uint32_t localMisalignment = localIndex % alignElements;
            if (localMisalignment != 0) {
                headCount = alignElements - localMisalignment;
                if (headCount > runLength) {
                    headCount = runLength;
                }
            }
            for (uint32_t i = 0; i < headCount; ++i) {
                local.SetValue(localIndex + i, gm.GetValue(inputOffset + i));
            }
            localIndex += headCount;
            inputOffset += headCount;
            runLength -= headCount;

            const uint32_t dmaCount = runLength / alignElements * alignElements;
            if (dmaCount > 0) {
                AscendC::DataCopyExtParams copyParams{
                    1,
                    static_cast<uint32_t>(dmaCount * sizeof(DT_X1)),
                    0,
                    0,
                    0};
                AscendC::DataCopyPadExtParams<DT_X1> padParams{false, 0, 0, 0};
                AscendC::DataCopyPad(local[localIndex], gm[inputOffset], copyParams, padParams);
                localIndex += dmaCount;
                inputOffset += dmaCount;
                runLength -= dmaCount;
            }

            for (uint32_t i = 0; i < runLength; ++i) {
                local.SetValue(localIndex + i, gm.GetValue(inputOffset + i));
            }
            localIndex += runLength;
        }
    }

    // 所有向量计算统一向上补齐到 128 元素。对于 half 是 256B，对于 float/int32
    // 是 512B，均满足 Compare 的 256B 对齐约束；补齐部分位于已分配 UB 内且不会写回 GM。
    __aicore__ inline uint32_t GetAlignedComputeLength() const {
        return (processLength + LESS_EQUAL_VECTOR_ELEMENTS - 1U) /
            LESS_EQUAL_VECTOR_ELEMENTS * LESS_EQUAL_VECTOR_ELEMENTS;
    }

    // Compare 输出为压缩 bitmask。Select 模式1连续消费这些 bit，向量生成 half 0/1，
    // 再 Cast 为 uint8_t bool 字节，从而删除原来的逐元素 GetValue/SetValue 展开循环。
    __aicore__ inline void ExpandCompareMaskVector(AscendC::LocalTensor<uint8_t> yLocal,
                                                   uint32_t computeLength) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> oneLocal = calcBuffer1.Get<half>();
        AscendC::LocalTensor<half> selectedLocal = calcBuffer2.Get<half>();
        AscendC::Duplicate(oneLocal, static_cast<half>(1.0f), computeLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(selectedLocal,
                        yLocal,
                        oneLocal,
                        static_cast<half>(0.0f),
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        computeLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal,
                      selectedLocal,
                      AscendC::RoundMode::CAST_NONE,
                      computeLength);
    }

    // 两个标量输入无需 Compare；仍使用 Duplicate + Cast 一次性生成 bool tensor。
    __aicore__ inline void FillBoolVector(AscendC::LocalTensor<uint8_t> yLocal,
                                          uint8_t value,
                                          uint32_t computeLength) {
        AscendC::LocalTensor<half> valueLocal = calcBuffer1.Get<half>();
        const half vectorValue = value == 0U ?
            static_cast<half>(0.0f) : static_cast<half>(1.0f);
        AscendC::Duplicate(valueLocal, vectorValue, computeLength);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(yLocal,
                      valueLocal,
                      AscendC::RoundMode::CAST_NONE,
                      computeLength);
    }

    // CopyIn：连续输入走 DMA；复杂广播只收集非标量输入，标量在 Compute 中向量填充。
    __aicore__ inline void CopyIn(uint32_t globalOffset) {
        if (x1IsScalar && x2IsScalar) {
            return;
        }

        if (x1IsScalar) {
            AscendC::LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();
            if (x2IsLinear) {
                CopyLinearToLocal(x2Local, x2Gm, globalOffset);
            } else {
                CopyBroadcastToLocal(x2Local,
                                     x2Gm,
                                     globalOffset,
                                     tilingData.x2ContiguousSpan,
                                     false);
            }
            x2Queue.EnQue(x2Local);
            return;
        }

        if (x2IsScalar) {
            AscendC::LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
            if (x1IsLinear) {
                CopyLinearToLocal(x1Local, x1Gm, globalOffset);
            } else {
                CopyBroadcastToLocal(x1Local,
                                     x1Gm,
                                     globalOffset,
                                     tilingData.x1ContiguousSpan,
                                     true);
            }
            x1Queue.EnQue(x1Local);
            return;
        }

        if (x1IsLinear && !x2IsLinear) {
            AscendC::LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
            AscendC::LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();
            CopyLinearToLocal(x1Local, x1Gm, globalOffset);

            CopyBroadcastToLocal(x2Local,
                                 x2Gm,
                                 globalOffset,
                                 tilingData.x2ContiguousSpan,
                                 false);

            x1Queue.EnQue(x1Local);
            x2Queue.EnQue(x2Local);
            return;
        }

        if (x2IsLinear && !x1IsLinear) {
            AscendC::LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
            AscendC::LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();
            CopyLinearToLocal(x2Local, x2Gm, globalOffset);

            CopyBroadcastToLocal(x1Local,
                                 x1Gm,
                                 globalOffset,
                                 tilingData.x1ContiguousSpan,
                                 true);

            x1Queue.EnQue(x1Local);
            x2Queue.EnQue(x2Local);
            return;
        }

        AscendC::LocalTensor<DT_X1> x1Local = x1Queue.AllocTensor<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue.AllocTensor<DT_X1>();
        if (tilingData.contiguous != 0) {
            CopyLinearToLocal(x1Local, x1Gm, globalOffset);
            CopyLinearToLocal(x2Local, x2Gm, globalOffset);
        } else {
            CopyBroadcastToLocal(x1Local,
                                 x1Gm,
                                 globalOffset,
                                 tilingData.x1ContiguousSpan,
                                 true);
            CopyBroadcastToLocal(x2Local,
                                 x2Gm,
                                 globalOffset,
                                 tilingData.x2ContiguousSpan,
                                 false);
        }
        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    // Compute：所有数据类型均以向量指令完成比较和 bool 字节生成。
    __aicore__ inline void Compute() {
        AscendC::LocalTensor<uint8_t> yLocal = yQueue.AllocTensor<uint8_t>();
        const uint32_t computeLength = GetAlignedComputeLength();

        if (x1IsScalar && x2IsScalar) {
            const uint8_t value = ScalarLessEqual(x1ScalarValue, x2ScalarValue) ? 1U : 0U;
            FillBoolVector(yLocal, value, computeLength);
            yQueue.EnQue(yLocal);
            return;
        }

        if (x1IsScalar) {
            AscendC::LocalTensor<DT_X1> x2Local = x2Queue.DeQue<DT_X1>();
            if constexpr (std::is_same_v<DT_X1, half> || std::is_same_v<DT_X1, float>) {
                // x1 <= x2 等价于 x2 >= x1Scalar。
                AscendC::CompareScalar(
                    yLocal, x2Local, x1ScalarValue, AscendC::CMPMODE::GE, computeLength);
            } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
                // min(x1, x2) == x1 等价于 x1 <= x2；Compare 对 int32 支持 EQ。
                AscendC::LocalTensor<int32_t> minLocal = calcBuffer1.Get<int32_t>();
                AscendC::LocalTensor<int32_t> scalarLocal = calcBuffer2.Get<int32_t>();
                AscendC::Duplicate(scalarLocal, static_cast<int32_t>(x1ScalarValue), computeLength);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Min(minLocal, scalarLocal, x2Local, static_cast<int32_t>(computeLength));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(
                    yLocal, minLocal, scalarLocal, AscendC::CMPMODE::EQ, computeLength);
            } else {
                // AICore禁止运行时int8标量直接转换为half；该小众分支保留标量比较。
                for (uint32_t i = 0; i < processLength; ++i) {
                    yLocal.SetValue(i,
                        ScalarLessEqual(x1ScalarValue, x2Local.GetValue(i)) ? 1U : 0U);
                }
                yQueue.EnQue(yLocal);
                x2Queue.FreeTensor(x2Local);
                return;
            }
            ExpandCompareMaskVector(yLocal, computeLength);
            yQueue.EnQue(yLocal);
            x2Queue.FreeTensor(x2Local);
            return;
        }

        if (x2IsScalar) {
            AscendC::LocalTensor<DT_X1> x1Local = x1Queue.DeQue<DT_X1>();
            if constexpr (std::is_same_v<DT_X1, half> || std::is_same_v<DT_X1, float>) {
                AscendC::CompareScalar(
                    yLocal, x1Local, x2ScalarValue, AscendC::CMPMODE::LE, computeLength);
            } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
                AscendC::LocalTensor<int32_t> minLocal = calcBuffer1.Get<int32_t>();
                AscendC::LocalTensor<int32_t> scalarLocal = calcBuffer2.Get<int32_t>();
                AscendC::Duplicate(scalarLocal, static_cast<int32_t>(x2ScalarValue), computeLength);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Min(minLocal, x1Local, scalarLocal, static_cast<int32_t>(computeLength));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Compare(
                    yLocal, minLocal, x1Local, AscendC::CMPMODE::EQ, computeLength);
            } else {
                // 同上：避免AICore不支持的运行时int8标量到half转换。
                for (uint32_t i = 0; i < processLength; ++i) {
                    yLocal.SetValue(i,
                        ScalarLessEqual(x1Local.GetValue(i), x2ScalarValue) ? 1U : 0U);
                }
                yQueue.EnQue(yLocal);
                x1Queue.FreeTensor(x1Local);
                return;
            }
            ExpandCompareMaskVector(yLocal, computeLength);
            yQueue.EnQue(yLocal);
            x1Queue.FreeTensor(x1Local);
            return;
        }

        AscendC::LocalTensor<DT_X1> x1Local = x1Queue.DeQue<DT_X1>();
        AscendC::LocalTensor<DT_X1> x2Local = x2Queue.DeQue<DT_X1>();
        if constexpr (std::is_same_v<DT_X1, half> || std::is_same_v<DT_X1, float>) {
            AscendC::Compare(
                yLocal, x1Local, x2Local, AscendC::CMPMODE::LE, computeLength);
        } else if constexpr (std::is_same_v<DT_X1, int32_t>) {
            AscendC::LocalTensor<int32_t> minLocal = calcBuffer1.Get<int32_t>();
            AscendC::Min(minLocal, x1Local, x2Local, static_cast<int32_t>(computeLength));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                yLocal, minLocal, x1Local, AscendC::CMPMODE::EQ, computeLength);
        } else {
            AscendC::LocalTensor<half> lhsLocal = calcBuffer1.Get<half>();
            AscendC::LocalTensor<half> rhsLocal = calcBuffer2.Get<half>();
            AscendC::Cast(lhsLocal, x1Local, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::Cast(rhsLocal, x2Local, AscendC::RoundMode::CAST_NONE, computeLength);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Compare(
                yLocal, lhsLocal, rhsLocal, AscendC::CMPMODE::LE, computeLength);
        }
        ExpandCompareMaskVector(yLocal, computeLength);
        yQueue.EnQue(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }
    // CopyOut：将 bool 结果写回 GM，DataCopyPad 可处理非 32B 对齐尾块。
    __aicore__ inline void CopyOut(uint32_t globalOffset) {
        AscendC::LocalTensor<uint8_t> yLocal = yQueue.DeQue<uint8_t>();

        AscendC::DataCopyExtParams copyParams{
            1,
            static_cast<uint32_t>(processLength * sizeof(uint8_t)),
            0,
            0,
            0};
        AscendC::DataCopyPad(yGm[globalOffset], yLocal, copyParams);
        yQueue.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, LESS_EQUAL_BUFFER_NUM> x1Queue;
    AscendC::TQue<AscendC::QuePosition::VECIN, LESS_EQUAL_BUFFER_NUM> x2Queue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, LESS_EQUAL_BUFFER_NUM> yQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuffer1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuffer2;

    AscendC::GlobalTensor<DT_X1> x1Gm;
    AscendC::GlobalTensor<DT_X1> x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;

    LessEqualTilingData tilingData;
    uint32_t blockDim;
    uint32_t tileLength;
    uint32_t startOffset;
    uint32_t coreLength;
    uint32_t processLength;

    uint32_t x1IsScalar;
    uint32_t x2IsScalar;
    uint32_t x1IsLinear;
    uint32_t x2IsLinear;
    DT_X1 x1ScalarValue;
    DT_X1 x2ScalarValue;
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1,
                                      GM_ADDR x2,
                                      GM_ADDR y,
                                      GM_ADDR workspace,
                                      GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);

    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}


