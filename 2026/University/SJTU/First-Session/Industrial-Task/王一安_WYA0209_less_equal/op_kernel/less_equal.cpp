// LessEqual Kernel 侧实现。
// 支持广播机制的逐元素比较算子：y = (x1 <= x2)
// 输入类型：float16/float32/int32/int8，输出类型：bool
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2U;
}

template <typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                GM_ADDR workspace, GM_ADDR tiling) {
        // 获取 tiling 数据
        REGISTER_TILING_DEFAULT(LessEqualTilingData);
        GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tilingData, tiling);

        totalLength_ = tilingData.totalLength;
        tileLength_ = tilingData.tileLength;
        ndim_ = tilingData.ndim;
        blockLength_ = tilingData.blockLength;

        // 复制 stride 信息到本地（用于广播寻址）
        for (uint32_t i = 0; i < ndim_ && i < MAX_NDIM; i++) {
            outShape_[i] = tilingData.outShape[i];
            stride0_[i] = tilingData.stride0[i];
            stride1_[i] = tilingData.stride1[i];
        }

        // 计算当前核的偏移和长度
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        coreOffset_ = blockIdx * blockLength_;

        if (coreOffset_ >= totalLength_) {
            coreLength_ = 0U;
            return;
        }

        const uint64_t remaining = totalLength_ - coreOffset_;
        coreLength_ = remaining < blockLength_ ? remaining : blockLength_;

        // 设置全局内存指针
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x1),
                              static_cast<uint32_t>(totalLength_));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x2),
                              static_cast<uint32_t>(totalLength_));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y),
                             static_cast<uint32_t>(totalLength_));

        // 初始化管道缓冲区
        // 输入队列：两个输入各需一个缓冲区（双缓冲）
        pipe_.InitBuffer(x1Queue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(x2Queue_, BUFFER_NUM, tileLength_ * sizeof(T));
        // 输出队列：bool 类型
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, tileLength_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        if (coreLength_ == 0U) {
            return;
        }

        const uint64_t tileCount =
            (coreLength_ + static_cast<uint64_t>(tileLength_) - 1U) /
            static_cast<uint64_t>(tileLength_);

        for (uint64_t tileIdx = 0U; tileIdx < tileCount; ++tileIdx) {
            const uint64_t offset = tileIdx * static_cast<uint64_t>(tileLength_);
            const uint64_t remaining = coreLength_ - offset;
            const uint32_t validLength = static_cast<uint32_t>(
                remaining < tileLength_ ? remaining : tileLength_);

            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    // 从全局内存拷贝数据到本地内存（处理广播寻址）
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t validLength) {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();

        const uint64_t globalOffset = coreOffset_ + offset;

        // 对于简单情况（无广播，两个输入形状相同），使用 DataCopy
        // 对于广播情况，需要逐元素寻址
        if (IsBroadcastNeeded()) {
            // 广播情况：逐元素拷贝
            for (uint32_t i = 0; i < validLength; i++) {
                uint64_t outIdx = globalOffset + i;
                uint64_t inIdx0 = ComputeInputIndex(outIdx, stride0_);
                uint64_t inIdx1 = ComputeInputIndex(outIdx, stride1_);

                // 使用 scalar 方式逐个拷贝（广播场景通常数据量不大）
                x1Local.SetValue(i, x1Gm_[inIdx0]);
                x2Local.SetValue(i, x2Gm_[inIdx1]);
            }
        } else {
            // 无广播：使用高效的 DataCopy
            DataCopy(x1Local, x1Gm_[globalOffset], validLength);
            DataCopy(x2Local, x2Gm_[globalOffset], validLength);
        }

        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    // 核心计算：逐元素比较 x1 <= x2
    __aicore__ inline void Compute(uint32_t validLength) {
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();

        // 使用 Compare 指令进行逐元素比较
        // CMP_LE: 小于等于比较，结果存入 yLocal（bool/uint8 类型）
        Compare(yLocal, x1Local, x2Local, CMP_LE, validLength);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    // 将结果从本地内存拷贝到全局内存
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t validLength) {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();

        const uint64_t globalOffset = coreOffset_ + offset;

        if (IsBroadcastNeeded()) {
            // 广播情况：逐元素拷贝
            for (uint32_t i = 0; i < validLength; i++) {
                yGm_[globalOffset + i] = yLocal.GetValue(i);
            }
        } else {
            // 无广播：使用高效的 DataCopy
            DataCopy(yGm_[globalOffset], yLocal, validLength);
        }

        yQueue_.FreeTensor(yLocal);
    }

    // 判断是否需要广播处理
    __aicore__ inline bool IsBroadcastNeeded() const {
        // 如果任何 stride 不为标准的连续布局，则需要广播
        // 简单判断：检查 stride0 或 stride1 是否有 0 值（广播维度）
        for (uint32_t i = 0; i < ndim_; i++) {
            if (stride0_[i] == 0 || stride1_[i] == 0) {
                // 检查对应的输出维度是否大于 1（真正的广播）
                if (outShape_[i] > 1) {
                    return true;
                }
            }
        }
        return false;
    }

    // 根据 flat output index 计算输入 tensor 的 flat index
    // 使用预计算的 stride 进行广播寻址
    __aicore__ inline uint64_t ComputeInputIndex(
        uint64_t outFlatIdx, const uint32_t *strides) const
    {
        uint64_t inFlatIdx = 0;
        uint64_t remaining = outFlatIdx;

        // 从最外层到最内层逐维计算
        for (uint32_t d = 0; d < ndim_; d++) {
            uint32_t dimSize = outShape_[d];
            uint64_t coord = remaining / dimSize;  // 该维的坐标（从外层看）
            remaining = remaining % dimSize;

            // 但我们需要从内层到外层的 stride
            // 重新计算：从内层开始
        }

        // 更简单的方法：从内层到外层分解 flat index
        uint64_t idx = outFlatIdx;
        for (int d = (int)ndim_ - 1; d >= 0; d--) {
            uint32_t dimSize = outShape_[d];
            uint64_t coord = idx % dimSize;
            idx = idx / dimSize;
            inFlatIdx += coord * strides[d];
        }

        return inFlatIdx;
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;

    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<uint8_t> yGm_;

    uint64_t totalLength_ = 0U;
    uint64_t blockLength_ = 0U;
    uint64_t coreOffset_ = 0U;
    uint64_t coreLength_ = 0U;
    uint32_t tileLength_ = 0U;
    uint32_t ndim_ = 0U;

    // 广播相关参数（从 tiling 数据复制）
    uint32_t outShape_[MAX_NDIM] = {0};
    uint32_t stride0_[MAX_NDIM] = {0};
    uint32_t stride1_[MAX_NDIM] = {0};
};

template <typename DT_X>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    KernelLessEqual<DT_X> kernel;
    kernel.Init(x1, x2, y, workspace, tiling);
    kernel.Process();
}
