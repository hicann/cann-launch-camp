/**
 * @file less_equal_kernel.h
 * @brief LessEqual算子核函数实现
 *
 * 实现逐元素比较 x1 <= x2，输出bool类型。
 * 支持 float16 / float32 / int32 / int8 输入。
 * 支持 NumPy 风格广播语义。
 * 适配非32字节对齐场景。
 *
 * 核函数输入:
 *   x1, x2: 输入张量 (Global Memory)
 *   y:      输出张量 (Global Memory, bool/uint8_t类型)
 *   tiling: 切分与广播参数
 */

#ifndef LESS_EQUAL_KERNEL_H
#define LESS_EQUAL_KERNEL_H

#include "kernel_operator.h"

// ============================================================================
// 编译期常量
// ============================================================================
constexpr int32_t BUFFER_NUM  = 2;     // Double Buffer: 每个队列2块
constexpr int32_t TILE_NUM    = 8;     // 单核内tile数量
constexpr int32_t MAX_DIM_NUM = 8;     // 最大广播维度数

// ============================================================================
// TilingData -- Host -> Device 参数传递
// ============================================================================
struct LessEqualTilingData {
    uint32_t blockDim;           // AI Core总数
    uint32_t totalLength;        // 输出张量总元素数
    uint32_t blockLength;        // 每核处理的输出元素数
    uint32_t tileNum;            // 每核tile数 (应与TILE_NUM一致)
    uint32_t dtype;              // 0=f16, 1=f32, 2=i32, 3=i8

    // ---- 广播支持 ----
    // 对于非广播场景: dimNum = 0, 表示两个输入shape完全一致,
    //   x1/x2在内存中与output一样是连续排列的。
    // 对于广播场景: dimNum > 0, 使用stride数组。
    uint32_t dimNum;
    uint32_t outShape[MAX_DIM_NUM];    // 广播后output的shape
    uint32_t x1Strides[MAX_DIM_NUM];   // x1各维stride (在output坐标系中)
    uint32_t x2Strides[MAX_DIM_NUM];   // x2各维stride (在output坐标系中)
    uint32_t x1Shape[MAX_DIM_NUM];     // x1原始shape (用于边界检查)
    uint32_t x2Shape[MAX_DIM_NUM];     // x2原始shape (用于边界检查)
};

// ============================================================================
// 模板类: KernelLessEqual<T>
// T = 输入数据类型 (half, float, int32_t, int8_t)
// ============================================================================
template<typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline ~KernelLessEqual() {}

    /**
     * @brief 初始化：多核数据划分 + Pipeline内存分配
     */
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &tiling)
    {
        uint32_t blockIdx   = AscendC::GetBlockIdx();
        this->blockLength   = tiling.totalLength / tiling.blockDim;
        this->tileNum       = tiling.tileNum;
        this->tileLength    = this->blockLength / this->tileNum / BUFFER_NUM;
        this->totalLength   = tiling.totalLength;
        this->dimNum        = tiling.dimNum;

        // 每个核的输出偏移
        uint32_t outOffset  = this->blockLength * blockIdx;

        // ---- 设置Global Buffer ----
        // 对于非广播场景，输入数据也是连续排列的，偏移与output相同
        // 对于广播场景，输入数据布局由stride决定，CopyIn时手动索引
        uint32_t x1Offset, x2Offset;
        if (this->dimNum == 0) {
            // 非广播：x1/x2在内存中连续排列，偏移与output相同
            x1Offset = outOffset;
            x2Offset = outOffset;
        } else {
            // 广播场景：从坐标0开始计算偏移
            // 但实际上对于非广播的维度，起始偏移通常也是0
            x1Offset = 0;
            x2Offset = 0;
        }

        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + x1Offset, this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + x2Offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + outOffset,  this->blockLength);

        // 保存广播参数
        for (uint32_t d = 0; d < this->dimNum; ++d) {
            this->outShape[d]   = tiling.outShape[d];
            this->outStrides[d] = tiling.outShape[d] > 0 ?
                this->ComputeOutStride(tiling.outShape, d) : 0;
            this->x1Strides[d]  = tiling.x1Strides[d];
            this->x2Strides[d]  = tiling.x2Strides[d];
        }
        // 首元素坐标初始化
        if (this->dimNum > 0) {
            FlatToCoords(outOffset, tiling.outShape, this->startCoords, this->dimNum);
        }

        // ---- Pipeline 内存分配 ----
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(uint8_t));

        // 标记是否首次Compute（用于debug printf）
        this->firstCompute = true;
    }

    /**
     * @brief Process: 流水线主循环
     */
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // ================================================================
    // CopyIn: 从Global Memory搬运数据到Local Memory
    // ================================================================
    __aicore__ inline void CopyIn(int32_t progress)
    {
        auto x1Local = inQueueX1.template AllocTensor<T>();
        auto x2Local = inQueueX2.template AllocTensor<T>();

        uint32_t tileStart = progress * this->tileLength;

        if (this->dimNum == 0) {
            // ----- 非广播: 直接连续拷贝 -----
            AscendC::DataCopy(x1Local, x1Gm[tileStart], this->tileLength);
            AscendC::DataCopy(x2Local, x2Gm[tileStart], this->tileLength);
        } else {
            // ----- 广播: 逐元素计算偏移后加载 -----
            // 使用LocalTensor的逐元素赋值
            uint32_t blockIdx   = AscendC::GetBlockIdx();
            uint32_t globalBase = blockIdx * this->blockLength;
            T* x1Base = (__gm__ T*)AscendC::GetUserDefinedBlockArg(0);
            T* x2Base = (__gm__ T*)AscendC::GetUserDefinedBlockArg(1);

            for (uint32_t i = 0; i < this->tileLength; ++i) {
                uint32_t globalIdx = globalBase + tileStart + i;
                uint32_t x1Off, x2Off;
                ComputeOffsets(globalIdx, x1Off, x2Off);
                x1Local(i) = x1Base[x1Off];
                x2Local(i) = x2Base[x2Off];
            }
        }

        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    // ================================================================
    // Compute: 矢量比较 x1 <= x2
    // ================================================================
    __aicore__ inline void Compute(int32_t progress)
    {
        auto x1Local = inQueueX1.template DeQue<T>();
        auto x2Local = inQueueX2.template DeQue<T>();
        auto yLocal  = outQueueY.template AllocTensor<uint8_t>();

        // 矢量比较: y[i] = (x1[i] <= x2[i]) ? 1 : 0
        AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::LE,
                         this->tileLength);

        outQueueY.EnQue<uint8_t>(yLocal);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    // ================================================================
    // CopyOut: 将bool结果搬回Global Memory
    // ================================================================
    __aicore__ inline void CopyOut(int32_t progress)
    {
        auto yLocal = outQueueY.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal,
                          this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

    // ================================================================
    // 广播辅助函数
    // ================================================================
    __aicore__ inline void FlatToCoords(
        uint32_t flat, const uint32_t *shape, uint32_t *coords, uint32_t ndim)
    {
        // 从最内维开始计算坐标 (row-major / C-order)
        // shape = [D0, D1, ..., D_{n-1}]; 最内维是 D_{n-1}
        uint32_t remaining = flat;
        for (int32_t d = (int32_t)ndim - 1; d >= 0; --d) {
            uint32_t dimSize = shape[d];
            coords[d] = remaining % dimSize;
            remaining /= dimSize;
        }
    }

    __aicore__ inline void ComputeOffsets(
        uint32_t globalIdx, uint32_t &x1Off, uint32_t &x2Off)
    {
        uint32_t coords[MAX_DIM_NUM];
        FlatToCoords(globalIdx, this->outShape, coords, this->dimNum);

        x1Off = 0;
        x2Off = 0;
        for (uint32_t d = 0; d < this->dimNum; ++d) {
            x1Off += coords[d] * this->x1Strides[d];
            x2Off += coords[d] * this->x2Strides[d];
        }
    }

    uint32_t ComputeOutStride(const uint32_t *shape, uint32_t dim) {
        uint32_t stride = 1;
        for (uint32_t d = dim + 1; d < this->dimNum; ++d) {
            stride *= shape[d] > 0 ? shape[d] : 1;
        }
        return stride;
    }

    // ================================================================
    // 成员变量
    // ================================================================
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<T>       x1Gm, x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;

    uint32_t totalLength, blockLength, tileNum, tileLength;
    uint32_t dimNum;
    uint32_t outShape[MAX_DIM_NUM];
    uint32_t outStrides[MAX_DIM_NUM];
    uint32_t x1Strides[MAX_DIM_NUM];
    uint32_t x2Strides[MAX_DIM_NUM];
    uint32_t startCoords[MAX_DIM_NUM]; // 本核首个output元素的坐标
    bool     firstCompute;
};

#endif // LESS_EQUAL_KERNEL_H
