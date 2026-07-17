/**
 * @file less_equal_kernel.cpp
 * @brief LessEqual算子 - 核函数入口 + 验证用Host代码
 *
 * 核函数入口: 按dtype分发的 __global__ __aicore__ 函数
 * Host侧代码: CPU仿真调试用的 main 函数
 */

#include "less_equal_kernel.h"
#include "tikicpulib.h"
#include "data_utils.h"

// ============================================================================
// 核函数入口 (按dtype实例化)
// ============================================================================

// 使用宏简化多dtype核函数的注册
// KERNEL_TASK_TYPE_DEFAULT 声明当前核为矢量计算内核
#define DEFINE_LESS_EQUAL_KERNEL(kernel_name, dtype_t)                  \
    __global__ __aicore__ void kernel_name(                             \
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,                             \
        GM_ADDR workspace, GM_ADDR tiling)                              \
    {                                                                    \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                 \
        auto tilingData = *reinterpret_cast<LessEqualTilingData*>(tiling); \
        KernelLessEqual<dtype_t> op;                                    \
        op.Init(x1, x2, y, tilingData);                                 \
        op.Process();                                                   \
    }

DEFINE_LESS_EQUAL_KERNEL(less_equal_float16, half)
DEFINE_LESS_EQUAL_KERNEL(less_equal_float32, float)
DEFINE_LESS_EQUAL_KERNEL(less_equal_int32,   int32_t)
DEFINE_LESS_EQUAL_KERNEL(less_equal_int8,    int8_t)

// ============================================================================
// CPU侧验证用 Host Code (CPU仿真调试)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// ---- 根据dtype选择核函数 ----
static void LaunchLessEqualKernel(
    const LessEqualTilingData &tiling,
    uint8_t *x1Data, uint8_t *x2Data, uint8_t *yData)
{
    uint32_t numBlocks = tiling.blockDim;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    switch (tiling.dtype) {
    case 0: // float16
        ICPU_RUN_KF(less_equal_float16, numBlocks, x1Data, x2Data, yData,
                    nullptr, (uint8_t*)&tiling);
        break;
    case 1: // float32
        ICPU_RUN_KF(less_equal_float32, numBlocks, x1Data, x2Data, yData,
                    nullptr, (uint8_t*)&tiling);
        break;
    case 2: // int32
        ICPU_RUN_KF(less_equal_int32, numBlocks, x1Data, x2Data, yData,
                    nullptr, (uint8_t*)&tiling);
        break;
    case 3: // int8
        ICPU_RUN_KF(less_equal_int8, numBlocks, x1Data, x2Data, yData,
                    nullptr, (uint8_t*)&tiling);
        break;
    default:
        ERROR_LOG("Unsupported dtype: %u", tiling.dtype);
        break;
    }
}

// ---- 广播shape计算 ----
static std::vector<uint32_t> BroadcastShapes(
    const std::vector<uint32_t> &s1,
    const std::vector<uint32_t> &s2)
{
    size_t ndim = std::max(s1.size(), s2.size());
    std::vector<uint32_t> result(ndim, 1);
    for (size_t i = 0; i < ndim; ++i) {
        uint32_t d1 = (i < s1.size()) ? s1[s1.size() - 1 - i] : 1;
        uint32_t d2 = (i < s2.size()) ? s2[s2.size() - 1 - i] : 1;
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            ERROR_LOG("Incompatible shapes for broadcast: dim %zu: %u vs %u",
                      i, d1, d2);
        }
        result[ndim - 1 - i] = std::max(d1, d2);
    }
    return result;
}

// ---- 计算stride ----
static std::vector<uint32_t> ComputeStrides(const std::vector<uint32_t> &shape) {
    std::vector<uint32_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// ---- 将原始shape对齐到广播后的ndim ----
static std::vector<uint32_t> AlignShape(
    const std::vector<uint32_t> &shape, size_t ndim)
{
    std::vector<uint32_t> result(ndim, 1);
    size_t offset = ndim - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        result[offset + i] = shape[i];
    }
    return result;
}

// ---- 填充TilingData ----
static void FillTilingData(
    LessEqualTilingData &tiling,
    const std::vector<uint32_t> &x1Shape,
    const std::vector<uint32_t> &x2Shape,
    uint32_t dtype)
{
    auto outShape = BroadcastShapes(x1Shape, x2Shape);
    size_t ndim = outShape.size();

    // 计算总元素数
    uint32_t totalLength = 1;
    for (auto d : outShape) totalLength *= d;

    // 对齐shape到广播ndim
    auto x1Aligned = AlignShape(x1Shape, ndim);
    auto x2Aligned = AlignShape(x2Shape, ndim);

    auto outStrides = ComputeStrides(outShape);
    auto x1Strides  = ComputeStrides(x1Aligned);
    auto x2Strides  = ComputeStrides(x2Aligned);

    // x1Stride[i] = (x1在当前维度大小为1) ? 0 : x1Stride[i]
    // 即：广播维度stride为0
    // 实际上x1Stride是x1在自己shape空间内的stride
    // 在output中，x1的坐标和output坐标相同，但广播维度的offset贡献为0
    // 所以x1在output空间的stride = (x1Shape[d]==1) ? 0 : x1Strides[d]
    for (size_t d = 0; d < ndim; ++d) {
        uint32_t x1BroadStride = (x1Aligned[d] == 1) ? 0 : x1Strides[d];
        uint32_t x2BroadStride = (x2Aligned[d] == 1) ? 0 : x2Strides[d];
        // 但这样不对！stride需要反映在output空间中的步长，而非原始空间
        // 正确的做法：在output空间中，output坐标增加1沿维度d，
        // x1的偏移增加 x1Strides[d]（如果维度大小>1）或0（如果广播）
    }

    uint32_t blockDim = 8;  // 默认8核
    uint32_t blockLength = (totalLength + blockDim - 1) / blockDim;
    // 对齐到tile粒度的整数倍
    uint32_t tileGranularity = TILE_NUM * BUFFER_NUM;
    blockLength = ((blockLength + tileGranularity - 1) / tileGranularity) * tileGranularity;

    tiling.blockDim    = blockDim;
    tiling.totalLength = totalLength;
    tiling.blockLength = blockLength;
    tiling.tileNum     = TILE_NUM;
    tiling.dtype       = dtype;
    tiling.dimNum      = (x1Shape == x2Shape) ? 0 : (uint32_t)ndim;

    // 填充shape/stride数组
    if (tiling.dimNum > 0) {
        for (size_t d = 0; d < ndim && d < MAX_DIM_NUM; ++d) {
            tiling.outShape[d] = outShape[d];
            // stride在output坐标系中: 沿维度d移动1步, x1偏移多少
            tiling.x1Strides[d] = (x1Aligned[d] == 1) ? 0 : x1Strides[d];
            tiling.x2Strides[d] = (x2Aligned[d] == 1) ? 0 : x2Strides[d];
            tiling.x1Shape[d]   = x1Aligned[d];
            tiling.x2Shape[d]   = x2Aligned[d];
        }
    }
}

// ---- 生成测试数据 (CPU侧验证) ----
template<typename T>
static std::vector<T> GenTestData(size_t n, T minVal, T maxVal) {
    std::vector<T> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = minVal + (T)((double)(rand() % 10000) / 10000.0 * (maxVal - minVal));
    }
    return data;
}

static std::vector<uint8_t> ComputeGolden(
    const void *x1, const void *x2, size_t n, uint32_t dtype)
{
    std::vector<uint8_t> golden(n);
    switch (dtype) {
    case 0: { // float16
        auto *p1 = (const half*)x1, *p2 = (const half*)x2;
        for (size_t i = 0; i < n; ++i)
            golden[i] = ((float)p1[i] <= (float)p2[i]) ? 1 : 0;
        break;
    }
    case 1: { // float32
        auto *p1 = (const float*)x1, *p2 = (const float*)x2;
        for (size_t i = 0; i < n; ++i)
            golden[i] = (p1[i] <= p2[i]) ? 1 : 0;
        break;
    }
    case 2: { // int32
        auto *p1 = (const int32_t*)x1, *p2 = (const int32_t*)x2;
        for (size_t i = 0; i < n; ++i)
            golden[i] = (p1[i] <= p2[i]) ? 1 : 0;
        break;
    }
    case 3: { // int8
        auto *p1 = (const int8_t*)x1, *p2 = (const int8_t*)x2;
        for (size_t i = 0; i < n; ++i)
            golden[i] = (p1[i] <= p2[i]) ? 1 : 0;
        break;
    }
    }
    return golden;
}

// ---- 主函数 ----
int32_t main(int32_t argc, char *argv[])
{
    INFO_LOG("=== LessEqual Operator CPU Debug Test ===");

    // --- 测试配置 ---
    constexpr uint32_t dtype = 1;  // 0=f16, 1=f32, 2=i32, 3=i8
    constexpr size_t N = 8 * 2048; // 总元素数
    std::vector<uint32_t> x1Shape = {8, 2048};
    std::vector<uint32_t> x2Shape = {8, 2048};

    // --- 准备Tiling数据 ---
    LessEqualTilingData tiling;
    FillTilingData(tiling, x1Shape, x2Shape, dtype);
    INFO_LOG("totalLength=%u, blockLength=%u, blockDim=%u",
             tiling.totalLength, tiling.blockLength, tiling.blockDim);

    // --- 分配和初始化数据 ---
    constexpr size_t elemSize = sizeof(float); // for float32
    size_t totalBytes = N * elemSize;
    size_t outBytes   = N * sizeof(uint8_t);

    uint8_t *x1Data = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t *x2Data = (uint8_t*)AscendC::GmAlloc(totalBytes);
    uint8_t *yData  = (uint8_t*)AscendC::GmAlloc(outBytes);

    // 填充测试数据
    auto x1 = GenTestData<float>(N, -10.0f, 10.0f);
    auto x2 = GenTestData<float>(N, -10.0f, 10.0f);
    memcpy(x1Data, x1.data(), totalBytes);
    memcpy(x2Data, x2.data(), totalBytes);
    memset(yData, 0, outBytes);

    // --- 启动核函数 ---
    LaunchLessEqualKernel(tiling, x1Data, x2Data, yData);

    // --- 验证结果 ---
    auto golden = ComputeGolden(x1Data, x2Data, N, dtype);
    size_t errorCount = 0;
    for (size_t i = 0; i < N; ++i) {
        if (yData[i] != golden[i]) {
            if (errorCount < 20) {
                ERROR_LOG("Mismatch at [%zu]: got %u, expected %u",
                          i, yData[i], golden[i]);
            }
            ++errorCount;
        }
    }

    if (errorCount == 0) {
        INFO_LOG("[SUCCESS] All %zu elements pass!", N);
    } else {
        ERROR_LOG("[FAILED] %zu / %zu mismatches", errorCount, N);
    }

    // --- 清理 ---
    AscendC::GmFree(x1Data);
    AscendC::GmFree(x2Data);
    AscendC::GmFree(yData);

    return errorCount > 0 ? 1 : 0;
}
