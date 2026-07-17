/**
 * @file less_equal_test.cpp
 * @brief LessEqual算子 - CPU仿真调试
 *
 * 支持: float16 / float32 / int32 / int8
 * 编译: cd build && cmake .. -Dsoc_version=Ascend910B4 && make -j
 * 运行: ./less_equal_test
 */

#include "kernel_operator.h"
#include "data_utils.h"
#include "tikicpulib.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <type_traits>

// ============================================================================
// 编译期常量
// ============================================================================
constexpr int32_t USE_CORE_NUM = 8;          // 模拟AI Core数量
constexpr int32_t BUFFER_NUM   = 2;          // Double Buffer
constexpr int32_t TILE_NUM     = 8;          // 单核tile数
constexpr int32_t TOTAL_LENGTH = 8 * 2048;   // 数据总元素数
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_LENGTH  = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

// ============================================================================
// KernelLessEqual<T>: Vector编程范式
// ============================================================================
template<typename T>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}
    __aicore__ inline ~KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->offset = BLOCK_LENGTH * blockIdx;

        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + this->offset, BLOCK_LENGTH);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + this->offset, BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ uint8_t*)y + this->offset, BLOCK_LENGTH);

        pipe.InitBuffer(inQ1, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(inQ2, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe.InitBuffer(outQ, BUFFER_NUM, TILE_LENGTH * sizeof(uint8_t));
    }

    __aicore__ inline void Process() {
        constexpr int32_t loop = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loop; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t p) {
        auto x1L = inQ1.template AllocTensor<T>();
        auto x2L = inQ2.template AllocTensor<T>();
        AscendC::DataCopy(x1L, x1Gm[p * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(x2L, x2Gm[p * TILE_LENGTH], TILE_LENGTH);
        inQ1.EnQue(x1L);
        inQ2.EnQue(x2L);
    }

    __aicore__ inline void Compute(int32_t p) {
        auto x1L = inQ1.template DeQue<T>();
        auto x2L = inQ2.template DeQue<T>();
        auto yL  = outQ.template AllocTensor<uint8_t>();

        // 逐元素标量比较，适用于CPU仿真调试
        // (NPU上 float16/float32 可改用 AscendC::Compare + Select 展开比特压缩结果)
        for (int32_t i = 0; i < TILE_LENGTH; ++i) {
            yL(i) = (x1L(i) <= x2L(i)) ? 1 : 0;
        }

        outQ.EnQue<uint8_t>(yL);
        inQ1.FreeTensor(x1L);
        inQ2.FreeTensor(x2L);
    }

    __aicore__ inline void CopyOut(int32_t p) {
        auto yL = outQ.DeQue<uint8_t>();
        AscendC::DataCopy(yGm[p * TILE_LENGTH], yL, TILE_LENGTH);
        outQ.FreeTensor(yL);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  BUFFER_NUM> inQ1, inQ2;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQ;
    AscendC::GlobalTensor<T>       x1Gm, x2Gm;
    AscendC::GlobalTensor<uint8_t> yGm;
    uint32_t offset;
};

// ============================================================================
// 核函数入口 (4种dtype)
// ============================================================================
__global__ __aicore__ void less_equal_f16(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelLessEqual<half> op; op.Init(x1, x2, y); op.Process();
}
__global__ __aicore__ void less_equal_f32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelLessEqual<float> op; op.Init(x1, x2, y); op.Process();
}
__global__ __aicore__ void less_equal_i32(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelLessEqual<int32_t> op; op.Init(x1, x2, y); op.Process();
}
__global__ __aicore__ void less_equal_i8(GM_ADDR x1, GM_ADDR x2, GM_ADDR y) {
    KernelLessEqual<int8_t> op; op.Init(x1, x2, y); op.Process();
}

// ============================================================================
// 测试辅助
// ============================================================================

template<typename T>
static std::vector<T> MakeData(size_t n, double lo, double hi) {
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        double val = lo + (double)(rand() % 10000) / 10000.0 * (hi - lo);
        v[i] = T(val);
    }
    return v;
}

// int8 特化 (范围: -100 到 100)
template<>
std::vector<int8_t> MakeData<int8_t>(size_t n, double lo, double hi) {
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = int8_t(lo + (double)(rand() % 10000) / 10000.0 * (hi - lo));
    }
    return v;
}

template<typename T>
static std::vector<uint8_t> ComputeGolden(const T *x1, const T *x2, size_t n) {
    std::vector<uint8_t> g(n);
    for (size_t i = 0; i < n; ++i) g[i] = (x1[i] <= x2[i]) ? 1 : 0;
    return g;
}

template<typename T>
static int RunDtypeTest(const char *name, double lo, double hi, size_t elemSize) {
    INFO_LOG("--- Test: %s ---", name);

    size_t inBytes  = TOTAL_LENGTH * elemSize;
    size_t outBytes = TOTAL_LENGTH * sizeof(uint8_t);

    auto x1Host = MakeData<T>(TOTAL_LENGTH, lo, hi);
    auto x2Host = MakeData<T>(TOTAL_LENGTH, lo, hi);

    uint8_t *x1Dev = (uint8_t*)AscendC::GmAlloc(inBytes);
    uint8_t *x2Dev = (uint8_t*)AscendC::GmAlloc(inBytes);
    uint8_t *yDev  = (uint8_t*)AscendC::GmAlloc(outBytes);
    memcpy(x1Dev, x1Host.data(), inBytes);
    memcpy(x2Dev, x2Host.data(), inBytes);
    memset(yDev, 0, outBytes);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    if constexpr (std::is_same_v<T, half>)
        ICPU_RUN_KF(less_equal_f16, USE_CORE_NUM, x1Dev, x2Dev, yDev);
    else if constexpr (std::is_same_v<T, float>)
        ICPU_RUN_KF(less_equal_f32, USE_CORE_NUM, x1Dev, x2Dev, yDev);
    else if constexpr (std::is_same_v<T, int32_t>)
        ICPU_RUN_KF(less_equal_i32, USE_CORE_NUM, x1Dev, x2Dev, yDev);
    else if constexpr (std::is_same_v<T, int8_t>)
        ICPU_RUN_KF(less_equal_i8, USE_CORE_NUM, x1Dev, x2Dev, yDev);

    auto golden = ComputeGolden((T*)x1Dev, (T*)x2Dev, TOTAL_LENGTH);
    uint8_t *y = yDev;
    size_t errors = 0;
    for (size_t i = 0; i < TOTAL_LENGTH; ++i) {
        if (y[i] != golden[i]) {
            if (errors < 10)
                printf("  [%zu] got=%u exp=%u  x1=%g x2=%g\n",
                       i, y[i], golden[i], double(x1Host[i]), double(x2Host[i]));
            ++errors;
        }
    }

    AscendC::GmFree(x1Dev); AscendC::GmFree(x2Dev); AscendC::GmFree(yDev);

    if (errors == 0)
        INFO_LOG("[PASS] %s: all %d correct", name, TOTAL_LENGTH);
    else
        ERROR_LOG("[FAIL] %s: %zu / %d errors", name, errors, TOTAL_LENGTH);
    return errors > 0 ? 1 : 0;
}

// ============================================================================
// main
// ============================================================================
int32_t main(int32_t argc, char *argv[]) {
    srand(42);
    int failed = 0;

    INFO_LOG("==============================================");
    INFO_LOG("  LessEqual CPU Debug Tests");
    INFO_LOG("  Cores=%d  Total=%d  Block=%d  Tile=%d",
             USE_CORE_NUM, TOTAL_LENGTH, BLOCK_LENGTH, TILE_LENGTH);
    INFO_LOG("==============================================");

    failed += RunDtypeTest<float>("float32", -100.0, 100.0, sizeof(float));
    failed += RunDtypeTest<half>("float16", -50.0, 50.0, sizeof(half));
    failed += RunDtypeTest<int32_t>("int32", -1000.0, 1000.0, sizeof(int32_t));
    failed += RunDtypeTest<int8_t>("int8", -100.0, 100.0, sizeof(int8_t));

    INFO_LOG("==============================================");
    if (failed == 0) INFO_LOG("  ALL TESTS PASSED!");
    else ERROR_LOG("  %d TEST(S) FAILED!", failed);
    INFO_LOG("==============================================");
    return failed;
}
