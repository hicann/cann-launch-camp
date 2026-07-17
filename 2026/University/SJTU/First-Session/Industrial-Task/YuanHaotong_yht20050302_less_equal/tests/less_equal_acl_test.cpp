#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "aclnn_less_equal.h"

#define CHECK_ACL(expr)                                                                            \
    do {                                                                                           \
        aclError _ret = (expr);                                                                    \
        if (_ret != ACL_SUCCESS) {                                                                 \
            std::cerr << "ACL call failed: " #expr << ", ret=" << static_cast<int>(_ret)           \
                      << ", line=" << __LINE__ << std::endl;                                       \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

#define CHECK_ACLNN(expr)                                                                          \
    do {                                                                                           \
        aclnnStatus _ret = (expr);                                                                 \
        if (_ret != 0) {                                                                           \
            std::cerr << "ACLNN call failed: " #expr << ", ret=" << static_cast<int>(_ret)         \
                      << ", line=" << __LINE__ << std::endl;                                       \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

namespace {
int g_failedCases = 0;

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kTileAlign = 32;
constexpr uint32_t kMinTileLength = 32;
constexpr uint32_t kSmallBlockTileLength = 256;
constexpr uint32_t kSmallBlockThreshold = 128;
constexpr uint32_t kInt32SmallBlockThreshold = 216;
constexpr uint32_t kSmallTotalBlockDim = 1;
constexpr uint64_t kFullCoreTotalThreshold = 4096;
constexpr uint64_t kFloat32FullCoreMinTotal = 4097;
constexpr uint64_t kTargetBlockLength = 1024;
constexpr uint32_t kDefaultAivCores = 25;
constexpr uint64_t kDefaultUbBytes = 192 * 1024;

struct Float16Bits {
    uint16_t value;
};

bool DebugEnabled()
{
    const char *env = std::getenv("LESS_EQUAL_DEBUG");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

uint64_t ReadEnvUint64(const char *name, uint64_t defaultValue)
{
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const uint64_t value = std::strtoull(env, &end, 10);
    return (end == env || *end != '\0') ? defaultValue : value;
}

uint16_t FloatToHalfBits(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    uint32_t mant = x & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant = (mant | 0x800000U) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000U) >> 13));
}

float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000U)) << 16;
    uint32_t exp = (h >> 10) & 0x1fU;
    uint32_t mant = h & 0x03ffU;
    uint32_t out = 0;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400U) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffU;
            out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f = 0.0f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

template <typename T>
aclDataType AclDtype();

template <>
aclDataType AclDtype<float>()
{
    return ACL_FLOAT;
}

template <>
aclDataType AclDtype<int32_t>()
{
    return ACL_INT32;
}

template <>
aclDataType AclDtype<int8_t>()
{
    return ACL_INT8;
}

template <>
aclDataType AclDtype<Float16Bits>()
{
    return ACL_FLOAT16;
}

template <typename T>
bool LessEqualHost(T a, T b)
{
    return a <= b;
}

template <>
bool LessEqualHost(Float16Bits a, Float16Bits b)
{
    return HalfBitsToFloat(a.value) <= HalfBitsToFloat(b.value);
}

template <typename T>
void PrintValue(T value)
{
    if constexpr (std::is_same_v<T, int8_t>) {
        std::cout << static_cast<int>(value);
    } else {
        std::cout << value;
    }
}

template <>
void PrintValue(Float16Bits value)
{
    std::cout << HalfBitsToFloat(value.value) << "(0x" << std::hex << value.value << std::dec << ")";
}

template <typename T>
std::string TypeName();

template <>
std::string TypeName<float>()
{
    return "float";
}

template <>
std::string TypeName<int32_t>()
{
    return "int32";
}

template <>
std::string TypeName<int8_t>()
{
    return "int8";
}

template <>
std::string TypeName<Float16Bits>()
{
    return "float16";
}

template <typename T>
uint32_t TypeSize()
{
    return sizeof(T);
}

uint32_t AlignUp32(uint32_t bytes)
{
    return (bytes + 31U) & ~31U;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return value / align * align;
}

template <typename T>
uint64_t GetUbBytesForTile(uint32_t tileLength)
{
    const uint32_t inputTypeSize = TypeSize<T>();
    uint64_t bytes = 0;
    bytes += static_cast<uint64_t>(kBufferNum) * AlignUp32(tileLength * inputTypeSize);
    bytes += static_cast<uint64_t>(kBufferNum) * AlignUp32(tileLength * inputTypeSize);
    bytes += static_cast<uint64_t>(kBufferNum) * AlignUp32(tileLength * sizeof(uint8_t));

    if constexpr (std::is_same_v<T, Float16Bits>) {
        bytes += AlignUp32(tileLength * sizeof(Float16Bits));
    } else if constexpr (std::is_same_v<T, float>) {
        bytes += AlignUp32(tileLength * sizeof(float));
        bytes += AlignUp32(tileLength * sizeof(Float16Bits));
    } else if constexpr (std::is_same_v<T, int8_t>) {
        bytes += 3ULL * AlignUp32(tileLength * sizeof(Float16Bits));
    } else {
        bytes += AlignUp32(tileLength * sizeof(int32_t));
        bytes += AlignUp32(tileLength * sizeof(Float16Bits));
        bytes += AlignUp32(tileLength * sizeof(float));
    }
    return bytes;
}

template <typename T>
uint32_t CalcTileLength(uint64_t ubSize)
{
    const uint32_t inputTypeSize = TypeSize<T>();
    uint32_t low = kMinTileLength;
    uint32_t high = AlignDown(static_cast<uint32_t>(std::max<uint64_t>(ubSize / inputTypeSize, low)), kTileAlign);

    if (high < low || GetUbBytesForTile<T>(low) > ubSize) {
        return low;
    }

    uint32_t best = low;
    while (low <= high) {
        const uint32_t mid = AlignDown(low + (high - low) / 2, kTileAlign);
        if (mid < kMinTileLength) {
            break;
        }
        if (GetUbBytesForTile<T>(mid) <= ubSize) {
            best = mid;
            low = mid + kTileAlign;
        } else {
            high = mid - kTileAlign;
        }
    }
    return best;
}

template <typename T>
uint32_t SelectBlockDim(uint64_t total, uint32_t aivCores)
{
    if (total == 0) {
        return 1;
    }
    const uint32_t availableCores = (aivCores == 0) ? 1 : aivCores;
    if constexpr (std::is_same_v<T, Float16Bits>) {
        return static_cast<uint32_t>(std::min<uint64_t>(kSmallTotalBlockDim, total));
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
    }
    if constexpr (std::is_same_v<T, float>) {
        if (total >= kFloat32FullCoreMinTotal) {
            return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
        }
    }
    if (total <= 1024) {
        return static_cast<uint32_t>(std::min<uint64_t>(kSmallTotalBlockDim, total));
    }
    if (total <= kFullCoreTotalThreshold) {
        return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, total)));
    }
    const uint64_t neededCores = (total + kTargetBlockLength - 1) / kTargetBlockLength;
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(availableCores, neededCores)));
}

template <typename T>
uint32_t SelectTileLength(uint64_t ubBytes, uint64_t maxBlockLength)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        if (maxBlockLength <= kInt32SmallBlockThreshold) {
            return kSmallBlockTileLength;
        }
    }
    if (maxBlockLength <= kSmallBlockThreshold) {
        return kSmallBlockTileLength;
    }
    return CalcTileLength<T>(ubBytes);
}

std::string ShapeString(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return "[]";
    }
    std::string text = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            text += ",";
        }
        text += std::to_string(shape[i]);
    }
    text += "]";
    return text;
}

uint64_t Numel(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), uint64_t{1}, [](uint64_t a, int64_t b) {
        return a * static_cast<uint64_t>(b);
    });
}

std::vector<int64_t> Strides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

std::vector<int64_t> OutputShape(const std::vector<int64_t> &x1Shape, const std::vector<int64_t> &x2Shape)
{
    if (x1Shape != x2Shape) {
        throw std::runtime_error("LessEqual test expects identical input shapes");
    }
    return x1Shape;
}

template <typename T>
std::vector<T> MakeInput(uint64_t size, int seed)
{
    std::vector<T> data(size);
    for (uint64_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, float>) {
            data[i] = static_cast<float>((static_cast<int64_t>(i * 17 + seed) % 23) - 11) / 3.0f;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            data[i] = static_cast<int32_t>((static_cast<int64_t>(i * 13 + seed) % 97) - 48);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            data[i] = static_cast<int8_t>((static_cast<int64_t>(i * 7 + seed) % 61) - 30);
        } else {
            const float f = static_cast<float>((static_cast<int64_t>(i * 11 + seed) % 31) - 15) / 4.0f;
            data[i].value = FloatToHalfBits(f);
        }
    }
    return data;
}

template <typename T>
std::vector<uint8_t> Reference(const std::vector<T> &x1, const std::vector<T> &x2,
    const std::vector<int64_t> &x1Shape, const std::vector<int64_t> &x2Shape)
{
    auto outShape = OutputShape(x1Shape, x2Shape);
    std::vector<uint8_t> y(Numel(outShape), 0);
    for (uint64_t i = 0; i < y.size(); ++i) {
        y[i] = LessEqualHost(x1[i], x2[i]) ? 1U : 0U;
    }
    return y;
}

aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *data)
{
    std::vector<int64_t> storageShape = shape.empty() ? std::vector<int64_t>{1} : shape;
    std::vector<int64_t> viewShape = shape.empty() ? std::vector<int64_t>{} : shape;
    std::vector<int64_t> stride = Strides(storageShape);
    return aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stride.data(), 0, ACL_FORMAT_ND,
        storageShape.data(), storageShape.size(), data);
}

template <typename T>
void PrintEstimatedTiling(const std::string &name, const std::vector<int64_t> &shape)
{
    const uint64_t total = Numel(shape);
    const uint32_t aivCores = static_cast<uint32_t>(ReadEnvUint64("LESS_EQUAL_AIV_CORES", kDefaultAivCores));
    const uint64_t ubBytes = ReadEnvUint64("LESS_EQUAL_UB_BYTES", kDefaultUbBytes);
    const uint32_t blockDim = SelectBlockDim<T>(total, aivCores);
    const uint64_t baseBlockLength = total / blockDim;
    const uint64_t tailBlockNum = total % blockDim;
    const uint64_t maxBlockLength = baseBlockLength + (tailBlockNum > 0 ? 1 : 0);
    const uint32_t dynamicTileLength = CalcTileLength<T>(ubBytes);
    const uint32_t tileLength = SelectTileLength<T>(ubBytes, maxBlockLength);
    const uint32_t tileNum = static_cast<uint32_t>((maxBlockLength + tileLength - 1) / tileLength);

    std::cout << "[TILING] " << name << " dtype=" << TypeName<T>() << " shape=" << ShapeString(shape)
              << " total=" << total << " aivCores=" << aivCores << " ubBytes=" << ubBytes
              << " blockDim=" << blockDim << " baseBlockLength=" << baseBlockLength
              << " tailBlockNum=" << tailBlockNum << " maxBlockLength=" << maxBlockLength
              << " tileLength=" << tileLength << " dynamicTileLength=" << dynamicTileLength
              << " tileNumPerMaxBlock=" << tileNum << std::endl;
}

template <typename T>
void RunCase(const std::string &name, const std::vector<int64_t> &x1Shape, const std::vector<int64_t> &x2Shape)
{
    auto outShape = OutputShape(x1Shape, x2Shape);
    PrintEstimatedTiling<T>(name, outShape);
    auto x1Host = MakeInput<T>(Numel(x1Shape), 3);
    auto x2Host = MakeInput<T>(Numel(x2Shape), 9);
    auto yExpect = Reference(x1Host, x2Host, x1Shape, x2Shape);
    std::vector<uint8_t> yGot(yExpect.size(), 0);

    void *x1Dev = nullptr;
    void *x2Dev = nullptr;
    void *yDev = nullptr;
    void *workspace = nullptr;
    const size_t x1Bytes = x1Host.size() * sizeof(T);
    const size_t x2Bytes = x2Host.size() * sizeof(T);
    const size_t yBytes = yGot.size() * sizeof(uint8_t);
    CHECK_ACL(aclrtMalloc(&x1Dev, x1Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&x2Dev, x2Bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&yDev, yBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(x1Dev, x1Bytes, x1Host.data(), x1Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(x2Dev, x2Bytes, x2Host.data(), x2Bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemset(yDev, yBytes, DebugEnabled() ? 0xA5 : 0, yBytes));

    aclTensor *x1Tensor = CreateTensor(x1Shape, AclDtype<T>(), x1Dev);
    aclTensor *x2Tensor = CreateTensor(x2Shape, AclDtype<T>(), x2Dev);
    aclTensor *yTensor = CreateTensor(outShape, ACL_BOOL, yDev);
    if (x1Tensor == nullptr || x2Tensor == nullptr || yTensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACLNN(aclnnLessEqualGetWorkspaceSize(x1Tensor, x2Tensor, yTensor, &workspaceSize, &executor));
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACLNN(aclnnLessEqual(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(yGot.data(), yBytes, yDev, yBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    size_t mismatch = 0;
    size_t first = 0;
    for (size_t i = 0; i < yGot.size(); ++i) {
        if ((yGot[i] != 0) != (yExpect[i] != 0)) {
            if (mismatch == 0) {
                first = i;
            }
            ++mismatch;
        }
    }
    std::cout << "[" << (mismatch == 0 ? "PASS" : "FAIL") << "] " << name << " dtype=" << TypeName<T>()
              << " elements=" << yGot.size() << " mismatch=" << mismatch;
    if (mismatch != 0) {
        std::cout << " first=" << first << " got=" << static_cast<int>(yGot[first])
                  << " expect=" << static_cast<int>(yExpect[first]);
    }
    std::cout << std::endl;
    if (DebugEnabled() && mismatch != 0) {
        std::cout << "  debug: first_index=" << first << " x1=";
        PrintValue(x1Host[first]);
        std::cout << " x2=";
        PrintValue(x2Host[first]);
        std::cout << " host_le=" << static_cast<int>(LessEqualHost(x1Host[first], x2Host[first]))
                  << " raw_y=" << static_cast<int>(yGot[first]) << std::endl;
        std::cout << "  debug: first outputs";
        const size_t limit = std::min<size_t>(yGot.size(), 16);
        for (size_t i = 0; i < limit; ++i) {
            std::cout << " " << static_cast<int>(yGot[i]);
        }
        std::cout << std::endl;
    }
    if (mismatch != 0) {
        ++g_failedCases;
    }

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACLNN(aclDestroyTensor(x1Tensor));
    CHECK_ACLNN(aclDestroyTensor(x2Tensor));
    CHECK_ACLNN(aclDestroyTensor(yTensor));
    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACL(aclrtFree(x1Dev));
    CHECK_ACL(aclrtFree(x2Dev));
    CHECK_ACL(aclrtFree(yDev));
}

template <typename T>
void RunAllForType()
{
    RunCase<T>("same-shape-small", {8}, {8});
    RunCase<T>("same-shape-one-tile", {256}, {256});
    RunCase<T>("same-shape-tail-tile", {263}, {263});
    RunCase<T>("same-shape-multicore", {8192}, {8192});
    RunCase<T>("same-shape-2d", {4, 7}, {4, 7});
    RunCase<T>("same-shape-4d", {2, 3, 5, 7}, {2, 3, 5, 7});
}
}  // namespace

int main(int argc, char **argv)
{
    int deviceId = 0;
    if (argc > 1) {
        deviceId = std::atoi(argv[1]);
    }

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));

    RunAllForType<Float16Bits>();
    RunAllForType<float>();
    RunAllForType<int32_t>();
    RunAllForType<int8_t>();

    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (g_failedCases != 0) {
        std::cerr << "Total failed cases: " << g_failedCases << std::endl;
        return 2;
    }
    return 0;
}
