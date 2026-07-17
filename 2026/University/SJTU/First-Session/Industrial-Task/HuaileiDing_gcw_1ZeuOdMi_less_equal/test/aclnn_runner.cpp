#include "acl/acl.h"
#include "aclnn_less_equal.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
bool Check(aclError status, const char *operation)
{
    if (status == ACL_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed, error code: " << status << std::endl;
    return false;
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    return strides;
}

aclTensor *CreateTensor(const std::vector<int64_t> &shape, aclDataType dtype, void *deviceData)
{
    const std::vector<int64_t> strides = ContiguousStrides(shape);
    return aclCreateTensor(shape.data(), shape.size(), dtype,
        strides.data(), 0, ACL_FORMAT_ND, shape.data(), shape.size(), deviceData);
}

bool RunFloat32Case(const std::string &name,
    const std::vector<float> &x1Host, const std::vector<int64_t> &x1Shape,
    const std::vector<float> &x2Host, const std::vector<int64_t> &x2Shape,
    const std::vector<uint8_t> &expected, const std::vector<int64_t> &outputShape,
    aclrtStream stream)
{
    void *x1Device = nullptr;
    void *x2Device = nullptr;
    void *outputDevice = nullptr;
    void *workspaceDevice = nullptr;
    aclTensor *x1Tensor = nullptr;
    aclTensor *x2Tensor = nullptr;
    aclTensor *outputTensor = nullptr;
    bool success = false;

    const size_t x1Bytes = x1Host.size() * sizeof(float);
    const size_t x2Bytes = x2Host.size() * sizeof(float);
    const size_t outputBytes = expected.size() * sizeof(uint8_t);

    if (!Check(aclrtMalloc(&x1Device, x1Bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(x1)") ||
        !Check(aclrtMalloc(&x2Device, x2Bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(x2)") ||
        !Check(aclrtMalloc(&outputDevice, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(output)")) {
        goto cleanup;
    }
    if (!Check(aclrtMemcpy(x1Device, x1Bytes, x1Host.data(), x1Bytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(x1)") ||
        !Check(aclrtMemcpy(x2Device, x2Bytes, x2Host.data(), x2Bytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy(x2)")) {
        goto cleanup;
    }

    x1Tensor = CreateTensor(x1Shape, ACL_FLOAT, x1Device);
    x2Tensor = CreateTensor(x2Shape, ACL_FLOAT, x2Device);
    outputTensor = CreateTensor(outputShape, ACL_BOOL, outputDevice);
    if (x1Tensor == nullptr || x2Tensor == nullptr || outputTensor == nullptr) {
        std::cerr << "aclCreateTensor failed" << std::endl;
        goto cleanup;
    }

    {
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        if (!Check(aclnnLessEqualGetWorkspaceSize(
                x1Tensor, x2Tensor, outputTensor, &workspaceSize, &executor),
                "aclnnLessEqualGetWorkspaceSize")) {
            goto cleanup;
        }
        if (workspaceSize > 0 &&
            !Check(aclrtMalloc(&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(workspace)")) {
            goto cleanup;
        }
        if (!Check(aclnnLessEqual(workspaceDevice, workspaceSize, executor, stream),
                "aclnnLessEqual") ||
            !Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
            goto cleanup;
        }
    }

    {
        std::vector<uint8_t> actual(expected.size(), 0);
        if (!Check(aclrtMemcpy(actual.data(), outputBytes, outputDevice, outputBytes,
                ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy(output)")) {
            goto cleanup;
        }
        if (actual != expected) {
            std::cerr << "FAIL " << name << ": expected";
            for (uint8_t value : expected) {
                std::cerr << ' ' << static_cast<int>(value);
            }
            std::cerr << ", actual";
            for (uint8_t value : actual) {
                std::cerr << ' ' << static_cast<int>(value);
            }
            std::cerr << std::endl;
            goto cleanup;
        }
    }

    std::cout << "PASS " << name << std::endl;
    success = true;

cleanup:
    if (workspaceDevice != nullptr) aclrtFree(workspaceDevice);
    if (outputTensor != nullptr) aclDestroyTensor(outputTensor);
    if (x2Tensor != nullptr) aclDestroyTensor(x2Tensor);
    if (x1Tensor != nullptr) aclDestroyTensor(x1Tensor);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (x2Device != nullptr) aclrtFree(x2Device);
    if (x1Device != nullptr) aclrtFree(x1Device);
    return success;
}
}  // namespace

int main(int argc, char **argv)
{
    const int32_t deviceId = argc > 1 ? std::atoi(argv[1]) : 0;
    if (!Check(aclInit(nullptr), "aclInit") ||
        !Check(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        return 1;
    }

    aclrtStream stream = nullptr;
    if (!Check(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    bool success = RunFloat32Case("float32_non_aligned",
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37},
        {37},
        {19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
         19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
         19, 19, 19},
        {37},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {37}, stream);

    success = RunFloat32Case("float32_broadcast",
        {1, 2, 3, 4}, {2, 2},
        {2, 3}, {2},
        {1, 1, 0, 0}, {2, 2}, stream) && success;

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return success ? 0 : 1;
}
