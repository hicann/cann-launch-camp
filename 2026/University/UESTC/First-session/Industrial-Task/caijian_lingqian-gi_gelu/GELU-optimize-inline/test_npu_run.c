/*
 * 在 NPU 上运行自定义 GELU 算子，与 PyTorch 参考值对比。
 * 编译：gcc -o test_npu_run test_npu_run.c -I/usr/local/Ascend/cann-8.5.0/include \
 *          -L/usr/local/Ascend/cann-8.5.0/lib64 -lascendcl -lstdc++ -lm -ldl
 * 运行：ACL_CUSTOM_OPP_PATH=./build/packages/vendors ./test_npu_run
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <dlfcn.h>
#include "aclnn/acl_meta.h"
#include "acl/acl_base_rt.h"    // aclDataType enum (ACL_FLOAT, ACL_FLOAT16)

// 自定义算子库生成的 aclnnGelu 头文件
#include "autogen/aclnn_gelu.h"

// ACL 错误码检查
#define ACL_CHECK(expr)                                     \
    do {                                                    \
        aclError __ret = (expr);                            \
        if (__ret != ACL_SUCCESS) {                         \
            fprintf(stderr, "[FAIL] %s:%d: %s -> %d\n",     \
                    __FILE__, __LINE__, #expr, __ret);      \
            exit(1);                                        \
        }                                                   \
    } while (0)

// ACL 数据类型常量
#define ACL_FLOAT  0
#define ACL_FLOAT16  1
#define ACL_ND  2  // aclFormat

// 测试用例
typedef struct {
    const char *name;
    int64_t n;
    int dtype;  // ACL_FLOAT or ACL_FLOAT16
} TestCase;

TestCase test_cases[] = {
    {"tiny",    128,    ACL_FLOAT},
    {"small",   2048,   ACL_FLOAT},
    {"medium",  8192,   ACL_FLOAT16},
    {"large",   65536,  ACL_FLOAT},
    {"huge",    262144, ACL_FLOAT16},
};
const int num_cases = sizeof(test_cases) / sizeof(test_cases[0]);

// PyTorch GELU 参考值 (CPU 计算 tanh 近似，与 Kernel 使用的 Tanh5 算法一致)
float gelu_ref_float(float x) {
    // GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    float c = 0.797884583f;  // √(2/π)
    float k = 0.044715f;
    float x3 = x * x * x;
    float inner = c * (x + k * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

void run_test(int64_t n, int dtype, const char *name, aclrtStream stream) {
    int elem_size = (dtype == ACL_FLOAT16) ? 2 : 4;
    int64_t nbytes = n * elem_size;

    // 1. 分配输入/输出 device 内存
    void *dev_input = NULL, *dev_output = NULL;
    ACL_CHECK(aclrtMalloc(&dev_input, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dev_output, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // 2. 生成输入数据 (host)
    void *host_input = malloc(nbytes);
    void *host_output = malloc(nbytes);
    if (!host_input || !host_output) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    srand(42);
    if (dtype == ACL_FLOAT) {
        float *data = (float *)host_input;
        for (int64_t i = 0; i < n; i++) {
            data[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;  // [-4, 4]
        }
    } else {
        // float16: 用 float 生成再转
        float *tmp = (float *)malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; i++) {
            tmp[i] = (float)rand() / RAND_MAX * 8.0f - 4.0f;
        }
        // float32 -> float16 (截断)
        uint16_t *fp16 = (uint16_t *)host_input;
        for (int64_t i = 0; i < n; i++) {
            // 简单的 float -> half 转换 (IEEE 754)
            uint32_t u;
            memcpy(&u, &tmp[i], sizeof(u));
            uint16_t sign = (u >> 16) & 0x8000;
            int exp = ((u >> 23) & 0xff) - 127 + 15;
            uint32_t mant = u & 0x007fffff;
            if (exp >= 31) {
                fp16[i] = sign | 0x7c00;  // Inf
            } else if (exp <= 0) {
                fp16[i] = sign | ((mant | 0x00800000) >> (1 - exp));
            } else {
                fp16[i] = sign | (exp << 10) | (mant >> 13);
            }
        }
        free(tmp);
    }

    // 3. host -> device
    ACL_CHECK(aclrtMemcpy(dev_input, nbytes, host_input, nbytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // 4. 创建 ACL Tensor
    int64_t shape[1] = {n};
    aclTensor *tensor_in = aclCreateTensor_NEW(shape, 1, dtype, n,
                                           dev_input, ACL_ND, NULL, 0);
    aclTensor *tensor_out = aclCreateTensor_NEW(shape, 1, dtype, n,
                                            dev_output, ACL_ND, NULL, 0);
    if (!tensor_in || !tensor_out) {
        fprintf(stderr, "aclCreateTensor_NEW failed\n");
        exit(1);
    }

    // 5. 两段式调用: GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = NULL;
    ACL_CHECK(aclnnGeluGetWorkspaceSize(tensor_in, tensor_out,
                                        &workspace_size, &executor));

    // 6. 分配 workspace
    void *workspace = NULL;
    if (workspace_size > 0) {
        ACL_CHECK(aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 7. 执行
    ACL_CHECK(aclnnGelu(workspace, workspace_size, executor, stream));

    // 8. 同步
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // 9. device -> host
    ACL_CHECK(aclrtMemcpy(host_output, nbytes, dev_output, nbytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));

    // 10. 与 PyTorch GELU 参考值对比
    double max_diff = 0.0;
    double sum_diff = 0.0;
    float rtol = (dtype == ACL_FLOAT16) ? 1e-2f : 1e-5f;

    if (dtype == ACL_FLOAT) {
        float *in = (float *)host_input;
        float *out = (float *)host_output;
        for (int64_t i = 0; i < n; i++) {
            float ref = gelu_ref_float(in[i]);
            float diff = fabsf(out[i] - ref);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
        }
    } else {
        uint16_t *in = (uint16_t *)host_input;
        uint16_t *out = (uint16_t *)host_output;
        for (int64_t i = 0; i < n; i++) {
            // 精确转换 half -> float
            uint32_t sign = (in[i] >> 15) & 0x1;
            int exp = (in[i] >> 10) & 0x1f;
            uint32_t mant = in[i] & 0x03ff;
            float v_in, v_out;
            if (exp == 0) {
                v_in = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2, -14);
            } else {
                v_in = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2, exp - 15);
            }
            uint32_t sign_o = (out[i] >> 15) & 0x1;
            int exp_o = (out[i] >> 10) & 0x1f;
            uint32_t mant_o = out[i] & 0x03ff;
            if (exp_o == 0) {
                v_out = (sign_o ? -1.0f : 1.0f) * (mant_o / 1024.0f) * powf(2, -14);
            } else {
                v_out = (sign_o ? -1.0f : 1.0f) * (1.0f + mant_o / 1024.0f) * powf(2, exp_o - 15);
            }
            float ref = gelu_ref_float(v_in);
            float diff = fabsf(v_out - ref);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
        }
    }

    double avg_diff = sum_diff / n;
    int passed = max_diff < rtol;
    const char *status = passed ? "PASS" : "FAIL";

    printf("  %s %s | n=%ld dtype=%s | max_diff=%.2e avg_diff=%.2e\n",
           passed ? "✅" : "❌", name, (long)n,
           (dtype == ACL_FLOAT16) ? "fp16" : "fp32",
           max_diff, avg_diff);
    if (!passed) {
        printf("  > rtol=%.0e exceeded!\n", rtol);
    }

    // 11. 清理
    aclDestroyTensor(tensor_in);
    aclDestroyTensor(tensor_out);
    aclrtFree(dev_input);
    aclrtFree(dev_output);
    if (workspace) aclrtFree(workspace);
    free(host_input);
    free(host_output);
}

int main() {
    printf("============================================================\n");
    printf("GELU 自定义算子 — NPU 实测 (C 版)\n");
    printf("============================================================\n\n");

    // 设置自定义算子路径
    char *custom_opp = getenv("ASCEND_CUSTOM_OPP_PATH");
    printf("ASCEND_CUSTOM_OPP_PATH=%s\n\n", custom_opp ? custom_opp : "(not set)");

    // 1. 初始化 ACL
    printf("[1/3] 初始化 AscendCL...\n");
    ACL_CHECK(aclInit(NULL));
    ACL_CHECK(aclrtSetDevice(0));

    aclrtContext context = NULL;
    ACL_CHECK(aclrtCreateContext(&context, 0));

    aclrtStream stream = NULL;
    ACL_CHECK(aclrtCreateStream(&stream));
    printf("  [OK]\n\n");

    // 2. 运行测试
    printf("[2/3] 在 NPU 上运行 GELU 算子...\n\n");
    for (int i = 0; i < num_cases; i++) {
        run_test(test_cases[i].n, test_cases[i].dtype, test_cases[i].name, stream);
    }

    // 3. 清理
    printf("\n[3/3] 清理资源...\n");
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(0);
    aclFinalize();
    printf("  [OK]\n\n");

    printf("============================================================\n");
    printf("✅ 测试完成！\n");
    printf("============================================================\n");
    return 0;
}
