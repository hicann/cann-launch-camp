/**
 * 通过 aclopExecuteV2 在 NPU 上运行自定义 GELU 算子。
 * 先加载 libascend_all_ops.so 注册算子，再调用 aclopExecuteV2。
 * 编译: g++ -o run_gelu run_gelu.c -I/usr/local/Ascend/cann-9.1.0/include \
 *        -L/usr/local/Ascend/cann-9.1.0/lib64 -lascendcl -lm -ldl
 * 运行: ASCEND_CUSTOM_OPP_PATH=<path> ./run_gelu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include "acl/acl.h"

#define CHECK(expr) do { \
    aclError __e = (expr); \
    if (__e != ACL_SUCCESS) { \
        fprintf(stderr, "[FAIL] %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, __e); \
        exit(1); \
    } \
} while(0)

int main() {
    printf("====================================\n");
    printf("GELU NPU Test (aclopExecuteV2)\n");
    printf("CANN 9.1.0\n");
    printf("====================================\n\n");

    // 0. 先加载 libascend_all_ops.so（含 REG_OP Gelu 注册 + tiling + kernel）
    printf("[0] Loading libs...\n");
    char *cwd = getenv("PWD");
    char path[512];
    snprintf(path, sizeof(path), "%s/build/autogen/libascend_all_ops.so", cwd ? cwd : ".");
    void *hdl = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    printf("  libascend_all_ops: %s\n", hdl ? "OK" : "FAILED");

    // 加载 cust_opapi（含 aclnnGelu 接口）
    snprintf(path, sizeof(path), "%s/build/packages/vendors/custom/op_api/lib/libcust_opapi.so", cwd ? cwd : ".");
    void *hdl2 = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    printf("  libcust_opapi: %s\n", hdl2 ? "OK" : "FAILED");

    // 1. Init ACL
    CHECK(aclInit(NULL));
    CHECK(aclrtSetDevice(0));
    aclrtContext ctx;
    CHECK(aclrtCreateContext(&ctx, 0));
    aclrtStream stream;
    CHECK(aclrtCreateStream(&stream));
    printf("[OK] ACL initialized\n\n");

    // 2. Test: 128 float32
    printf("Test: tiny 128 float32\n");
    int64_t n = 128;
    int64_t nbytes = n * sizeof(float);

    srand(42);
    float *h_in = (float*)malloc(nbytes);
    float *h_out = (float*)malloc(nbytes);
    for (int i = 0; i < n; i++) h_in[i] = ((float)rand()/RAND_MAX)*8.0f - 4.0f;

    void *d_in, *d_out;
    CHECK(aclrtMalloc(&d_in, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&d_out, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d_in, nbytes, h_in, nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t shape[1] = {n};
    aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_ND);
    aclDataBuffer *inputBuf = aclCreateDataBuffer(d_in, nbytes);
    aclTensorDesc *outputDesc = aclCreateTensorDesc(ACL_FLOAT, 1, shape, ACL_FORMAT_ND);
    aclDataBuffer *outputBuf = aclCreateDataBuffer(d_out, nbytes);

    printf("Calling aclopExecuteV2(Gelu)...\n");
    aclError ret = aclopExecuteV2("Gelu",
                                  1, &inputDesc, &inputBuf,
                                  1, &outputDesc, &outputBuf,
                                  NULL, stream);
    printf("aclopExecuteV2 returned: %d\n", ret);

    if (ret == ACL_SUCCESS) {
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(h_out, nbytes, d_out, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));
        double md = 0.0;
        for (int i = 0; i < n; i++) {
            double d = fabs(h_out[i] - (0.5f * h_in[i] * (1.0f + erff((float)(h_in[i] * 0.70710678f)))));
            if (d > md) md = d;
        }
        printf("  ✅ PASS | max_diff=%.2e\n", md);
        float g0 = 0.5f * h_in[n/2] * (1.0f + erff((float)(h_in[n/2] * 0.70710678f)));
        printf("  GELU(%.2f)=%.8f (ref=%.8f)\n", h_in[n/2], h_out[n/2], g0);
    } else {
        const char *reason = "UNKNOWN";
        if (ret == 100024) reason = "OP_NOT_FOUND";
        if (ret == 161001) reason = "PARAM_NULLPTR";
        if (ret == 561002) reason = "TILING_ERROR";
        printf("  ❌ FAIL: error %d (%s)\n", ret, reason);
    }

    // Cleanup
    aclDestroyTensorDesc(inputDesc);
    aclDestroyDataBuffer(inputBuf);
    aclDestroyTensorDesc(outputDesc);
    aclDestroyDataBuffer(outputBuf);
    aclrtFree(d_in); aclrtFree(d_out);
    free(h_in); free(h_out);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    if (hdl) dlclose(hdl);
    if (hdl2) dlclose(hdl2);
    printf("\nDone.\n");
    return 0;
}
