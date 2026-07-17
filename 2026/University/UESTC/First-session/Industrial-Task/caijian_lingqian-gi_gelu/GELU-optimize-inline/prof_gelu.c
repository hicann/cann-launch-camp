/**
 * GELU Profiling Test — 纯 dlopen 方式
 *
 * 加载顺序：
 *   1. libascend_all_ops.so（无 devlib 依赖，安全）
 *   2. aclInit（此时 LD_LIBRARY_PATH 不含 devlib）
 *   3. dlopen 按需加载 libascend_hal.so + libcust_opapi.so
 *   4. 用 dlsym 获取 acl 函数指针 + 运行算子
 *
 * 编译（不链接任何 CANN 库，纯 dlopen 方式）：
 *   gcc -o prof_gelu prof_gelu.c -lm -ldl
 *
 * 运行：
 *   export ASCEND_CUSTOM_OPP_PATH=...
 *   ./prof_gelu
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <stdint.h>

// ==== ACL 类型定义（只从头文件获取类型，不链接库）====
typedef int aclError;
typedef void* aclrtContext;
typedef void* aclrtStream;
typedef int aclDataType;
typedef int aclFormat;
typedef void* aclTensorDesc;
typedef void* aclDataBuffer;
#define ACL_SUCCESS 0
#define ACL_FLOAT 0
#define ACL_FLOAT16 1
#define ACL_FORMAT_ND 2
#define ACL_MEM_MALLOC_HUGE_FIRST 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2

// ==== 常量 ====
#define CANN_LIB "/usr/local/Ascend/cann-8.5.0"
#define DEVLIBSO CANN_LIB "/aarch64-linux/devlib/linux/aarch64/libascend_hal.so"
#define OPSO_PATH "/opt/atomgit/GELU/build_out/autogen/libascend_all_ops.so"

// 函数指针类型
typedef aclError (*AclInitFn)(const char*);
typedef aclError (*AclrtSetDeviceFn)(int32_t);
typedef aclError (*AclrtCreateContextFn)(aclrtContext*, int32_t);
typedef aclError (*AclrtCreateStreamFn)(aclrtStream*);
typedef aclError (*AclrtMallocFn)(void**, uint64_t, int32_t);
typedef aclError (*AclrtMemcpyFn)(void*, uint64_t, void*, uint64_t, int32_t);
typedef aclError (*AclrtFreeFn)(void*);
typedef aclError (*AclrtSynchronizeStreamFn)(aclrtStream);
typedef aclError (*AclrtDestroyStreamFn)(aclrtStream);
typedef aclError (*AclrtDestroyContextFn)(aclrtContext);
typedef aclError (*AclrtResetDeviceFn)(int32_t);
typedef aclError (*AclFinalizeFn)();
typedef aclTensorDesc* (*AclCreateTensorDescFn)(aclDataType, int, const int64_t*, aclFormat);
typedef aclDataBuffer* (*AclCreateDataBufferFn)(void*, uint64_t);
typedef aclError (*AclDestroyTensorDescFn)(aclTensorDesc*);
typedef aclError (*AclDestroyDataBufferFn)(aclDataBuffer*);
typedef aclError (*AclOpExecuteV2Fn)(const char*, int, const aclTensorDesc**, const aclDataBuffer**,
                                      int, const aclTensorDesc**, const aclDataBuffer**,
                                      aclTensorDesc*, aclrtStream);

// ==== 测试用例 ====
typedef struct {
    const char *name;
    int64_t n;
    int dtype; // ACL_FLOAT or ACL_FLOAT16
} TestCase;

static TestCase TESTS[] = {
    {"tiny_128_f32",   128,     ACL_FLOAT},
    {"small_2048_f32", 2048,    ACL_FLOAT},
    {"medium_8192_f16",8192,    ACL_FLOAT16},
    {"large_65536_f32",65536,   ACL_FLOAT},
    {"huge_262144_f16",262144,  ACL_FLOAT16},
};
static const int NTESTS = sizeof(TESTS) / sizeof(TESTS[0]);

// float <-> half
static uint16_t f2h(float f) {
    uint32_t u; memcpy(&u, &f, sizeof(u));
    uint16_t s = (u>>16)&0x8000;
    int e = ((u>>23)&0xff)-127+15;
    uint32_t m = u&0x007fffff;
    if (e>=31) return s|0x7c00;
    if (e<=0)  return s|((m|0x00800000)>>(1-e));
    return s|(e<<10)|(m>>13);
}
static float h2f(uint16_t h) {
    uint32_t s=(h>>15)&1, e=(h>>10)&0x1f, m=h&0x03ff;
    float v;
    if (e==0) v=(s?-1:1)*(m/1024.0f)*powf(2,-14);
    else      v=(s?-1:1)*(1.0f+m/1024.0f)*powf(2,e-15);
    return v;
}
static float gelu_ref_f32(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678f));
}

static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s: %s\n", msg, dlerror());
    exit(1);
}

static void *load_so(const char *path, int mode) {
    void *h = dlopen(path, mode);
    if (!h) fprintf(stderr, "  dlopen(%s): %s\n", path, dlerror());
    return h;
}

int main() {
    printf("====================================\n");
    printf("GELU Profiling Test (dlopen mode)\n");
    printf("====================================\n\n");

    // [0] 只加载 libascend_all_ops.so（无 devlib 依赖，绝对安全）
    printf("[0] Loading op registration library...\n");
    void *hdl_all = load_so(OPSO_PATH, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_all) return 1;
    printf("  libascend_all_ops.so: OK\n");

    // 加载 devlib 目录下的 libascendcl.so（关键！此版本无 libascend_hal.so 依赖）
    // devlib/libascendcl.so 只依赖 libstdc++/libm/等基础库，因此不会触发冲突
    char so_path[512];
    snprintf(so_path, sizeof(so_path), "%s/aarch64-linux/devlib/linux/aarch64/libascendcl.so", CANN_LIB);
    void *hdl_ac = load_so(so_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_ac) {
        fprintf(stderr, "  Fallback to lib64/libascendcl.so...\n");
        snprintf(so_path, sizeof(so_path), "%s/lib64/libascendcl.so", CANN_LIB);
        hdl_ac = load_so(so_path, RTLD_LAZY | RTLD_GLOBAL);
        if (!hdl_ac) return 1;
    }
    printf("  libascendcl.so (devlib): OK");

    AclInitFn aclInit = (AclInitFn)dlsym(hdl_ac, "aclInit");
    if (!aclInit) { fprintf(stderr, "  aclInit symbol not found\n"); return 1; }

    // aclInit（此时没有 devlib 库在符号表中）
    aclError ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "  aclInit failed: %d\n", ret);
        return 1;
    }
    printf("  [OK] aclInit passed\n");

    // 加载 libnnopbase.so
    void *hdl_nn = NULL;
    snprintf(so_path, sizeof(so_path), "%s/lib64/libnnopbase.so", CANN_LIB);
    hdl_nn = load_so(so_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_nn) return 1;
    printf("  libnnopbase.so: OK\n");

    // [1] 加载函数指针
    printf("\n[1] Getting function pointers...\n");
    aclInit = (AclInitFn)dlsym(hdl_ac, "aclInit");
    AclrtSetDeviceFn aclrtSetDevice = (AclrtSetDeviceFn)dlsym(hdl_ac, "aclrtSetDevice");
    AclrtCreateContextFn aclrtCreateContext = (AclrtCreateContextFn)dlsym(hdl_ac, "aclrtCreateContext");
    AclrtCreateStreamFn aclrtCreateStream = (AclrtCreateStreamFn)dlsym(hdl_ac, "aclrtCreateStream");
    AclrtMallocFn aclrtMalloc = (AclrtMallocFn)dlsym(hdl_ac, "aclrtMalloc");
    AclrtMemcpyFn aclrtMemcpy = (AclrtMemcpyFn)dlsym(hdl_ac, "aclrtMemcpy");
    AclrtFreeFn aclrtFree = (AclrtFreeFn)dlsym(hdl_ac, "aclrtFree");
    AclrtSynchronizeStreamFn aclrtSynchronizeStream = (AclrtSynchronizeStreamFn)dlsym(hdl_ac, "aclrtSynchronizeStream");
    AclrtDestroyStreamFn aclrtDestroyStream = (AclrtDestroyStreamFn)dlsym(hdl_ac, "aclrtDestroyStream");
    AclrtDestroyContextFn aclrtDestroyContext = (AclrtDestroyContextFn)dlsym(hdl_ac, "aclrtDestroyContext");
    AclrtResetDeviceFn aclrtResetDevice = (AclrtResetDeviceFn)dlsym(hdl_ac, "aclrtResetDevice");
    AclFinalizeFn aclFinalize = (AclFinalizeFn)dlsym(hdl_ac, "aclFinalize");
    AclCreateTensorDescFn aclCreateTensorDesc = (AclCreateTensorDescFn)dlsym(hdl_ac, "aclCreateTensorDesc");
    AclCreateDataBufferFn aclCreateDataBuffer = (AclCreateDataBufferFn)dlsym(hdl_ac, "aclCreateDataBuffer");
    AclDestroyTensorDescFn aclDestroyTensor = (AclDestroyTensorDescFn)dlsym(hdl_nn, "aclDestroyTensor");
    AclDestroyDataBufferFn aclDestroyDataBuffer = (AclDestroyDataBufferFn)dlsym(hdl_ac, "aclDestroyDataBuffer");
    AclOpExecuteV2Fn aclopExecuteV2 = (AclOpExecuteV2Fn)dlsym(hdl_ac, "aclopExecuteV2");

    if (!aclInit || !aclrtSetDevice || !aclrtCreateContext || !aclrtCreateStream ||
        !aclrtMalloc || !aclrtMemcpy || !aclrtFree || !aclrtSynchronizeStream ||
        !aclrtDestroyStream || !aclrtDestroyContext || !aclrtResetDevice || !aclFinalize) {
        fprintf(stderr, "  aclInit=%p setDevice=%p createContext=%p createStream=%p\n",
                (void*)aclInit, (void*)aclrtSetDevice, (void*)aclrtCreateContext, (void*)aclrtCreateStream);
        fprintf(stderr, "  malloc=%p memcpy=%p free=%p sync=%p\n",
                (void*)aclrtMalloc, (void*)aclrtMemcpy, (void*)aclrtFree, (void*)aclrtSynchronizeStream);
        fprintf(stderr, "  destroyStream=%p destroyCtx=%p resetDevice=%p finalize=%p\n",
                (void*)aclrtDestroyStream, (void*)aclrtDestroyContext, (void*)aclrtResetDevice, (void*)aclFinalize);
        fprintf(stderr, "FATAL: ACL functions not found\n");
        return 1;
    }
    if (!aclCreateTensorDesc || !aclCreateDataBuffer ||
        !aclDestroyTensor || !aclDestroyDataBuffer) {
        fprintf(stderr, "  createTensorDesc=%p createDataBuf=%p destroyTensor=%p destroyDataBuf=%p\n",
                (void*)aclCreateTensorDesc, (void*)aclCreateDataBuffer, (void*)aclDestroyTensor, (void*)aclDestroyDataBuffer);
        fprintf(stderr, "FATAL: Tensor functions not found\n");
        return 1;
    }
    if (!aclopExecuteV2) {
        fprintf(stderr, "  aclopExecuteV2=%p\n", (void*)aclopExecuteV2);
        fprintf(stderr, "FATAL: aclopExecuteV2 not found\n");
        return 1;
    }
    printf("  All function pointers: OK\n");

    // [2] Set device, create context/stream
    printf("\n[2] Setting device and creating context/stream...\n");
    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "setDevice: %d\n", ret); return 1; }
    aclrtContext ctx;
    ret = aclrtCreateContext(&ctx, 0);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "createContext: %d\n", ret); return 1; }
    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) { fprintf(stderr, "createStream: %d\n", ret); return 1; }
    printf("  [OK] ACL initialized\n\n");

    // [3] 运行测试用例
    printf("\n[3] Running tests...\n\n");
    int all_pass = 1;

    for (int t = 0; t < NTESTS; t++) {
        const char *name = TESTS[t].name;
        int64_t n = TESTS[t].n;
        int dtype = TESTS[t].dtype;
        int elem_size = (dtype == ACL_FLOAT) ? 4 : 2;
        int64_t nbytes = n * elem_size;

        printf("  --- %s (%ld elements) ---\n", name, (long)n);

        // 准备 host 数据
        void *h_in = malloc(nbytes);
        void *h_out = malloc(nbytes);
        if (!h_in || !h_out) { fprintf(stderr, "malloc failed\n"); return 1; }

        srand(42);
        if (dtype == ACL_FLOAT) {
            float *p = (float*)h_in;
            for (int64_t i = 0; i < n; i++)
                p[i] = ((float)rand()/RAND_MAX)*8.0f - 4.0f;
        } else {
            uint16_t *p = (uint16_t*)h_in;
            for (int64_t i = 0; i < n; i++) {
                float v = ((float)rand()/RAND_MAX)*8.0f - 4.0f;
                p[i] = f2h(v);
            }
        }

        // device 内存
        void *d_in = NULL, *d_out = NULL;
        ret = aclrtMalloc(&d_in, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "  malloc input failed: %d\n", ret); all_pass=0; goto cleanup; }
        ret = aclrtMalloc(&d_out, nbytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "  malloc output failed: %d\n", ret); all_pass=0; goto cleanup; }
        ret = aclrtMemcpy(d_in, nbytes, h_in, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) { fprintf(stderr, "  memcpy H2D failed: %d\n", ret); all_pass=0; goto cleanup; }

        // tensor 描述
        int64_t shape[1] = {n};
        aclTensorDesc *inputDesc = aclCreateTensorDesc(dtype, 1, shape, ACL_FORMAT_ND);
        aclDataBuffer *inputBuf = aclCreateDataBuffer(d_in, nbytes);
        aclTensorDesc *outputDesc = aclCreateTensorDesc(dtype, 1, shape, ACL_FORMAT_ND);
        aclDataBuffer *outputBuf = aclCreateDataBuffer(d_out, nbytes);

        // 执行算子
        ret = aclopExecuteV2("Gelu",
                              1, (const aclTensorDesc**)&inputDesc, (const aclDataBuffer**)&inputBuf,
                              1, (const aclTensorDesc**)&outputDesc, (const aclDataBuffer**)&outputBuf,
                              NULL, stream);

        if (ret != ACL_SUCCESS) {
            printf("  FAIL | aclopExecuteV2 error=%d\n", ret);
            all_pass = 0;
        } else {
            aclrtSynchronizeStream(stream);
            aclrtMemcpy(h_out, nbytes, d_out, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);

            // 验证
            double max_diff = 0.0;
            if (dtype == ACL_FLOAT) {
                float *pin = (float*)h_in, *pout = (float*)h_out;
                for (int64_t i = 0; i < n; i++) {
                    double d = fabs(pout[i] - gelu_ref_f32(pin[i]));
                    if (d > max_diff) max_diff = d;
                }
            } else {
                uint16_t *pin = (uint16_t*)h_in, *pout = (uint16_t*)h_out;
                for (int64_t i = 0; i < n; i++) {
                    double d = fabs(h2f(pout[i]) - gelu_ref_f32(h2f(pin[i])));
                    if (d > max_diff) max_diff = d;
                }
            }
            double rtol = (dtype == ACL_FLOAT) ? 1e-5 : 1e-2;
            int ok = (max_diff < rtol);
            printf("  %s | max_diff=%.2e | rtol=%.0e\n",
                   ok ? "  PASS" : "  FAIL", max_diff, rtol);
            if (!ok) all_pass = 0;
        }

        // 清理
        aclDestroyTensor(inputDesc);
        aclDestroyDataBuffer(inputBuf);
        aclDestroyTensor(outputDesc);
        aclDestroyDataBuffer(outputBuf);
    cleanup:
        if (d_in) aclrtFree(d_in);
        if (d_out) aclrtFree(d_out);
        free(h_in);
        free(h_out);
    }

    // [4] 清理
    printf("\n[4] Cleanup...\n");
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    if (hdl_all) dlclose(hdl_all);
    if (hdl_ac) dlclose(hdl_ac);
    if (hdl_nn) dlclose(hdl_nn);

    printf("\n====================================\n");
    printf("%s\n", all_pass ? "All tests PASSED" : "Some tests FAILED");
    printf("====================================\n");
    return all_pass ? 0 : 1;
}
