/**
 * 通过 aclnnGelu API 在 NPU 上运行自定义 GELU 算子。
 * 
 * 关键修复：先加载 libascend_all_ops.so（注册 OP_ADD），再 aclInit，
 * 最后才加载 libcust_opapi.so（含 aclnnGelu 接口）。
 * 顺序不能错——libcust_opapi.so 依赖链带 driver 的 libascend_hal.so，
 * 在 aclInit 之前加载会导致冲突（error 500000）。
 *
 * 编译:
 *   g++ -o run_gelu_aclnn run_gelu_aclnn.c \
 *       -I/opt/conda/Ascend/cann-9.0.0/include \
 *       -L/opt/conda/Ascend/cann-9.0.0/lib64 \
 *       -lascendcl -lnnopbase -lm -ldl \
 *       -Wl,-rpath,/opt/conda/Ascend/cann-9.0.0/lib64
 *
 * 运行:
 *   ASCEND_CUSTOM_OPP_PATH=/path/to/build/packages/vendors \
 *   LD_LIBRARY_PATH=/opt/conda/Ascend/cann-9.0.0/lib64:/opt/conda/Ascend/cann-9.0.0/aarch64-linux/devlib/linux/aarch64 \
 *   ./run_gelu_aclnn
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <unistd.h>
#include "acl/acl.h"

// 手动定义 aclnn 相关类型（避免依赖 aclnn/acl_meta.h，该头文件在某些 CANN 版本中不可用）
typedef int aclnnStatus;
typedef void aclTensor;
typedef void aclOpExecutor;

#define CHECK(expr) do { \
    aclError __e = (expr); \
    if (__e != ACL_SUCCESS) { \
        fprintf(stderr, "[FAIL] %s:%d: %s -> %d\n", __FILE__, __LINE__, #expr, __e); \
        exit(1); \
    } \
} while(0)

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

// 函数指针类型
typedef aclTensor* (*AclCreateTensorFn)(const int64_t*, uint64_t, aclDataType, const int64_t*, int64_t, aclFormat, const int64_t*, uint64_t, void*);
typedef aclnnStatus (*AclDestroyTensorFn)(const aclTensor*);

int main() {

    AclCreateTensorFn myAclCreateTensor = NULL;
    AclDestroyTensorFn myAclDestroyTensor = NULL;

    char cwd[512];
    if (!getcwd(cwd, sizeof(cwd))) {
        strcpy(cwd, ".");
    }
    char path[1024];

    // ====== Step 0: 先加载 libascend_all_ops.so（只含 op 注册，无 driver 依赖）======
    snprintf(path, sizeof(path), "%s/build/autogen/libascend_all_ops.so", cwd);
    void *hdl_reg = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_reg) {
        fprintf(stderr, "FATAL: dlopen(%s) failed: %s\n", path, dlerror());
        return 1;
    }

    // ====== Step 1: aclInit（在加载 libnnopbase/libcust_opapi 之前）======
    CHECK(aclInit(NULL));
    CHECK(aclrtSetDevice(0));
    aclrtContext ctx; CHECK(aclrtCreateContext(&ctx, 0));
    aclrtStream stream; CHECK(aclrtCreateStream(&stream));

    // ====== Step 2: 加载 libnnopbase.so（提供 aclCreateTensor）======
    // 注意：libnnopbase.so 依赖 libascend_hal.so，该库位于 devlib 目录。
    // 但 devlib 路径不能加入运行时 LD_LIBRARY_PATH（会导致 aclInit 失败 500000）。
    // 所以用绝对路径直接指定依赖查找路径。
    void *hdl_nnop = NULL;
    const char *toolkit = getenv("ASCEND_TOOLKIT_HOME") 
        ? getenv("ASCEND_TOOLKIT_HOME") : "/opt/conda/Ascend/cann-9.0.0";
    // 方法1: 从 lib64 找（运行时的路径已有 lib64 在 LD_LIBRARY_PATH/RPATH）
    hdl_nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_nnop) {
        char ldpath[1024];
        strcpy(ldpath, toolkit);
        strcat(ldpath, "/lib64:");
        strcat(ldpath, toolkit);
        strcat(ldpath, "/aarch64-linux/devlib/linux/aarch64");
        setenv("LD_LIBRARY_PATH", ldpath, 1);
        hdl_nnop = dlopen("libnnopbase.so", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (hdl_nnop) {
        myAclCreateTensor = (AclCreateTensorFn)dlsym(hdl_nnop, "aclCreateTensor");
        myAclDestroyTensor = (AclDestroyTensorFn)dlsym(hdl_nnop, "aclDestroyTensor");
    }

    // ====== Step 3: 加载 tiling 库（被 aclnnGelu 内部使用）======
    // 注意：libcust_opmaster_rt2.0.so 依赖 libascend_hal.so（在 devlib 目录）。
    // 但 devlib 不能在 aclInit 前加入 LD_LIBRARY_PATH。所以 aclInit 后才加入。
    char ldpath[2048];
    const char *cur_ld = getenv("LD_LIBRARY_PATH");
    if (!cur_ld) cur_ld = "";
    strcpy(ldpath, cur_ld);
    strcat(ldpath, ":");
    strcat(ldpath, toolkit);
    strcat(ldpath, "/aarch64-linux/devlib/linux/aarch64");
    setenv("LD_LIBRARY_PATH", ldpath, 1);

    void *hdl_tiling = NULL;
    char tiling_path[512];
    strcpy(tiling_path, cwd);
    strcat(tiling_path, "/build/packages/vendors/custom/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so");
    hdl_tiling = dlopen(tiling_path, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl_tiling) {
        // 回退到 build/op_host 目录
        snprintf(path, sizeof(path), "%s/build/op_host/libcust_opmaster_rt2.0.so", cwd);
        hdl_tiling = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    }

    // ====== Step 4: 加载 libcust_opapi.so（aclnnGelu 接口）======
    void *hdl = NULL;
    snprintf(path, sizeof(path), "%s/build/packages/vendors/custom/op_api/lib/libcust_opapi.so", cwd);
    hdl = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!hdl) {
        // Fallback to op_host/libcust_opapi.so
        snprintf(path, sizeof(path), "%s/build/op_host/libcust_opapi.so", cwd);
        hdl = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!hdl) {
        fprintf(stderr, "FATAL: dlopen libcust_opapi.so failed: %s\n", dlerror());
        if (hdl_reg) dlclose(hdl_reg);
        return 1;
    }

    // dlsym aclnnGelu 函数
    typedef aclnnStatus (*GetWsSizeFn)(const aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
    typedef aclnnStatus (*ExecFn)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    GetWsSizeFn aclnnGeluGetWorkspaceSize_fn = (GetWsSizeFn)dlsym(hdl, "aclnnGeluGetWorkspaceSize");
    ExecFn aclnnGelu_fn = (ExecFn)dlsym(hdl, "aclnnGelu");
    if (!aclnnGeluGetWorkspaceSize_fn || !aclnnGelu_fn) {
        return 1;
    }

    // ====== Test 1: 128 float32 ======
    int64_t n = 128;
    int64_t nbytes = n * sizeof(float);

    srand(42);
    float *h_in = (float*)malloc(nbytes);
    float *h_out = (float*)malloc(nbytes);
    for (int i = 0; i < n; i++) h_in[i] = ((float)rand()/RAND_MAX)*8-4;

    void *d_in, *d_out;
    CHECK(aclrtMalloc(&d_in, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&d_out, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d_in, nbytes, h_in, nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t shape[1] = {n};
    aclTensor *t_in  = myAclCreateTensor(shape, 1, ACL_FLOAT, NULL, 0, ACL_FORMAT_ND, shape, 1, d_in);
    aclTensor *t_out = myAclCreateTensor(shape, 1, ACL_FLOAT, NULL, 0, ACL_FORMAT_ND, shape, 1, d_out);

    uint64_t ws_size = 0;
    aclOpExecutor *executor = NULL;
    aclnnStatus ret = aclnnGeluGetWorkspaceSize_fn(t_in, t_out, &ws_size, &executor);

    if (ret == 0) {
        void *ws = NULL;
        if (ws_size > 0) CHECK(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
        ret = aclnnGelu_fn(ws, ws_size, executor, stream);
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(h_out, nbytes, d_out, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

        double md = 0;
        for (int i = 0; i < n; i++) {
            double d = fabs(h_out[i] - (h_in[i]*0.5*(1+erf(h_in[i]/sqrt(2)))));
            if (d > md) md = d;
        }
    } else {
        fprintf(stderr, "[FAIL] GetWorkspaceSize returned %d\n", ret);
        return 1;
    }

    myAclDestroyTensor(t_in);
    myAclDestroyTensor(t_out);
    aclrtFree(d_in); aclrtFree(d_out);
    free(h_in); free(h_out);

    // ====== Test 2: 8192 float16 ======
    n = 8192; nbytes = n * 2;
    shape[0] = n;  // 更新 shape 到当前 n

    uint16_t *h16_in = (uint16_t*)malloc(nbytes);
    uint16_t *h16_out = (uint16_t*)malloc(nbytes);
    srand(42);
    for (int i = 0; i < n; i++) {
        float v = ((float)rand()/RAND_MAX)*8-4;
        h16_in[i] = f2h(v);
    }

    CHECK(aclrtMalloc(&d_in, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&d_out, nbytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d_in, nbytes, h16_in, nbytes, ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *t16_in  = myAclCreateTensor(shape, 1, ACL_FLOAT16, NULL, 0, ACL_FORMAT_ND, shape, 1, d_in);
    aclTensor *t16_out = myAclCreateTensor(shape, 1, ACL_FLOAT16, NULL, 0, ACL_FORMAT_ND, shape, 1, d_out);

    ws_size = 0; executor = NULL;
    ret = aclnnGeluGetWorkspaceSize_fn(t16_in, t16_out, &ws_size, &executor);

    if (ret == 0) {
        void *ws = NULL;
        if (ws_size > 0) CHECK(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
        ret = aclnnGelu_fn(ws, ws_size, executor, stream);
        CHECK(aclrtSynchronizeStream(stream));
        CHECK(aclrtMemcpy(h16_out, nbytes, d_out, nbytes, ACL_MEMCPY_DEVICE_TO_HOST));

        double md = 0;
        for (int i = 0; i < n; i++) {
            float v_in = h2f(h16_in[i]);
            float v_out = h2f(h16_out[i]);
            double d = fabs(v_out - v_in*0.5*(1+erf(v_in/sqrt(2))));
            if (d > md) md = d;
        }
    } else {
        fprintf(stderr, "[FAIL] GetWorkspaceSize (fp16) returned %d\n", ret);
    }

    myAclDestroyTensor(t16_in);
    myAclDestroyTensor(t16_out);
    aclrtFree(d_in); aclrtFree(d_out);
    free(h16_in); free(h16_out);

    // Cleanup
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    if (hdl_reg) dlclose(hdl_reg);
    if (hdl_tiling) dlclose(hdl_tiling);
    if (hdl) dlclose(hdl);
    return 0;
}
