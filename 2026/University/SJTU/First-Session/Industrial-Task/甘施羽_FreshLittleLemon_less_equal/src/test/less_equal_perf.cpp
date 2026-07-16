// LessEqual 性能计时：对若干形状多次调用 aclnnLessEqual 并测平均耗时（μs）。
#include <acl/acl.h>
#include <aclnn_less_equal.h>

#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#define ACL_CHECK(call)                                                            \
    do {                                                                           \
        aclError _ret = (call);                                                    \
        if (_ret != ACL_SUCCESS) {                                                 \
            std::cerr << "ACL error " << _ret << " @" << __LINE__ << std::endl;    \
            return 1;                                                              \
        }                                                                          \
    } while (0)

enum DType { DT_F16, DT_F32, DT_I32, DT_I8 };
static aclDataType AclType(DType t) {
    switch (t) { case DT_F16: return ACL_FLOAT16; case DT_F32: return ACL_FLOAT;
                 case DT_I32: return ACL_INT32; case DT_I8: return ACL_INT8; }
    return ACL_FLOAT;
}
static size_t TypeSize(DType t) {
    switch (t) { case DT_F16: return 2; case DT_F32: return 4; case DT_I32: return 4; case DT_I8: return 1; }
    return 4;
}
static std::vector<int64_t> Contig(const std::vector<int64_t> &s) {
    std::vector<int64_t> st(s.size(), 1);
    for (int i = (int)s.size() - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1];
    return st;
}
static int64_t Numel(const std::vector<int64_t> &s) { int64_t n = 1; for (auto d : s) n *= d; return n; }

static aclrtStream g_stream = nullptr;

static double TimeCase(const std::string &name, std::vector<int64_t> s1, std::vector<int64_t> s2,
                       std::vector<int64_t> so, DType t, int iters) {
    int64_t n1 = Numel(s1), n2 = Numel(s2), no = Numel(so);
    void *d1, *d2, *dy;
    aclrtMalloc(&d1, n1 * TypeSize(t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&d2, n2 * TypeSize(t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dy, no, ACL_MEM_MALLOC_HUGE_FIRST);
    auto st1 = Contig(s1), st2 = Contig(s2), sto = Contig(so);
    aclTensor *x1 = aclCreateTensor(s1.data(), s1.size(), AclType(t), st1.data(), 0, ACL_FORMAT_ND, s1.data(), s1.size(), d1);
    aclTensor *x2 = aclCreateTensor(s2.data(), s2.size(), AclType(t), st2.data(), 0, ACL_FORMAT_ND, s2.data(), s2.size(), d2);
    aclTensor *y = aclCreateTensor(so.data(), so.size(), ACL_BOOL, sto.data(), 0, ACL_FORMAT_ND, so.data(), so.size(), dy);

    // warmup + 持有 workspace
    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    aclnnLessEqualGetWorkspaceSize(x1, x2, y, &ws, &exec);
    void *wsp = nullptr; if (ws > 0) aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST);
    aclnnLessEqual(wsp, ws, exec, g_stream);
    aclrtSynchronizeStream(g_stream);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        uint64_t ws2 = 0; aclOpExecutor *e2 = nullptr;
        aclnnLessEqualGetWorkspaceSize(x1, x2, y, &ws2, &e2);
        aclnnLessEqual(wsp, ws2, e2, g_stream);
    }
    aclrtSynchronizeStream(g_stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;

    std::cout << "  " << name << "  numel=" << no << "  avg=" << us << " us (含host下发)" << std::endl;
    if (wsp) aclrtFree(wsp);
    aclDestroyTensor(x1); aclDestroyTensor(x2); aclDestroyTensor(y);
    aclrtFree(d1); aclrtFree(d2); aclrtFree(dy);
    return us;
}

int main() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    ACL_CHECK(aclrtCreateStream(&g_stream));
    int N = 200;
    std::cout << "==== LessEqual 性能 (avg over " << N << " iters) ====" << std::endl;
    // 小张量（对应测试点1/4，延迟敏感）
    TimeCase("small_f32_256",  {256}, {256}, {256}, DT_F32, N);
    TimeCase("small_f16_256",  {256}, {256}, {256}, DT_F16, N);
    TimeCase("small_f32_1024", {1024}, {1024}, {1024}, DT_F32, N);
    TimeCase("small_i32_1024", {1024}, {1024}, {1024}, DT_I32, N);
    TimeCase("small_i8_1024",  {1024}, {1024}, {1024}, DT_I8, N);
    // 中等
    TimeCase("mid_f32_65536",  {65536}, {65536}, {65536}, DT_F32, N);
    TimeCase("mid_f16_65536",  {65536}, {65536}, {65536}, DT_F16, N);
    // 大（对应测试点5）
    TimeCase("big_f32_1M",     {1048576}, {1048576}, {1048576}, DT_F32, N);
    TimeCase("big_f16_4M",     {4194304}, {4194304}, {4194304}, DT_F16, N);
    // 广播
    TimeCase("bcast_f32_1Kx1K", {1024, 1024}, {1024}, {1024, 1024}, DT_F32, N);
    ACL_CHECK(aclrtDestroyStream(g_stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    return 0;
}
