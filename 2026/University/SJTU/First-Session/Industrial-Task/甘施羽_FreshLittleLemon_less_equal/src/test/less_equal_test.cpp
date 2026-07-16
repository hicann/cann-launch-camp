// LessEqual 算子测试：通过 aclnn 二段式接口直接调用自定义算子，与 CPU golden 对拍。
#include <acl/acl.h>
#include <aclnn_less_equal.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define ACL_CHECK(call)                                                            \
    do {                                                                           \
        aclError _ret = (call);                                                    \
        if (_ret != ACL_SUCCESS) {                                                 \
            std::cerr << "ACL error " << _ret << " at " << __FILE__ << ":"         \
                      << __LINE__ << std::endl;                                    \
            return false;                                                          \
        }                                                                          \
    } while (0)

#define ACLNN_CHECK(call)                                                          \
    do {                                                                           \
        aclnnStatus _ret = (call);                                                 \
        if (_ret != 0) {                                                           \
            std::cerr << "ACLNN error " << _ret << " at " << __FILE__ << ":"       \
                      << __LINE__ << std::endl;                                    \
            return false;                                                          \
        }                                                                          \
    } while (0)

enum DType { DT_F16, DT_F32, DT_I32, DT_I8 };

static aclDataType AclType(DType t) {
    switch (t) {
        case DT_F16: return ACL_FLOAT16;
        case DT_F32: return ACL_FLOAT;
        case DT_I32: return ACL_INT32;
        case DT_I8:  return ACL_INT8;
    }
    return ACL_FLOAT;
}
static size_t TypeSize(DType t) {
    switch (t) {
        case DT_F16: return 2;
        case DT_F32: return 4;
        case DT_I32: return 4;
        case DT_I8:  return 1;
    }
    return 4;
}

// ---- fp16 <-> fp32 极简转换（够用于测试数据构造与比较）----
static uint16_t F32ToF16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}
static float F16ToF32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

struct Shape {
    std::vector<int64_t> dims;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

// 广播后输出 shape
static Shape BroadcastShape(const Shape &a, const Shape &b) {
    size_t ra = a.dims.size(), rb = b.dims.size();
    size_t r = std::max(ra, rb);
    Shape out;
    out.dims.resize(r);
    for (size_t i = 0; i < r; i++) {
        int64_t da = (i + ra >= r) ? a.dims[i - (r - ra)] : 1;
        int64_t db = (i + rb >= r) ? b.dims[i - (r - rb)] : 1;
        out.dims[i] = std::max(da, db);
    }
    return out;
}

// 计算连续 stride
static std::vector<int64_t> ContigStride(const std::vector<int64_t> &shape) {
    std::vector<int64_t> s(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; i--) s[i] = s[i + 1] * shape[i + 1];
    return s;
}

// CPU golden：广播 x1<=x2，输出到 out(0/1)。以 double 承载数值。
static void GoldenLE(const std::vector<double> &x1, const Shape &s1,
                     const std::vector<double> &x2, const Shape &s2,
                     const Shape &so, std::vector<uint8_t> &out) {
    size_t r = so.dims.size();
    auto st1full = ContigStride(s1.dims);
    auto st2full = ContigStride(s2.dims);
    // padded stride（左补 1，广播维=0）
    std::vector<int64_t> bstr1(r, 0), bstr2(r, 0);
    size_t r1 = s1.dims.size(), r2 = s2.dims.size();
    for (size_t d = 0; d < r; d++) {
        if (d + r1 >= r) {
            size_t i = d - (r - r1);
            bstr1[d] = (s1.dims[i] == 1) ? 0 : st1full[i];
        }
        if (d + r2 >= r) {
            size_t i = d - (r - r2);
            bstr2[d] = (s2.dims[i] == 1) ? 0 : st2full[i];
        }
    }
    int64_t total = so.numel();
    auto ostr = ContigStride(so.dims);
    out.resize(total);
    for (int64_t f = 0; f < total; f++) {
        int64_t rem = f, off1 = 0, off2 = 0;
        for (size_t d = 0; d < r; d++) {
            int64_t idx = rem / ostr[d];
            rem = rem % ostr[d];
            off1 += idx * bstr1[d];
            off2 += idx * bstr2[d];
        }
        out[f] = (x1[off1] <= x2[off2]) ? 1 : 0;
    }
}

// 将 double 值数组编码为设备字节
static std::vector<uint8_t> EncodeInput(const std::vector<double> &v, DType t) {
    std::vector<uint8_t> buf(v.size() * TypeSize(t));
    for (size_t i = 0; i < v.size(); i++) {
        if (t == DT_F32) {
            float f = (float)v[i];
            std::memcpy(&buf[i * 4], &f, 4);
        } else if (t == DT_F16) {
            uint16_t h = F32ToF16((float)v[i]);
            std::memcpy(&buf[i * 2], &h, 2);
        } else if (t == DT_I32) {
            int32_t x = (int32_t)v[i];
            std::memcpy(&buf[i * 4], &x, 4);
        } else {
            int8_t x = (int8_t)v[i];
            std::memcpy(&buf[i], &x, 1);
        }
    }
    return buf;
}

static aclTensor *MakeTensor(const Shape &s, DType t, void *devPtr) {
    auto stride = ContigStride(s.dims);
    return aclCreateTensor(s.dims.data(), s.dims.size(), AclType(t), stride.data(), 0,
                           ACL_FORMAT_ND, s.dims.data(), s.dims.size(), devPtr);
}

static int g_pass = 0, g_fail = 0;
static aclrtStream g_stream = nullptr;

// 单个用例：给定两输入 shape、dtype 与数值生成器
static bool RunCase(const std::string &name, const Shape &s1, const Shape &s2, DType t,
                    // 数值：用 index 生成，覆盖大小关系
                    double (*gen1)(int64_t), double (*gen2)(int64_t)) {
    Shape so = BroadcastShape(s1, s2);
    int64_t n1 = s1.numel(), n2 = s2.numel(), no = so.numel();

    std::vector<double> v1(n1), v2(n2);
    for (int64_t i = 0; i < n1; i++) v1[i] = gen1(i);
    for (int64_t i = 0; i < n2; i++) v2[i] = gen2(i);

    std::vector<uint8_t> golden;
    GoldenLE(v1, s1, v2, s2, so, golden);

    auto b1 = EncodeInput(v1, t);
    auto b2 = EncodeInput(v2, t);

    void *d1 = nullptr, *d2 = nullptr, *dy = nullptr;
    size_t bytes1 = b1.size() ? b1.size() : 1;
    size_t bytes2 = b2.size() ? b2.size() : 1;
    size_t bytesY = (size_t)(no ? no : 1);  // bool = 1 byte
    ACL_CHECK(aclrtMalloc(&d1, bytes1, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d2, bytes2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&dy, bytesY, ACL_MEM_MALLOC_HUGE_FIRST));
    if (b1.size()) ACL_CHECK(aclrtMemcpy(d1, bytes1, b1.data(), b1.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    if (b2.size()) ACL_CHECK(aclrtMemcpy(d2, bytes2, b2.data(), b2.size(), ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *x1T = MakeTensor(s1, t, d1);
    aclTensor *x2T = MakeTensor(s2, t, d2);
    aclTensor *yT = MakeTensor(so, DT_F16 /*placeholder*/, dy);
    // y 必须是 bool；单独用 bool 类型创建
    aclDestroyTensor(yT);
    {
        auto stride = ContigStride(so.dims);
        yT = aclCreateTensor(so.dims.data(), so.dims.size(), ACL_BOOL, stride.data(), 0,
                             ACL_FORMAT_ND, so.dims.data(), so.dims.size(), dy);
    }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    bool ok = true;
    const char *stage = "";
    aclnnStatus st = aclnnLessEqualGetWorkspaceSize(x1T, x2T, yT, &wsSize, &exec);
    if (st != 0) { ok = false; stage = "GetWorkspaceSize"; }
    void *ws = nullptr;
    if (ok && wsSize > 0) {
        if (aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { ok = false; stage = "wsMalloc"; }
    }
    if (ok) {
        aclnnStatus r = aclnnLessEqual(ws, wsSize, exec, g_stream);
        if (r != 0) { ok = false; stage = "aclnnLessEqual"; }
    }
    if (ok) {
        aclError r = aclrtSynchronizeStream(g_stream);
        if (r != ACL_SUCCESS) { ok = false; stage = "Synchronize"; }
    }

    std::vector<uint8_t> hostY(bytesY, 0xEE);
    if (ok && no > 0) {
        if (aclrtMemcpy(hostY.data(), bytesY, dy, bytesY, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ok = false;
    }

    // 比较（bool 精确匹配）
    int64_t mism = 0;
    int64_t firstBad = -1;
    if (ok) {
        for (int64_t i = 0; i < no; i++) {
            uint8_t got = hostY[i] ? 1 : 0;
            if (got != golden[i]) {
                mism++;
                if (firstBad < 0) firstBad = i;
            }
        }
    }
    bool pass = ok && (mism == 0);

    std::cout << "  " << (pass ? "PASS" : "FAIL") << "  " << name
              << "  (numel=" << no << ", mismatch=" << mism;
    if (!ok) std::cout << ", FAILSTAGE=" << stage;
    if (firstBad >= 0)
        std::cout << ", first@" << firstBad << " got=" << (int)(hostY[firstBad] ? 1 : 0)
                  << " exp=" << (int)golden[firstBad];
    std::cout << ")" << std::endl;

    if (ws) aclrtFree(ws);
    aclDestroyTensor(x1T);
    aclDestroyTensor(x2T);
    aclDestroyTensor(yT);
    aclrtFree(d1);
    aclrtFree(d2);
    aclrtFree(dy);

    if (pass) g_pass++; else g_fail++;
    return pass;
}

// 数值生成器
static double genRamp(int64_t i) { return (double)(i % 13) - 6.0; }      // -6..6 循环
static double genConst2(int64_t) { return 2.0; }
static double genRampB(int64_t i) { return (double)(i % 7) - 3.0; }
static double genI8(int64_t i) { return (double)((i % 255) - 128); }     // int8 全域
static double genI8b(int64_t i) { return (double)((i * 3 % 255) - 128); }
// 极值生成器
static double genI8min(int64_t) { return -128.0; }
static double genI8max(int64_t) { return 127.0; }
static double genI32min(int64_t) { return -2147483648.0; }
static double genI32max(int64_t) { return 2147483647.0; }
static double genF32big(int64_t i) { return (i % 2) ? 1e30 : -1e30; }
static double genF32small(int64_t i) { return (i % 3) - 1.0; }

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) { std::cerr << "aclInit failed\n"; return 1; }
    aclrtSetDevice(0);
    aclrtCreateStream(&g_stream);

    std::cout << "==== LessEqual 测试 ====" << std::endl;

    // ---- 等形状 fast path，多 dtype ----
    RunCase("f32_1d_16", {{16}}, {{16}}, DT_F32, genRamp, genRampB);
    RunCase("f32_1d_unalign_17", {{17}}, {{17}}, DT_F32, genRamp, genRampB);
    RunCase("f16_1d_100", {{100}}, {{100}}, DT_F16, genRamp, genRampB);
    RunCase("i32_1d_50", {{50}}, {{50}}, DT_I32, genRamp, genRampB);
    RunCase("i8_1d_300", {{300}}, {{300}}, DT_I8, genI8, genI8b);
    RunCase("f32_2d_3x4", {{3, 4}}, {{3, 4}}, DT_F32, genRamp, genConst2);
    RunCase("f32_4d", {{2, 3, 4, 5}}, {{2, 3, 4, 5}}, DT_F32, genRamp, genRampB);
    RunCase("f16_5d_batch", {{2, 2, 3, 4, 5}}, {{2, 2, 3, 4, 5}}, DT_F16, genRamp, genRampB);
    RunCase("i32_equal", {{64}}, {{64}}, DT_I32, genRamp, genRamp);  // x1==x2 全 True
    RunCase("f32_large_10000", {{10000}}, {{10000}}, DT_F32, genRamp, genRampB);
    RunCase("f32_n1", {{1}}, {{1}}, DT_F32, genRamp, genConst2);

    // ---- 广播 path ----
    RunCase("bcast_scalar", {{2, 2}}, {{1}}, DT_F32, genRamp, genConst2);           // 张量 vs 标量
    RunCase("bcast_vec_mat", {{2, 2}}, {{2}}, DT_F32, genRamp, genConst2);          // 矩阵 vs 向量(尾维)
    RunCase("bcast_col", {{3, 4}}, {{3, 1}}, DT_F32, genRamp, genRampB);            // 尾维广播
    RunCase("bcast_row", {{3, 4}}, {{1, 4}}, DT_F32, genRamp, genRampB);            // 外维广播
    RunCase("bcast_3d", {{2, 3, 4}}, {{3, 1}}, DT_F32, genRamp, genRampB);          // 高维不同 rank
    RunCase("bcast_both", {{4, 1, 5}}, {{1, 3, 1}}, DT_F32, genRamp, genRampB);     // 双向广播
    RunCase("bcast_i32", {{5, 6}}, {{6}}, DT_I32, genRamp, genRampB);
    RunCase("bcast_i8", {{4, 300}}, {{300}}, DT_I8, genI8, genI8b);
    RunCase("bcast_f16", {{3, 100}}, {{1, 100}}, DT_F16, genRamp, genRampB);
    RunCase("bcast_scalar_lhs", {{1}}, {{3, 4}}, DT_F32, genConst2, genRamp);       // 标量在左

    // ---- 边界数值 ----
    RunCase("edge_i8_min_vs_max", {{64}}, {{64}}, DT_I8, genI8min, genI8max);       // -128 <= 127
    RunCase("edge_i8_max_vs_min", {{64}}, {{64}}, DT_I8, genI8max, genI8min);       // 127 <= -128 (F)
    RunCase("edge_i32_min_max", {{40}}, {{40}}, DT_I32, genI32min, genI32max);
    RunCase("edge_i32_max_min", {{40}}, {{40}}, DT_I32, genI32max, genI32min);
    RunCase("edge_i32_equal_max", {{40}}, {{40}}, DT_I32, genI32max, genI32max);    // 相等 -> True
    RunCase("edge_f32_bignum", {{128}}, {{128}}, DT_F32, genF32big, genF32small);
    RunCase("edge_f16_ramp", {{500}}, {{500}}, DT_F16, genRamp, genRampB);

    // ---- 空张量 ----
    RunCase("empty_1d", {{0}}, {{0}}, DT_F32, genRamp, genRampB);
    RunCase("empty_2d", {{0, 4}}, {{0, 4}}, DT_F32, genRamp, genRampB);

    // ---- 大规模 & 非对齐组合 ----
    RunCase("large_i32_9999", {{9999}}, {{9999}}, DT_I32, genRamp, genRampB);
    RunCase("bcast_large", {{100, 100}}, {{100}}, DT_F32, genRamp, genRampB);
    RunCase("unalign_3d_7x9x13", {{7, 9, 13}}, {{7, 9, 13}}, DT_F16, genRamp, genRampB);

    std::cout << "==== 汇总: PASS=" << g_pass << " FAIL=" << g_fail << " ====" << std::endl;

    aclrtDestroyStream(g_stream);
    aclrtResetDevice(0);
    aclFinalize();
    return g_fail == 0 ? 0 : 1;
}
