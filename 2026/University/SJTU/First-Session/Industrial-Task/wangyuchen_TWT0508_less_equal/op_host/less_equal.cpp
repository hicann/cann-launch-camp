#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace {

constexpr uint32_t kAlign    = 256;   // tile size alignment
constexpr uint32_t kMaxTile  = 4096;  // max elements per tile
constexpr uint32_t kAligned  = 32;    // per-core work alignment
constexpr uint64_t kUbSlack  = 2048;  // UB reserved headroom

inline bool MulOverflow(uint64_t a, uint64_t b, uint64_t &out) {
    if (a && b > UINT64_MAX / a) return false;
    out = a * b; return true;
}

inline uint64_t CeilDiv(uint64_t n, uint64_t d) {
    return d ? (n + d - 1) / d : 0;
}

inline uint64_t RoundUp(uint64_t v, uint64_t m) {
    return m ? CeilDiv(v, m) * m : v;
}

inline bool IsDtypeOk(ge::DataType dt) {
    return dt == ge::DT_FLOAT16 || dt == ge::DT_FLOAT ||
           dt == ge::DT_INT32   || dt == ge::DT_INT8;
}

// ---- broadcast shape and stride ----

// get the size of an input along a given axis, padding with 1 on the left
uint64_t SafeDim(const gert::Shape &s, uint32_t outRank, uint32_t axis) {
    uint32_t inRank = static_cast<uint32_t>(s.GetDimNum());
    if (axis + inRank < outRank) return 1;
    int64_t v = s.GetDim(axis + inRank - outRank);
    return v < 0 ? UINT64_MAX : static_cast<uint64_t>(v);
}

// compute the output shape after broadcasting
bool BuildShape(const gert::Shape &a, const gert::Shape &b,
                uint64_t raw[LESS_EQUAL_RAW_DIMS],
                uint32_t &rank, uint64_t &total) {
    uint32_t ra = static_cast<uint32_t>(a.GetDimNum());
    uint32_t rb = static_cast<uint32_t>(b.GetDimNum());
    rank = std::max(ra, rb);
    if (rank > LESS_EQUAL_RAW_DIMS) return false;
    total = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t da = SafeDim(a, rank, i);
        uint64_t db = SafeDim(b, rank, i);
        if (da == UINT64_MAX || db == UINT64_MAX) return false;
        if (da != db && da != 1 && db != 1)  return false;
        uint64_t d = (da == 1) ? db : da;
        raw[i] = d;
        if (d > static_cast<uint64_t>(INT64_MAX)) return false;
        if (!MulOverflow(total, d, total))       return false;
    }
    return true;
}

// compute broadcast strides for one input
// stride = 0 means this dim is broadcast (size is 1 but output size > 1)
bool BuildStrides(const gert::Shape &s, const uint64_t raw[LESS_EQUAL_RAW_DIMS],
                  uint32_t rank, uint64_t out[LESS_EQUAL_RAW_DIMS]) {
    uint32_t inRank = static_cast<uint32_t>(s.GetDimNum());
    if (inRank > rank) return false;
    for (uint32_t i = 0; i < rank; ++i) out[i] = 0;
    uint64_t step = 1;
    for (uint32_t r = inRank; r > 0; --r) {
        uint32_t ia = r - 1;
        uint32_t oa = rank - inRank + ia;
        int64_t dv  = s.GetDim(ia);
        if (dv < 0) return false;
        uint64_t d = static_cast<uint64_t>(dv);
        out[oa] = (d == 1 && raw[oa] != 1) ? 0 : step;
        if (!MulOverflow(step, d, step)) return false;
    }
    return true;
}

// ---- dimension collapse ----
// merge adjacent dims that are both scalar-broadcast or both contiguous
// after collapse, the last dim is always a contiguous run (or scalar),
// so the kernel never needs per-element scatter/gather

bool Collapse(const uint64_t shape[LESS_EQUAL_RAW_DIMS],
              const uint64_t st1[LESS_EQUAL_RAW_DIMS],
              const uint64_t st2[LESS_EQUAL_RAW_DIMS],
              uint32_t rawRank, LessEqualTilingData &td) {
    if (rawRank == 0) { td.ndim = 0; return true; }

    uint64_t sh[LESS_EQUAL_RAW_DIMS];
    uint64_t p1[LESS_EQUAL_RAW_DIMS];
    uint64_t p2[LESS_EQUAL_RAW_DIMS];
    uint32_t cr = 1;
    sh[0] = shape[0]; p1[0] = st1[0]; p2[0] = st2[0];

    for (uint32_t i = 1; i < rawRank; ++i) {
        uint32_t prev = cr - 1;
        uint64_t inner = shape[i];

        bool bothScalar = (p1[prev] == 0 && st1[i] == 0 &&
                           p2[prev] == 0 && st2[i] == 0);
        bool bothDense  = (p1[prev] != 0 && st1[i] != 0 &&
                           p1[prev] == st1[i] * inner &&
                           p2[prev] != 0 && st2[i] != 0 &&
                           p2[prev] == st2[i] * inner);
        if (bothScalar || bothDense) {
            if (!MulOverflow(sh[prev], inner, sh[prev])) return false;
            p1[prev] = st1[i];
            p2[prev] = st2[i];
        } else {
            sh[cr] = inner; p1[cr] = st1[i]; p2[cr] = st2[i];
            ++cr;
        }
    }

    if (cr > LESS_EQUAL_MAX_DIMS) return false;
    td.ndim = cr;
    for (uint32_t i = 0; i < cr; ++i) {
        if (sh[i] > UINT32_MAX || p1[i] > UINT32_MAX || p2[i] > UINT32_MAX)
            return false;
        td.shape[i] = static_cast<uint32_t>(sh[i]);
        td.s1[i]    = static_cast<uint32_t>(p1[i]);
        td.s2[i]    = static_cast<uint32_t>(p2[i]);
    }
    return true;
}

// ---- UB budget and tile selection ----

uint64_t CalcUb(ge::DataType dt, uint32_t tile) {
    uint32_t ib = static_cast<uint32_t>(ge::GetSizeByDataType(dt));
    // queues: x1[2] + x2[2] + y[2] = 4*input_bytes + 2*output_bytes
    uint64_t q  = static_cast<uint64_t>(tile) * (4ULL * ib + 2ULL);
    // common compute: ones + zeros + outHalf = 3 half arrays = 6 bytes/elem
    uint64_t cc = static_cast<uint64_t>(tile) * 6ULL;
    // extra for int8/int32 path
    uint64_t dc = (dt == ge::DT_INT8 || dt == ge::DT_INT32)
                    ? static_cast<uint64_t>(tile) * 4ULL : 0ULL;
    // compare bitmask: tile bits, rounded up to 32B
    uint64_t mk = RoundUp(CeilDiv(tile, 8), 32);
    return q + cc + dc + mk + kUbSlack;
}

uint32_t PickTile(ge::DataType dt, uint64_t perCoreWork, uint64_t ub) {
    uint64_t w = std::max<uint64_t>(perCoreWork, kAlign);
    uint32_t t = static_cast<uint32_t>(
        std::min<uint64_t>(kMaxTile, RoundUp(w, kAlign)));
    while (t > kAlign && CalcUb(dt, t) > ub) t -= kAlign;
    return t;
}

}  // namespace

// ==================== host-side tiling ====================

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
    if (!ctx) return ge::GRAPH_FAILED;

    const gert::Tensor *t1 = ctx->GetRequiredInputTensor(0);
    const gert::Tensor *t2 = ctx->GetRequiredInputTensor(1);
    const auto *st1 = ctx->GetInputShape(0);
    const auto *st2 = ctx->GetInputShape(1);
    LessEqualTilingData *td = ctx->GetTilingData<LessEqualTilingData>();
    if (!t1 || !t2 || !st1 || !st2 || !td) return ge::GRAPH_FAILED;

    ge::DataType dt = t1->GetDataType();
    if (dt != t2->GetDataType() || !IsDtypeOk(dt)) return ge::GRAPH_FAILED;
    ASCENDC_TPL_SEL_PARAM(ctx, static_cast<uint32_t>(dt));

    const gert::Shape &sh1 = st1->GetStorageShape();
    const gert::Shape &sh2 = st2->GetStorageShape();

    // step 1: broadcast shape and strides
    uint64_t rawShape[LESS_EQUAL_RAW_DIMS] = {};
    uint64_t rawS1[LESS_EQUAL_RAW_DIMS]    = {};
    uint64_t rawS2[LESS_EQUAL_RAW_DIMS]    = {};
    uint32_t rawRank = 0;
    uint64_t total   = 0;

    if (!BuildShape(sh1, sh2, rawShape, rawRank, total) ||
        !BuildStrides(sh1, rawShape, rawRank, rawS1) ||
        !BuildStrides(sh2, rawShape, rawRank, rawS2) ||
        total > UINT32_MAX)
        return ge::GRAPH_FAILED;

    *td = {};
    td->count = static_cast<uint32_t>(total);

    // empty tensor
    if (total == 0) {
        td->mode  = LE_MODE_DIRECT;
        td->tile  = kAlign;
        td->cores = 1;
        ctx->SetBlockDim(1);
        size_t *ws = ctx->GetWorkspaceSizes(1);
        if (ws) ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // hardware info
    auto plat  = platform_ascendc::PlatformAscendC(ctx->GetPlatformInfo());
    int32_t nc = plat.GetCoreNumAiv();
    uint32_t ncores = (nc > 0) ? static_cast<uint32_t>(nc) : 1;
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);

    // check if shapes are identical
    bool same = (sh1.GetDimNum() == sh2.GetDimNum());
    if (same) {
        for (size_t i = 0; i < sh1.GetDimNum(); ++i)
            if (sh1.GetDim(i) != sh2.GetDim(i)) { same = false; break; }
    }

    if (same) {
        // ---- direct mode ----
        td->mode    = LE_MODE_DIRECT;
        uint64_t t  = CeilDiv(total, ncores);
        uint64_t ap = RoundUp(t, kAligned);
        if (ap > UINT32_MAX) return ge::GRAPH_FAILED;
        td->perCore = static_cast<uint32_t>(ap);
        td->cores   = static_cast<uint32_t>(CeilDiv(total, ap));
        uint64_t pw = std::min<uint64_t>(ap, total);
        td->tile    = PickTile(dt, pw, ub);
    } else {
        // ---- broadcast mode ----
        td->mode = LE_MODE_BROADCAST;
        if (!Collapse(rawShape, rawS1, rawS2, rawRank, *td) || td->ndim == 0)
            return ge::GRAPH_FAILED;
        // last dim is the contiguous run, leading dims form "rows"
        td->tail = td->shape[td->ndim - 1];
        uint64_t rows = 1;
        for (uint32_t i = 0; i + 1 < td->ndim; ++i)
            if (!MulOverflow(rows, td->shape[i], rows)) return ge::GRAPH_FAILED;
        if (rows > UINT32_MAX) return ge::GRAPH_FAILED;
        td->rows    = static_cast<uint32_t>(rows);
        td->perCore = static_cast<uint32_t>(CeilDiv(rows, ncores));
        td->cores   = static_cast<uint32_t>(CeilDiv(rows, td->perCore));
        uint64_t pw = std::min<uint64_t>(td->tail,
                        static_cast<uint64_t>(td->perCore) * td->tail);
        td->tile    = PickTile(dt, pw, ub);
    }

    ctx->SetBlockDim(td->cores);
    size_t *ws = ctx->GetWorkspaceSizes(1);
    if (ws) ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

// ==================== shape and type inference ====================

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *ctx) {
    if (!ctx) return GRAPH_FAILED;
    const gert::Shape *s1 = ctx->GetInputShape(0);
    const gert::Shape *s2 = ctx->GetInputShape(1);
    gert::Shape *out = ctx->GetOutputShape(0);
    uint64_t raw[LESS_EQUAL_RAW_DIMS];
    uint32_t rank; uint64_t total;
    if (!s1 || !s2 || !out ||
        !BuildShape(*s1, *s2, raw, rank, total))
        return GRAPH_FAILED;
    out->SetDimNum(rank);
    for (uint32_t i = 0; i < rank; ++i)
        out->SetDim(i, static_cast<int64_t>(raw[i]));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
    if (!ctx) return GRAPH_FAILED;
    ctx->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}

}  // namespace ge

// ==================== operator registration ====================

namespace ops {

class LessEqual : public OpDef {
public:
    explicit LessEqual(const char *name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(LessEqual);

}  // namespace ops
