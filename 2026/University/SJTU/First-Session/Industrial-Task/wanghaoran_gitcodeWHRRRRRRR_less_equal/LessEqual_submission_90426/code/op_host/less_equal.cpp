// Host 侧：InferShape + Tiling（自己实现，不折叠维度）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {

    static inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
        return (b == 0) ? 0 : (a + b - 1) / b;
    }
    static inline uint32_t AlignUpU(uint32_t a, uint32_t m) {
        return (m == 0) ? a : ((a + m - 1) / m) * m;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (num_cores_aiv == 0) { num_cores_aiv = 1; }

        // 数据类型选择（保持框架既定写法）
        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x1 = tensor_x1->GetDataType();
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        // 读取原始形状（OriginShape，对应 InferShape 的输出布局）
        const gert::Shape &x1s = context->GetInputShape(0)->GetOriginShape();
        const gert::Shape &x2s = context->GetInputShape(1)->GetOriginShape();
        const gert::Shape &ys  = context->GetOutputShape(0)->GetOriginShape();

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();

        size_t n1 = x1s.GetDimNum();
        size_t n2 = x2s.GetDimNum();
        int64_t outElems = ys.GetShapeSize();

        // ---- 空张量：kernel 早返回 ----
        if (outElems == 0) {
            tiling->mode = 0;
            tiling->totalElements = 0;
            tiling->tileSize = 4096;
            tiling->blockDim = 1;
            tiling->perCore = 0;
            tiling->ndim = 0;
            tiling->lastDimLen = 0;
            tiling->totalRows = 0;
            for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
                tiling->outShape[i] = 0;
                tiling->x1Stride[i] = 0;
                tiling->x2Stride[i] = 0;
            }
            context->SetBlockDim(1);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ---- FAST 检测：rank 相同且每一维都相等（无 size-1 广播）----
        bool fast = (n1 == n2);
        if (fast) {
            for (size_t i = 0; i < n1; ++i) {
                if (x1s.GetDim(i) != x2s.GetDim(i)) { fast = false; break; }
            }
        }

        const uint32_t TILE = 4096;

        // mode=0(FAST)/2(SCALAR) 共用的按元素切核 tiling。
        // ndimVal：FAST 传 0；SCALAR 传 1(x1标量)/2(x2标量)，复用 ndim 字段承载，
        // 避免新增结构体字段导致三文件不同步。
        auto fastLikeTiling = [&](uint32_t mode, uint32_t total, uint32_t ndimVal) {
            uint32_t perCore = AlignUpU(CeilDivU(total, num_cores_aiv), 32);
            if (perCore == 0) { perCore = 32; }
            uint32_t blockDim = CeilDivU(total, perCore);
            if (blockDim == 0) { blockDim = 1; }

            tiling->mode = mode;
            tiling->totalElements = total;
            tiling->tileSize = TILE;
            tiling->blockDim = blockDim;
            tiling->perCore = perCore;
            tiling->ndim = ndimVal;
            tiling->lastDimLen = 0;
            tiling->totalRows = 0;
            for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
                tiling->outShape[i] = 0;
                tiling->x1Stride[i] = 0;
                tiling->x2Stride[i] = 0;
            }
            context->SetBlockDim(blockDim);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
        };

        if (fast) {
            uint32_t total = static_cast<uint32_t>(outElems);
            // perCore 对齐 32 个元素：保证每核 GM 起点 32 字节对齐（任意 dtype）
            fastLikeTiling(0, total, 0);
            return ge::GRAPH_SUCCESS;
        }

        // ---- SCALAR 检测：一个输入为标量，按输出总元素数切核 ----
        bool x1Scalar = (x1s.GetShapeSize() == 1);
        bool x2Scalar = (x2s.GetShapeSize() == 1);
        if (x1Scalar || x2Scalar) {
            uint32_t total = static_cast<uint32_t>(outElems);
            fastLikeTiling(2, total, x1Scalar ? 1u : 2u);
            return ge::GRAPH_SUCCESS;
        }

        // ---- BCAST 路径：右对齐广播，计算 od / 广播 stride（广播维 stride=0）----
        size_t nmax = (n1 > n2) ? n1 : n2;
        int64_t d1[64];
        int64_t d2[64];
        int64_t od[64];
        for (size_t i = 0; i < nmax; ++i) {
            int64_t a = (i + n1 >= nmax) ? x1s.GetDim(i - (nmax - n1)) : 1;
            int64_t b = (i + n2 >= nmax) ? x2s.GetDim(i - (nmax - n2)) : 1;
            d1[i] = a;
            d2[i] = b;
            od[i] = (a == 1) ? b : a;
        }

        int64_t s1[64];
        int64_t s2[64];
        {
            int64_t acc1 = 1;
            for (int i = (int)nmax - 1; i >= 0; --i) {
                if (d1[i] == 1) { s1[i] = 0; }
                else { s1[i] = acc1; acc1 *= d1[i]; }
            }
            int64_t acc2 = 1;
            for (int i = (int)nmax - 1; i >= 0; --i) {
                if (d2[i] == 1) { s2[i] = 0; }
                else { s2[i] = acc2; acc2 *= d2[i]; }
            }
        }

        // ---- 折叠相邻可合并维（从内向外交叠）----
        // 合并条件：
        // 1) 两边在该维都广播（s1==0 && s2==0），且已合并的内邻维也广播；
        // 2) 两边在该维都连续非广播（s1[i] == s1[inner]*od[inner] && s2 同）。
        // 折叠后 ndim 更小、lastDimLen 更宽，BCAST 路径每行处理更多连续元素。
        uint32_t cShape[64];
        int64_t  cs1[64];
        int64_t  cs2[64];
        int cnt = 0;
        for (int i = (int)nmax - 1; i >= 0; --i) {
            if (cnt > 0) {
                int j = cnt - 1;
                bool bothBcast = (s1[i] == 0 && s2[i] == 0 && cs1[j] == 0 && cs2[j] == 0);
                bool contig =
                    (s1[i] != 0 && cs1[j] != 0 && s1[i] == cs1[j] * (int64_t)cShape[j]) &&
                    (s2[i] != 0 && cs2[j] != 0 && s2[i] == cs2[j] * (int64_t)cShape[j]);
                if (bothBcast) {
                    cShape[j] = (uint32_t)((int64_t)cShape[j] * od[i]);
                    continue;
                }
                if (contig) {
                    cShape[j] = (uint32_t)((int64_t)cShape[j] * od[i]);
                    cs1[j] = s1[i];
                    cs2[j] = s2[i];
                    continue;
                }
            }
            cShape[cnt] = (uint32_t)od[i];
            cs1[cnt] = s1[i];
            cs2[cnt] = s2[i];
            cnt++;
        }
        // 当前 cShape 是 innermost-first，反转为 outermost-first
        int ndimI = cnt;
        int64_t fShape[64];
        int64_t fs1[64];
        int64_t fs2[64];
        for (int i = 0; i < ndimI; ++i) {
            fShape[i] = cShape[ndimI - 1 - i];
            fs1[i] = cs1[ndimI - 1 - i];
            fs2[i] = cs2[ndimI - 1 - i];
        }

        uint32_t ndim = static_cast<uint32_t>(ndimI);
        uint32_t lastDimLen = static_cast<uint32_t>(fShape[ndim - 1]);
        uint32_t totalRows = 1;
        for (uint32_t i = 0; i + 1 < ndim; ++i) {
            totalRows *= static_cast<uint32_t>(fShape[i]);
        }

        uint32_t perCore = CeilDivU(totalRows, num_cores_aiv);
        if (perCore == 0) { perCore = 1; }
        uint32_t blockDim = CeilDivU(totalRows, perCore);
        if (blockDim == 0) { blockDim = 1; }

        tiling->mode = 1;
        tiling->totalElements = static_cast<uint32_t>(outElems);
        tiling->tileSize = TILE;
        tiling->blockDim = blockDim;
        tiling->perCore = perCore;
        tiling->ndim = ndim;
        tiling->lastDimLen = lastDimLen;
        tiling->totalRows = totalRows;
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            if (i < ndim) {
                tiling->outShape[i] = static_cast<uint32_t>(fShape[i]);
                tiling->x1Stride[i] = static_cast<int32_t>(fs1[i]);
                tiling->x2Stride[i] = static_cast<int32_t>(fs2[i]);
            } else {
                tiling->outShape[i] = 0;
                tiling->x1Stride[i] = 0;
                tiling->x2Stride[i] = 0;
            }
        }
        context->SetBlockDim(blockDim);
        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *s1 = context->GetInputShape(0);
        const gert::Shape *s2 = context->GetInputShape(1);
        gert::Shape *y = context->GetOutputShape(0);
        size_t n1 = s1->GetDimNum();
        size_t n2 = s2->GetDimNum();
        size_t n = (n1 > n2) ? n1 : n2;
        y->SetDimNum(n);
        for (size_t i = 0; i < n; ++i) {
            // i = 0 对应最外层（tail 维），需要右对齐
            int64_t dd1 = (i < n1) ? s1->GetDim(n1 - 1 - i) : 1;
            int64_t dd2 = (i < n2) ? s2->GetDim(n2 - 1 - i) : 1;
            int64_t o = (dd1 == 1) ? dd2 : dd1;
            y->SetDim(n - 1 - i, o);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_BOOL);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

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
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LessEqual);
}  // namespace ops
