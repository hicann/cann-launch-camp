// Host侧Tiling实现 - 优化版：精简冗余初始化，简化广播路径
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

        // dtype 选择
        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x1 = tensor_x1->GetDataType();
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        // 通过 StorageShape 读取输入形状
        const gert::Shape &x1s = context->GetInputShape(0)->GetStorageShape();
        const gert::Shape &x2s = context->GetInputShape(1)->GetStorageShape();
        const gert::Shape &ys  = context->GetOutputShape(0)->GetStorageShape();

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();

        size_t n1 = x1s.GetDimNum();
        size_t n2 = x2s.GetDimNum();
        int64_t outElems = ys.GetShapeSize();

        // ---- 空张量：kernel 立即返回 ----
        if (outElems == 0) {
            tiling->mode = 0;
            tiling->totalElements = 0;
            tiling->tileSize = 4096;
            tiling->blockDim = 1;
            tiling->perCore = 0;
            tiling->ndim = 0;
            tiling->lastDimLen = 0;
            tiling->totalRows = 0;
            tiling->bufferNum = 1;
            context->SetBlockDim(1);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ---- FAST 检测：相同秩且所有维度相等（无 size-1 广播）----
        bool fast = (n1 == n2);
        if (fast) {
            for (size_t i = 0; i < n1; ++i) {
                if (x1s.GetDim(i) != x2s.GetDim(i)) { fast = false; break; }
            }
        }

        if (fast) {
            uint32_t total = static_cast<uint32_t>(outElems);
            // perCore 对齐到 32 元素：保证任意数据类型的 GM 起始地址对齐
            uint32_t perCore = AlignUpU(CeilDivU(total, num_cores_aiv), 32);
            if (perCore == 0) { perCore = 32; }
            uint32_t blockDim = CeilDivU(total, perCore);
            if (blockDim == 0) { blockDim = 1; }

            // 自适应 tile：基于 perCore 工作量，减少小张量的缓冲/初始化开销
            // 下限 256 (Compare 的 RoundUp256 要求)，上限 4096
            uint32_t TILE = (perCore <= 256u) ? 256u :
                            (perCore <= 4096u) ? AlignUpU(perCore, 256u) : 4096u;

            tiling->mode = 0;
            tiling->totalElements = total;
            tiling->tileSize = TILE;
            tiling->blockDim = blockDim;
            tiling->perCore = perCore;
            tiling->bufferNum = (perCore > TILE) ? 2U : 1U;
            // BCAST 字段不需要初始化 — kernel 在 mode==0 时不读取它们
            tiling->ndim = 0;
            tiling->lastDimLen = 0;
            tiling->totalRows = 0;
            context->SetBlockDim(blockDim);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ---- BCAST 路径 ----
        size_t nmax = (n1 > n2) ? n1 : n2;

        // 右对齐维度数组。d1[i]/d2[i] 中 i=0 为最外层维度
        int64_t d1[64], d2[64], od[64];
        for (size_t i = 0; i < nmax; ++i) {
            int64_t a = (i + n1 >= nmax) ? x1s.GetDim(i - (nmax - n1)) : 1;
            int64_t b = (i + n2 >= nmax) ? x2s.GetDim(i - (nmax - n2)) : 1;
            d1[i] = a;
            d2[i] = b;
            od[i] = (a == 1) ? b : a;
        }

        // 计算每个输入的广播步长：stride=0 表示该输入维度为 1（广播）
        int64_t s1[64], s2[64];
        {
            int64_t acc = 1;
            for (int i = (int)nmax - 1; i >= 0; --i) {
                s1[i] = (d1[i] == 1) ? 0 : acc;
                if (d1[i] != 1) { acc *= d1[i]; }
            }
        }
        {
            int64_t acc = 1;
            for (int i = (int)nmax - 1; i >= 0; --i) {
                s2[i] = (d2[i] == 1) ? 0 : acc;
                if (d2[i] != 1) { acc *= d2[i]; }
            }
        }

        // 合并相邻维度：
        // - 两者均为广播 (s1==0 && s2==0)
        // - 两者均为连续非广播 (contig)
        uint32_t cShape[64];
        int64_t  cs1[64], cs2[64];
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
                    // 不覆盖 cs1[j]/cs2[j]：合并后应保留内层 stride
                    continue;
                }
            }
            cShape[cnt] = (uint32_t)od[i];
            cs1[cnt] = s1[i];
            cs2[cnt] = s2[i];
            cnt++;
        }

        // 反转：从内层优先变为外层优先
        int ndim = cnt;
        uint32_t fShape[64];
        int64_t  fs1[64], fs2[64];
        for (int i = 0; i < ndim; ++i) {
            fShape[i] = cShape[ndim - 1 - i];
            fs1[i] = cs1[ndim - 1 - i];
            fs2[i] = cs2[ndim - 1 - i];
        }

        // 极端情况保护：若合并后仍超出 LE_MAX_DIM，截断丢弃外层维度
        if (ndim > (int)LE_MAX_DIM) { ndim = (int)LE_MAX_DIM; }

        uint32_t lastDimLen = fShape[ndim - 1];
        uint32_t totalRows = 1;
        for (int i = 0; i < ndim - 1; ++i) { totalRows *= fShape[i]; }

        uint32_t perCore = CeilDivU(totalRows, num_cores_aiv);
        if (perCore == 0) { perCore = 1; }
        uint32_t blockDim = CeilDivU(totalRows, perCore);
        if (blockDim == 0) { blockDim = 1; }

        // 自适应 tile：基于最内层维度长度，减少小张量的缓冲/初始化开销
        // 下限 256 (Compare 的 RoundUp256 要求)，上限 4096
        uint32_t TILE = (lastDimLen <= 256u) ? 256u :
                        (lastDimLen <= 4096u) ? AlignUpU(lastDimLen, 256u) : 4096u;

        tiling->mode = 1;
        tiling->totalElements = static_cast<uint32_t>(outElems);
        tiling->tileSize = TILE;
        tiling->blockDim = blockDim;
        tiling->perCore = perCore;
        tiling->bufferNum = (lastDimLen > TILE) ? 2U : 1U;
        tiling->ndim = (uint32_t)ndim;
        tiling->lastDimLen = lastDimLen;
        tiling->totalRows = totalRows;
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            tiling->outShape[i]  = ((int)i < ndim) ? fShape[i] : 0;
            tiling->x1Stride[i]  = ((int)i < ndim) ? (int32_t)fs1[i] : 0;
            tiling->x2Stride[i]  = ((int)i < ndim) ? (int32_t)fs2[i] : 0;
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
            int64_t d1 = (i < n1) ? s1->GetDim(n1 - 1 - i) : 1;
            int64_t d2 = (i < n2) ? s2->GetDim(n2 - 1 - i) : 1;
            y->SetDim(n - 1 - i, (d1 == 1) ? d2 : d1);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
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
