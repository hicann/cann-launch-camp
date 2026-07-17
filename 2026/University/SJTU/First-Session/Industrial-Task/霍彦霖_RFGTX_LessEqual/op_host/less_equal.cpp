// Host侧Tiling实现
#include <vector>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv < 1) {
            num_cores_aiv = 1;
        }

        // 输入张量与数据类型
        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x1 = tensor_x1->GetDataType();
        const gert::Shape &s1 = tensor_x1->GetStorageShape();
        const gert::Shape &s2 = tensor_x2->GetStorageShape();

        // ---- 1. 右对齐计算广播后形状 ----
        size_t r1 = s1.GetDimNum();
        size_t r2 = s2.GetDimNum();
        size_t r = r1 > r2 ? r1 : r2;
        if (r == 0) {
            r = 1;  // 标量视为 shape=[1]
        }
        std::vector<uint32_t> outv(r), a1(r), a2(r);
        std::vector<uint8_t> b1(r), b2(r);
        for (size_t i = 0; i < r; i++) {
            int64_t d1 = (i < r1) ? s1.GetDim(r1 - 1 - i) : 1;
            int64_t d2 = (i < r2) ? s2.GetDim(r2 - 1 - i) : 1;
            if (d1 < 0) d1 = 1;  // 兜底(动态shape不在本题范围)
            if (d2 < 0) d2 = 1;
            int64_t o;
            if (d1 == d2)      o = d1;
            else if (d1 == 1)  o = d2;
            else if (d2 == 1)  o = d1;
            else               return ge::GRAPH_FAILED;  // 无法广播
            size_t pos = r - 1 - i;
            outv[pos] = static_cast<uint32_t>(o);
            a1[pos] = static_cast<uint32_t>(d1);
            a2[pos] = static_cast<uint32_t>(d2);
        }
        for (size_t d = 0; d < r; d++) {
            b1[d] = (a1[d] == 1 && outv[d] != 1) ? 1 : 0;
            b2[d] = (a2[d] == 1 && outv[d] != 1) ? 1 : 0;
        }

        // ---- 2. 合并广播模式相同的相邻维度, 降低维度数 ----
        std::vector<uint32_t> mOut, mA1, mA2;
        std::vector<uint8_t> mB1, mB2;
        for (size_t d = 0; d < r; d++) {
            if (!mOut.empty() && mB1.back() == b1[d] && mB2.back() == b2[d]) {
                mOut.back() *= outv[d];
                mA1.back() *= a1[d];
                mA2.back() *= a2[d];
            } else {
                mOut.push_back(outv[d]);
                mA1.push_back(a1[d]);
                mA2.push_back(a2[d]);
                mB1.push_back(b1[d]);
                mB2.push_back(b2[d]);
            }
        }
        size_t rank = mOut.size();

        uint64_t total = 1;
        for (auto v : mOut) {
            total *= v;
        }

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        for (uint32_t i = 0; i < LE_MAX_DIM; i++) {
            tiling->outShape[i] = 1;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }

        // ---- 3. 空张量场景 ----
        if (total == 0) {
            tiling->totalLength = 0;
            tiling->mode = 0;
            tiling->blockDim = 1;
            tiling->rank = 1;
            tiling->lastDim = 0;
            tiling->numRows = 0;
            tiling->x1LastBroad = 0;
            tiling->x2LastBroad = 0;
            context->SetBlockDim(1);
            size_t *ws = context->GetWorkspaceSizes(1);
            ws[0] = 0;
            uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
            ASCENDC_TPL_SEL_PARAM(context, DT_X1);
            return ge::GRAPH_SUCCESS;
        }

        if (rank > LE_MAX_DIM) {
            return ge::GRAPH_FAILED;
        }

        tiling->totalLength = static_cast<uint32_t>(total);

        bool mode0 = (rank == 1 && mB1[0] == 0 && mB2[0] == 0);
        if (mode0) {
            // 纯逐元素: 按总元素数分核
            tiling->mode = 0;
            uint32_t bd = num_cores_aiv;
            if (static_cast<uint32_t>(total) < bd) {
                bd = static_cast<uint32_t>(total);
            }
            if (bd < 1) bd = 1;
            tiling->blockDim = bd;
            tiling->rank = 1;
            tiling->lastDim = static_cast<uint32_t>(total);
            tiling->numRows = 1;
            tiling->x1LastBroad = 0;
            tiling->x2LastBroad = 0;
            context->SetBlockDim(bd);
        } else {
            // 广播: 按行(最内层维度之外)分核
            tiling->mode = 1;
            tiling->rank = static_cast<uint32_t>(rank);
            uint32_t lastDim = mOut[rank - 1];
            uint64_t numRows = total / lastDim;
            tiling->lastDim = lastDim;
            tiling->numRows = static_cast<uint32_t>(numRows);
            tiling->x1LastBroad = mB1[rank - 1];
            tiling->x2LastBroad = mB2[rank - 1];

            uint64_t run1 = 1, run2 = 1;
            for (int d = static_cast<int>(rank) - 1; d >= 0; d--) {
                tiling->outShape[d] = mOut[d];
                tiling->x1Stride[d] = mB1[d] ? 0 : static_cast<uint32_t>(run1);
                tiling->x2Stride[d] = mB2[d] ? 0 : static_cast<uint32_t>(run2);
                run1 *= mA1[d];  // 广播维 mA=1, 不影响步长
                run2 *= mA2[d];
            }

            uint32_t bd = num_cores_aiv;
            if (static_cast<uint32_t>(numRows) < bd) {
                bd = static_cast<uint32_t>(numRows);
            }
            if (bd < 1) bd = 1;
            tiling->blockDim = bd;
            context->SetBlockDim(bd);
        }

        size_t *ws = context->GetWorkspaceSizes(1);
        ws[0] = 0;

        // 配置 tiling key: 按数据类型区分 kernel 模板实例
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *s1 = context->GetInputShape(0);
        const gert::Shape *s2 = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);
        if (s1 == nullptr || s2 == nullptr || y_shape == nullptr) {
            return GRAPH_FAILED;
        }
        size_t r1 = s1->GetDimNum();
        size_t r2 = s2->GetDimNum();
        size_t r = r1 > r2 ? r1 : r2;
        if (r == 0) {
            y_shape->SetDimNum(0);
            return GRAPH_SUCCESS;
        }
        y_shape->SetDimNum(r);
        for (size_t i = 0; i < r; i++) {
            int64_t d1 = (i < r1) ? s1->GetDim(r1 - 1 - i) : 1;
            int64_t d2 = (i < r2) ? s2->GetDim(r2 - 1 - i) : 1;
            int64_t o;
            if (d1 == d2)      o = d1;
            else if (d1 == 1)  o = d2;
            else if (d2 == 1)  o = d1;
            else               return GRAPH_FAILED;
            y_shape->SetDim(r - 1 - i, o);
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
