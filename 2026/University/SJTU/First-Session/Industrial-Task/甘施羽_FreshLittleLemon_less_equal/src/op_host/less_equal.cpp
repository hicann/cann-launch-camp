// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
    static inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
        return b == 0 ? 0 : (a + b - 1) / b;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 1. 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores_aiv = static_cast<uint32_t>(platform.GetCoreNumAiv());
        if (num_cores_aiv == 0) {
            num_cores_aiv = 1;
        }

        // 2. 输入信息
        const gert::StorageShape *shape_x1 = context->GetInputShape(0);
        const gert::StorageShape *shape_x2 = context->GetInputShape(1);
        const gert::StorageShape *shape_y = context->GetOutputShape(0);
        ge::DataType dtype_x1 = context->GetInputDesc(0)->GetDataType();
        uint32_t dtype_size = static_cast<uint32_t>(ge::GetSizeByDataType(dtype_x1));

        const gert::Shape &s1 = shape_x1->GetStorageShape();
        const gert::Shape &s2 = shape_x2->GetStorageShape();
        const gert::Shape &sy = shape_y->GetStorageShape();

        uint32_t rank1 = static_cast<uint32_t>(s1.GetDimNum());
        uint32_t rank2 = static_cast<uint32_t>(s2.GetDimNum());
        uint32_t rankY = static_cast<uint32_t>(sy.GetDimNum());

        // 输出总元素数（以输出 shape 为准）
        uint32_t totalLength = 1;
        for (uint32_t i = 0; i < rankY; i++) {
            totalLength *= static_cast<uint32_t>(sy.GetDim(i));
        }

        // 3. 判断是否需要广播：x1、x2 形状完全一致则走 fast path
        bool sameShape = (rank1 == rank2);
        if (sameShape) {
            for (uint32_t i = 0; i < rank1; i++) {
                if (s1.GetDim(i) != s2.GetDim(i)) {
                    sameShape = false;
                    break;
                }
            }
        }
        uint32_t needBroadcast = sameShape ? 0u : 1u;

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        tiling->totalLength = totalLength;
        tiling->needBroadcast = needBroadcast;

        if (needBroadcast == 1) {
            // 对齐维度：右对齐，左侧补 1，维度数取输出 rank（不超过 MAX_DIM）
            uint32_t ndim = rankY;
            if (ndim > LESS_EQUAL_MAX_DIM) {
                ndim = LESS_EQUAL_MAX_DIM;  // 约束内不会触发
            }
            tiling->ndim = ndim;

            // padded 输入 shape（左补 1）
            uint32_t shp1[LESS_EQUAL_MAX_DIM];
            uint32_t shp2[LESS_EQUAL_MAX_DIM];
            uint32_t shpO[LESS_EQUAL_MAX_DIM];
            for (uint32_t d = 0; d < ndim; d++) {
                // 从右往左对齐
                int32_t idx1 = static_cast<int32_t>(rank1) - static_cast<int32_t>(ndim) + static_cast<int32_t>(d);
                int32_t idx2 = static_cast<int32_t>(rank2) - static_cast<int32_t>(ndim) + static_cast<int32_t>(d);
                shp1[d] = (idx1 >= 0) ? static_cast<uint32_t>(s1.GetDim(idx1)) : 1u;
                shp2[d] = (idx2 >= 0) ? static_cast<uint32_t>(s2.GetDim(idx2)) : 1u;
                shpO[d] = static_cast<uint32_t>(sy.GetDim(d));
            }

            // 输出累积步长 dimStride[d] = prod(shpO[d+1..])
            uint32_t dimStride[LESS_EQUAL_MAX_DIM];
            dimStride[ndim - 1] = 1;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; d--) {
                dimStride[d] = dimStride[d + 1] * shpO[d + 1];
            }

            // 输入连续步长（基于 padded shape），被广播维（shp==1）步长置 0
            uint32_t contig1[LESS_EQUAL_MAX_DIM];
            uint32_t contig2[LESS_EQUAL_MAX_DIM];
            contig1[ndim - 1] = 1;
            contig2[ndim - 1] = 1;
            for (int32_t d = static_cast<int32_t>(ndim) - 2; d >= 0; d--) {
                contig1[d] = contig1[d + 1] * shp1[d + 1];
                contig2[d] = contig2[d + 1] * shp2[d + 1];
            }

            for (uint32_t d = 0; d < ndim; d++) {
                tiling->shapeOut[d] = shpO[d];
                tiling->dimStride[d] = dimStride[d];
                tiling->strideX1[d] = (shp1[d] == 1) ? 0u : contig1[d];
                tiling->strideX2[d] = (shp2[d] == 1) ? 0u : contig2[d];
            }
            // 未使用的高维填 0
            for (uint32_t d = ndim; d < LESS_EQUAL_MAX_DIM; d++) {
                tiling->shapeOut[d] = 1;
                tiling->dimStride[d] = 0;
                tiling->strideX1[d] = 0;
                tiling->strideX2[d] = 0;
            }
        } else {
            tiling->ndim = 0;
            for (uint32_t d = 0; d < LESS_EQUAL_MAX_DIM; d++) {
                tiling->shapeOut[d] = 1;
                tiling->dimStride[d] = 0;
                tiling->strideX1[d] = 0;
                tiling->strideX2[d] = 0;
            }
        }

        // 4. UB 分块：预算 192KB。按 kernel 实际 buffer 用量（每元素字节）估算：
        //    inQueueX1/X2: 2*sizeof(T) 各 double buffer -> 4*sizeof(T)
        //    outQueueY(int8) double buffer -> 2
        //    maskBuf(uint8) -> 1
        //    oneBuf/zeroBuf/resBuf(half) -> 3*2 = 6
        //    minBuf(int32, 仅 int32) -> 4
        //    castX1/X2(half, 仅 int8) -> 4
        //    上界(int32): 16+2+1+6+4=29；(int8): 4+2+1+6+4=17；统一 4*sizeof(T)+16 留余量
        constexpr uint32_t UB_SIZE = 192 * 1024;
        constexpr uint32_t RESERVED = 16 * 1024;
        uint32_t ubBudget = UB_SIZE - RESERVED;

        uint32_t perElem = 4u * dtype_size + 16u;
        uint32_t tileLength = ubBudget / perElem;
        // 向下对齐到 256 元素，保证向量指令效率与 32B 对齐
        tileLength = (tileLength / 256u) * 256u;
        if (tileLength < 256u) {
            tileLength = 256u;
        }
        tiling->tileLength = tileLength;

        // 5. 多核切分：按输出元素一维展平。
        //    向量粒度 256 元素，每核至少分一个「块」，避免小张量时空转核过多、
        //    以及每核不足 256 时被 256 对齐放大的冗余计算。
        constexpr uint32_t GRAIN = 256u;
        uint32_t numGrains = (totalLength + GRAIN - 1) / GRAIN;  // 需要的 256 块数
        uint32_t blockDim = std::min(num_cores_aiv, numGrains);
        if (blockDim == 0) {
            blockDim = 1;  // 空张量场景，至少 1 核（不产出数据）
        }
        tiling->blockDim = blockDim;

        // 6. TilingKey（按 x1 数据类型区分 kernel 实例）
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        // 7. 启动核数
        context->SetBlockDim(blockDim);

        // 8. workspace
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        const gert::Shape *x2_shape = context->GetInputShape(1);
        gert::Shape *y_shape = context->GetOutputShape(0);

        size_t rank1 = x1_shape->GetDimNum();
        size_t rank2 = x2_shape->GetDimNum();
        size_t rankY = (rank1 > rank2) ? rank1 : rank2;

        y_shape->SetDimNum(rankY);
        // NumPy 广播：从最右维对齐，逐维取 max（要求相等或其一为 1）
        for (size_t d = 0; d < rankY; d++) {
            int64_t i1 = static_cast<int64_t>(rank1) - static_cast<int64_t>(rankY) + static_cast<int64_t>(d);
            int64_t i2 = static_cast<int64_t>(rank2) - static_cast<int64_t>(rankY) + static_cast<int64_t>(d);
            int64_t d1 = (i1 >= 0) ? x1_shape->GetDim(i1) : 1;
            int64_t d2 = (i2 >= 0) ? x2_shape->GetDim(i2) : 1;
            int64_t od = (d1 > d2) ? d1 : d2;
            // 若某维为 1，广播到另一维；否则应相等（非法组合由运行时判定）
            if (d1 == 1) od = d2;
            else if (d2 == 1) od = d1;
            y_shape->SetDim(d, od);
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出恒为 bool
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
