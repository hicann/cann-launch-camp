// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <limits>

#include "../op_kernel/less_equal_custom_tiling.h"
#include "../op_kernel/tiling_key_less_equal_custom.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t num_cores = platform.GetCoreNumAiv();

        // 获取UB大小
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
        if (tensor_x1 == nullptr || tensor_x2 == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x1 = tensor_x1->GetDataType();
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        const gert::StorageShape *shape_x1 = context->GetInputShape(0);
        const gert::StorageShape *shape_x2 = context->GetInputShape(1);
        if (shape_x1 == nullptr || shape_x2 == nullptr) {
            return ge::GRAPH_FAILED;
        }

        int32_t dims_x1 = shape_x1->GetStorageShape().GetDimNum();
        int32_t dims_x2 = shape_x2->GetStorageShape().GetDimNum();
        int32_t max_dims = dims_x1 > dims_x2 ? dims_x1 : dims_x2;
        // 标量也统一成一维 [1]，避免 Kernel 出现负下标。
        if (max_dims == 0) {
            max_dims = 1;
        }

        if (max_dims > MAX_DIMS) {
            return ge::GRAPH_FAILED;
        }

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->dims = max_dims;

        // 广播形状对齐（右对齐，左侧补1）
        for (int32_t i = 0; i < max_dims; i++) {
            tiling->x1Shape[i] = 1;
            tiling->x2Shape[i] = 1;
        }
        for (int32_t i = 0; i < dims_x1; i++) {
            tiling->x1Shape[max_dims - dims_x1 + i] = shape_x1->GetStorageShape().GetDim(i);
        }
        for (int32_t i = 0; i < dims_x2; i++) {
            tiling->x2Shape[max_dims - dims_x2 + i] = shape_x2->GetStorageShape().GetDim(i);
        }

        // 验证广播兼容性并计算输出形状
        uint64_t totalLength = 1;
        for (int32_t i = 0; i < max_dims; i++) {
            int32_t d1 = tiling->x1Shape[i];
            int32_t d2 = tiling->x2Shape[i];
            if (d1 != d2 && d1 != 1 && d2 != 1) {
                return ge::GRAPH_FAILED;  // 广播不兼容
            }
            // 不能用 max：空维 0 与 1 广播后的维度应为 0，而不是 1。
            tiling->outShape[i] = d1 == 1 ? d2 : d1;
            const uint64_t outDim = static_cast<uint64_t>(tiling->outShape[i]);
            if (outDim != 0 && totalLength > std::numeric_limits<uint64_t>::max() / outDim) {
                return ge::GRAPH_FAILED;
            }
            totalLength *= outDim;
        }

        // 计算stride（用于Kernel广播索引映射）
        for (int32_t i = max_dims - 1; i >= 0; i--) {
            if (i == max_dims - 1) {
                tiling->x1Stride[i] = 1;
                tiling->x2Stride[i] = 1;
            } else {
                tiling->x1Stride[i] = tiling->x1Stride[i + 1] * tiling->x1Shape[i + 1];
                tiling->x2Stride[i] = tiling->x2Stride[i + 1] * tiling->x2Shape[i + 1];
            }
        }

        tiling->totalLength = totalLength;
        tiling->dtype = static_cast<int32_t>(dtype_x1);

        // 动态设置核数：多核并行加速，每核至少4096元素避免调度开销
        constexpr uint32_t MIN_ELEM_PER_CORE = 4096;
        uint32_t blockDim = 1;
        if (totalLength > MIN_ELEM_PER_CORE) {
            const uint64_t neededCores = totalLength / MIN_ELEM_PER_CORE +
                (totalLength % MIN_ELEM_PER_CORE != 0);
            blockDim = static_cast<uint32_t>(std::min<uint64_t>(num_cores, neededCores));
        }
        context->SetBlockDim(blockDim);

        // 计算 UB 分块。Kernel 最坏需要双缓冲输入/输出和四块向量工作区，
        // 按每元素 32B 预算可覆盖 float/int32 分支并留出框架开销。
        uint32_t elemSize = 4;
        if (dtype_x1 == ge::DT_FLOAT16) elemSize = 2;
        else if (dtype_x1 == ge::DT_INT8) elemSize = 1;
        uint32_t usableUb = static_cast<uint32_t>(ub_size * 7 / 8);
        uint32_t chunkSize = usableUb / 32;
        constexpr uint32_t ALIGN_BYTES = 32;
        uint32_t alignElems = ALIGN_BYTES / elemSize;
        if (alignElems < 1) alignElems = 1;
        chunkSize = (chunkSize / alignElems) * alignElems;
        if (chunkSize < alignElems) chunkSize = alignElems;
        // DataCopyParams.blockLen 为 uint16_t，限制单次字节数避免截断。
        uint32_t maxCopyElements = 65535 / elemSize;
        if (chunkSize > maxCopyElements) chunkSize = maxCopyElements;
        chunkSize = (chunkSize / alignElems) * alignElems;

        tiling->blockDim = blockDim;
        tiling->ubChunkSize = chunkSize;

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
        if (x1_shape == nullptr || x2_shape == nullptr || y_shape == nullptr) {
            return GRAPH_FAILED;
        }

        int32_t dims_x1 = x1_shape->GetDimNum();
        int32_t dims_x2 = x2_shape->GetDimNum();
        int32_t max_dims = dims_x1 > dims_x2 ? dims_x1 : dims_x2;
        if (dims_x1 < 0 || dims_x2 < 0 || max_dims > MAX_DIMS) {
            return GRAPH_FAILED;
        }

        // 广播形状推导：右对齐，逐维度取最大值
        int64_t out_dims[MAX_DIMS];
        for (int32_t i = 0; i < max_dims; i++) {
            out_dims[i] = 1;
        }
        for (int32_t i = 0; i < dims_x1; i++) {
            out_dims[max_dims - dims_x1 + i] = x1_shape->GetDim(i);
        }
        for (int32_t i = 0; i < dims_x2; i++) {
            int32_t idx = max_dims - dims_x2 + i;
            int64_t d2 = x2_shape->GetDim(i);
            const int64_t d1 = out_dims[idx];
            if (d1 != d2 && d1 != 1 && d2 != 1) {
                return GRAPH_FAILED;
            }
            out_dims[idx] = d1 == 1 ? d2 : d1;
        }

        y_shape->SetDimNum(max_dims);
        for (int32_t i = 0; i < max_dims; i++) {
            y_shape->SetDim(i, out_dims[i]);
        }
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class LessEqualCustom : public OpDef {
    public:
        explicit LessEqualCustom(const char *name) : OpDef(name) {
            this->Input("x1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("x2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(LessEqualCustom);
}  // namespace ops
