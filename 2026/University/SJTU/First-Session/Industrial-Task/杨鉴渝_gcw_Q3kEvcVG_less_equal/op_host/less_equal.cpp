// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        uint32_t totalLength = tensor_x1->GetShapeSize();
        ge::DataType dtype_x1 = tensor_x1->GetDataType();

        // tiling key: dtype dispatch
        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        // 填充tiling结构体
        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
        tiling->totalLength = totalLength;
        tiling->length      = totalLength;
        tiling->blockDim    = num_cores;
        tiling->tileNum     = 8;
        tiling->blockLength = totalLength / num_cores;
        tiling->dtype       = DT_X1;

        // 配置启动核数和workspace
        context->SetBlockDim(num_cores);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Tensor *tensor_x1 = context->GetInputTensor(0);
        if (tensor_x1 == nullptr) {
            return GRAPH_FAILED;
        }
        const gert::TensorShape &shape_x1 = tensor_x1->GetShape();
        std::vector<int64_t> outDims;
        for (size_t i = 0; i < shape_x1.GetDimNum(); ++i) {
            outDims.push_back(shape_x1.GetDim(i));
        }
        gert::TensorShape outShape(outDims);
        gert::Tensor *tensor_y = context->GetOutputTensor(0);
        if (tensor_y == nullptr) {
            return GRAPH_FAILED;
        }
        tensor_y->SetShape(outShape);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        gert::Tensor *tensor_y = context->GetOutputTensor(0);
        if (tensor_y == nullptr) {
            return GRAPH_FAILED;
        }
        tensor_y->SetDataType(ge::DT_BOOL);
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
