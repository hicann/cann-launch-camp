// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_min = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_max = context->GetRequiredInputTensor(2);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t length_min = tensor_min->GetShapeSize();
        uint32_t length_max = tensor_max->GetShapeSize();

        bool isScalarMin = (length_min <= 1);
        bool isScalarMax = (length_max <= 1);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        constexpr uint32_t BLOCK_BYTES = 32;
        uint32_t elementsPerBlock = BLOCK_BYTES / static_cast<uint32_t>(dtype_size_x);
        uint32_t blockDim = static_cast<uint32_t>(num_cores_aiv);
        if (blockDim == 0) blockDim = 1;
        
        if (length_x < elementsPerBlock && blockDim > 1) blockDim = 1;
        uint32_t bufferCount;
        if (isScalarMin && isScalarMax) bufferCount = 4;
        else if (isScalarMin || isScalarMax) bufferCount = 6;
        else bufferCount = 8;
        uint32_t maxTileElements = static_cast<uint32_t>(ub_size / (bufferCount * static_cast<uint64_t>(dtype_size_x)));
        uint32_t tileLength = maxTileElements / elementsPerBlock * elementsPerBlock;
        if (tileLength < elementsPerBlock) tileLength = elementsPerBlock;

        uint32_t roughBlockLen = (length_x + blockDim - 1) / blockDim;
        uint32_t blockLength = ((roughBlockLen + tileLength - 1) / tileLength) * tileLength;
        if (blockLength == 0) blockLength = tileLength;

        uint32_t usedCoreNum = (length_x + blockLength - 1) / blockLength;
        if (usedCoreNum == 0) usedCoreNum = 1;

        if (tileLength > blockLength) {
            tileLength = blockLength;
        }

        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->length = length_x;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        tiling->isScalarMin = isScalarMin ? 1 : 0;
        tiling->isScalarMax = isScalarMax ? 1 : 0;

        context->SetBlockDim(usedCoreNum);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) {
            return GRAPH_FAILED;
        }
        *yShape = *xShape; 
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0)); 
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class ClipByValue : public OpDef {
    public:
        explicit ClipByValue(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("clip_value_min")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("clip_value_max")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(ClipByValue);
}  // namespace ops
