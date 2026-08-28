#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/clip_by_value_tiling.h"

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
        uint32_t dt_size = ge::GetSizeByDataType(dtype_x); 
        uint32_t total_length = tensor_x->GetShapeSize(); 

        uint32_t is_min_scalar = (tensor_min->GetShapeSize() == 1) ? 1 : 0;
        uint32_t is_max_scalar = (tensor_max->GetShapeSize() == 1) ? 1 : 0;

        uint32_t elements_per_32b = 32 / dt_size;

        uint32_t usedCoreNum = num_cores_aiv;
        uint32_t coreData = total_length / usedCoreNum;
        coreData = (coreData / elements_per_32b) * elements_per_32b; 
        
        if (coreData == 0) {
            usedCoreNum = 1;
            coreData = (total_length / elements_per_32b) * elements_per_32b; 
        }
        
        uint32_t coreDataTail = total_length - coreData * (usedCoreNum - 1);

        uint32_t ub_block = ub_size / 5; 
        uint32_t tileLength = (ub_block / dt_size) / elements_per_32b * elements_per_32b;

        ClipByValueTilingData tiling;
        tiling.set_usedCoreNum(usedCoreNum);
        tiling.set_coreData(coreData);
        tiling.set_coreDataTail(coreDataTail);
        tiling.set_tileLength(tileLength);
        tiling.set_is_min_scalar(is_min_scalar);
        tiling.set_is_max_scalar(is_max_scalar);
        // 新增：把数据类型的枚举值传进 Kernel
        tiling.set_dtype(static_cast<uint32_t>(dtype_x));
        
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        context->SetBlockDim(usedCoreNum);
        context->GetWorkspaceSizes(1)[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
} 

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape* x_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        if (x_shape != nullptr && y_shape != nullptr) {
            *y_shape = *x_shape;
        }
        return ge::GRAPH_SUCCESS;
    }
    
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
        return ge::GRAPH_SUCCESS;
    }
} 

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
}