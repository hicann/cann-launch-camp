#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t coreNumAiv = platform.GetCoreNumAiv();
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t totalLength = static_cast<uint32_t>(tensor_x->GetShapeSize());

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        constexpr uint32_t MIN_LENGTH_PER_CORE = 64;
        int32_t coreNum = (coreNumAiv > 0) ? coreNumAiv : 1;
        if (totalLength == 0) {
            coreNum = 1;
        } else {
            uint32_t maxCoreByWorkload = (totalLength + MIN_LENGTH_PER_CORE - 1) / MIN_LENGTH_PER_CORE;
            if (maxCoreByWorkload < 1) {
                maxCoreByWorkload = 1;
            }
            if (static_cast<uint32_t>(coreNum) > maxCoreByWorkload) {
                coreNum = static_cast<int32_t>(maxCoreByWorkload);
            }
            if (static_cast<uint32_t>(coreNum) > totalLength) {
                coreNum = static_cast<int32_t>(totalLength);
            }
        }

        uint32_t everyCoreLength = totalLength / static_cast<uint32_t>(coreNum);
        uint32_t tailBlockNum = totalLength % static_cast<uint32_t>(coreNum);

        uint32_t formerNum, tailNum, formerLength, tailLength;
        if (tailBlockNum == 0) {
            formerNum = 0;
            tailNum = static_cast<uint32_t>(coreNum);
            formerLength = everyCoreLength;
            tailLength = everyCoreLength;
        } else {
            formerNum = tailBlockNum;
            tailNum = static_cast<uint32_t>(coreNum) - formerNum;
            formerLength = everyCoreLength + 1;
            tailLength = everyCoreLength;
        }

        constexpr uint32_t RESERVED_BUFFER_NUM = 5;
        uint32_t perElemBytes = dtype_size_x * RESERVED_BUFFER_NUM;
        uint32_t tileLength = (perElemBytes > 0) ? static_cast<uint32_t>(ubSize) / perElemBytes : 1;
        if (tileLength == 0) {
            tileLength = 1;
        }
        constexpr uint32_t ALIGN_BYTES = 32;
        uint32_t alignElem = (dtype_size_x > 0) ? (ALIGN_BYTES / dtype_size_x) : 1;
        if (alignElem == 0) {
            alignElem = 1;
        }
        if (tileLength >= alignElem) {
            tileLength = (tileLength / alignElem) * alignElem;
        }
        if (tileLength == 0) {
            tileLength = 1;
        }

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        tiling->totalLength = totalLength;
        tiling->coreNum = static_cast<uint32_t>(coreNum);
        tiling->formerNum = formerNum;
        tiling->tailNum = tailNum;
        tiling->formerLength = formerLength;
        tiling->tailLength = tailLength;
        tiling->tileLength = tileLength;

        context->SetBlockDim(static_cast<uint32_t>(coreNum));

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class FastGelu : public OpDef {
    public:
        explicit FastGelu(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}