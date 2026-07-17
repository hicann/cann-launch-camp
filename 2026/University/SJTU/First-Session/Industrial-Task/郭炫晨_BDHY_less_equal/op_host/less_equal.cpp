// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"

namespace optiling {
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t BUFFER_NUM = 16;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
        if (tensor_x1 == nullptr || tensor_x2 == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x1 = tensor_x1->GetDataType();
        int64_t dtype_size_x1 = ge::GetSizeByDataType(dtype_x1);
        if (dtype_size_x1 <= 0) dtype_size_x1 = 1;
        uint64_t length_x1 = tensor_x1->GetShapeSize();

        uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
        ASCENDC_TPL_SEL_PARAM(context, DT_X1);

        LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();

        uint64_t totalLength = length_x1;
        uint32_t pad32 = BLOCK_SIZE;

        uint32_t padMax = ub_size / BUFFER_NUM / dtype_size_x1;
        padMax = (padMax / (2 * BLOCK_SIZE)) * (2 * BLOCK_SIZE);
        if (padMax == 0) padMax = 2 * BLOCK_SIZE;

        uint32_t coreNum = static_cast<uint32_t>(num_cores_aiv);
        if (coreNum == 0) coreNum = 1;

        tiling->totalLength = totalLength;

        if (totalLength * static_cast<uint64_t>(dtype_size_x1) <= BLOCK_SIZE) {
            tiling->blockLengthMean = pad32;
            tiling->blockLengthEnd = pad32;
            tiling->tileNumMean = 1;
            tiling->tileNumEnd = 1;
            tiling->tileLengthMean = pad32;
            tiling->tileLengthEnd = pad32;
            context->SetBlockDim(1);
        } else {
            if (totalLength <= padMax) {
                uint32_t alignedLen = ((totalLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
                tiling->blockLengthMean = alignedLen;
                tiling->blockLengthEnd = alignedLen;
                tiling->tileNumMean = 1;
                tiling->tileNumEnd = 1;
                tiling->tileLengthMean = alignedLen;
                tiling->tileLengthEnd = static_cast<uint32_t>(totalLength);
                context->SetBlockDim(1);
            } else {
                uint32_t maxBlockLength = totalLength / coreNum;
                maxBlockLength = ((maxBlockLength + padMax - 1) / padMax) * padMax;

                while (coreNum > 1 && maxBlockLength * (coreNum - 1) >= totalLength) {
                    coreNum--;
                    maxBlockLength = totalLength / coreNum;
                    maxBlockLength = ((maxBlockLength + padMax - 1) / padMax) * padMax;
                }

                uint32_t tileNumMean = maxBlockLength / padMax;
                uint32_t blockLengthEnd = totalLength - maxBlockLength * (coreNum - 1);
                uint32_t tileNumEnd = (blockLengthEnd + padMax - 1) / padMax;
                if (tileNumEnd == 0) tileNumEnd = 1;
                uint32_t tileLengthEnd = blockLengthEnd - (tileNumEnd - 1) * padMax;

                tiling->blockLengthMean = maxBlockLength;
                tiling->blockLengthEnd = blockLengthEnd;
                tiling->tileNumMean = tileNumMean;
                tiling->tileNumEnd = tileNumEnd;
                tiling->tileLengthMean = padMax;
                tiling->tileLengthEnd = tileLengthEnd;
                context->SetBlockDim(static_cast<int32_t>(coreNum));
            }
        }

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
    }
}

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
}
