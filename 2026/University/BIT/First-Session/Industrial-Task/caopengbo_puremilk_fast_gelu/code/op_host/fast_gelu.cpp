#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {

constexpr uint32_t TILING_BUFFER_NUM = 2;
constexpr uint32_t TILING_TMP_FLOAT_BUF_CNT = 4;
constexpr uint32_t UB_RESERVED_RATIO_NUM = 9;
constexpr uint32_t UB_RESERVED_RATIO_DEN = 10;
constexpr uint32_t BLOCK_SIZE_BYTES = 32;
constexpr uint32_t SMALL_FLOAT32_SIGMOID_THRESHOLD = 24;

static uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}

static uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return align == 0 ? value : ((value + align - 1) / align) * align;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();

    int dtype_size_x = ge::GetSizeByDataType(dtype_x);
    if (dtype_size_x <= 0) {
        dtype_size_x = 4;
    }

    uint32_t length_x = static_cast<uint32_t>(tensor_x->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = length_x;

    uint32_t bytesPerElem =
        TILING_BUFFER_NUM * static_cast<uint32_t>(dtype_size_x) * 2u +
        TILING_TMP_FLOAT_BUF_CNT * static_cast<uint32_t>(sizeof(float));

    uint64_t usableUbSize = ub_size * UB_RESERVED_RATIO_NUM / UB_RESERVED_RATIO_DEN;
    uint32_t tileDataNum = bytesPerElem > 0 ? static_cast<uint32_t>(usableUbSize / bytesPerElem) : 1u;

    if (tileDataNum == 0) {
        tileDataNum = 1;
    }

    uint32_t alignElem = std::max<uint32_t>(
        BLOCK_SIZE_BYTES / static_cast<uint32_t>(dtype_size_x), 1u);

    if (tileDataNum > alignElem) {
        tileDataNum = (tileDataNum / alignElem) * alignElem;
    }

    if (length_x > 0 && tileDataNum > length_x) {
        tileDataNum = AlignUp(length_x, alignElem);
    }

    tiling->tileDataNum = tileDataNum;

    uint32_t coreNum = static_cast<uint32_t>(num_cores_aiv > 0 ? num_cores_aiv : 1);
    uint32_t usedCoreNum = 1;

    if (length_x > 0) {
        if (dtype_x == ge::DT_FLOAT && length_x <= SMALL_FLOAT32_SIGMOID_THRESHOLD) {
            usedCoreNum = 1;
        } else {
            usedCoreNum = std::min<uint32_t>(coreNum, CeilDiv(length_x, alignElem));

            for (uint32_t i = 0; i < 4; ++i) {
                uint32_t perCore = CeilDiv(length_x, usedCoreNum);
                uint32_t alignedPerCore = AlignUp(perCore, alignElem);
                uint32_t newUsedCoreNum = CeilDiv(length_x, alignedPerCore);

                if (newUsedCoreNum == usedCoreNum || newUsedCoreNum == 0) {
                    break;
                }

                usedCoreNum = newUsedCoreNum;
            }

            usedCoreNum = std::max<uint32_t>(usedCoreNum, 1u);
        }
    }

    context->SetBlockDim(usedCoreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FastGelu : public OpDef {
public:
    explicit FastGelu(const char *name) : OpDef(name)
    {
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

}  // namespace ops
