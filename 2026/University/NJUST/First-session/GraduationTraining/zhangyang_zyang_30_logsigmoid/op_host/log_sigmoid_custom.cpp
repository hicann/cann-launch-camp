#include "../op_kernel/kernel_log_sigmoid_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

/**
 * @brief  Tiling function – computes per-core workload split and packs
 *         tiling constants so the kernel can index its slice.
 *
 *  tiling->totalLength   = N  (total element count)
 *  tiling->blockDim      = number of AI cores
 *  tiling->tileLength    = ceil(N / blockDim), aligned to 32
 *  tiling->dataType      = 0 (float32) / 1 (float16) / 2 (bfloat16)
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // --- 1. Obtain pointer to tiling data struct in constant memory ---
    LogSigmoidCustomTilingData *tiling =
        context->GetTilingData<LogSigmoidCustomTilingData>();

    // --- 2. Compute total element count from input shape ---
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); ++i) {
        totalLength *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = totalLength;

    // --- 3. Determine data type ---
    //     datatype mapping: 0=float32  1=float16  2=bfloat16
    auto dt = context->GetInputDataType(0);
    if (dt == ge::DT_FLOAT) {
        tiling->dataType = 0;
    } else if (dt == ge::DT_FLOAT16) {
        tiling->dataType = 1;
    } else if (dt == ge::DT_BF16) {
        tiling->dataType = 2;
    } else {
        // Unknown types fall back to float32
        tiling->dataType = 0;
    }

    // --- 4. Determine block count ---
    //     Use a heuristic: each core ideally handles at least 32 elements.
    //     For ascend910b there are typically 32 AI cores; for smaller
    //     workloads we reduce the block count so cores don't sit idle.
    constexpr uint32_t kMaxBlockDim = 32u;   // ascend910b: 32 AI cores
    constexpr uint32_t kMinElemsPerCore = 32u;

    uint32_t blockDim = kMaxBlockDim;
    // Don't use more cores than the total number of elements
    while (blockDim > 1 && totalLength < blockDim * kMinElemsPerCore) {
        blockDim /= 2;
    }
    if (blockDim > totalLength) {
        blockDim = totalLength;
    }
    if (blockDim == 0) {
        blockDim = 1;
    }
    tiling->blockDim = blockDim;
    context->SetBlockDim(blockDim);

    // --- 5. Compute tile length (elements per core), aligned to 32 ---
    //     32-element alignment is required by the Ascend C vector engine.
    uint32_t tileLength = (totalLength + blockDim - 1) / blockDim;  // ceil
    tileLength = (tileLength + 31u) / 32u * 32u;                    // align to 32
    if (tileLength == 0) {
        tileLength = 32;
    }
    tiling->tileLength = tileLength;

    // --- 6. Workspace ---
    //     All intermediate buffers are allocated in on-chip UB (L1).
    //     No extra global-memory workspace is needed.
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

// ============================================================================
//  Shape & DataType inference – output mirrors input
// ============================================================================

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

// ============================================================================
//  Operator registration
// ============================================================================

namespace ops {

class LogSigmoidCustom : public OpDef {
public:
    explicit LogSigmoidCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(LogSigmoidCustom);

} // namespace ops
