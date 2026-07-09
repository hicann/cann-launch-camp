// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

#include <algorithm>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取硬件平台信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNum();

    // 获取算子输入信息
    uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    ge::DataType dtype_x = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dtype_x, typeLength);

    // 配置tiling key
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 32字节对齐处理，适配非对齐场景
    const uint32_t BLOCK_SIZE = 32;
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

    // 获取UB大小
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 计算每个tile的块数：根据UB大小和实际缓冲区数量
    const uint32_t BUFFER_NUM = 2;       // 双缓冲流水
    const uint32_t TOTAL_SLOTS = 2 * BUFFER_NUM;  // = 4（无tmpBuf，复用yLocal做临时计算）
    // 每个元素在UB中占用的等效字节数（考虑所有缓冲区）
    uint32_t bytesPerElementInUb = TOTAL_SLOTS * typeLength;

    // 计算UB能容纳的最大元素数，并对齐到BLOCK_SIZE
    uint32_t maxElementsByUb = static_cast<uint32_t>(ubSize / bytesPerElementInUb);
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;
    uint32_t tileDataNum = (maxElementsByUb / elementsPerBlock) * elementsPerBlock;
    tileDataNum = std::max(tileDataNum, elementsPerBlock);
    tileDataNum = std::min(tileDataNum, static_cast<uint32_t>(8192));

    // ── 核数优化：小数据合并到更少核上 ──
    // 核启动/同步开销在小数据场景下占主导，减少核数让每个核有足够 tile 发挥双缓冲流水优势
    // 策略：保证每个核至少有 2 个 tile，否则合并到更少的核上
    uint32_t maxTileNum = (inputNum + tileDataNum - 1) / tileDataNum;
    uint32_t minCoreByTile = (maxTileNum + 1) / 2;   // 每核至少 2 tile → 核数 ≤ tile数/2
    minCoreByTile = std::max(minCoreByTile, static_cast<uint32_t>(1));

    // 最终核数 = min(可用核数, 块数, tile约束)
    coreNum = std::min(coreNum, totalBlockNum);
    coreNum = std::min(coreNum, minCoreByTile);
    coreNum = std::max(coreNum, static_cast<uint32_t>(1));
    context->SetBlockDim(coreNum);

    // 填充tiling结构体
    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    tiling->length = inputNum;
    tiling->blockNum = coreNum;
    tiling->tileDataNum = tileDataNum;

    // 配置workspace大小（无需额外workspace）
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

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
}  // namespace ops