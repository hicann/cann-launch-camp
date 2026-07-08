// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace {
constexpr uint64_t BLOCK_BYTES = 32;        // 910b 一个 dataBlock = 32B
constexpr uint64_t UB_RESERVED = 8192;      // UB 预留（kfc 等已扣过，再留余量）
constexpr uint64_t TILE_CAP_HARD_MAX = 65536;  // 单 tile 元素数硬上限（安全兜底，正常运行不触发）

// 峰值 UB 占用（BUFFER_NUM=2）：
//   fp32: xQue(2×4)+yQue(2×4)+t1(4)                    = 20 B/elem
//   fp16: xQue(2×2)+yQue(2×2)+xF(4)+t1(4)              = 16 B/elem
//   通用式：4*dtypeSize + 4 + (fp16 ? 4 : 0)
//   （xf 路径复用输入 xF 作输出缓冲，已省掉 yF；新等价推导仅用 1 个 temp。）
inline uint64_t UbBytesPerElem(uint32_t dtypeSize) {
    return 4ULL * dtypeSize + 4ULL + ((dtypeSize == 2) ? 4ULL : 0ULL);
}

inline uint64_t CeilDiv(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);        // 2 或 4
        uint64_t length_x = (uint64_t)tensor_x->GetShapeSize();   // 展平后总元素数

        // 配置 tiling key：按输入 dtype 派发 kernel 模板特化（half / float）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->dtypeSize = (uint32_t)dtype_size_x;

        uint32_t coreNum = (num_cores_aiv > 0) ? (uint32_t)num_cores_aiv : 1u;
        uint64_t blockElems = BLOCK_BYTES / (uint64_t)dtype_size_x;  // fp32=8, fp16=16

        // tileLength 由 UB 预算推导，向下对齐到 blockElems
        uint64_t ubPerElem = UbBytesPerElem((uint32_t)dtype_size_x);
        uint64_t ub_safe = (ub_size > UB_RESERVED) ? (ub_size - UB_RESERVED) : ub_size;
        uint64_t tileCapMax = ub_safe / ubPerElem;
        if (tileCapMax > TILE_CAP_HARD_MAX) {
            tileCapMax = TILE_CAP_HARD_MAX;
        }
        uint64_t maxTileLength = (tileCapMax / blockElems) * blockElems;
        if (maxTileLength == 0) {
            maxTileLength = blockElems;  // 兜底
        }

        // 空张量短路：单核 no-op
        if (length_x == 0) {
            tiling->totalLength  = 0;
            tiling->usedCoreNum  = 1;
            tiling->tileLength    = maxTileLength;
            tiling->tilesPerCore  = 0;
            tiling->tailCoreNum   = 0;
            context->SetBlockDim(1);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        uint64_t totalTiles = CeilDiv(length_x, maxTileLength);
        uint32_t usedCoreNum = coreNum;
        uint64_t finalTileLength = maxTileLength;

        if (totalTiles < (uint64_t)coreNum) {
            // 数据量很小，UB够大：直接使用对应的总Tile数作为核数
            usedCoreNum = (uint32_t)totalTiles;
            if (usedCoreNum == 0) {
                usedCoreNum = 1;
            }
        } else {
            // 数据量大：启动自适应 Even Tiling 负载均衡
            // 1. 均分总元素：计算平均每个核分担的 workload
            uint64_t maxElemsPerCore = CeilDiv(length_x, (uint64_t)coreNum);
            // 2. 均分切块：计算每个核需要几次 Tile 才能处理完上述 workload
            uint64_t tilesPerCoreOpt = CeilDiv(maxElemsPerCore, maxTileLength);
            // 3. 完美回溯：重新计算达到此切片次数所需的精确 Tile 大小
            uint64_t optTileLength = CeilDiv(maxElemsPerCore, tilesPerCoreOpt);
            // 4. 硬件强制对齐 (32B)
            optTileLength = CeilDiv(optTileLength, blockElems) * blockElems;
            
            // 最终定型，防止意外超限
            finalTileLength = (optTileLength > maxTileLength) ? maxTileLength : optTileLength;
            totalTiles = CeilDiv(length_x, finalTileLength);
        }

        uint32_t tilesPerCore = (uint32_t)(totalTiles / usedCoreNum);
        uint32_t tailCoreNum  = (uint32_t)(totalTiles % usedCoreNum);  // 前 tailCoreNum 个核各 +1 tile

        tiling->totalLength  = length_x;
        tiling->usedCoreNum  = usedCoreNum;
        tiling->tileLength    = finalTileLength;
        tiling->tilesPerCore  = tilesPerCore;
        tiling->tailCoreNum   = tailCoreNum;

        context->SetBlockDim(usedCoreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 shape 与输入一致
        const auto inputShape = context->GetInputShape(0);
        auto outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 dtype 与输入一致
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
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(FastGelu);
}  // namespace ops
