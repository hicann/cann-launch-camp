#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

#include <vector>

namespace optiling {
    constexpr uint32_t UB_RESERVED_BYTES = 8 * 1024;
    constexpr uint32_t FP16_LARGE_TILE_THRESHOLD = 2 * 1024 * 1024;
    constexpr uint32_t TARGET_ELEMENTS_PER_CORE = 8192;

    struct TileSelection {
        uint32_t tileLength;
        uint32_t tanhTmpSize;
    };

    static uint32_t CeilDiv(uint32_t value, uint32_t divisor)
    {
        return (value + divisor - 1) / divisor;
    }

    static uint32_t SelectTmpSize(uint32_t maxValue, uint32_t minValue, uint32_t available)
    {
        if (maxValue == 0 || minValue == 0 || available == 0) {
            return 0;
        }
        uint32_t selected = maxValue;
        if (selected > available) {
            selected = available;
        }
        if (available < minValue) {
            return 0;
        }
        if (selected < minValue) {
            selected = minValue;
        }
        return selected;
    }

    static uint32_t GetAvailableTmpBytes(platform_ascendc::PlatformAscendC& platform,
        uint32_t dtypeSize, uint32_t tileLength, bool needWorkBuffer)
    {
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        uint64_t mainBufferBytes = static_cast<uint64_t>(DOUBLE_BUFFER) * tileLength * dtypeSize * 2;
        uint64_t workBufferBytes = needWorkBuffer ? static_cast<uint64_t>(tileLength) * dtypeSize : 0;
        uint64_t usedBytes = mainBufferBytes + workBufferBytes + UB_RESERVED_BYTES;
        if (ubSize <= usedBytes) {
            return 0;
        }
        uint64_t available = ubSize - usedBytes;
        return available > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(available);
    }

    static uint32_t SelectBlockNum(uint32_t totalLength, uint32_t maxCores, uint32_t alignUnit)
    {
        if (totalLength == 0 || maxCores == 0) {
            return 1;
        }

        uint32_t blockNum = CeilDiv(totalLength, TARGET_ELEMENTS_PER_CORE);
        uint32_t maxUsefulCores = CeilDiv(totalLength, alignUnit);
        if (blockNum == 0) {
            blockNum = 1;
        }
        if (blockNum > maxCores) {
            blockNum = maxCores;
        }
        if (blockNum > maxUsefulCores) {
            blockNum = maxUsefulCores;
        }
        return blockNum > 0 ? blockNum : 1;
    }

    static TileSelection SelectTileLength(platform_ascendc::PlatformAscendC& platform,
        uint32_t dtypeSize, uint32_t totalLength)
    {
        static constexpr uint32_t FP32_TILE_CANDIDATES[] = {
            TILE_LENGTH_FP32, 6144, 4096, 2048
        };
        static constexpr uint32_t FP16_TILE_CANDIDATES[] = {
            12288, TILE_LENGTH_FP16, 8192, 6144
        };
        static constexpr uint32_t FP16_LARGE_TILE_CANDIDATES[] = {
            24576, TILE_LENGTH_FP16_LARGE, 18432, 16384, 12288, TILE_LENGTH_FP16, 8192
        };

        const uint32_t* candidates = FP32_TILE_CANDIDATES;
        uint32_t candidateCount = sizeof(FP32_TILE_CANDIDATES) / sizeof(FP32_TILE_CANDIDATES[0]);
        uint32_t defaultTileLength = TILE_LENGTH_FP32;
        if (dtypeSize == 2) {
            if (totalLength >= FP16_LARGE_TILE_THRESHOLD) {
                candidates = FP16_LARGE_TILE_CANDIDATES;
                candidateCount = sizeof(FP16_LARGE_TILE_CANDIDATES) / sizeof(FP16_LARGE_TILE_CANDIDATES[0]);
                defaultTileLength = TILE_LENGTH_FP16_LARGE;
            } else {
                candidates = FP16_TILE_CANDIDATES;
                candidateCount = sizeof(FP16_TILE_CANDIDATES) / sizeof(FP16_TILE_CANDIDATES[0]);
                defaultTileLength = TILE_LENGTH_FP16;
            }
        }

        TileSelection fallback = {defaultTileLength, 0};
        bool fallbackReady = false;
        for (uint32_t i = 0; i < candidateCount; ++i) {
            uint32_t candidate = candidates[i];
            std::vector<int64_t> tanhShapeVec = {static_cast<int64_t>(candidate)};
            ge::Shape tanhShape(tanhShapeVec);
            uint32_t tanhTmpMaxSize = 0;
            uint32_t tanhTmpMinSize = 0;
            AscendC::GetTanhMaxMinTmpSize(tanhShape, dtypeSize, false,
                tanhTmpMaxSize, tanhTmpMinSize);

            uint32_t availableTmpBytes = GetAvailableTmpBytes(platform, dtypeSize,
                candidate, dtypeSize == 4);
            uint32_t tanhTmpSize = SelectTmpSize(tanhTmpMaxSize, tanhTmpMinSize,
                availableTmpBytes);

            if (candidate == defaultTileLength) {
                fallback = {candidate, tanhTmpSize};
                fallbackReady = true;
            }
            if (tanhTmpSize > 0) {
                return {candidate, tanhTmpSize};
            }
        }

        return fallbackReady ? fallback : TileSelection{defaultTileLength, 0};
    }

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t numCoresAiv = platform.GetCoreNumAiv();

        const gert::Tensor *tensorInput = context->GetRequiredInputTensor(0);
        ge::DataType dtype = tensorInput->GetDataType();
        int dtypeSize = ge::GetSizeByDataType(dtype);
        uint32_t totalLength = tensorInput->GetShapeSize();

        uint32_t dtInputX = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, dtInputX);

        TileSelection tileSelection = SelectTileLength(platform,
            static_cast<uint32_t>(dtypeSize), totalLength);

        constexpr uint32_t ALIGN_UNIT_F32 = 8;  // 32B / 4B
        constexpr uint32_t ALIGN_UNIT_F16 = 16; // 32B / 2B
        uint32_t alignUnit = (dtypeSize == 2) ? ALIGN_UNIT_F16 : ALIGN_UNIT_F32;
        uint32_t maxCoresAiv = numCoresAiv > 0 ? static_cast<uint32_t>(numCoresAiv) : 1;
        uint32_t blockNum = SelectBlockNum(totalLength, maxCoresAiv, alignUnit);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        uint32_t alignedLength = (totalLength + alignUnit - 1) & ~(alignUnit - 1);
        tiling->blockNum = blockNum;
        tiling->totalLength = totalLength;
        tiling->alignedLength = alignedLength;
        tiling->alignUnit = alignUnit;
        tiling->numPerCore = (totalLength / blockNum) & ~(alignUnit - 1);
        if (tiling->numPerCore == 0) tiling->numPerCore = alignUnit;
        tiling->tailNumLastCore = totalLength - tiling->numPerCore * (blockNum - 1);
        tiling->tileLength = tileSelection.tileLength;
        tiling->tanhTmpSize = tileSelection.tanhTmpSize;

        context->SetBlockDim(blockNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}
