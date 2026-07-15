    // Host侧Tiling实现
    #include "register/op_def_registry.h"

    #include "tiling/platform/platform_ascendc.h"

    #include "../op_kernel/gelu_tiling.h"

    #include "../op_kernel/tiling_key_gelu.h"

    namespace optiling {
        static constexpr uint32_t BLOCK_SIZE = 32;

        static ge::graphStatus TilingFunc(gert::TilingContext *context) {
            auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
            uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
            uint64_t ubSize;
            platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

            const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
            ge::DataType dtype = tensorX->GetDataType();
            // GetShapeSize() 返回 int64_t, 用 64 位承载以支持超大张量(避免 32 位截断)
            uint64_t inputNum = static_cast<uint64_t>(tensorX->GetShapeSize());

            uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype);
            ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

            const bool isHalf = (dtype == ge::DT_FLOAT16);
            const uint32_t typeLength = isHalf ? 2u : 4u;
            const uint32_t blockElems = BLOCK_SIZE / typeLength;

            // 字节数/块数用 64 位, 避免 inputNum*typeLength 在 32 位下溢出
            uint64_t inputLength = inputNum * static_cast<uint64_t>(typeLength);
            uint64_t inputLengthAlign32 = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
            uint64_t totalBlockNum = inputLengthAlign32 / BLOCK_SIZE;

            // half: 三档选核 ≤1024 单核 / ≤8192 每核≥1024 / >8192 满核
            // float: 平滑扩核 ≤1024 单核 / 其余 ceil(n/1024) 上限 maxCores, 消除 8193 处 8核→40核 cliff
            uint32_t maxCores = (coreNum == 0) ? 1u : coreNum;
            constexpr uint32_t kTinyTensorElems = 1024;
            constexpr uint32_t kMinElemsPerCore = 1024;
            if (isHalf) {
                constexpr uint32_t kSmallTensorElems = 8192;
                if (inputNum <= kTinyTensorElems) {
                    coreNum = 1;
                } else if (inputNum <= kSmallTensorElems) {
                    coreNum = static_cast<uint32_t>((inputNum + kMinElemsPerCore - 1) / kMinElemsPerCore);
                } else {
                    coreNum = maxCores;
                }
            } else {
                if (inputNum <= kTinyTensorElems) {
                    coreNum = 1;
                } else {
                    coreNum = static_cast<uint32_t>((inputNum + kMinElemsPerCore - 1) / kMinElemsPerCore);
                }
            }
            if (coreNum > maxCores) {
                coreNum = maxCores;
            }
            if (totalBlockNum > 0 && static_cast<uint64_t>(coreNum) > totalBlockNum) {
                coreNum = static_cast<uint32_t>(totalBlockNum);
            }
            if (coreNum == 0) {
                coreNum = 1;
            }

            uint64_t everyCoreBlockNum = totalBlockNum / coreNum;
            // 余数块数 < coreNum(<=核数), 恒小, 安全收窄为 32 位
            uint32_t tailBlockNum = static_cast<uint32_t>(totalBlockNum % coreNum);

            // 队列深度需与 kernel 侧 kQueDepth 一致: 均为双缓冲(实测三缓冲挤小 tile 反而更慢, 已回退)
            //   half : 2*2*2 = 8B/elem(双缓冲 in/out), Gelu 框架自申请 tmp
            //   float: 2*4*2 + 4 = 20B/elem(双缓冲 in/out + sigmoid fp32 工作区)
            constexpr uint32_t kKernelBufferNum = 2;
            uint32_t ownBytesPerElem = kKernelBufferNum * typeLength * 2u;
            if (!isHalf) {
                ownBytesPerElem += static_cast<uint32_t>(sizeof(float));
            }
            constexpr uint32_t kReservedUb = 8 * 1024;
            uint64_t usable = (ubSize > kReservedUb) ? (ubSize - kReservedUb) : ubSize;
            // 自有缓冲固定占 50% UB, 其余留给高阶 API 自动临时区.
            // 注: 实测调高比例(65%/自适应分档)会全线变慢或落入评测噪声, 故 50% 为最优, 不再改动.
            uint32_t tileDataNum = static_cast<uint32_t>((usable / 2) / ownBytesPerElem);
            tileDataNum = (tileDataNum / blockElems) * blockElems;
            if (tileDataNum == 0) {
                tileDataNum = blockElems;
            }
            uint32_t tileBlockNum = tileDataNum / blockElems;

            // 每核块数/元素数用 64 位, 避免超大张量单核负载在 32 位下溢出
            uint64_t smallCoreBlockNum = everyCoreBlockNum;
            uint64_t bigCoreBlockNum = everyCoreBlockNum + 1;
            uint64_t smallCoreDataNum = smallCoreBlockNum * blockElems;
            uint64_t bigCoreDataNum = bigCoreBlockNum * blockElems;

            uint64_t maxCoreBlockNum = (tailBlockNum > 0) ? bigCoreBlockNum : smallCoreBlockNum;
            // 循环次数与每片块数受 UB 与核数约束, 恒小于 2^32, 安全收窄为 32 位
            uint32_t finalTileNum = static_cast<uint32_t>((maxCoreBlockNum + tileBlockNum - 1) / tileBlockNum);
            if (finalTileNum == 0) {
                finalTileNum = 1;
            }

            uint32_t bigBaseTileBlock = static_cast<uint32_t>(bigCoreBlockNum / finalTileNum);
            uint32_t bigRemTiles = static_cast<uint32_t>(bigCoreBlockNum % finalTileNum);
            uint32_t smallBaseTileBlock = static_cast<uint32_t>(smallCoreBlockNum / finalTileNum);
            uint32_t smallRemTiles = static_cast<uint32_t>(smallCoreBlockNum % finalTileNum);

            GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
            tiling->length = inputNum;
            tiling->tailBlockNum = tailBlockNum;
            tiling->bigCoreDataNum = bigCoreDataNum;
            tiling->smallCoreDataNum = smallCoreDataNum;
            tiling->smallCoreBaseOffset = bigCoreDataNum * tailBlockNum;
            tiling->tileDataNum = tileDataNum;
            tiling->finalTileNum = finalTileNum;
            tiling->bigBaseTileBlock = bigBaseTileBlock;
            tiling->bigRemTiles = bigRemTiles;
            tiling->smallBaseTileBlock = smallBaseTileBlock;
            tiling->smallRemTiles = smallRemTiles;

            context->SetBlockDim(coreNum);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }
    }  // namespace optiling

    namespace ge {
        static graphStatus InferShape(gert::InferShapeContext *context) {
            const gert::Shape *x_shape = context->GetInputShape(0);
            gert::Shape *y_shape = context->GetOutputShape(0);
            *y_shape = *x_shape;
            return GRAPH_SUCCESS;
        }
        static graphStatus InferDataType(gert::InferDataTypeContext *context) {
            context->SetOutputDataType(0, context->GetInputDataType(0));
            return ge::GRAPH_SUCCESS;
        }
    }  // namespace ge

    namespace ops {
        class Gelu : public OpDef {
        public:
            explicit Gelu(const char *name) : OpDef(name) {
                this->Input("input_x")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
                this->Output("output")
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
        OP_ADD(Gelu);
    }  // namespace ops
