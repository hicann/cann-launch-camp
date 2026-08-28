// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/clip_by_value_tiling.h"
#include "../op_kernel/tiling_key_clip_by_value.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 获取算子输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_clip_value_min = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_clip_value_max = context->GetRequiredInputTensor(2);

        // 获取数据类型
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);

        // 获取元素个数
        uint32_t length_x = tensor_x->GetShapeSize();

        // 判断clip_value_min/max是否为标量
        uint32_t min_size = tensor_clip_value_min->GetShapeSize();
        uint32_t max_size = tensor_clip_value_max->GetShapeSize();
        uint32_t minIsScalar = (min_size == 1) ? 1 : 0;
        uint32_t maxIsScalar = (max_size == 1) ? 1 : 0;

        // 配置tiling key, 实现kernel侧不同数据类型的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 每个元素的字节数 & 32字节对齐的元素个数
        uint32_t elementSize = dtype_size_x;
        uint32_t alignNum = 32 / elementSize;

        // 原始总元素数
        uint32_t totalLength = length_x;

        // blockSize：根据广播模式动态计算，充分利用UB空间
        uint32_t queueCount = 2 + (minIsScalar ? 0 : 1) + (maxIsScalar ? 0 : 1);
        uint32_t ubBudget = 192 * 1024 * 7 / 8;
        uint32_t blockSize;

        // 先用 BUFFER_NUM=3 估算
        uint32_t maxByUb3 = ubBudget / (queueCount * 3) / elementSize;
        blockSize = maxByUb3;
        blockSize = (blockSize / alignNum) * alignNum;
        if (blockSize > 8192) blockSize = 8192;
        if (blockSize < alignNum) blockSize = alignNum;

        // 总块数（向上取整）
        uint32_t totalBlockNum = (totalLength + blockSize - 1) / blockSize;

        // 小数据场景(totalBlockNum<=1)：Kernel侧bufNum=1，blockSize可放大3倍
        if (totalBlockNum <= 1 && totalLength > 0) {
            uint32_t maxByUb1 = ubBudget / queueCount / elementSize;
            maxByUb1 = (maxByUb1 / alignNum) * alignNum;
            if (maxByUb1 > 32768) maxByUb1 = 32768;
            if (maxByUb1 < alignNum) maxByUb1 = alignNum;
            if (maxByUb1 > 0) {
                blockSize = maxByUb1;
                totalBlockNum = (totalLength + blockSize - 1) / blockSize;
            }
        }

        // 决定启动核数：不超过可用核数，也不超过总块数
        uint32_t coreNum = (uint32_t)num_cores_aiv;
        if (coreNum == 0) {
            coreNum = 1;
        }
        if (totalBlockNum > 0 && coreNum > totalBlockNum) {
            coreNum = totalBlockNum;
        }
        if (totalBlockNum == 0) {
            coreNum = 1;
        }

        // 将 totalBlockNum 个对齐块尽量均匀地分到 coreNum 个核上
        uint32_t avgBlocksPerCore = totalBlockNum / coreNum;
        uint32_t remBlocks = totalBlockNum % coreNum;

        // 填充tiling结构体
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->totalLength = totalLength;
        tiling->alignNum = alignNum;
        tiling->blockSize = blockSize;
        tiling->totalBlockNum = totalBlockNum;
        tiling->coreNum = coreNum;
        tiling->avgBlocksPerCore = avgBlocksPerCore;
        tiling->remBlocks = remBlocks;
        tiling->minIsScalar = minIsScalar;
        tiling->maxIsScalar = maxIsScalar;

        // 配置启动核数
        context->SetBlockDim(coreNum);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        *output_shape = *input_shape;

        return ge::GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        auto input_x = context->GetRequiredInputDesc(0);
        ge::DataType dtype = input_x->GetDataType();

        context->SetOutputDataType(0, dtype);

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
