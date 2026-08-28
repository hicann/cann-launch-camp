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
        if (num_cores_aiv <= 0) {
            num_cores_aiv = 1;
        }
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取算子输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_clip_value_min = context->GetRequiredInputTensor(1);
        const gert::Tensor *tensor_clip_value_max = context->GetRequiredInputTensor(2);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        // 检测 clip_value_min / clip_value_max 是否为标量
        uint32_t min_is_scalar = (tensor_clip_value_min->GetShapeSize() == 1) ? 1 : 0;
        uint32_t max_is_scalar = (tensor_clip_value_max->GetShapeSize() == 1) ? 1 : 0;

        // 计算 SCALAR_MODE 编码（0/1/2/3 四种模式，对应 4 个 kernel 编译期特化）
        //   bit0=1 表示 min 是张量，bit1=1 表示 max 是张量
        //   0: 全标量   1: min张量,max标量   2: min标量,max张量   3: 双张量
        uint32_t scalar_mode = (min_is_scalar ? 0 : 1) | (max_is_scalar ? 0 : 2);

        // 配置 tiling key（按 DT_X + SCALAR_MODE 选择 kernel 特化版本）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X, scalar_mode);

        // ========== 多核切分 ==========
        uint32_t align_elements = 32 / static_cast<uint32_t>(dtype_size_x);
        if (align_elements == 0) {
            align_elements = 1;
        }
 // ========== 多核切分 ==========
        if (align_elements == 0) {
            align_elements = 1;
        }

        uint32_t block_length;
        if (length_x == 0) {
            block_length = 0;
            num_cores_aiv = 1;
        } else if (static_cast<uint32_t>(num_cores_aiv) >= length_x) {
            // 数据量小于核数，每核处理1个元素
            block_length = 1;
            num_cores_aiv = static_cast<int32_t>(length_x);
        } else {
            block_length = (length_x + num_cores_aiv - 1) / num_cores_aiv;
            // 对齐到 32 字节，确保每核 GM 起始地址对齐，DataCopy 可用快速路径
            block_length = (block_length + align_elements - 1) / align_elements * align_elements;
            // 对齐后重新计算实际核数，避免尾核负载严重不均
            uint32_t actual_cores = (length_x + block_length - 1) / block_length;
            if (actual_cores < static_cast<uint32_t>(num_cores_aiv)) {
                num_cores_aiv = static_cast<int32_t>(actual_cores);
            }
        }

        // 预留 1024B（标量 buffer + pipe 内部管理 + TQue 书签）
        uint32_t reserved_ub = 1024;
        uint32_t available_ub = static_cast<uint32_t>(ub_size) - reserved_ub;

        // ========== 直通检测：小数据单 tile 直通，跳过 TQue 流水线 ==========
        // 条件：x+y(+min+max) 全部能一次装入 UB
        // 注：广播张量由框架补齐 GM 分配，读取时不会越界，无需 same_length 检查
        uint32_t total_bufs = 2;  // x + y
        if (!min_is_scalar) total_bufs++;
        if (!max_is_scalar) total_bufs++;
        uint32_t data_bytes = length_x * static_cast<uint32_t>(dtype_size_x);
        bool pass_through = (data_bytes * total_bufs + reserved_ub <= available_ub);

        uint32_t buffer_depth = 2;  // 固定双缓冲（放弃自适应三缓冲）
        uint32_t buffer_count;
        uint32_t tile_length;

        if (pass_through) {
            // 直通模式：单核、一次搬运、无 TQue 开销
            num_cores_aiv = 1;
            block_length = length_x;
            buffer_depth = 0;  // 哨兵，kernel 走直通路径
            buffer_count = 0;
            tile_length = block_length;
        } else {
            // ========== UB 切分（统一双缓冲）==========
            if (min_is_scalar && max_is_scalar) {
                buffer_count = 4;  // x2 + y2
            } else if (min_is_scalar || max_is_scalar) {
                buffer_count = 6;  // x2 + y2 + 1 个张量 ×2
            } else {
                buffer_count = 8;  // x2 + y2 + min2 + max2（始终双缓冲）
            }

            tile_length = available_ub / (buffer_count * static_cast<uint32_t>(dtype_size_x));
            tile_length = (tile_length / align_elements) * align_elements;
            if (tile_length == 0) {
                tile_length = align_elements;
            }
        }
 // 填充 tiling 结构体
        ClipByValueTilingData *tiling = context->GetTilingData<ClipByValueTilingData>();
        tiling->totalLength = length_x;
        tiling->blockLength = block_length;
        tiling->tileLength = tile_length;
        tiling->minIsScalar = min_is_scalar;
        tiling->maxIsScalar = max_is_scalar;
        tiling->bufferDepth = buffer_depth;

        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置 workspace 大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const ge::DataType xDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, xDtype);
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