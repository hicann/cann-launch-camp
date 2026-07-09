// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fast_gelu_tiling.h"
#include "../op_kernel/tiling_key_fast_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 1. 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // 2. 获取输入 Tensor 信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype_x = tensor_x->GetDataType();
    int32_t dtype_size = ge::GetSizeByDataType(dtype_x);
    if (dtype_size <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 3. 设置 tiling key，用于 kernel 侧区分 float16 / float32
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 4. 获取输入元素总数
    uint64_t total_length = static_cast<uint64_t>(tensor_x->GetShapeSize());

    // 5. 计算 32B 对齐元素数
    // float16: 32 / 2 = 16
    // float32: 32 / 4 = 8
    uint32_t align_elem_num = static_cast<uint32_t>(32 / dtype_size);
    if (align_elem_num == 0) {
        align_elem_num = 1;
    }

    // 6. 按 32B block 计算总 block 数
    uint64_t total_block_num = 0;
    if (total_length > 0) {
        total_block_num = (total_length + align_elem_num - 1) / align_elem_num;
    }

    // 7. 实际使用核数
    

    uint32_t small_mode = 0;
    uint32_t used_core_num = 1;

    if (total_length > 0 && total_length <= 2048) {
        small_mode = 1;
        used_core_num = 1;
    } else {
        if (total_block_num > 0) {
            uint64_t core_num_u64 = static_cast<uint64_t>(num_cores_aiv);
            uint64_t used_core_num_u64 =
                total_block_num < core_num_u64 ? total_block_num : core_num_u64;
            used_core_num = static_cast<uint32_t>(used_core_num_u64);
        }
    }

    // 8. 多核均匀分配 32B block
    uint64_t base_block_num = 0;
    uint32_t tail_block_num = 0;

    if (total_block_num > 0 && used_core_num > 0) {
        base_block_num = total_block_num / used_core_num;
        tail_block_num = static_cast<uint32_t>(total_block_num % used_core_num);
    }

    // 9. 计算 tileLength
    //
    // 当前最优基线：
    //   total_length <= 24576 时使用 32768；
    //   其他情况使用 16384。
    //
    // 已验证：
    //   24576 明显比 8192 / 28672 / 32768 / block_num 分流更稳。
    uint32_t tile_length = 16384;

    if (total_length > 0 && total_length <= 24576) {
        tile_length = 32768;
    }

    if (small_mode == 1) {
        // 小 shape：tileLength 覆盖整个输入，并向上对齐到 32B 元素数。
        uint64_t small_tile = total_length;
        if (small_tile == 0) {
            small_tile = align_elem_num;
        }

        small_tile = ((small_tile + align_elem_num - 1) / align_elem_num) * align_elem_num;
        tile_length = static_cast<uint32_t>(small_tile);
    } else {
        if (ub_size > 0) {
            uint64_t reserve_size = 8192;
            uint64_t usable_ub_size =
                ub_size > reserve_size ? (ub_size - reserve_size) : ub_size;

            // kernel 当前主要使用 x/y/den 三份 UB buffer
            uint64_t max_tile_by_ub =
                usable_ub_size / (3 * static_cast<uint64_t>(dtype_size));

            if (max_tile_by_ub > 0 && max_tile_by_ub < tile_length) {
                tile_length = static_cast<uint32_t>(max_tile_by_ub);
            }
        }

        // tileLength 向下对齐到 32B 元素数，保证 full tile 可走普通 DataCopy
        if (tile_length >= align_elem_num) {
            tile_length = (tile_length / align_elem_num) * align_elem_num;
        }

        if (tile_length < align_elem_num) {
            tile_length = align_elem_num;
        }
    }

    // 10. 填充 tiling 参数
    FastGeluTilingData *tiling = context->GetTilingData<FastGeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength = total_length;
    tiling->usedCoreNum = used_core_num;
    tiling->alignElemNum = align_elem_num;
    tiling->baseBlockNum = base_block_num;
    tiling->tailBlockNum = tail_block_num;
    tiling->tileLength = tile_length;
    tiling->smallMode = small_mode;

    // 11. 设置启动核数
    context->SetBlockDim(used_core_num);

    // 12. 当前算子不需要 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);

    if (x_shape == nullptr || y_shape == nullptr) {
        return GRAPH_FAILED;
    }

    *y_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    ge::DataType x_dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, x_dtype);
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FastGelu);
}  // namespace ops