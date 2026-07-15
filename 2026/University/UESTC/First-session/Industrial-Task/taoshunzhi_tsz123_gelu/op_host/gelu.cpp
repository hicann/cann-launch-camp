#include <algorithm>
#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"
namespace optiling {
    // 用于双缓冲的缓冲区数量
    constexpr uint32_t BUFFER_NUM = 2;
    // 单次tiling处理的最大元素数（8192个元素，通常对应16KB? 实际以字节计算为准）
    constexpr uint32_t MAX_TILE_LENGTH = 8192;
    // UB（统一缓冲区）预留空间，确保系统或其他操作有足够剩余空间
    constexpr uint32_t UB_RESERVED_SIZE = 32 * 1024;
    // 向下对齐函数：将value按align向下取整，常用于内存对齐
    static uint32_t AlignDown(uint32_t value, uint32_t align) {
        if (align == 0) {
            return value;
        }
        return value / align * align;
    }
    // Tiling函数：在AICore上执行时，将数据切分为多个tile，并分配计算资源
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取昇腾平台相关信息（核心数、UB大小等）
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();          // AIV核心总数
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size); // UB总大小
        // 获取输入张量信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType(); // 数据类型（FP16/FP32）
        int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x); // 每个元素的字节数
        uint32_t length_input_x = tensor_input_x->GetShapeSize();   // 元素总个数
        // 将数据类型传递给tiling key（用于编译时选择特定kernel）
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);
        const uint32_t typeSize = static_cast<uint32_t>(dtype_size_input_x);
        // 根据数据类型计算最小对齐单位：确保tile长度能整除该值，便于向量化访问
        const uint32_t elementPerBlock = std::max<uint32_t>(1, 32 / typeSize);
        // 对于FP16，计算过程中可能需要额外空间（例如转换为float）——此处为2个float大小
        const uint32_t calcBytes = dtype_input_x == ge::DT_FLOAT16 ? sizeof(float) * 2 : 0;
        // 每个tile所需的总内存：输入、输出各一份，加上双缓冲（BUFFER_NUM）以及额外计算缓存
        const uint32_t bytesPerElement = typeSize * BUFFER_NUM * 2 + calcBytes;
        // 可用的UB大小（预留一部分系统使用）
        const uint64_t usableUb = ub_size > UB_RESERVED_SIZE ? ub_size - UB_RESERVED_SIZE : ub_size / 2;
        // 根据可用UB计算单个tile能容纳的最大元素数
        uint32_t tileLength = static_cast<uint32_t>(usableUb / bytesPerElement);
        // 对齐到elementPerBlock，并限制在MAX_TILE_LENGTH范围内
        tileLength = AlignDown(std::min<uint32_t>(tileLength, MAX_TILE_LENGTH), elementPerBlock);
        // 至少处理一个block
        tileLength = std::max<uint32_t>(tileLength, elementPerBlock);
        // 将tiling参数存储到自定义结构体中，供kernel使用
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;        // 总元素数
        tiling->tileLength = tileLength;        // 每个tile的元素数
        // 计算并行计算的核数（blockDim）：不超过总核数，同时根据数据量决定使用多少核
        const uint32_t blockDim = std::max<uint32_t>(1, std::min<uint32_t>(static_cast<uint32_t>(num_cores_aiv),
            std::max<uint32_t>(length_input_x, 1)));
        context->SetBlockDim(blockDim);         // 设置并行核数
        // 当前算子不需要额外workspace，置为0
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling
namespace ge {
    // 形状推导：输入输出形状相同（逐元素操作）
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    // 数据类型推导：输出数据类型与输入一致
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge
namespace ops {
    // Gelu算子定义类，继承自OpDef
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            // 定义输入：input_x，必须存在，支持FP16和FP32，格式为ND
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            // 定义输出：output，属性与输入相同
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            // 设置形状和数据类型推导函数
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            // 指定AICore后端的tiling函数，并声明支持ascend910b平台
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    // 全局注册Gelu算子
    OP_ADD(Gelu);
}  // namespace ops
