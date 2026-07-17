#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

/**
 * @brief 匿名命名空间，存放本文件内部使用的辅助函数
 *        包括广播维度计算、内存对齐、连续运行区间判断等工具函数
 */
namespace {

/**
 * @brief 计算两个维度在广播后的输出维度
 *
 * 遵循 NumPy 广播规则：
 *   - 两个维度相等，输出该维度
 *   - 其中一个为 1，输出另一个
 *   - 其他情况无法广播，返回 false
 *
 * @param d1 [in]  第一个输入张量在当前轴上的维度大小
 * @param d2 [in]  第二个输入张量在当前轴上的维度大小
 * @param out [out] 广播后的输出维度大小
 * @return true  广播成功，out 已写入有效值
 * @return false 广播失败（维度不兼容或存在负值）
 */
inline bool GetBroadcastDim(int64_t d1, int64_t d2, int64_t &out)
{
    // 负值维度非法，直接返回失败
    if (d1 < 0 || d2 < 0)
        return false;
    // 两维度相同，直接取该值
    if (d1 == d2) {
        out = d1;
        return true;
    }
    // d1 为 1（标量广播），取 d2
    if (d1 == 1) {
        out = d2;
        return true;
    }
    // d2 为 1（标量广播），取 d1
    if (d2 == 1) {
        out = d1;
        return true;
    }
    // 两个维度不同且都不为 1，无法广播
    return false;
}

/**
 * @brief 将值 v 向上对齐到 a 的倍数
 *
 * @param v [in] 待对齐的值
 * @param a [in] 对齐粒度（必须 > 0）
 * @return uint64_t 对齐后的值，满足 result >= v 且 result % a == 0
 */
inline uint64_t AlignUp(uint64_t v, uint64_t a)
{
    return ((v + a - 1) / a) * a;
}

/**
 * @brief 尝试将当前轴扩展到连续运行区间（vector run）中
 *
 * 在从最内层轴向外逐轴扫描时，判断当前轴是否可以并入一段连续的
 * 向量运行区间。一个轴能被纳入连续区间需满足以下条件之一：
 *   - stride == 0       → 该轴上为标量广播（SCALAR）
 *   - stride == currentRun → 该轴上数据在内存中连续（CONTIGUOUS）
 *
 * 同时要求同一张量所有纳入区间的轴具有相同的运行模式
 * （全标量或全连续），否则区间在此轴截断。
 *
 * @param stride     [in]  当前轴的步长（0 表示标量广播）
 * @param dim        [in]  当前轴的维度大小
 * @param currentRun [in]  当前已累积的连续运行区间长度
 * @param state      [in,out] 当前运行模式状态：
 *                             -1 表示尚未初始化；
 *                             其他值为已确定的模式（SCALAR 或 CONTIGUOUS）
 * @return true  当前轴可纳入连续运行区间
 * @return false 当前轴不可纳入，应在此截断
 */
inline bool ExtendRun(uint64_t stride, uint64_t dim, uint64_t currentRun, int32_t &state)
{
    // 维度 <= 1 的轴不影响连续性，直接跳过
    if (dim <= 1)
        return true;

    int32_t axisState = -1;
    // stride 为 0 → 标量广播模式
    if (stride == 0)
        axisState = static_cast<int32_t>(LESS_EQUAL_RUN_SCALAR);
    // stride 等于当前累积长度 → 连续模式
    else if (stride == currentRun)
        axisState = static_cast<int32_t>(LESS_EQUAL_RUN_CONTIGUOUS);
    // 既不是标量也不是连续，无法扩展
    else
        return false;

    // 首次设置状态
    if (state < 0) {
        state = axisState;
        return true;
    }
    // 后续轴的模式必须与已确定模式一致
    return state == axisState;
}

}  // namespace

/**
 * @brief Tiling 函数命名空间
 *        负责在 host 侧计算 LessEqual 算子的所有 tiling 参数，
 *        包括广播后的输出形状、各输入步长、数据分块策略、
 *        多核切分方案等，并写入 TilingData 供 kernel 侧使用。
 */
namespace optiling {

/**
 * @brief LessEqual 算子的 Tiling 主函数
 *
 * 该函数完成以下工作：
 *   1. 获取平台信息（可用 AI Vector 核心数）
 *   2. 读取两个输入张量的形状与数据类型，执行校验
 *   3. 对齐两个输入的维度（右对齐填充 1），计算广播后的输出形状
 *   4. 计算每个输入在各轴上的步长（broadcast strides）
 *   5. 根据输入与输出的元素总数关系，确定计算模式
 *      （DIRECT / X1_SCALAR / X2_SCALAR / GENERAL_BROADCAST）
 *   6. 从最内层轴向外扫描，计算连续运行区间长度及运行模式
 *   7. 根据 totalLength 选择多核切分策略并设置 BlockDim
 *   8. 计算每个核处理的 blockLength（32 字节对齐）
 *   9. 将所有 tiling 参数写入 TilingData 结构体
 *
 * @param context [in] Tiling 上下文指针，由框架提供，
 *                     包含平台信息、输入张量描述、
 *                     以及 TilingData 输出缓冲区
 * @return ge::graphStatus GRAPH_SUCCESS 表示成功，GRAPH_FAILED 表示失败
 */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // ---- 1. 获取平台信息 ----
    // 从上下文中获取平台信息并构造 PlatformAscendC 对象
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 获取可用的 AI Vector 核心数，至少为 1
    const uint32_t availableCores = static_cast<uint32_t>(std::max<int32_t>(1, platform.GetCoreNumAiv()));

    // ---- 2. 读取输入张量信息 ----
    // 获取第一个输入张量 x1 的描述
    const gert::Tensor *x1 = context->GetRequiredInputTensor(0);
    // 获取第二个输入张量 x2 的描述
    const gert::Tensor *x2 = context->GetRequiredInputTensor(1);

    // 校验：输入指针非空且两个输入数据类型一致
    if (x1 == nullptr || x2 == nullptr || x1->GetDataType() != x2->GetDataType())
        return ge::GRAPH_FAILED;

    // 获取 x1 和 x2 的原始形状（OriginShape，未经任何排布转换）
    const auto shape1 = x1->GetOriginShape();
    const auto shape2 = x2->GetOriginShape();

    // dim1: x1 的维度数; dim2: x2 的维度数
    const uint32_t dim1 = shape1.GetDimNum();
    const uint32_t dim2 = shape2.GetDimNum();
    // rank: 两者中较大的维度数，作为广播后的统一维度数
    const uint32_t rank = std::max(dim1, dim2);

    // 维度数超过预定义上限，返回失败
    if (rank > LESS_EQUAL_MAX_DIMS)
        return ge::GRAPH_FAILED;

    // ---- 3. 维度对齐与输出形状计算 ----
    // aligned1 / aligned2: 将 x1 / x2 的形状右对齐到 rank 维，高位补 1
    // outDims: 广播后输出在各轴上的维度
    uint64_t aligned1[LESS_EQUAL_MAX_DIMS] = {0};
    uint64_t aligned2[LESS_EQUAL_MAX_DIMS] = {0};
    uint64_t outDims[LESS_EQUAL_MAX_DIMS] = {0};

    // 初始化所有维度为 1（用于高位填充）
    for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
        aligned1[i] = 1;
        aligned2[i] = 1;
        outDims[i] = 1;
    }

    // 将 x1 的各维度值填入 aligned1 的右对齐位置
    for (uint32_t i = 0; i < dim1; ++i) {
        const int64_t value = shape1.GetDim(i);
        if (value < 0)
            return ge::GRAPH_FAILED;
        aligned1[rank - dim1 + i] = static_cast<uint64_t>(value);
    }

    // 将 x2 的各维度值填入 aligned2 的右对齐位置
    for (uint32_t i = 0; i < dim2; ++i) {
        const int64_t value = shape2.GetDim(i);
        if (value < 0)
            return ge::GRAPH_FAILED;
        aligned2[rank - dim2 + i] = static_cast<uint64_t>(value);
    }

    // 逐轴计算广播后的输出维度，并累乘得到总元素数 totalLength
    uint64_t totalLength = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t outputDim = 0;
        // 对每个轴调用 GetBroadcastDim 进行广播计算
        if (!GetBroadcastDim(static_cast<int64_t>(aligned1[i]), static_cast<int64_t>(aligned2[i]), outputDim))
            return ge::GRAPH_FAILED;
        outDims[i] = static_cast<uint64_t>(outputDim);
        // 溢出检查：确保 totalLength * outDims[i] 不会溢出
        if (outDims[i] != 0 && totalLength > std::numeric_limits<uint64_t>::max() / outDims[i])
            return ge::GRAPH_FAILED;
        totalLength *= outDims[i];
    }

    // x1Length / x2Length: 两个输入张量的实际元素总数
    const uint64_t x1Length = static_cast<uint64_t>(x1->GetShapeSize());
    const uint64_t x2Length = static_cast<uint64_t>(x2->GetShapeSize());

    // ---- 4. 写入 TilingData 基本信息 ----
    // 获取 TilingData 输出缓冲区
    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr)
        return ge::GRAPH_FAILED;

    // 写入总元素数和维度数
    tiling->totalLength = totalLength;
    tiling->rank = rank;

    // ---- 5. 设置 tileLength（每个核内单次处理的元素块大小） ----
    // 小数据（<= 2048）时按 32 字节对齐一次性处理
    // 大数据时固定 4096 元素为一个 tile
    if (totalLength <= 2048) {
        tiling->tileLength = static_cast<uint32_t>(AlignUp(totalLength, 32));
    } else {
        tiling->tileLength = 4096;
    }

    // ---- 6. 计算各输入的广播步长 ----
    // 初始化 outDims、x1Strides、x2Strides 数组
    for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
        tiling->outDims[i] = 1;
        tiling->x1Strides[i] = 0;
        tiling->x2Strides[i] = 0;
    }

    // 从最内层轴（最右侧）向外逐轴计算步长
    // stride: 当前轴的步长 = 更内层所有轴维度值的乘积
    // 如果某轴维度为 1（广播轴），步长设为 0
    uint64_t stride1 = 1, stride2 = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
        const uint32_t index = static_cast<uint32_t>(axis);
        tiling->outDims[index] = outDims[index];
        // aligned1[index] == 1 → x1 在此轴广播，步长为 0
        tiling->x1Strides[index] = (aligned1[index] == 1) ? 0 : stride1;
        // aligned2[index] == 1 → x2 在此轴广播，步长为 0
        tiling->x2Strides[index] = (aligned2[index] == 1) ? 0 : stride2;
        // 更新步长：乘以当前轴的维度值
        stride1 *= aligned1[index];
        stride2 *= aligned2[index];
    }

    // ---- 7. 确定计算模式 ----
    // x1Direct / x2Direct: 输入元素数是否等于输出元素数（即该输入无需广播）
    const bool x1Direct = (x1Length == totalLength);
    const bool x2Direct = (x2Length == totalLength);

    // 根据输入与输出的元素数关系选择模式：
    //   DIRECT          → 两个输入都无需广播
    //   X1_SCALAR       → x1 为标量（1 个元素），x2 无需广播
    //   X2_SCALAR       → x2 为标量，x1 无需广播
    //   GENERAL_BROADCAST → 通用广播模式
    if (x1Direct && x2Direct)
        tiling->mode = LESS_EQUAL_DIRECT;
    else if (x1Length == 1 && x2Direct)
        tiling->mode = LESS_EQUAL_X1_SCALAR;
    else if (x2Length == 1 && x1Direct)
        tiling->mode = LESS_EQUAL_X2_SCALAR;
    else
        tiling->mode = LESS_EQUAL_GENERAL_BROADCAST;

    // ---- 8. 计算连续运行区间（vector run） ----
    // vectorRunLength: 从最内层轴开始，能连续处理的元素数
    // x1State / x2State: x1 / x2 在连续区间内的运行模式
    uint64_t vectorRunLength = 1;
    int32_t x1State = -1, x2State = -1;

    // totalLength 不为 0 时才需要扫描
    if (totalLength != 0) {
        // 从最内层轴向外逐轴扫描
        for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
            const uint32_t index = static_cast<uint32_t>(axis);
            const uint64_t dim = outDims[index];
            // 尝试将 x1 和 x2 的当前轴纳入连续区间
            if (!ExtendRun(tiling->x1Strides[index], dim, vectorRunLength, x1State) ||
                !ExtendRun(tiling->x2Strides[index], dim, vectorRunLength, x2State))
                break;  // 无法扩展，截断
            // 成功扩展，乘入当前轴维度
            vectorRunLength *= dim;
        }
    }

    // 写入连续运行区间长度和运行模式
    tiling->vectorRunLength = vectorRunLength;
    // x1 运行模式：标量或连续
    tiling->x1RunMode =
        (x1State == static_cast<int32_t>(LESS_EQUAL_RUN_SCALAR)) ? LESS_EQUAL_RUN_SCALAR : LESS_EQUAL_RUN_CONTIGUOUS;
    // x2 运行模式：标量或连续
    tiling->x2RunMode =
        (x2State == static_cast<int32_t>(LESS_EQUAL_RUN_SCALAR)) ? LESS_EQUAL_RUN_SCALAR : LESS_EQUAL_RUN_CONTIGUOUS;

    // ---- 9. 多核切分策略 ----
    // SMALL_DATA: 小数据阈值，低于此值时只使用单核
    // ELEMS_PER_CORE: 每个核最少处理的元素数
    constexpr uint64_t SMALL_DATA = 2048;
    constexpr uint64_t ELEMS_PER_CORE = 1024;

    // 默认单核
    uint32_t coreNum = 1;
    if (totalLength > SMALL_DATA) {
        // 期望核数 = 总元素数 / 每核元素数（向上取整）
        const uint64_t requested = (totalLength + ELEMS_PER_CORE - 1) / ELEMS_PER_CORE;
        // 实际核数 = min(可用核数, 期望核数)，且至少为 1
        coreNum = static_cast<uint32_t>(std::min<uint64_t>(availableCores, std::max<uint64_t>(1, requested)));
    }

    // 设置 BlockDim（核数）
    context->SetBlockDim(coreNum);

    // ---- 10. 计算 blockLength（每个核处理的数据块大小） ----
    uint64_t blockLength = 0;
    if (totalLength > 0) {
        // 均分到各核：totalLength / coreNum（向上取整）
        blockLength = (totalLength + coreNum - 1) / coreNum;
        // 对齐到 32 字节（256 bit，AI Core 向量指令的对齐要求）
        blockLength = AlignUp(blockLength, 32);
    }
    tiling->blockLength = blockLength;

    // ---- 11. Workspace 与模板选择 ----
    // 申请 1 个 workspace 大小（当前算子不需要 workspace，设为 0）
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    // 获取输入数据类型，用于模板参数选择
    const uint32_t dtype = static_cast<uint32_t>(x1->GetDataType());
    // 根据数据类型选择对应的 kernel 模板
    ASCENDC_TPL_SEL_PARAM(context, dtype);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

/**
 * @brief 图引擎（Graph Engine）命名空间
 *        包含算子的形状推导与数据类型推导函数
 */
namespace ge {

/**
 * @brief LessEqual 算子的输出形状推导函数
 *
 * 根据两个输入的形状，按照广播规则推导输出形状。
 * 从最右侧轴开始逐轴对齐，对每对维度调用 GetBroadcastDim
 * 计算输出维度，最终构建完整的输出 Shape。
 *
 * @param context [in] 形状推导上下文，提供输入形状并接收输出形状
 * @return graphStatus GRAPH_SUCCESS 表示成功，GRAPH_FAILED 表示失败
 */
static graphStatus InferShape(gert::InferShapeContext *context)
{
    // 获取两个输入的形状和输出形状的指针
    const gert::Shape *shape1 = context->GetInputShape(0);
    const gert::Shape *shape2 = context->GetInputShape(1);
    gert::Shape *output = context->GetOutputShape(0);
    // 校验指针非空
    if (shape1 == nullptr || shape2 == nullptr || output == nullptr)
        return GRAPH_FAILED;

    // dim1 / dim2: 两个输入的维度数
    const int64_t dim1 = shape1->GetDimNum();
    const int64_t dim2 = shape2->GetDimNum();
    // rank: 较大的维度数
    const int64_t rank = std::max(dim1, dim2);

    // 维度数超过上限，返回失败
    if (rank > static_cast<int64_t>(LESS_EQUAL_MAX_DIMS))
        return GRAPH_FAILED;

    // result: 存储推导出的输出各维度值，初始全为 1
    std::vector<int64_t> result(static_cast<size_t>(rank), 1);
    // 从最右侧轴（right=0）开始逐轴向左对齐
    for (int64_t right = 0; right < rank; ++right) {
        // 从右向左取各输入的维度值，不存在则为 1
        const int64_t d1 = (right < dim1) ? shape1->GetDim(dim1 - 1 - right) : 1;
        const int64_t d2 = (right < dim2) ? shape2->GetDim(dim2 - 1 - right) : 1;
        int64_t outDim = 0;
        // 广播计算
        if (!GetBroadcastDim(d1, d2, outDim))
            return GRAPH_FAILED;
        // 将结果写入对应位置（从右向左填充）
        result[static_cast<size_t>(rank - 1 - right)] = outDim;
    }
    // 设置输出形状的维度数和各维度值
    output->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i)
        output->SetDim(i, result[static_cast<size_t>(i)]);
    return GRAPH_SUCCESS;
}

/**
 * @brief LessEqual 算子的输出数据类型推导函数
 *
 * LessEqual 的输出始终为布尔类型（DT_BOOL），
 * 与输入数据类型无关。
 *
 * @param context [in] 数据类型推导上下文
 * @return graphStatus GRAPH_SUCCESS 表示成功
 */
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // 输出固定为 bool 类型
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}

}  // namespace ge

/**
 * @brief 算子定义命名空间
 *        使用 OpDef 框架注册 LessEqual 算子，
 *        声明输入/输出的参数类型、数据类型、格式，
 *        并绑定 Tiling 函数和形状推导函数。
 */
namespace ops {

/**
 * @brief LessEqual 算子定义类
 *
 * 继承自 OpDef，在构造函数中完成算子的完整注册：
 *   1. 声明两个必需输入 x1、x2（支持 FLOAT16/FLOAT/INT32/INT8）
 *   2. 声明一个必需输出 y（固定为 BOOL 类型）
 *   3. 绑定形状推导函数 InferShape 和数据类型推导函数 InferDataType
 *   4. 绑定 Tiling 函数并指定支持的硬件平台（ascend910b）
 */
class LessEqual : public OpDef {
public:
    /**
     * @brief 构造函数，完成算子注册
     *
     * @param name [in] 算子名称，通常为 "LessEqual"
     */
    explicit LessEqual(const char *name) : OpDef(name)
    {
        // ---- 输入 x1 声明 ----
        // 参数类型：必需（REQUIRED）
        // 支持数据类型：FLOAT16, FLOAT, INT32, INT8
        // 支持格式：ND（任意维度）
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 输入 x2 声明 ----
        // 参数类型与数据类型格式要求与 x1 相同
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 输出 y 声明 ----
        // 参数类型：必需（REQUIRED）
        // 数据类型：固定为 BOOL
        // 格式：ND
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 绑定推导函数 ----
        // 设置形状推导函数和数据类型推导函数
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // ---- 绑定 Tiling 函数并指定硬件平台 ----
        // AICore 上设置 Tiling 函数，并添加 ascend910b 平台配置
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

// 注册算子到算子注册表
OP_ADD(LessEqual);

}  // namespace ops