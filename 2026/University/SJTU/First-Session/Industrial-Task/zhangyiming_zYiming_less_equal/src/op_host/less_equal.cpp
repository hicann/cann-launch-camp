#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <cstdint>
#include <limits>

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"

/**
 * @file less_equal.cpp
 * @brief LessEqual算子Host侧实现
 * 
 * 该文件实现了LessEqual算子在Host侧的核心逻辑，包括：
 * - Tiling配置计算（分块策略和并行处理配置）
 * - 形状推断（InferShape）
 * - 数据类型推断（InferDataType）
 * - 算子定义与注册
 * 
 * Host侧负责根据输入张量的形状和硬件特性，计算最优的Tiling参数，
 * 并传递给Kernel侧进行实际计算。
 */

namespace optiling {

/**
 * @brief 无符号整数向上取整除法
 * 
 * 计算 ceil(a / b)，即向上取整的除法结果。
 * 用于计算分块时的核心数量和每核心处理量。
 * 
 * @param a 被除数
 * @param b 除数
 * @return 向上取整的商
 */
static inline uint32_t CeilDivU(uint32_t a, uint32_t b) {
    if (b == 0) return 0;
    return static_cast<uint32_t>((static_cast<uint64_t>(a) + b - 1) / b);
}

/**
 * @brief 无符号整数向上对齐
 * 
 * 将数值a向上对齐到m的倍数。
 * 用于内存对齐和分块大小计算。
 * 
 * @param a 原始数值
 * @param m 对齐基数
 * @return 对齐后的数值
 */
static inline uint32_t AlignUpU(uint32_t a, uint32_t m) {
    return (m == 0) ? a : ((a + m - 1) / m) * m;
}

/**
 * @brief 安全的64位无符号整数乘法
 * 
 * 在乘法前检查是否会发生溢出，
 * 用于计算总元素数和维度乘积时的安全检查。
 * 
 * @param a 乘数1
 * @param b 乘数2
 * @param out 乘积输出
 * @return 是否成功（无溢出）
 */
static bool SafeMultiplyU64(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    out = a * b;
    return true;
}

/**
 * @brief LessEqual算子Tiling配置计算函数
 * 
 * 根据输入张量的形状和AI Core硬件特性，计算最优的Tiling参数，
 * 包括分块大小、核心数量、处理模式等配置。
 * 
 * @param context Tiling上下文，包含输入张量信息和硬件平台信息
 * @return GRAPH_SUCCESS表示成功，GRAPH_FAILED表示失败
 */
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    if (context == nullptr) return ge::GRAPH_FAILED;

    // 获取硬件平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (num_cores == 0) num_cores = 1;

    // 获取输入张量信息
    const gert::Tensor* tensor_x1 = context->GetRequiredInputTensor(0);
    const gert::Tensor* tensor_x2 = context->GetRequiredInputTensor(1);
    if (tensor_x1 == nullptr || tensor_x2 == nullptr) return ge::GRAPH_FAILED;

    // 验证数据类型一致性
    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    ge::DataType dtype_x2 = tensor_x2->GetDataType();
    if (dtype_x1 != dtype_x2) return ge::GRAPH_FAILED;
    // 验证数据类型是否支持
    if (dtype_x1 != ge::DT_FLOAT16 && dtype_x1 != ge::DT_FLOAT &&
        dtype_x1 != ge::DT_INT32 && dtype_x1 != ge::DT_INT8) {
        return ge::GRAPH_FAILED;
    }

    // 设置模板参数选择（用于编译期类型实例化）
    uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
    ASCENDC_TPL_SEL_PARAM(context, DT_X1);

    // 获取输入输出张量的形状信息
    const gert::Shape& x1s = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& x2s = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape& ys = context->GetOutputShape(0)->GetStorageShape();

    // 获取Tiling数据结构指针
    LessEqualTilingData* tiling = context->GetTilingData<LessEqualTilingData>();
    if (tiling == nullptr) return ge::GRAPH_FAILED;

    // 计算输出张量的总元素数
    int64_t outElems = ys.GetShapeSize();

    // 空张量处理：直接返回零配置
    if (outElems == 0) {
        tiling->mode = 0;
        tiling->totalLen = 0;
        tiling->tileLen = 0;
        tiling->blockDim = 1;
        tiling->perCore = 0;
        tiling->ndim = 0;
        tiling->lastDimLen = 0;
        tiling->totalRows = 0;

        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }

        context->SetBlockDim(1);

        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

    // 判断是否使用fast模式（输入形状完全相同时）
    size_t n1 = x1s.GetDimNum();
    size_t n2 = x2s.GetDimNum();

    bool fast = (n1 == n2);

    if (fast) {
        for (size_t i = 0; i < n1; ++i) {
            if (x1s.GetDim(i) != x2s.GetDim(i)) {
                fast = false;
                break;
            }
        }
    }

    // 获取数据类型大小
    int dtype_size = ge::GetSizeByDataType(dtype_x1);
    if (dtype_size <= 0) return ge::GRAPH_FAILED;

    // 计算每个元素所需的UB空间（字节）
    // 不同数据类型的比较操作需要不同的中间缓冲区
    uint32_t perElemUB;

    if (dtype_x1 == ge::DT_INT32) {
        perElemUB = 27;  // int32需要额外的Min计算缓冲区
    } else if (dtype_x1 == ge::DT_FLOAT) {
        perElemUB = 25;  // float32直接比较
    } else {
        perElemUB = 15;  // float16和int8需要较少空间
    }

    // 获取UB（Unified Buffer）大小
    uint64_t ub_size;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ub_size);

    // 预留8KB的UB空间给系统使用
    constexpr uint64_t UB_RESERVE = 8192;
    uint64_t usable_ub =
        ub_size > UB_RESERVE
            ? ub_size - UB_RESERVE
            : ub_size;

    // 基础块大小（AI Core向量指令的最小处理单位）
    const uint32_t BLOCK = 32;

    // 计算每个Block所需的UB空间
    uint32_t perBlockUB =
        perElemUB *
        (BLOCK / static_cast<uint32_t>(dtype_size));

    // 计算可用UB能容纳的Block数量
    uint32_t tileBlockNum =
        static_cast<uint32_t>(usable_ub) /
        perBlockUB;

    if (tileBlockNum == 0) tileBlockNum = 1;

    // 计算每个Tile的元素数量
    uint32_t tileLen =
        tileBlockNum *
        (BLOCK / static_cast<uint32_t>(dtype_size));

    // Tile长度上限（防止过大导致UB溢出）
    constexpr uint32_t TILE_LEN_CAP = 8192;

    if (tileLen > TILE_LEN_CAP) {
        tileLen = TILE_LEN_CAP;
    }

    // 总元素数（转换为uint32）
    uint32_t total =
        static_cast<uint32_t>(outElems);

    // 如果总元素数小于tileLen，调整tileLen
    if (tileLen > total) {
        tileLen = total;
    }

    // Fast模式：输入形状完全相同，使用简单的一维分块策略
    if (fast) {
        // 计算每个核心处理的元素数（向上对齐到32）
        uint32_t perCore =
            AlignUpU(
                CeilDivU(total, num_cores),
                32U);

        if (perCore == 0) {
            perCore = 32U;
        }

        // 计算所需的核心数
        uint32_t blockDim =
            CeilDivU(total, perCore);

        if (blockDim == 0) {
            blockDim = 1;
        }

        // 设置Fast模式的Tiling参数
        tiling->mode = 0;
        tiling->totalLen = total;
        tiling->tileLen = tileLen;
        tiling->blockDim = blockDim;
        tiling->perCore = perCore;
        tiling->ndim = 0;
        tiling->lastDimLen = 0;
        tiling->totalRows = 0;

        // 初始化形状和步长数组（Fast模式不需要）
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }

        // 设置并行核心数
        context->SetBlockDim(blockDim);

        // 工作空间大小（本算子不需要工作空间）
        size_t* ws = context->GetWorkspaceSizes(1);
        if (ws != nullptr) ws[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

    // Broadcast模式：输入形状不同，需要计算广播后的形状和步长

    // 获取最大维度数
    size_t nmax = (n1 > n2) ? n1 : n2;

    // 扩展输入维度到相同长度，不足的维度补1
    int64_t d1[64];  // x1各维度大小（扩展后）
    int64_t d2[64];  // x2各维度大小（扩展后）
    int64_t od[64];  // 广播后各维度大小

    for (size_t i = 0; i < nmax; ++i) {
        // 从右侧开始对齐，不足的维度视为1
        int64_t a =
            (i + n1 >= nmax)
                ? x1s.GetDim(i - (nmax - n1))
                : 1;

        int64_t b =
            (i + n2 >= nmax)
                ? x2s.GetDim(i - (nmax - n2))
                : 1;

        d1[i] = a;
        d2[i] = b;
        // 广播规则：若其中一个为1，则取另一个；否则必须相等
        od[i] = (a == 1) ? b : a;
    }

    // 计算各维度的步长（用于内存访问）
    int64_t s1[64];  // x1各维度步长
    int64_t s2[64];  // x2各维度步长

    {
        int64_t acc1 = 1;

        // 从右向左计算x1的步长
        for (int i = static_cast<int>(nmax) - 1;
             i >= 0;
             --i) {
            // 若维度为1（广播维度），步长为0
            s1[i] = (d1[i] == 1) ? 0 : acc1;

            if (d1[i] != 1) {
                acc1 *= d1[i];
            }
        }

        int64_t acc2 = 1;

        // 从右向左计算x2的步长
        for (int i = static_cast<int>(nmax) - 1;
             i >= 0;
             --i) {
            // 若维度为1（广播维度），步长为0
            s2[i] = (d2[i] == 1) ? 0 : acc2;

            if (d2[i] != 1) {
                acc2 *= d2[i];
            }
        }
    }

    // 压缩连续维度：将连续的维度合并以减少Kernel侧的维度处理开销
    uint32_t cShape[64];  // 压缩后的形状
    int64_t cs1[64];      // 压缩后的x1步长
    int64_t cs2[64];      // 压缩后的x2步长

    int cnt = 0;

    // 从右向左遍历，合并可压缩的维度
    for (int i = static_cast<int>(nmax) - 1;
         i >= 0;
         --i) {
        if (cnt > 0) {
            int j = cnt - 1;

            // 判断是否为双向广播维度（可以合并）
            bool bothBcast =
                s1[i] == 0 &&
                s2[i] == 0 &&
                cs1[j] == 0 &&
                cs2[j] == 0;

            // 判断是否为连续维度（内存布局连续，可以合并）
            bool contig =
                s1[i] != 0 &&
                cs1[j] != 0 &&
                s1[i] ==
                    cs1[j] *
                    static_cast<int64_t>(cShape[j]) &&
                s2[i] != 0 &&
                cs2[j] != 0 &&
                s2[i] ==
                    cs2[j] *
                    static_cast<int64_t>(cShape[j]);

            // 双向广播维度合并
            if (bothBcast) {
                cShape[j] =
                    static_cast<uint32_t>(
                        static_cast<int64_t>(cShape[j]) *
                        od[i]);

                continue;
            }

            // 连续维度合并
            if (contig) {
                cShape[j] =
                    static_cast<uint32_t>(
                        static_cast<int64_t>(cShape[j]) *
                        od[i]);

                cs1[j] = s1[i];
                cs2[j] = s2[i];

                continue;
            }
        }

        // 无法合并，新增一个维度
        cShape[cnt] =
            static_cast<uint32_t>(od[i]);

        cs1[cnt] = s1[i];
        cs2[cnt] = s2[i];

        ++cnt;
    }

        int ndim = cnt;

    // 翻转维度顺序：从右向左遍历得到的维度需要翻转回原始顺序
    uint32_t fShape[64];  // 最终形状（翻转后）
    int64_t fs1[64];      // 最终x1步长（翻转后）
    int64_t fs2[64];      // 最终x2步长（翻转后）

    for (int i = 0; i < ndim; ++i) {
        fShape[i] =
            cShape[ndim - 1 - i];

        fs1[i] =
            cs1[ndim - 1 - i];

        fs2[i] =
            cs2[ndim - 1 - i];
    }

    // 维度数限制：超过LE_MAX_DIM时合并前两维
    while (ndim > static_cast<int>(LE_MAX_DIM)) {
        // 合并第一维和第二维
        fShape[1] =
            static_cast<uint32_t>(
                static_cast<int64_t>(fShape[0]) *
                fShape[1]);

        // 左移所有维度
        for (int i = 1; i < ndim; ++i) {
            fShape[i - 1] = fShape[i];
            fs1[i - 1] = fs1[i];
            fs2[i - 1] = fs2[i];
        }

        --ndim;
    }

    // 最后一维长度（作为每行的长度）
    uint32_t lastDimLen =
        fShape[ndim - 1];

    // 计算总行数（除最后一维外的维度乘积）
    uint32_t totalRows = 1;

    for (int i = 0; i < ndim - 1; ++i) {
        uint64_t prod = 0;

        if (!SafeMultiplyU64(
                totalRows,
                fShape[i],
                prod)) {
            return ge::GRAPH_FAILED;
        }

        totalRows =
            static_cast<uint32_t>(prod);
    }

    // 目标每Block处理的元素数（用于计算所需核心数）
    uint64_t targetPerBlock = 2048;

    // 计算期望的核心数
    uint64_t desired =
        (static_cast<uint64_t>(total) +
         targetPerBlock - 1) /
        targetPerBlock;

    // 实际使用的核心数（不超过可用核心数）
    uint32_t blockDim =
        static_cast<uint32_t>(
            std::min(
                desired,
                static_cast<uint64_t>(num_cores)));

    if (blockDim == 0) {
        blockDim = 1;
    }

    // 计算每个核心处理的行数
    uint32_t perCoreRows =
        CeilDivU(totalRows, blockDim);

    if (perCoreRows == 0) {
        perCoreRows = 1;
    }

    // 设置Broadcast模式的Tiling参数
    tiling->mode = 1;
    tiling->totalLen = total;
    tiling->tileLen = tileLen;
    tiling->blockDim = blockDim;
    tiling->perCore = perCoreRows;
    tiling->ndim = static_cast<uint32_t>(ndim);
    tiling->lastDimLen = lastDimLen;
    tiling->totalRows = totalRows;

    // 填充形状和步长数组
    for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
        if (static_cast<int>(i) < ndim) {
            tiling->outShape[i] = fShape[i];
            tiling->x1Stride[i] =
                static_cast<int32_t>(fs1[i]);
            tiling->x2Stride[i] =
                static_cast<int32_t>(fs2[i]);
        } else {
            tiling->outShape[i] = 0;
            tiling->x1Stride[i] = 0;
            tiling->x2Stride[i] = 0;
        }
    }

    // 设置并行核心数
    context->SetBlockDim(blockDim);

    // 工作空间大小（本算子不需要工作空间）
    size_t* ws =
        context->GetWorkspaceSizes(1);

    if (ws != nullptr) {
        ws[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

/**
 * @brief LessEqual算子形状推断函数
 * 
 * 根据输入张量的形状，计算输出张量的形状，
 * 支持NumPy/TensorFlow标准的广播语义。
 * 
 * @param context 形状推断上下文
 * @return GRAPH_SUCCESS表示成功，GRAPH_FAILED表示失败
 */
static graphStatus InferShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 获取输入输出形状
    const gert::Shape* s1 =
        context->GetInputShape(0);

    const gert::Shape* s2 =
        context->GetInputShape(1);

    gert::Shape* y =
        context->GetOutputShape(0);

    if (s1 == nullptr ||
        s2 == nullptr ||
        y == nullptr) {
        return GRAPH_FAILED;
    }

    // 获取输入维度数
    size_t n1 = s1->GetDimNum();
    size_t n2 = s2->GetDimNum();
    size_t n = (n1 > n2) ? n1 : n2;

    // 设置输出维度数
    y->SetDimNum(n);

    // 从右向左逐维计算输出维度（支持广播）
    for (size_t i = 0; i < n; ++i) {
        // 获取当前维度的大小（不足则补1）
        int64_t dd1 =
            (i < n1)
                ? s1->GetDim(n1 - 1 - i)
                : 1;

        int64_t dd2 =
            (i < n2)
                ? s2->GetDim(n2 - 1 - i)
                : 1;

        // 广播规则：若其中一个为1，则取另一个
        int64_t outputDim =
            (dd1 == 1) ? dd2 : dd1;

        y->SetDim(
            n - 1 - i,
            outputDim);
    }

    return GRAPH_SUCCESS;
}

/**
 * @brief LessEqual算子数据类型推断函数
 * 
 * LessEqual算子的输出数据类型始终为bool类型，
 * 与输入数据类型无关。
 * 
 * @param context 数据类型推断上下文
 * @return GRAPH_SUCCESS表示成功，GRAPH_FAILED表示失败
 */
static graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    // 输出类型始终为bool
    context->SetOutputDataType(
        0,
        ge::DT_BOOL);

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

/**
 * @brief LessEqual算子定义类
 * 
 * 该类定义了LessEqual算子的完整配置，包括：
 * - 输入参数（x1, x2）及其数据类型和格式
 * - 输出参数（y）及其数据类型和格式
 * - 形状推断函数和数据类型推断函数
 * - AI Core Tiling配置函数
 * 
 * 算子功能：逐元素比较x1 <= x2，返回bool类型结果。
 */
class LessEqual : public OpDef {
public:
    /**
     * @brief 构造函数，注册算子配置
     * 
     * @param name 算子名称
     */
    explicit LessEqual(const char* name)
        : OpDef(name)
    {
        // 定义输入x1：必选，支持float16/float32/int32/int8，ND格式
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT,
                ge::DT_INT32,
                ge::DT_INT8
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        // 定义输入x2：必选，支持float16/float32/int32/int8，ND格式
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT,
                ge::DT_INT32,
                ge::DT_INT8
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        // 定义输出y：必选，bool类型，ND格式
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_BOOL,
                ge::DT_BOOL,
                ge::DT_BOOL,
                ge::DT_BOOL
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        // 设置形状推断和数据类型推断函数
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // 配置AI Core执行：设置Tiling函数和硬件平台
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

/**
 * @brief 注册LessEqual算子到算子库
 * 
 * OP_ADD宏将LessEqual算子注册到Ascend算子框架中，
 * 使其可以被ACLNN或GraphEngine调用。
 */
OP_ADD(LessEqual);

}  // namespace ops