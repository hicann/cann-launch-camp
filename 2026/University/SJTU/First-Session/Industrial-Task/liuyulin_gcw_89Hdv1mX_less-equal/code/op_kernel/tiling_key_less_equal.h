#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

/**
 * @brief 声明 LessEqual 算子的 Tiling 模板参数
 *
 * 该宏声明了 LessEqual 算子在 Kernel 侧所需的模板参数列表。
 * 编译器在编译阶段会根据此声明生成对应的模板参数组合。
 *
 * 参数说明：
 *   - LessEqual       : 算子名称，用于标识当前模板参数所属的算子
 *   - DT_X1           : 模板参数名，表示输入 X1 的数据类型
 *   - C_DT_FLOAT16    : 支持的数据类型之一，半精度浮点数（float16）
 *   - C_DT_FLOAT      : 支持的数据类型之一，单精度浮点数（float32）
 *   - C_DT_INT32      : 支持的数据类型之一，32位有符号整数（int32）
 *   - C_DT_INT8       : 支持的数据类型之一，8位有符号整数（int8）
 *   - ASCENDC_TPL_INPUT(0) : 指定该模板参数关联的输入索引为 0（即第一个输入张量 X1）
 */
ASCENDC_TPL_ARGS_DECL(
    LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(
        DT_X1,              /* 模板参数名：输入 X1 的数据类型 */
        C_DT_FLOAT16,       /* 支持类型：float16 */
        C_DT_FLOAT,         /* 支持类型：float32 */
        C_DT_INT32,         /* 支持类型：int32 */
        C_DT_INT8,          /* 支持类型：int8 */
        ASCENDC_TPL_INPUT(0) /* 关联输入索引：0，表示第一个输入张量 */
    ),
);

/**
 * @brief 定义 LessEqual 算子的 Tiling 模板参数选择器
 *
 * 该宏定义了编译时需要生成的模板实例组合。
 * 每个 ASCENDC_TPL_ARGS_SEL 对应一种数据类型的 Kernel 实例化方案，
 * 编译器将为下列每种数据类型分别生成一份 Kernel 二进制代码：
 *
 *   1. DT_X1 = C_DT_FLOAT16 : 输入 X1 为 float16 类型的实例
 *   2. DT_X1 = C_DT_FLOAT   : 输入 X1 为 float32 类型的实例
 *   3. DT_X1 = C_DT_INT32   : 输入 X1 为 int32 类型的实例
 *   4. DT_X1 = C_DT_INT8    : 输入 X1 为 int8 类型的实例
 *
 * 运行时 Tiling 阶段，Host 侧会根据用户传入的实际输入数据类型，
 * 选择对应的模板实例进行执行。
 */
ASCENDC_TPL_SEL(
    /* 选择器 1：DT_X1 = float16 */
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(
            DT_X1,
            C_DT_FLOAT16
        ),
    ),
    /* 选择器 2：DT_X1 = float32 */
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(
            DT_X1,
            C_DT_FLOAT
        ),
    ),
    /* 选择器 3：DT_X1 = int32 */
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(
            DT_X1,
            C_DT_INT32
        ),
    ),
    /* 选择器 4：DT_X1 = int8 */
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(
            DT_X1,
            C_DT_INT8
        ),
    ),
);
