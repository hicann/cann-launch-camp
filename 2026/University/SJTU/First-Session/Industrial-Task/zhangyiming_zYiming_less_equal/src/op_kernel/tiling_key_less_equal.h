#pragma once
#include "ascendc/host_api/tiling/template_argument.h"

/**
 * @file tiling_key_less_equal.h
 * @brief LessEqual算子模板参数声明与选择
 * 
 * 该文件定义了LessEqual算子支持的数据类型模板参数，
 * 用于Ascend C算子框架进行编译期类型选择。
 */

/**
 * @brief LessEqual算子模板参数声明
 * 
 * 声明DT_X1为模板参数，支持以下四种数据类型：
 * - C_DT_FLOAT16: 半精度浮点数
 * - C_DT_FLOAT: 单精度浮点数
 * - C_DT_INT32: 32位整数
 * - C_DT_INT8: 8位整数
 */
ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
);

/**
 * @brief LessEqual算子模板参数选择器
 * 
 * 定义运行时数据类型到模板参数的映射关系，
 * 使得算子可以根据实际输入数据类型动态选择对应的模板实例。
 */
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ),
);
