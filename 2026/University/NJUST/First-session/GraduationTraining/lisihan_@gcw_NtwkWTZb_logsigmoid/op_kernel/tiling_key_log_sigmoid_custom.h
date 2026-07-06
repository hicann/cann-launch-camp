#ifndef TILING_KEY_LOG_SIGMOID_CUSTOM_H
#define TILING_KEY_LOG_SIGMOID_CUSTOM_H
#include "ascendc/host_api/tiling/template_argument.h"

// 定义模板参数
ASCENDC_TPL_ARGS_DECL(LogSigmoidCustom,
    // D_T_X: 输入x的数据类型，支持float32/float16/bf16
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16,
                              ASCENDC_TPL_INPUT(0)),
    // D_T_Y: 输出y的数据类型，支持float32/float16/bf16
    ASCENDC_TPL_DATATYPE_DECL(D_T_Y, C_DT_FLOAT, C_DT_FLOAT16, C_DT_BF16,
                              ASCENDC_TPL_OUTPUT(0)),
);

// 定义合法组合：输入输出类型必须一致
ASCENDC_TPL_SEL(
    // 组合1: float32 → float32 (case1, case2)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT),
    ),
    // 组合2: float16 → float16 (case3, case4)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_FLOAT16),
    ),
    // 组合3: bf16 → bf16 (case5, case6)
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16),
        ASCENDC_TPL_DATATYPE_SEL(D_T_Y, C_DT_BF16),
    ),
);

#endif // TILING_KEY_LOG_SIGMOID_CUSTOM_H
