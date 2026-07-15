#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 声明 Gelu kernel 的模板参数：
// 1. DT_INPUT_X 根据输入 dtype 选择 float16 / float32 实例；
// 2. USE_TBUF_PATH 根据 tiling 结果选择 TBuf 单 tile 路径或队列多 tile 路径。
ASCENDC_TPL_ARGS_DECL(Gelu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(USE_TBUF_PATH, 0, 1),
);

// 注册可生成的模板组合，Host 侧通过 ASCENDC_TPL_SEL_PARAM 选择具体实例。
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(USE_TBUF_PATH, 0, 1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT),
        ASCENDC_TPL_BOOL_SEL(USE_TBUF_PATH, 0, 1),
    ),
);
