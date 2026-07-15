#pragma once  // 防止头文件被重复包含

// 引入 AscendC 框架中用于定义模板参数选择（Tiling 参数）的宏和工具
#include "ascendc/host_api/tiling/template_argument.h"


ASCENDC_TPL_ARGS_DECL(Gelu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT),
);


ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);
