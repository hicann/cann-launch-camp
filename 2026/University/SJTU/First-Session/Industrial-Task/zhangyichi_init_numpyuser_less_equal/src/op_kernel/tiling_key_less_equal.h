// 以输入 dtype 作为 TilingKey 模板参数，x1/x2 在原型中按位配对为同一类型。
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ),
);
