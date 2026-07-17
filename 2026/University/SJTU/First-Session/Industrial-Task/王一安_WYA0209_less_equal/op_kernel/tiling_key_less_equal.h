// TilingKey 模板定义的头文件
// 支持 float16, float32, int32, int8 四种数据类型
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ),
);
