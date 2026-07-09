// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// 只按数据类型选模板；小/大 kernel 的分派放到运行时（读 tiling.isSmallShape），
// 以避免引入不确定的 ASCENDC_TPL_UINT_* 宏。
ASCENDC_TPL_ARGS_DECL(FastGelu,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);
