// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(QmmCustom,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1, C_DT_INT8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1, C_DT_INT8),
    ),
);