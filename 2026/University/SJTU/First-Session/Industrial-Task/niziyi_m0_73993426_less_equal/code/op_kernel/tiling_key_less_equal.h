// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define LESS_EQUAL_SCH_NORMAL 0
#define LESS_EQUAL_SCH_LINEAR_DB_ONES 1
#define LESS_EQUAL_SCH_FAST_LIGHT 2
#define LESS_EQUAL_SCH_STATIC_HALF 3

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    // Four schedule values require two bits. V31 used one bit for three
    // values, so OPC rejected the template before compiling device code.
    ASCENDC_TPL_UINT_DECL(SCH_MODE, 2, ASCENDC_TPL_UI_LIST,
                          LESS_EQUAL_SCH_NORMAL,
                          LESS_EQUAL_SCH_LINEAR_DB_ONES,
                          LESS_EQUAL_SCH_FAST_LIGHT,
                          LESS_EQUAL_SCH_STATIC_HALF),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
        ASCENDC_TPL_UINT_SEL(SCH_MODE, ASCENDC_TPL_UI_LIST,
                            LESS_EQUAL_SCH_NORMAL,
                            LESS_EQUAL_SCH_LINEAR_DB_ONES,
                            LESS_EQUAL_SCH_FAST_LIGHT,
                            LESS_EQUAL_SCH_STATIC_HALF),
    ),
);
