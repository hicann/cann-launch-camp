// TilingKey template definition for LessEqual.
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

#define LESS_EQUAL_PATH_GENERAL 0
#define LESS_EQUAL_PATH_SMALL_SAME 1
#define LESS_EQUAL_PATH_TINY_SAME 2
#define LESS_EQUAL_PATH_TINY_X1_SCALAR 3
#define LESS_EQUAL_PATH_TINY_X2_SCALAR 4
#define LESS_EQUAL_PATH_MEDIUM_X1_SCALAR 5
#define LESS_EQUAL_PATH_MEDIUM_X2_SCALAR 6

ASCENDC_TPL_ARGS_DECL(LessEqual,
    ASCENDC_TPL_DATATYPE_DECL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
    ASCENDC_TPL_UINT_DECL(PATH_KIND, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                          LESS_EQUAL_PATH_GENERAL,
                          LESS_EQUAL_PATH_SMALL_SAME,
                          LESS_EQUAL_PATH_TINY_SAME,
                          LESS_EQUAL_PATH_TINY_X1_SCALAR,
                          LESS_EQUAL_PATH_TINY_X2_SCALAR,
                          LESS_EQUAL_PATH_MEDIUM_X1_SCALAR,
                          LESS_EQUAL_PATH_MEDIUM_X2_SCALAR),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X1, C_DT_FLOAT16, C_DT_FLOAT, C_DT_INT32, C_DT_INT8),
        ASCENDC_TPL_UINT_SEL(PATH_KIND, ASCENDC_TPL_UI_LIST,
                             LESS_EQUAL_PATH_GENERAL,
                             LESS_EQUAL_PATH_SMALL_SAME,
                             LESS_EQUAL_PATH_TINY_SAME,
                             LESS_EQUAL_PATH_TINY_X1_SCALAR,
                             LESS_EQUAL_PATH_TINY_X2_SCALAR,
                             LESS_EQUAL_PATH_MEDIUM_X1_SCALAR,
                             LESS_EQUAL_PATH_MEDIUM_X2_SCALAR),
    ),
);
