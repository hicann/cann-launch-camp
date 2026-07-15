#ifndef TILING_KEY_GELU_H
#define TILING_KEY_GELU_H
#include "ascendc/host_api/tiling/template_argument.h"
#include "register/tilingdata_base.h"
#include "gelu_tiling.h"

ASCENDC_TPL_ARGS_DECL(Gelu,
    ASCENDC_TPL_DATATYPE_DECL(IN_X_DTYPE, C_DT_FLOAT16, C_DT_FLOAT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(IN_X_DTYPE, C_DT_FLOAT16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DATATYPE_SEL(IN_X_DTYPE, C_DT_FLOAT))
);

namespace optiling {
    BEGIN_TILING_DATA_DEF(GeluTilingDataDef)
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(Gelu, GeluTilingDataDef)
}
#endif