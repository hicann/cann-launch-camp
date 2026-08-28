#ifndef CLIP_BY_VALUE_TILING_H
#define CLIP_BY_VALUE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ClipByValueTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);   
    TILING_DATA_FIELD_DEF(uint32_t, coreData);      
    TILING_DATA_FIELD_DEF(uint32_t, coreDataTail);  
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);    
    TILING_DATA_FIELD_DEF(uint32_t, is_min_scalar); 
    TILING_DATA_FIELD_DEF(uint32_t, is_max_scalar); 
    // 新增：数据类型透传
    TILING_DATA_FIELD_DEF(uint32_t, dtype); 
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ClipByValue, ClipByValueTilingData)
} // namespace optiling

#endif // CLIP_BY_VALUE_TILING_H