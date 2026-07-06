#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

// LogSigmoid 自定义算子 Tiling 参数结构体
// 注意：字段顺序与名称需与 Host 侧写入、Kernel 侧读取严格一致
struct LogSigmoidCustomTilingData {
    uint32_t smallCoreElemCnt;      
    uint32_t bigCoreElemCnt;       
    uint32_t bigTotalTileCnt;      
    uint32_t smallTotalTileCnt;    
    uint32_t elemPerTile;           
    uint32_t smallLastTileElemCnt;  
    uint32_t bigLastTileElemCnt;    
    uint32_t extraBlockCnt;        
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
