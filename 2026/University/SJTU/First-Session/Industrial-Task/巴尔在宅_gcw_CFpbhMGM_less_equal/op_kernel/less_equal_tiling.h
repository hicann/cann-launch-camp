#pragma once
 	 
 	 #include <cstdint>
 	 
 	 #define LE_MAX_DIM 8
 	 
 	 struct LessEqualTilingData {
 	     // 模式：0=快速（无广播），1=广播
 	     uint32_t mode;
 	 
 	     // 总元素数
 	     uint64_t totalLen;
 	 
 	     // 每次处理的 tile 大小（元素个数）
 	     uint32_t tileLen;
 	 
 	     // NPU 核数
 	     uint32_t blockDim;
 	 
 	     // 每个核处理的元素/行数
 	     uint32_t perCore;
 	 
 	     // 广播时的维度数
 	     uint32_t ndim;
 	 
 	     // 最后一维长度
 	     uint32_t lastDimLen;
 	 
 	     // 总行数（广播模式下，除最后一维外的所有维度乘积）
 	     uint32_t totalRows;
 	 
 	     // 输出形状
 	     uint32_t outShape[LE_MAX_DIM];
 	 
 	     // 输入 x1 的 stride（支持广播，0 表示标量）
 	     int32_t x1Stride[LE_MAX_DIM];
 	 
 	     // 输入 x2 的 stride
 	     int32_t x2Stride[LE_MAX_DIM];
 	 };