#ifndef LOG_SIGMOID_CUSTOM_TILING_H
#define LOG_SIGMOID_CUSTOM_TILING_H

#include <cstdint>

/*
 * LogSigmoidCustomTilingData
 * ==========================
 * Tiling parameters passed from host to kernel via constant memory.
 *
 *   dataType mapping:
 *     0 -> DT_FLOAT   (float32)
 *     1 -> DT_FLOAT16 (half  / float16)
 *     2 -> DT_BF16    (bfloat16)
 *
 * For bfloat16 inputs the kernel must upcast to float32 internally,
 * compute LogSigmoid, then downcast the result back to bfloat16.
 * This is because the AI Core's vector engine works best at float32
 * precision, and the CPU / test framework has no native bfloat16 support.
 */
struct LogSigmoidCustomTilingData {
    uint32_t totalLength;   ///< total element count across all dimensions
    uint32_t blockDim;      ///< number of AI cores reserved for this launch
    uint32_t tileLength;    ///< elements per tile (ceil(totalLength/blockDim), 32-aligned)
    uint32_t dataType;      ///< 0=float32, 1=float16, 2=bfloat16
};

#endif // LOG_SIGMOID_CUSTOM_TILING_H
