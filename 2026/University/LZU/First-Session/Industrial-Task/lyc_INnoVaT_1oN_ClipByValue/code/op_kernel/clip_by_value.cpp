#include "clip_by_value_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

// 专门为标量尾部处理写的安全裁剪函数（解决half类型隐患，保证int32不丢精度）
template <typename D>
__aicore__ inline D SafeScalarClip(D val, D min_val, D max_val) {
    val = val < min_val ? min_val : val;
    val = val > max_val ? max_val : val;
    return val;
}

// 针对 half 类型的特化：转成 float 比较最安全
template <>
__aicore__ inline half SafeScalarClip<half>(half val, half min_val, half max_val) {
    float val_f = static_cast<float>(val);
    float min_f = static_cast<float>(min_val);
    float max_f = static_cast<float>(max_val);
    val_f = val_f < min_f ? min_f : val_f;
    val_f = val_f > max_f ? max_f : val_f;
    return static_cast<half>(val_f);
}

template <typename T>
class KernelClipByValue {
public:
    __aicore__ inline KernelClipByValue() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, 
                                ClipByValueTilingData* tiling_data) {
        uint32_t core_id = GetBlockIdx();
        if (core_id >= tiling_data->usedCoreNum) return;

        this->tileLength = tiling_data->tileLength;
        this->is_min_scalar = tiling_data->is_min_scalar;
        this->is_max_scalar = tiling_data->is_max_scalar;

        this->thisCoreLength = (core_id == tiling_data->usedCoreNum - 1) 
                                ? tiling_data->coreDataTail 
                                : tiling_data->coreData;
        uint32_t offset = core_id * tiling_data->coreData;

        GlobalTensor<T> minGmTemp, maxGmTemp;
        minGmTemp.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(clip_value_min), 1);
        maxGmTemp.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(clip_value_max), 1);
        if (is_min_scalar) this->min_scalar_val = minGmTemp.GetValue(0);
        if (is_max_scalar) this->max_scalar_val = maxGmTemp.GetValue(0);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, thisCoreLength);
        if (!is_min_scalar) minGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(clip_value_min) + offset, thisCoreLength);
        if (!is_max_scalar) maxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(clip_value_max) + offset, thisCoreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, thisCoreLength);

        uint32_t elements_per_32b = 32 / sizeof(T);
        this->alignedLength = (thisCoreLength / elements_per_32b) * elements_per_32b;
        this->scalarTailLength = thisCoreLength - this->alignedLength;

        if (this->alignedLength > 0) {
            pipe.InitBuffer(inQueueX, 1, tileLength * sizeof(T));
            if (!is_min_scalar) pipe.InitBuffer(inQueueMin, 1, tileLength * sizeof(T));
            if (!is_max_scalar) pipe.InitBuffer(inQueueMax, 1, tileLength * sizeof(T));
            pipe.InitBuffer(outQueueY, 1, tileLength * sizeof(T));
        }
    }

    __aicore__ inline void Process() {
        if (this->thisCoreLength == 0) return;

        uint32_t tileNum = alignedLength / tileLength;
        uint32_t tileTail = alignedLength % tileLength;

        for (uint32_t i = 0; i < tileNum; i++) {
            ProcessBlock(i * tileLength, tileLength);
        }
        if (tileTail > 0) {
            ProcessBlock(tileNum * tileLength, tileTail);
        }

        if (scalarTailLength > 0) {
            ProcessScalarTail();
        }
    }

private:
    __aicore__ inline void ProcessBlock(uint32_t offset, uint32_t length) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        
        DataCopy(xLocal, xGm[offset], length);
        inQueueX.EnQue(xLocal);

        if (!is_min_scalar) {
            LocalTensor<T> minLocal = inQueueMin.AllocTensor<T>();
            DataCopy(minLocal, minGm[offset], length);
            inQueueMin.EnQue(minLocal);
        }
        if (!is_max_scalar) {
            LocalTensor<T> maxLocal = inQueueMax.AllocTensor<T>();
            DataCopy(maxLocal, maxGm[offset], length);
            inQueueMax.EnQue(maxLocal);
        }

        LocalTensor<T> xCompute = inQueueX.DeQue<T>(); 
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();

        if (is_min_scalar) {
            Maxs(yLocal, xCompute, min_scalar_val, length);
        } else {
            LocalTensor<T> minCompute = inQueueMin.DeQue<T>();
            Max(yLocal, xCompute, minCompute, length);
            inQueueMin.FreeTensor(minCompute);
        }

        if (is_max_scalar) {
            Mins(yLocal, yLocal, max_scalar_val, length);
        } else {
            LocalTensor<T> maxCompute = inQueueMax.DeQue<T>();
            Min(yLocal, yLocal, maxCompute, length);
            inQueueMax.FreeTensor(maxCompute);
        }

        inQueueX.FreeTensor(xCompute);

        outQueueY.EnQue(yLocal); 
        LocalTensor<T> yOut = outQueueY.DeQue<T>(); 
        DataCopy(yGm[offset], yOut, length);
        outQueueY.FreeTensor(yOut); 
    }

    __aicore__ inline void ProcessScalarTail() {
        uint32_t start_idx = alignedLength;
        for (uint32_t i = 0; i < scalarTailLength; i++) {
            uint32_t idx = start_idx + i;
            T val = xGm.GetValue(idx);
            T current_min = is_min_scalar ? min_scalar_val : minGm.GetValue(idx);
            T current_max = is_max_scalar ? max_scalar_val : maxGm.GetValue(idx);
            
            // 使用安全比较函数，彻底规避所有精度和底层运算坑
            val = SafeScalarClip<T>(val, current_min, current_max);
            
            yGm.SetValue(idx, val);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueMin, inQueueMax;
    TQue<QuePosition::VECOUT, 1> outQueueY;

    GlobalTensor<T> xGm, minGm, maxGm, yGm;
    
    uint32_t tileLength;
    uint32_t thisCoreLength;
    uint32_t alignedLength;
    uint32_t scalarTailLength;

    uint32_t is_min_scalar;
    uint32_t is_max_scalar;
    T min_scalar_val;
    T max_scalar_val;
};

// 恢复无模板的标准 C 接口
extern "C" __global__ __aicore__ void clip_by_value(GM_ADDR x, GM_ADDR clip_value_min, GM_ADDR clip_value_max, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA_WITH_STRUCT(ClipByValueTilingData, tiling_data, tiling);
    
    // 通过 TilingData 透传的 dtype 进行动态分支，完美规避构建环境宏丢失的问题
    // CANN DataType 定义：0: float32, 1: float16 (half), 3: int32
    if (tiling_data.dtype == 1) { 
        KernelClipByValue<half> op;
        op.Init(x, clip_value_min, clip_value_max, y, &tiling_data);
        op.Process();
    } else if (tiling_data.dtype == 3) {
        KernelClipByValue<int32_t> op;
        op.Init(x, clip_value_min, clip_value_max, y, &tiling_data);
        op.Process();
    } else { 
        // 默认 fallback 到 float32 (dtype == 0)
        KernelClipByValue<float> op;
        op.Init(x, clip_value_min, clip_value_max, y, &tiling_data);
        op.Process();
    }
}