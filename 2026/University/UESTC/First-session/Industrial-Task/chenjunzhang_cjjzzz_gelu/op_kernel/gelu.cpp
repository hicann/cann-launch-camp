// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
__aicore__ inline void ComputeGelu(LocalTensor<T> outputLocal, LocalTensor<T> inputLocal, LocalTensor<T> tmpLocal,
                                   uint32_t count) {
    Muls(tmpLocal, inputLocal, static_cast<T>(0.7071067811865476f), count);
    Erf(tmpLocal, tmpLocal, count);
    Adds(tmpLocal, tmpLocal, static_cast<T>(1.0f), count);
    Mul(outputLocal, inputLocal, tmpLocal, count);
    Muls(outputLocal, outputLocal, static_cast<T>(0.5f), count);
}

template <>
__aicore__ inline void ComputeGelu<half>(LocalTensor<half> outputLocal, LocalTensor<half> inputLocal,
                                         LocalTensor<half> tmpLocal, uint32_t count) {
    (void)tmpLocal;
    Gelu(outputLocal, inputLocal, count);
}

template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t length, uint32_t tileLength) {
        length_ = length;
        tileLength_ = tileLength;

        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();
        if (blockNum == 0 || length_ == 0) {
            validBlockLength_ = 0;
            return;
        }

        uint32_t alignElems = Get32ByteAlignElems();
        uint32_t blockLength = AlignUp(DivCeil(length_, blockNum), alignElems);
        blockOffset_ = blockIdx * blockLength;
        if (blockOffset_ >= length_) {
            validBlockLength_ = 0;
            return;
        }
        validBlockLength_ = Min(blockLength, length_ - blockOffset_);

        if (validBlockLength_ == 0 || tileLength_ == 0) {
            return;
        }

        inputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + blockOffset_, validBlockLength_);
        outputGm_.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + blockOffset_, validBlockLength_);

        pipe_.InitBuffer(inputQue_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(outputQue_, BUFFER_NUM, tileLength_ * sizeof(DT_INPUT_X));
        pipe_.InitBuffer(calcBuf_, tileLength_ * sizeof(DT_INPUT_X));
    }
    __aicore__ inline void Process() {
        if (validBlockLength_ == 0 || tileLength_ == 0) {
            return;
        }
        ProcessDoubleBuffer();
    }
private:
    __aicore__ inline void ProcessDoubleBuffer() {
        uint32_t tileCount = DivCeil(validBlockLength_, tileLength_);
        CopyIn(0, GetTileLength(0));
        for (uint32_t tileIdx = 0; tileIdx < tileCount; ++tileIdx) {
            uint32_t nextTileIdx = tileIdx + 1;
            if (nextTileIdx < tileCount) {
                CopyIn(nextTileIdx * tileLength_, GetTileLength(nextTileIdx));
            }

            uint32_t offset = tileIdx * tileLength_;
            uint32_t count = GetTileLength(tileIdx);
            Compute(count);
            CopyOut(offset, count);
        }
    }

    __aicore__ inline uint32_t Min(uint32_t lhs, uint32_t rhs) {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t DivCeil(uint32_t lhs, uint32_t rhs) {
        return rhs == 0 ? 0 : (lhs + rhs - 1) / rhs;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) {
        return align == 0 ? value : (value + align - 1) / align * align;
    }

    __aicore__ inline uint32_t GetTileLength(uint32_t tileIdx) {
        uint32_t offset = tileIdx * tileLength_;
        return Min(tileLength_, validBlockLength_ - offset);
    }

    __aicore__ inline uint32_t Get32ByteAlignElems() {
        constexpr uint32_t ALIGN_BYTES = 32;
        return ALIGN_BYTES / sizeof(DT_INPUT_X);
    }

    __aicore__ inline bool Is32ByteAligned(uint32_t elemOffset, uint32_t elemCount) {
        uint32_t alignElems = Get32ByteAlignElems();
        return alignElems == 0 || (elemOffset % alignElems == 0 && elemCount % alignElems == 0);
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count) {
        LocalTensor<DT_INPUT_X> inputLocal = inputQue_.AllocTensor<DT_INPUT_X>();
        if (Is32ByteAligned(blockOffset_ + offset, count)) {
            DataCopy(inputLocal, inputGm_[offset], count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm_[offset], copyParams, padParams);
        }
        inputQue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count) {
        LocalTensor<DT_INPUT_X> inputLocal = inputQue_.DeQue<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> outputLocal = outputQue_.AllocTensor<DT_INPUT_X>();
        LocalTensor<DT_INPUT_X> tmpLocal = calcBuf_.Get<DT_INPUT_X>();

        ComputeGelu<DT_INPUT_X>(outputLocal, inputLocal, tmpLocal, count);

        outputQue_.EnQue(outputLocal);
        inputQue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count) {
        LocalTensor<DT_INPUT_X> outputLocal = outputQue_.DeQue<DT_INPUT_X>();
        if (Is32ByteAligned(blockOffset_ + offset, count)) {
            DataCopy(outputGm_[offset], outputLocal, count);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(DT_INPUT_X)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
        outputQue_.FreeTensor(outputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQue_;
    TBuf<QuePosition::VECCALC> calcBuf_;
    GlobalTensor<DT_INPUT_X> inputGm_;
    GlobalTensor<DT_INPUT_X> outputGm_;
    uint32_t length_ = 0;
    uint32_t tileLength_ = 0;
    uint32_t blockOffset_ = 0;
    uint32_t validBlockLength_ = 0;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.length, tiling_data.tileLength);
    op.Process();
}