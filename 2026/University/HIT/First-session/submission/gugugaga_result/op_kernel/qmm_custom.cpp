#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#include "qmm_custom_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t VEC_ROW_CHUNK = 16U;

__aicore__ inline uint32_t QmmCeilDivU32(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1U) / divisor;
}

__aicore__ inline uint32_t QmmMinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

struct QmmBlockInfo {
    uint32_t rowStart;
    uint32_t colStart;
    uint32_t actualM;
    uint32_t actualN;
    bool active;
};

__aicore__ inline QmmBlockInfo GetQmmBlockInfo(const TCubeTiling &tiling)
{
    const uint32_t mBlockNum = QmmCeilDivU32(tiling.M, tiling.singleCoreM);
    const uint32_t logicalBlockIdx = GetBlockIdx() / GetTaskRation();
    const uint32_t mBlockIdx = logicalBlockIdx % mBlockNum;
    const uint32_t nBlockIdx = logicalBlockIdx / mBlockNum;

    QmmBlockInfo info {};
    info.rowStart = mBlockIdx * tiling.singleCoreM;
    info.colStart = nBlockIdx * tiling.singleCoreN;
    info.active = info.rowStart < tiling.M && info.colStart < tiling.N;
    if (info.active) {
        info.actualM = QmmMinU32(tiling.singleCoreM, tiling.M - info.rowStart);
        info.actualN = QmmMinU32(tiling.singleCoreN, tiling.N - info.colStart);
    }
    return info;
}
}  // namespace

class KernelQmmInt32 {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, false>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;

    matmul::Matmul<AType, BType, CType, BiasType> matmulObj;
    TCubeTiling cubeTiling;

    __aicore__ inline KernelQmmInt32() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const QmmCustomTilingData &tilingData)
    {
        cubeTiling = tilingData.cubeTilingData;
        const QmmBlockInfo info = GetQmmBlockInfo(cubeTiling);
        rowStart_ = info.rowStart;
        colStart_ = info.colStart;
        actualM_ = info.actualM;
        actualN_ = info.actualN;
        active_ = info.active;
        if (!active_) {
            return;
        }

        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x1),
                                  static_cast<uint64_t>(cubeTiling.M) * cubeTiling.Ka);
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x2),
                                  static_cast<uint64_t>(cubeTiling.Kb) * cubeTiling.N);
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(y),
                                 static_cast<uint64_t>(cubeTiling.M) * cubeTiling.N);

        x1Global_ = x1Global_[static_cast<uint64_t>(rowStart_) * cubeTiling.Ka];
        // FRACTAL_NZ(int8): every N-column fractal owns a contiguous K span.
        x2Global_ = x2Global_[static_cast<uint64_t>(colStart_) * cubeTiling.Kb];
        yGlobal_ = yGlobal_[static_cast<uint64_t>(rowStart_) * cubeTiling.N + colStart_];
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }
        matmulObj.SetTensorA(x1Global_);
        matmulObj.SetTensorB(x2Global_);
        matmulObj.SetTail(actualM_, actualN_);
        // Fixpipe writes INT32 directly from L0C to GM, removing the previous
        // VECIN->VECOUT->GM round trip entirely.
        matmulObj.IterateAll(yGlobal_);
        matmulObj.End();
    }

private:
    uint32_t rowStart_ = 0U;
    uint32_t colStart_ = 0U;
    uint32_t actualM_ = 0U;
    uint32_t actualN_ = 0U;
    bool active_ = false;

    GlobalTensor<int8_t> x1Global_;
    GlobalTensor<int8_t> x2Global_;
    GlobalTensor<int32_t> yGlobal_;
};

class KernelQmmPertoken {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int8_t, false>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::NZ, int8_t, false>;
    using CType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t, false>;
    using BiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t, false>;

    matmul::Matmul<AType, BType, CType, BiasType> matmulObj;
    TCubeTiling cubeTiling;

    __aicore__ inline KernelQmmPertoken() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale,
                                GM_ADDR y, const QmmCustomTilingData &tilingData, TPipe *pipe)
    {
        cubeTiling = tilingData.cubeTilingData;
        pipe_ = pipe;
        const QmmBlockInfo info = GetQmmBlockInfo(cubeTiling);
        rowStart_ = info.rowStart;
        colStart_ = info.colStart;
        actualM_ = info.actualM;
        actualN_ = info.actualN;
        active_ = info.active;
        if (!active_) {
            return;
        }

        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x1),
                                  static_cast<uint64_t>(cubeTiling.M) * cubeTiling.Ka);
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x2),
                                  static_cast<uint64_t>(cubeTiling.Kb) * cubeTiling.N);
        scaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale), cubeTiling.N);
        pertokenGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pertokenScale), cubeTiling.M);
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(y),
                                 static_cast<uint64_t>(cubeTiling.M) * cubeTiling.N);

        x1Global_ = x1Global_[static_cast<uint64_t>(rowStart_) * cubeTiling.Ka];
        x2Global_ = x2Global_[static_cast<uint64_t>(colStart_) * cubeTiling.Kb];

        // The full cube result tile stays in VECIN.  Postprocessing uses only
        // 16 rows at a time, so a 128x256 cube tile still fits comfortably in UB.
        pipe_->InitBuffer(mmOutQueue_, 1,
                          cubeTiling.baseM * cubeTiling.baseN * sizeof(int32_t));
        pipe_->InitBuffer(scaleQueue_, 1, cubeTiling.baseN * sizeof(float));
        pipe_->InitBuffer(tokenScaleQueue_, 1, VEC_ROW_CHUNK * sizeof(float));
        pipe_->InitBuffer(outputQueue_, 1,
                          VEC_ROW_CHUNK * cubeTiling.baseN * sizeof(bfloat16_t));
        pipe_->InitBuffer(floatBuffer_,
                          VEC_ROW_CHUNK * cubeTiling.baseN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }

        matmulObj.SetTensorA(x1Global_);
        matmulObj.SetTensorB(x2Global_);
        matmulObj.SetTail(actualM_, actualN_);

        const uint32_t roundM = QmmCeilDivU32(actualM_, cubeTiling.baseM);
        uint32_t computeRound = 0U;
        while (matmulObj.template Iterate<true>()) {
            LocalTensor<int32_t> mmOut = mmOutQueue_.AllocTensor<int32_t>();
            matmulObj.template GetTensorC<true>(mmOut, false, true);
            mmOutQueue_.EnQue(mmOut);
            mmOut = mmOutQueue_.DeQue<int32_t>();

            const uint32_t tileM = computeRound % roundM;
            const uint32_t tileN = computeRound / roundM;
            const uint32_t tileRowStart = rowStart_ + tileM * cubeTiling.baseM;
            const uint32_t tileColStart = colStart_ + tileN * cubeTiling.baseN;
            const uint32_t validRows = QmmMinU32(cubeTiling.baseM, cubeTiling.M - tileRowStart);
            const uint32_t validCols = QmmMinU32(cubeTiling.baseN, cubeTiling.N - tileColStart);

            DequantAndCopyOut(mmOut, tileRowStart, tileColStart, validRows, validCols);
            mmOutQueue_.FreeTensor(mmOut);
            ++computeRound;
        }
        matmulObj.End();
    }

private:
    __aicore__ inline void DequantAndCopyOut(const LocalTensor<int32_t> &src,
                                             uint32_t tileRowStart, uint32_t tileColStart,
                                             uint32_t rows, uint32_t cols)
    {
        LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
        DataCopy(scaleLocal, scaleGlobal_[tileColStart], cols);
        scaleQueue_.EnQue(scaleLocal);
        scaleLocal = scaleQueue_.DeQue<float>();

        LocalTensor<float> floatLocal = floatBuffer_.Get<float>();

        for (uint32_t rowBase = 0U; rowBase < rows; rowBase += VEC_ROW_CHUNK) {
            const uint32_t chunkRows = QmmMinU32(VEC_ROW_CHUNK, rows - rowBase);
            const bool fullTokenChunk = chunkRows == VEC_ROW_CHUNK;
            LocalTensor<float> tokenScaleLocal;
            if (fullTokenChunk) {
                tokenScaleLocal = tokenScaleQueue_.AllocTensor<float>();
                DataCopy(tokenScaleLocal, pertokenGlobal_[tileRowStart + rowBase], VEC_ROW_CHUNK);
                tokenScaleQueue_.EnQue(tokenScaleLocal);
                tokenScaleLocal = tokenScaleQueue_.DeQue<float>();
            }

            LocalTensor<bfloat16_t> outputLocal = outputQueue_.AllocTensor<bfloat16_t>();
            for (uint32_t row = 0U; row < chunkRows; ++row) {
                const uint32_t srcOffset = (rowBase + row) * cubeTiling.baseN;
                const uint32_t localOffset = row * cubeTiling.baseN;
                const float tokenScale = fullTokenChunk
                    ? tokenScaleLocal.GetValue(row)
                    : pertokenGlobal_.GetValue(tileRowStart + rowBase + row);

                // Preserve the passing v4 arithmetic order exactly:
                // INT32 -> FP32 -> *scale[n] -> *pertoken_scale[m] -> BF16.
                Cast(floatLocal[localOffset], src[srcOffset], RoundMode::CAST_NONE, cols);
                PipeBarrier<PIPE_V>();
                Mul(floatLocal[localOffset], floatLocal[localOffset], scaleLocal, cols);
                PipeBarrier<PIPE_V>();
                Muls(floatLocal[localOffset], floatLocal[localOffset], tokenScale, cols);
                PipeBarrier<PIPE_V>();
                Cast(outputLocal[localOffset], floatLocal[localOffset], RoundMode::CAST_RINT, cols);
            }
            outputQueue_.EnQue(outputLocal);
            outputLocal = outputQueue_.DeQue<bfloat16_t>();

            DataCopyParams copyParams {
                static_cast<uint16_t>(chunkRows),
                static_cast<uint16_t>(cols * sizeof(bfloat16_t) / DEFAULT_C0_SIZE),
                static_cast<uint16_t>((cubeTiling.baseN - cols) * sizeof(bfloat16_t) /
                                      DEFAULT_C0_SIZE),
                static_cast<uint16_t>((cubeTiling.N - cols) * sizeof(bfloat16_t) /
                                      DEFAULT_C0_SIZE)
            };
            DataCopy(yGlobal_[static_cast<uint64_t>(tileRowStart + rowBase) * cubeTiling.N +
                              tileColStart],
                     outputLocal, copyParams);
            outputQueue_.FreeTensor(outputLocal);

            if (fullTokenChunk) {
                tokenScaleQueue_.FreeTensor(tokenScaleLocal);
            }
        }
        scaleQueue_.FreeTensor(scaleLocal);
    }

    TPipe *pipe_ = nullptr;
    uint32_t rowStart_ = 0U;
    uint32_t colStart_ = 0U;
    uint32_t actualM_ = 0U;
    uint32_t actualN_ = 0U;
    bool active_ = false;

    GlobalTensor<int8_t> x1Global_;
    GlobalTensor<int8_t> x2Global_;
    GlobalTensor<float> scaleGlobal_;
    GlobalTensor<float> pertokenGlobal_;
    GlobalTensor<bfloat16_t> yGlobal_;

    TQue<TPosition::VECIN, 1> mmOutQueue_;
    TQue<TPosition::VECIN, 1> scaleQueue_;
    TQue<TPosition::VECIN, 1> tokenScaleQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> floatBuffer_;
};

extern "C" __global__ __aicore__ void qmm_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale,
                                                   GM_ADDR pertoken_scale, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(QmmCustomTilingData);
    GET_TILING_DATA_WITH_STRUCT(QmmCustomTilingData, tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    if (tilingData.isPertoken == 0U) {
        TPipe pipe;
        KernelQmmInt32 op;
        op.Init(x1, x2, y, tilingData);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &op.cubeTiling);
        op.Process();
        pipe.Destroy();
    } else {
        TPipe pipe;
        KernelQmmPertoken op;
        op.Init(x1, x2, scale, pertoken_scale, y, tilingData, &pipe);
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &op.cubeTiling);
        op.Process();
        pipe.Destroy();
    }
}
