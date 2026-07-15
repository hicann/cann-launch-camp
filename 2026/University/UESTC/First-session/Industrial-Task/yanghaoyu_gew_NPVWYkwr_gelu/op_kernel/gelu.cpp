#include "kernel_operator.h"

#include "gelu_tiling.h"

#include "tiling_key_gelu.h"


#define GELU_CAT_IMPL(a, b) a##b
#define GELU_CAT(a, b) GELU_CAT_IMPL(a, b)


constexpr uint32_t BUFFER_NUM = 2;

constexpr uint32_t TILE_LENGTH = 4096;

constexpr float GELU_INV_SQRT2 = 0.70710678118654752440f;


template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
        return (a + b - 1) / b;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t a, uint32_t align) {
        return (a + align - 1) / align * align;
    }

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        uint32_t length
    ) {
        uint32_t coreIdx = AscendC::GetBlockIdx();

        uint32_t coreNum = AscendC::GetBlockNum();

        /*
         * 32B 对齐所需元素数：
         *
         * float32: 32 / 4 = 8
         * float16: 32 / 2 = 16
         */
        uint32_t alignNum = 32 / sizeof(DT_INPUT_X);

        uint32_t blockLength = 0;

        if (coreNum > 0) {
            blockLength = CeilDiv(length, coreNum);

            blockLength = AlignUp(blockLength, alignNum);
        }

        this->coreOffset = coreIdx * blockLength;

        if (this->coreOffset >= length) {
            this->coreLength = 0;
        } else {
            this->coreLength = length - this->coreOffset;

            if (this->coreLength > blockLength) {
                this->coreLength = blockLength;
            }
        }

        inputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(input_x) + this->coreOffset,
            this->coreLength
        );

        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_INPUT_X *>(output) + this->coreOffset,
            this->coreLength
        );

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_INPUT_X)
        );

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(DT_INPUT_X)
        );

        pipe.InitBuffer(
            tmpBufA,
            TILE_LENGTH * sizeof(DT_INPUT_X)
        );

        pipe.InitBuffer(
            tmpBufB,
            TILE_LENGTH * sizeof(DT_INPUT_X)
        );
    }

    __aicore__ inline void Process() {
        if (this->coreLength == 0) {
            return;
        }

        uint32_t fullLoop = this->coreLength / TILE_LENGTH;

        uint32_t tailLen = this->coreLength - fullLoop * TILE_LENGTH;

        /*
         * 完整 tile：
         * coreOffset 已经按 32B 对齐；
         * TILE_LENGTH 对 float16 / float32 都是 32B 整倍数；
         * 因此直接使用 DataCopy，不做额外判断。
         */
        for (uint32_t i = 0; i < fullLoop; ++i) {
            uint32_t offset = i * TILE_LENGTH;

            CopyInAligned(offset);

            Compute(TILE_LENGTH);

            CopyOutAligned(offset);
        }

        /*
         * 尾块：
         * tailLen 可能不是 32B 整倍数；
         * 单独走 DataCopyPad 兼容非对齐场景。
         */
        if (tailLen > 0) {
            uint32_t offset = fullLoop * TILE_LENGTH;

            CopyInTail(offset, tailLen);

            Compute(tailLen);

            CopyOutTail(offset, tailLen);
        }
    }

private:
    __aicore__ inline void CopyInAligned(uint32_t offset) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal =
            inQueueX.AllocTensor<DT_INPUT_X>();

        AscendC::DataCopy(
            xLocal,
            inputGm[offset],
            TILE_LENGTH
        );

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutAligned(uint32_t offset) {
        AscendC::LocalTensor<DT_INPUT_X> yLocal =
            outQueueY.DeQue<DT_INPUT_X>();

        AscendC::DataCopy(
            outputGm[offset],
            yLocal,
            TILE_LENGTH
        );

        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyInTail(
        uint32_t offset,
        uint32_t calcLen
    ) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal =
            inQueueX.AllocTensor<DT_INPUT_X>();

        uint32_t gmOffset = this->coreOffset + offset;

        uint32_t byteOffset = gmOffset * sizeof(DT_INPUT_X);

        uint32_t byteLen = calcLen * sizeof(DT_INPUT_X);

        if ((byteOffset % 32 == 0) && (byteLen % 32 == 0)) {
            AscendC::DataCopy(
                xLocal,
                inputGm[offset],
                calcLen
            );
        } else {
            AscendC::DataCopyExtParams copyParams;

            copyParams.blockCount = 1;
            copyParams.blockLen = byteLen;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams;

            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = static_cast<DT_INPUT_X>(0);

            AscendC::DataCopyPad(
                xLocal,
                inputGm[offset],
                copyParams,
                padParams
            );
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyOutTail(
        uint32_t offset,
        uint32_t calcLen
    ) {
        AscendC::LocalTensor<DT_INPUT_X> yLocal =
            outQueueY.DeQue<DT_INPUT_X>();

        uint32_t gmOffset = this->coreOffset + offset;

        uint32_t byteOffset = gmOffset * sizeof(DT_INPUT_X);

        uint32_t byteLen = calcLen * sizeof(DT_INPUT_X);

        if ((byteOffset % 32 == 0) && (byteLen % 32 == 0)) {
            AscendC::DataCopy(
                outputGm[offset],
                yLocal,
                calcLen
            );
        } else {
            AscendC::DataCopyExtParams copyParams;

            copyParams.blockCount = 1;
            copyParams.blockLen = byteLen;
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            AscendC::DataCopyPad(
                outputGm[offset],
                yLocal,
                copyParams
            );
        }

        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void Compute(uint32_t calcLen) {
        AscendC::LocalTensor<DT_INPUT_X> xLocal =
            inQueueX.DeQue<DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X> yLocal =
            outQueueY.AllocTensor<DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X> tmpA =
            tmpBufA.Get<DT_INPUT_X>();

        AscendC::LocalTensor<DT_INPUT_X> tmpB =
            tmpBufB.Get<DT_INPUT_X>();



        AscendC::Muls(
            tmpA,
            xLocal,
            static_cast<DT_INPUT_X>(GELU_INV_SQRT2),
            calcLen
        );

        AscendC::Erf(
            tmpB,
            tmpA,
            calcLen
        );

        AscendC::Adds(
            tmpB,
            tmpB,
            static_cast<DT_INPUT_X>(1.0f),
            calcLen
        );

        AscendC::Muls(
            tmpB,
            tmpB,
            static_cast<DT_INPUT_X>(0.5f),
            calcLen
        );

        AscendC::Mul(
            yLocal,
            xLocal,
            tmpB,
            calcLen
        );

        outQueueY.EnQue<DT_INPUT_X>(yLocal);

        inQueueX.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;

    AscendC::TQue<AscendC::QuePosition::GELU_CAT(VEC, OUT), BUFFER_NUM> outQueueY;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufA;

    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBufB;

    AscendC::GlobalTensor<DT_INPUT_X> inputGm;

    AscendC::GlobalTensor<DT_INPUT_X> outputGm;

    uint32_t coreLength = 0;

    uint32_t coreOffset = 0;
};


template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    REGISTER_TILING_DEFAULT(GeluTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        GeluTilingData,
        tiling_data,
        tiling
    );

    KernelGelu<DT_INPUT_X> op;

    op.Init(
        input_x,
        output,
        tiling_data.length
    );

    op.Process();
}