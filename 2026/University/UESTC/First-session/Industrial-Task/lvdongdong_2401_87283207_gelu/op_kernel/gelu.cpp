// Kernel侧核函数实现
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr float GELU_D1 = 1.59501576856f;
constexpr float GELU_D3 = 0.074011292044f;
constexpr float GELU_D5 = -0.00070303357684f;

// float 走"串行 TBuf 直通"的最大片数阈值:
//   实测结论——多 tile float 用例访存受限、必须靠 TQue 双缓冲重叠, 串行(补同步后无重叠)一律更慢;
//   仅单 tile(无需重叠)时串行才省掉 TQue 机制开销. 故设为 1: 单 tile 走串行直通, 其余全走 TQue.
constexpr uint32_t kSerialMaxTiles = 1;

// 综合版: float 按 tile 数分两条路, 各取所长(串行省开销 + TQue 保重叠, 均正确):
//   float, tileNum<=kSerialMaxTiles : 串行 TBuf 直通(手动 SetFlag/WaitFlag 补流水同步),
//                                     省去 TQue 的 Alloc/Free/EnQue/DeQue 开销, 利中小 float(计算受限);
//   float, tileNum> kSerialMaxTiles : TQue 双缓冲, 保留搬运/计算重叠, 利大 float(访存受限, 如 test6);
//   half : Gelu<false,false> + TQue 双缓冲(计算/路径固定不动).
// 注: TBuf 下高阶 DataCopy/向量 API 不会自动插入 MTE↔V 同步(实测无同步则 99.9% 错), 必须手动补齐.
template <class DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData &t)
    {
        this->tileDataNum = t.tileDataNum;
        this->tileNum = t.finalTileNum;

        uint32_t coreIdx = AscendC::GetBlockIdx();
        // GM 偏移与每核元素数用 64 位, 支持超大张量(避免 32 位偏移溢出)
        uint64_t globalOffset;
        if (coreIdx < t.tailBlockNum) {
            this->coreDataNum = t.bigCoreDataNum;
            this->baseTileBlock = t.bigBaseTileBlock;
            this->remTiles = t.bigRemTiles;
            globalOffset = t.bigCoreDataNum * coreIdx;
        } else {
            this->coreDataNum = t.smallCoreDataNum;
            this->baseTileBlock = t.smallBaseTileBlock;
            this->remTiles = t.smallRemTiles;
            globalOffset = t.smallCoreBaseOffset
                         + t.smallCoreDataNum * (coreIdx - t.tailBlockNum);
        }

        uint64_t realCoreDataNum = 0;
        if (t.length > globalOffset) {
            uint64_t remain = t.length - globalOffset;
            realCoreDataNum = (remain < this->coreDataNum) ? remain : this->coreDataNum;
        }
        this->realCoreDataNum = realCoreDataNum;

        xGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)input_x + globalOffset, realCoreDataNum);
        yGm.SetGlobalBuffer((__gm__ DT_INPUT_X *)output + globalOffset, realCoreDataNum);

        if constexpr (kInputIsFloat) {
            // 片数少 => 串行 TBuf 直通(含单 tile); 片数多 => TQue 双缓冲重叠
            this->useTBufPath = (this->tileNum <= kSerialMaxTiles);
            if (this->useTBufPath) {
                pipe.InitBuffer(xBuf, this->tileDataNum * sizeof(DT_INPUT_X));
                pipe.InitBuffer(yBuf, this->tileDataNum * sizeof(DT_INPUT_X));
            } else {
                pipe.InitBuffer(inQueueX, kQueDepth, this->tileDataNum * sizeof(DT_INPUT_X));
                pipe.InitBuffer(outQueueY, kQueDepth, this->tileDataNum * sizeof(DT_INPUT_X));
            }
            pipe.InitBuffer(calcBuf, this->tileDataNum * sizeof(float));
        } else {
            this->useTBufPath = false;
            pipe.InitBuffer(inQueueX, kQueDepth, this->tileDataNum * sizeof(DT_INPUT_X));
            pipe.InitBuffer(outQueueY, kQueDepth, this->tileDataNum * sizeof(DT_INPUT_X));
        }
    }

    __aicore__ inline void Process()
    {
        if (this->realCoreDataNum == 0) {
            return;
        }
        if (this->useTBufPath) {
            ProcessTBufSerial();
        } else {
            ProcessTiled();
        }
    }

private:
    static constexpr bool kInputIsFloat = (sizeof(DT_INPUT_X) == sizeof(float));
    // 队列深度固定双缓冲: 与 host 侧 UB 预算(float 20 B/elem, half 8 B/elem)一致.
    // 实测三缓冲(float=3)会挤小 tile 且与 host 预算不符, test6 反而更慢, 故回退.
    static constexpr int32_t kQueDepth = 2;

    // float 串行 TBuf 直通(手动流水同步版): 逐片复用同一对 TBuf, 省去 TQue 的 Alloc/Free/EnQue/DeQue 机制开销,
    // 但用 SetFlag/WaitFlag 显式补齐流水同步(TBuf 下高阶 API 不会自动插入 MTE↔V 同步):
    //   MTE2->V : 搬入完成再计算(RAW x);  V->MTE3 : 计算完成再搬出(RAW y);
    //   V->MTE2 : 上片计算读完 x 再让下片搬入覆盖(WAR x);  MTE3->V : 上片搬出读完 y 再让下片计算覆盖(WAR y).
    __aicore__ inline void ProcessTBufSerial()
    {
        if constexpr (kInputIsFloat) {
            constexpr uint32_t blockElems = 32u / sizeof(DT_INPUT_X);
            AscendC::LocalTensor<DT_INPUT_X> xLocal = xBuf.template Get<DT_INPUT_X>();
            AscendC::LocalTensor<DT_INPUT_X> yLocal = yBuf.template Get<DT_INPUT_X>();

            event_t evtMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
            event_t evtVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
            event_t evtMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
            event_t evtVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));

            uint64_t offset = 0;
            for (uint32_t i = 0; i < this->tileNum; i++) {
                if (offset >= this->realCoreDataNum) {
                    break;
                }
                uint32_t curBlocks = this->baseTileBlock + (i < this->remTiles ? 1u : 0u);
                uint32_t curLen = curBlocks * blockElems;
                uint64_t remain = this->realCoreDataNum - offset;
                if (curLen > remain) {
                    curLen = static_cast<uint32_t>(remain);
                }
                bool aligned = (curLen % blockElems == 0);

                if (i > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVToMte2);  // WAR x: 上片算完读 x
                }
                if (aligned) {
                    AscendC::DataCopy(xLocal, xGm[offset], curLen);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
                    AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
                    AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2ToV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2ToV);  // RAW x: 搬入完成

                if (i > 0) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evtMte3ToV);  // WAR y: 上片搬出读完 y
                }
                ComputeFloat(xLocal, yLocal, curLen);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVToMte2);  // x 可被下片覆盖
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVToMte3);  // RAW y: 计算完成

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVToMte3);
                if (aligned) {
                    AscendC::DataCopy(yGm[offset], yLocal, curLen);
                } else {
                    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
                    AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evtMte3ToV);  // y 可被下片计算覆盖

                offset += curBlocks * blockElems;
            }
            // 收尾: 消费最后一片遗留的 V->MTE2 / MTE3->V(每片各 set 一次, 循环内已 wait N-1 次)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVToMte2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evtMte3ToV);
        }
    }

    __aicore__ inline void ProcessTiled()
    {
        constexpr uint32_t blockElems = 32u / sizeof(DT_INPUT_X);
        uint64_t offset = 0;
        for (uint32_t i = 0; i < this->tileNum; i++) {
            if (offset >= this->realCoreDataNum) {
                break;
            }
            uint32_t curBlocks = this->baseTileBlock + (i < this->remTiles ? 1u : 0u);
            uint32_t curLen = curBlocks * blockElems;
            uint64_t remain = this->realCoreDataNum - offset;
            if (curLen > remain) {
                curLen = static_cast<uint32_t>(remain);
            }
            bool aligned = (curLen % blockElems == 0);
            CopyIn(offset, curLen, aligned);
            Compute(curLen);
            CopyOut(offset, curLen, aligned);
            offset += curBlocks * blockElems;
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t curLen, bool aligned)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.template AllocTensor<DT_INPUT_X>();
        if (aligned) {
            AscendC::DataCopy(xLocal, xGm[offset], curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_INPUT_X> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t curLen)
    {
        AscendC::LocalTensor<DT_INPUT_X> xLocal = inQueueX.template DeQue<DT_INPUT_X>();
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.template AllocTensor<DT_INPUT_X>();

        if constexpr (kInputIsFloat) {
            ComputeFloat(xLocal, yLocal, curLen);
        } else {
            AscendC::Gelu<DT_INPUT_X, false, false>(yLocal, xLocal, curLen);
        }

        outQueueY.template EnQue<DT_INPUT_X>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeFloat(AscendC::LocalTensor<DT_INPUT_X> xLocal,
                                        AscendC::LocalTensor<DT_INPUT_X> yLocal,
                                        uint32_t curLen)
    {
        if constexpr (kInputIsFloat) {
            AscendC::LocalTensor<float> tmp = calcBuf.Get<float>();
            AscendC::Mul(tmp, xLocal, xLocal, curLen);
            AscendC::Muls(yLocal, tmp, GELU_D5, curLen);
            AscendC::Adds(yLocal, yLocal, GELU_D3, curLen);
            AscendC::Mul(yLocal, yLocal, tmp, curLen);
            AscendC::Adds(yLocal, yLocal, GELU_D1, curLen);
            AscendC::Mul(yLocal, yLocal, xLocal, curLen);
            AscendC::Sigmoid(yLocal, yLocal, curLen);
            AscendC::Mul(yLocal, xLocal, yLocal, curLen);
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t curLen, bool aligned)
    {
        AscendC::LocalTensor<DT_INPUT_X> yLocal = outQueueY.template DeQue<DT_INPUT_X>();
        if (aligned) {
            AscendC::DataCopy(yGm[offset], yLocal, curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_INPUT_X)), 0, 0, 0};
            AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, kQueDepth> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, kQueDepth> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECIN> xBuf;
    AscendC::TBuf<AscendC::TPosition::VECOUT> yBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<DT_INPUT_X> xGm;
    AscendC::GlobalTensor<DT_INPUT_X> yGm;
    uint32_t tileDataNum = 0;
    uint64_t coreDataNum = 0;
    uint64_t realCoreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t baseTileBlock = 0;
    uint32_t remTiles = 0;
    bool useTBufPath = false;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data);
    op.Process();
}
