// Kernel侧核函数实现
// FastGelu: y = x * exp(0.851*(x-|x|)) / (1 + exp(-1.702*|x|))
// 调优策略：
//   1) 多核按 tile 均分（usedCoreNum=min(coreNum,totalTiles)，核间 tile 数差 ≤1，用满核、负载均衡）；
//   2) UB 双缓冲（BUFFER_NUM=2）驱动 MTE2/V/MTE3 三级流水重叠；
//   3) 算法等价降 Exp（最关键）：令 E=exp(-1.702·|x|)，
//        x≥0: div_up = x·e^0 = x      ；div_down = 1+E
//        x<0: div_up = x·e^{0.851·2x} = x·e^{-1.702·|x|} = x·E；div_down = 1+E
//      故 div_up = max(x, x·E)（E∈(0,1]：x≥0 取 x，x<0 取 x·E），y = max(x,x·E)/(1+E)。
//      两次 vexp 合并为一次（vexp 是 NPU 最贵的超越函数），且无需 mask/Select。
//      compute 仅 2 个 temp（t1=|x|->E->xE->div_up，t2=div_down），fp16 路径复用输入 xF 作输出省掉 yF，
//      UB/elem 降到 fp32=24 / fp16=20 -> tile 更大、循环开销与 DMA 起停摊销更优；
//   4) 全满 tile 走 DataCopy 快路径（32B 对齐），仅末核末 tile 用 DataCopyPad 处理非对齐尾巴；
//   5) fp16 输入内部升 fp32 计算、末尾 cast 回 fp16（满足双千分之一精度）。
#include "kernel_operator.h"

#include <type_traits>

#include "fast_gelu_tiling.h"
#include "tiling_key_fast_gelu.h"

// CANN 8.5 起 Ascend C 类型全部收入 AscendC:: 命名空间（全局别名不再自动提供），
// 故在此 using，使 LocalTensor/TPipe/TQue/TBuf/TPosition/intrinsic 等可直接裸用。
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // UB 双缓冲，使 MTE2/V/MTE3 三级流水重叠

template <class DT_X>
class KernelFastGelu {
public:
    __aicore__ inline KernelFastGelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FastGeluTilingData &tiling) {
        totalLength_  = tiling.totalLength;
        usedCoreNum_  = tiling.usedCoreNum;
        tileLength_   = tiling.tileLength;
        tilesPerCore_ = tiling.tilesPerCore;
        tailCoreNum_  = tiling.tailCoreNum;
        dtypeSize_    = tiling.dtypeSize;

        // 全量 GM 视图，核内按全局 offset 索引（DataCopyPad 的 xGm[offset]）
        xGm.SetGlobalBuffer((__gm__ DT_X *)x, totalLength_);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y, totalLength_);

        blockIdx_ = GetBlockIdx();

        // tile 缓冲容量（元素），32B 对齐（host 已保证 tileLength 对齐，此处幂等保护）
        uint32_t blockElems = 32u / dtypeSize_;  // fp32=8, fp16=16
        tileCap_ = (uint32_t)(((tileLength_ + blockElems - 1) / blockElems) * blockElems);
        if (tileCap_ == 0) {
            tileCap_ = blockElems;
        }

        uint32_t tileBytesX = tileCap_ * dtypeSize_;            // 每块 UB 缓冲字节（输入/输出 dtype）
        uint32_t tmpBytesF  = tileCap_ * (uint32_t)sizeof(float);  // fp32 临时缓冲字节

        pipe.InitBuffer(xQue, BUFFER_NUM, tileBytesX);
        pipe.InitBuffer(yQue, BUFFER_NUM, tileBytesX);
        pipe.InitBuffer(t1Buf, tmpBytesF);  // intermediate computation
        if constexpr (std::is_same<DT_X, half>::value) {
            // fp16 路径专用：输入升 fp32，计算后原地复用作输出降 fp16 前的暂存（省掉 yF）
            pipe.InitBuffer(xFBuf, tmpBytesF);
        }
    }

    __aicore__ inline void Process() {
        if (totalLength_ == 0) {
            return;  // 空张量：host 已设 SetBlockDim(1)，本核直接返回
        }

        int64_t coreIdx = blockIdx_;
        if ((uint64_t)coreIdx >= usedCoreNum_) {
            return;  // 防御：未被调度的核直接返回
        }

        int64_t tilesPerCore = (int64_t)tilesPerCore_;
        int64_t tailCoreNum  = (int64_t)tailCoreNum_;

        // 前 tailCoreNum 个核各多扛 1 tile；tile 起始连续排列，核间无空洞、无重叠。
        int64_t myTiles;
        int64_t myStartTile;
        if (coreIdx < tailCoreNum) {
            myTiles     = tilesPerCore + 1;
            myStartTile = coreIdx * (tilesPerCore + 1);
        } else {
            myTiles     = tilesPerCore;
            myStartTile = tailCoreNum * (tilesPerCore + 1) + (coreIdx - tailCoreNum) * tilesPerCore;
        }

        int64_t tileLen = (int64_t)tileLength_;
        int64_t offset  = myStartTile * tileLen;

        for (int64_t t = 0; t < myTiles; ++t) {
            int64_t valid = tileLen;
            if (offset + tileLen > (int64_t)totalLength_) {
                valid = (int64_t)totalLength_ - offset;  // 仅末核末 tile 可能非整
            }
            if (valid <= 0) {
                break;
            }
            ComputePerTile(offset, valid);
            offset += tileLen;
        }
    }

private:
    __aicore__ inline void ComputePerTile(int64_t offset, int64_t valid) {
        // 判断是否32 Bytes对齐，只要对齐即可走DataCopy快路径
        bool isAligned = (((uint64_t)valid * dtypeSize_) % 32 == 0);

        // GM -> UB (MTE2)
        auto xLocal = xQue.AllocTensor<DT_X>();
        if (isAligned) {
            DataCopy(xLocal, xGm[offset], (uint32_t)valid);  // 快路径：无 pad 开销
        } else {
            // 非对齐搬运：blockLen 按字节，支持 valid 非 32B 整倍
            DataCopyExtParams copyParams{};
            copyParams.blockCount = 1;
            copyParams.blockLen    = (uint32_t)(valid * (int64_t)dtypeSize_);
            copyParams.srcStride   = 0;
            copyParams.dstStride   = 0;
            DataCopyPadExtParams<DT_X> padParams{};
            padParams.isPad         = true;   // UB 尾巴填 0；计算由 calCount=valid 限定不受影响
            padParams.leftPadding   = 0;
            padParams.rightPadding  = 0;
            padParams.paddingValue  = (DT_X)0;
            DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        }
        xQue.EnQue(xLocal);
        xLocal = xQue.DeQue<DT_X>();  // 进入 V 流水

        auto yLocal = yQue.AllocTensor<DT_X>();

        // 微架构优化：将 valid 向上对齐到 32B 数据块（blockElems），消除 Vector 指令由于处理非对齐尾巴产生的动态 SetVectorMask 开销。
        // 因为 DataCopyPad 已经将 UB 尾部填充了 0，而 FastGelu(0) = 0，所以多计算的 padding 元素结果为 0，绝对安全。
        // 最后回写 GM 时只 copy 有效的 valid 长度，不会污染全局内存。
        uint32_t blockElems = 32u / dtypeSize_;
        uint32_t alignedValid = ((valid + blockElems - 1) / blockElems) * blockElems;

        if constexpr (std::is_same<DT_X, half>::value) {
            // fp16 -> fp32 计算 -> fp16
            auto xF = xFBuf.Get<float>();
            Cast(xF, xLocal, RoundMode::CAST_NONE, alignedValid);  // fp16->fp32 无损
            Compute(xF, xF, alignedValid);  // 复用 xF 作输出：Mul 读完 x 后 Div 才写回，V 管线顺序安全
            Cast(yLocal, xF, RoundMode::CAST_RINT, alignedValid);  // fp32->fp16 最近偶
        } else {
            // DT_X == float：xLocal/yLocal 本身即 fp32，直接计算
            Compute(xLocal, yLocal, alignedValid);
        }

        // UB -> GM (MTE3)：无 padParams，框架自动剥除填充
        yQue.EnQue(yLocal);
        yLocal = yQue.DeQue<DT_X>();
        if (isAligned) {
            DataCopy(yGm[offset], yLocal, (uint32_t)valid);  // 快路径
        } else {
            DataCopyExtParams copyParams{};
            copyParams.blockCount = 1;
            copyParams.blockLen    = (uint32_t)(valid * (int64_t)dtypeSize_);
            copyParams.srcStride   = 0;
            copyParams.dstStride   = 0;
            DataCopyPad(yGm[offset], yLocal, copyParams);
        }
        xQue.FreeTensor(xLocal);
        yQue.FreeTensor(yLocal);
    }

    // 极致优化：逐元素 FastGelu（全程 fp32）。
    // 基于数学等价推导：x * exp(0.851*(x-|x|)) / (1 + exp(-1.702*|x|)) 严格等价于 x * sigmoid(1.702*x)
    // 充分利用硬件高级指令，指令流从 4 条 (Muls->Exp->Adds->Div) 减为 3 条 (Muls->Sigmoid->Mul)。
    __aicore__ inline void Compute(LocalTensor<float> x, LocalTensor<float> y, int32_t calCount) {
        auto t1 = t1Buf.Get<float>();

        Muls(t1, x, 1.702f, calCount);        // t1 = 1.702 * x
        Sigmoid(t1, t1, calCount);            // t1 = 1 / (1 + exp(-1.702 * x))
        Mul(y, x, t1, calCount);              // y  = x * t1
    }

private:
    TPipe pipe;
    GlobalTensor<DT_X> xGm;
    GlobalTensor<DT_X> yGm;
    TQue<TPosition::VECIN, BUFFER_NUM> xQue;   // 输入 dtype
    TQue<TPosition::VECOUT, BUFFER_NUM> yQue;  // 输出 dtype
    TBuf<TPosition::VECCALC> t1Buf;            // intermediate calculation
    TBuf<TPosition::VECCALC> xFBuf;            // fp16 路径输入升 fp32 / 输出降 fp16 前的 fp32 暂存（复用）

    uint64_t totalLength_ = 0;
    uint32_t usedCoreNum_ = 1;
    uint64_t tileLength_ = 0;
    uint32_t tilesPerCore_ = 0;
    uint32_t tailCoreNum_ = 0;
    uint32_t dtypeSize_ = 4;
    uint32_t tileCap_ = 0;
    int64_t blockIdx_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void fast_gelu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FastGeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(FastGeluTilingData, tiling_data, tiling);
    KernelFastGelu<DT_X> op;
    op.Init(x, y, tiling_data);
    op.Process();
}
