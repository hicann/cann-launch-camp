// Kernel 侧：FAST（同形）+ BCAST（右对齐广播，朴素行解码）
#include "kernel_operator.h"

#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

// CANN 8.5.0：AscendC 核心 API 均在 AscendC 命名空间
using namespace AscendC;

// BCAST 多行打包专用缓冲上限：独立于全局 tileSize(4096)，放大到 8192 可在
// lastDimLen 中等(接近 4096)时让 rowsPerTile>1，成倍减少 tile 数与 queue 操作。
// 仅 BCAST 路径使用，FAST/SCALAR 仍用 qX1/qX2/qOut@4096，互不影响。
#define PACK_TILE 8192

template <class DT_X1>
class KernelLessEqual {
public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const LessEqualTilingData &td) {
        mode = td.mode;
        totalElements = td.totalElements;
        tileSize = td.tileSize;
        perCore = td.perCore;
        ndim = td.ndim;
        lastDimLen = td.lastDimLen;
        totalRows = td.totalRows;
        // mode=2(SCALAR) 时，ndim 字段承载：1=x1标量, 2=x2标量
        scalarInput = (mode == 2) ? td.ndim : 0;
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            outShape[i] = td.outShape[i];
            x1StrideArr[i] = td.x1Stride[i];
            x2StrideArr[i] = td.x2Stride[i];
            idx[i] = 0;
        }

        x1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x1));
        x2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x2));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));

        if (totalElements == 0) { return; }  // 空张量：无缓冲需分配

        // 队列双缓冲（BCAST / FAST / SCALAR 共用）
        pipe.InitBuffer(qX1, 2, tileSize * static_cast<uint32_t>(sizeof(DT_X1)));
        pipe.InitBuffer(qX2, 2, tileSize * static_cast<uint32_t>(sizeof(DT_X1)));
        pipe.InitBuffer(qOut, 2, tileSize * static_cast<uint32_t>(sizeof(int8_t)));
        // BCAST 多行打包专用缓冲（上限 PACK_TILE=8192），depth=1。
        // 仅在 BCAST 且末维长度可进入打包分支时分配，避免 FAST/SCALAR 路径承担额外 UB 分配与
        // 布局开销，从而防止其他测试点被连坐劣化。
        // depth=2 会导致 float32/int32 的 UB 占用超 240KB 而 TLE；depth=1 既符合当前单 tile
        // 顺序流程，又把 float32/int32 压回 ~200KB 安全区。
        if (mode == 1 && lastDimLen <= PACK_TILE) {
            pipe.InitBuffer(qX1B, 1, PACK_TILE * static_cast<uint32_t>(sizeof(DT_X1)));
            pipe.InitBuffer(qX2B, 1, PACK_TILE * static_cast<uint32_t>(sizeof(DT_X1)));
            pipe.InitBuffer(qOutB, 1, PACK_TILE * static_cast<uint32_t>(sizeof(int8_t)));
        }

        // 纯计算缓冲
        pipe.InitBuffer(bufMask, ((tileSize / 8) + 31) / 32 * 32);  // bit-packed mask
        pipe.InitBuffer(bufMinI32, tileSize * static_cast<uint32_t>(sizeof(int32_t)));
        pipe.InitBuffer(bufOnesHalf, tileSize * static_cast<uint32_t>(sizeof(half)));
        pipe.InitBuffer(bufZerosHalf, tileSize * static_cast<uint32_t>(sizeof(half)));
        pipe.InitBuffer(bufOutHalf, tileSize * static_cast<uint32_t>(sizeof(half)));
        pipe.InitBuffer(bufAHalf, tileSize * static_cast<uint32_t>(sizeof(half)));
        pipe.InitBuffer(bufBHalf, tileSize * static_cast<uint32_t>(sizeof(half)));

        // 预置全 1 / 全 0 的 half 向量：Select(vsel) 在 910B 上仅支持 half 的
        // dst/src0/src1，故 ones/zeros 以 half 形式提供，输出再 Cast 成 int8。
        LocalTensor<half> onesH = bufOnesHalf.Get<half>();
        LocalTensor<half> zerosH = bufZerosHalf.Get<half>();
        Duplicate(onesH, static_cast<half>(1.0), tileSize);
        Duplicate(zerosH, static_cast<half>(0.0), tileSize);
    }

    __aicore__ inline void Process() {
        if (totalElements == 0) { return; }
        if (mode == 0) {
            ProcessFast();
        } else if (mode == 2) {
            ProcessScalar();
        } else {
            ProcessBcast();
        }
    }

private:
    // 把计算量向上取到 256 的倍数：保证 bit-packed Compare mask 是 32 字节粒度，
    // 满足 half(256B)/float/int32(512B) 的对齐规则；只把真实 L 个元素写回 GM。
    __aicore__ inline uint32_t RoundUp256(uint32_t v) {
        return (v + 255) / 256 * 256;
    }

    // 由 aLocal/bLocal（DT_X1 原生）产出 Lc 路的 0/1 int8，写入 outI8。
    // 指令链：Compare 生成 bit-mask -> Select 展开成 half 的 0/1 -> Cast 成 int8。
    // 注意 910B 的 vsel/Select 仅支持 half 的 dst/src，故必须保留 half 中间缓冲。
    // aScalar/bScalar 为 true 表示该操作数整行只有元素 [0]（标量广播行），需在 half 域内展开。
    __aicore__ inline void ComputeTile(const LocalTensor<DT_X1> &aLocal,
                                       const LocalTensor<DT_X1> &bLocal,
                                       const LocalTensor<int8_t> &outI8,
                                       uint32_t Lc,
                                       bool aScalar, bool bScalar) {
        LocalTensor<uint8_t> mask = bufMask.Get<uint8_t>();
        LocalTensor<half> outHalf = bufOutHalf.Get<half>();

        // aScalar/bScalar 仅 int8 分支直接使用；其余 dtype 已在调用方把标量展开成整 tile 常量，
        // 故此处恒为张量-张量运算。void 转换仅为避免未使用参数告警。
        (void)aScalar;
        (void)bScalar;

        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            // 整数路线：x1 <= x2 等价于 Min(x1,x2) == x1，全精确整数运算。
            // 三种情形统一为张量-张量（标量操作数已在调用方 Duplicate 成整 tile 常量）。
            LocalTensor<int32_t> minBuf = bufMinI32.Get<int32_t>();
            Min(minBuf, aLocal, bLocal, Lc);
            Compare(mask, aLocal, minBuf, CMPMODE::EQ, Lc);
        } else if constexpr (std::is_same_v<DT_X1, int8_t>) {
            // int8 -> half 无损（|v| <= 127 在 half 中精确），在 half 域比较
            LocalTensor<half> aHalf = bufAHalf.Get<half>();
            LocalTensor<half> bHalf = bufBHalf.Get<half>();
            if (aScalar) {
                int8_t va = aLocal.GetValue(0);
                Duplicate(aHalf, static_cast<half>(static_cast<float>(static_cast<int32_t>(va))), Lc);
            } else {
                Cast(aHalf, aLocal, RoundMode::CAST_NONE, Lc);
            }
            if (bScalar) {
                int8_t vb = bLocal.GetValue(0);
                Duplicate(bHalf, static_cast<half>(static_cast<float>(static_cast<int32_t>(vb))), Lc);
            } else {
                Cast(bHalf, bLocal, RoundMode::CAST_NONE, Lc);
            }
            Compare(mask, aHalf, bHalf, CMPMODE::LE, Lc);
        } else {
            // half / float 直接张量-张量比较（标量操作数已在调用方 Duplicate 成整 tile 常量）。
            Compare(mask, aLocal, bLocal, CMPMODE::LE, Lc);
        }

        // bit-mask -> half 0/1 -> int8 0/1：
        // 910B 的 vsel/Select 在带字节掩码时仅接受 half 的 dst/src0/src1（不支持标量 src），
        // 故用预置的 half ones/zeros 张量做 VSEL_TENSOR_TENSOR_MODE，再 Cast 成 int8。
        LocalTensor<half> onesH = bufOnesHalf.Get<half>();
        LocalTensor<half> zerosH = bufZerosHalf.Get<half>();
        Select(outHalf, mask, onesH, zerosH, SELMODE::VSEL_TENSOR_TENSOR_MODE, Lc);
        Cast(outI8, outHalf, RoundMode::CAST_RINT, Lc);
    }

    // ---------------- FAST 路径 ----------------
    // 同形输入：按输出元素数切核，每个 tile 顺序执行搬运→计算→写回。
    __aicore__ inline void ProcessFast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t off = blockIdx * perCore;
        if (off >= totalElements) { return; }
        uint32_t myElems = perCore;
        if (off + myElems > totalElements) { myElems = totalElements - off; }

        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};
        uint32_t numTiles = (myElems + tileSize - 1) / tileSize;
        if (numTiles == 0) { return; }

        for (uint32_t i = 0; i < numTiles; ++i) {
            uint32_t t = i * tileSize;
            uint32_t L = tileSize;
            if (t + L > myElems) { L = myElems - t; }
            uint32_t Lc = RoundUp256(L);
            uint32_t base = off + t;

            LocalTensor<DT_X1> a = qX1.AllocTensor<DT_X1>();
            LocalTensor<DT_X1> b = qX2.AllocTensor<DT_X1>();
            DataCopyPad(a, x1Gm[base],
                        DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            DataCopyPad(b, x2Gm[base],
                        DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            qX1.EnQue(a);
            qX2.EnQue(b);

            a = qX1.DeQue<DT_X1>();
            b = qX2.DeQue<DT_X1>();

            LocalTensor<int8_t> out = qOut.AllocTensor<int8_t>();
            ComputeTile(a, b, out, Lc, false, false);

            qX1.FreeTensor(a);
            qX2.FreeTensor(b);
            qOut.EnQue(out);

            out = qOut.DeQue<int8_t>();
            DataCopyPad(yGm[base], out, DataCopyExtParams{1, L, 0, 0, 0});
            qOut.FreeTensor(out);
        }
    }

    // ---------------- SCALAR 广播路径 ----------------
    // 一个输入为标量：直接按输出元素数切核。每 tile 把标量操作数搬 1 个元素并展开到
    // 整 tile 常量，另一操作数正常搬运。
    __aicore__ inline void ProcessScalar() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t off = blockIdx * perCore;
        if (off >= totalElements) { return; }
        uint32_t myElems = perCore;
        if (off + myElems > totalElements) { myElems = totalElements - off; }

        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};
        uint32_t numTiles = (myElems + tileSize - 1) / tileSize;
        if (numTiles == 0) { return; }

        const GlobalTensor<DT_X1> &tensorGm = (scalarInput == 1) ? x2Gm : x1Gm;

        for (uint32_t i = 0; i < numTiles; ++i) {
            uint32_t t = i * tileSize;
            uint32_t L = tileSize;
            if (t + L > myElems) { L = myElems - t; }
            uint32_t Lc = RoundUp256(L);
            uint32_t base = off + t;

            LocalTensor<DT_X1> a;
            LocalTensor<DT_X1> b;
            if (scalarInput == 1) {
                a = qX1.AllocTensor<DT_X1>();
                DataCopyPad(a, x1Gm[0],
                            DataCopyExtParams{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
                qX1.EnQue(a);
                a = qX1.DeQue<DT_X1>();
                if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                    Duplicate(a, a.GetValue(0), Lc);
                }

                b = qX2.AllocTensor<DT_X1>();
                DataCopyPad(b, tensorGm[base],
                            DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
                qX2.EnQue(b);
                b = qX2.DeQue<DT_X1>();
            } else {
                a = qX1.AllocTensor<DT_X1>();
                DataCopyPad(a, tensorGm[base],
                            DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
                qX1.EnQue(a);
                a = qX1.DeQue<DT_X1>();

                b = qX2.AllocTensor<DT_X1>();
                DataCopyPad(b, x2Gm[0],
                            DataCopyExtParams{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
                qX2.EnQue(b);
                b = qX2.DeQue<DT_X1>();
                if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                    Duplicate(b, b.GetValue(0), Lc);
                }
            }

            LocalTensor<int8_t> out = qOut.AllocTensor<int8_t>();
            ComputeTile(a, b, out, Lc, (scalarInput == 1), (scalarInput == 2));

            qX1.FreeTensor(a);
            qX2.FreeTensor(b);
            qOut.EnQue(out);

            out = qOut.DeQue<int8_t>();
            DataCopyPad(yGm[base], out, DataCopyExtParams{1, L, 0, 0, 0});
            qOut.FreeTensor(out);
        }
    }

    // ---------------- BCAST 路径 ----------------
    // 初始化行状态：由 rowId 直接算出 x1/x2 行首的线性下标。
    __aicore__ inline void InitRowState(uint32_t r0) {
        x1Base = 0;
        x2Base = 0;
        uint32_t rem = r0;
        for (uint32_t d = 0; d + 1 < ndim; ++d) {
            uint32_t mult = 1;
            for (uint32_t e = d + 1; e + 1 < ndim; ++e) { mult *= outShape[e]; }
            uint32_t id = (mult == 0) ? 0 : (rem / mult);
            rem = (mult == 0) ? rem : (rem % mult);
            idx[d] = id;
            x1Base += static_cast<int32_t>(id) * x1StrideArr[d];
            x2Base += static_cast<int32_t>(id) * x2StrideArr[d];
        }
    }

    // 推进到下一行：仅通过加减 stride 更新行首下标，避免每行做除法。
    __aicore__ inline void AdvanceRowState() {
        if (ndim < 2) { return; }
        int32_t d = static_cast<int32_t>(ndim) - 2;
        idx[d]++;
        x1Base += x1StrideArr[d];
        x2Base += x2StrideArr[d];
        while (d > 0 && idx[d] >= outShape[d]) {
            x1Base -= static_cast<int32_t>(outShape[d]) * x1StrideArr[d];
            x2Base -= static_cast<int32_t>(outShape[d]) * x2StrideArr[d];
            idx[d] = 0;
            d--;
            idx[d]++;
            x1Base += x1StrideArr[d];
            x2Base += x2StrideArr[d];
        }
    }

    // 多行连续打包判断：折叠后相邻行在 GM 上连续，或该操作数是全局标量（rowStride==0）。
    __aicore__ inline bool CanPackRows(uint32_t L, int32_t sLast, int32_t rowStride,
                                       bool &isPackedScalar) {
        isPackedScalar = false;
        if (sLast != 0) {
            return (rowStride == static_cast<int32_t>(L) * sLast);
        } else {
            if (rowStride == 0) {
                isPackedScalar = true;
                return true;
            }
            return false;
        }
    }

    __aicore__ inline void ProcessBcast() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowOff = blockIdx * perCore;
        if (rowOff >= totalRows) { return; }
        uint32_t myRows = perCore;
        if (rowOff + myRows > totalRows) { myRows = totalRows - rowOff; }

        uint32_t L = lastDimLen;
        int32_t s1last = x1StrideArr[ndim - 1];
        int32_t s2last = x2StrideArr[ndim - 1];

        bool packRows = false;
        bool x1PackedScalar = false, x2PackedScalar = false;
        int32_t rowStrideX1 = 0, rowStrideX2 = 0;
        if (ndim >= 2 && L > 0 && L <= PACK_TILE) {
            rowStrideX1 = x1StrideArr[ndim - 2];
            rowStrideX2 = x2StrideArr[ndim - 2];
            bool x1Ok = CanPackRows(L, s1last, rowStrideX1, x1PackedScalar);
            bool x2Ok = CanPackRows(L, s2last, rowStrideX2, x2PackedScalar);
            packRows = x1Ok && x2Ok;
        }

        InitRowState(rowOff);
        uint32_t r = 0;
        while (r < myRows) {
            uint32_t rowId = rowOff + r;
            uint32_t outRowBase = rowId * L;

            if (packRows) {
                uint32_t rowsPerTile = PACK_TILE / L;
                if (rowsPerTile == 0) { rowsPerTile = 1; }
                uint32_t remainRows = myRows - r;
                if (rowsPerTile > remainRows) { rowsPerTile = remainRows; }
                uint32_t realLen = rowsPerTile * L;
                uint32_t Lc = RoundUp256(realLen);

                LocalTensor<DT_X1> aLocal;
                LocalTensor<DT_X1> bLocal;
                LoadOperandPacked(qX1B, x1Gm, x1Base, s1last, realLen, Lc, aLocal, x1PackedScalar);
                LoadOperandPacked(qX2B, x2Gm, x2Base, s2last, realLen, Lc, bLocal, x2PackedScalar);

                LocalTensor<int8_t> outI8 = qOutB.AllocTensor<int8_t>();
                ComputeTile(aLocal, bLocal, outI8, Lc, x1PackedScalar, x2PackedScalar);
                qX1B.FreeTensor(aLocal);
                qX2B.FreeTensor(bLocal);
                qOutB.EnQue(outI8);

                outI8 = qOutB.DeQue<int8_t>();
                DataCopyPad(yGm[outRowBase], outI8, DataCopyExtParams{1, realLen, 0, 0, 0});
                qOutB.FreeTensor(outI8);

                // 行连续时直接推进 base，避免 rowsPerTile 次 AdvanceRowState。
                x1Base += rowsPerTile * rowStrideX1;
                x2Base += rowsPerTile * rowStrideX2;
                r += rowsPerTile;
            } else {
                for (uint32_t c = 0; c < L; c += tileSize) {
                    uint32_t chunk = tileSize;
                    if (c + chunk > L) { chunk = L - c; }
                    uint32_t Lc = RoundUp256(chunk);

                    LocalTensor<DT_X1> aLocal;
                    LocalTensor<DT_X1> bLocal;
                    LoadOperand(qX1, x1Gm, x1Base, s1last, c, chunk, Lc, aLocal);
                    LoadOperand(qX2, x2Gm, x2Base, s2last, c, chunk, Lc, bLocal);

                    bool aScalar = (s1last == 0);
                    bool bScalar = (s2last == 0);
                    LocalTensor<int8_t> outI8 = qOut.AllocTensor<int8_t>();
                    ComputeTile(aLocal, bLocal, outI8, Lc, aScalar, bScalar);
                    qX1.FreeTensor(aLocal);
                    qX2.FreeTensor(bLocal);
                    qOut.EnQue(outI8);

                    outI8 = qOut.DeQue<int8_t>();
                    DataCopyPad(yGm[outRowBase + c], outI8, DataCopyExtParams{1, chunk, 0, 0, 0});
                    qOut.FreeTensor(outI8);
                }
                if (r + 1 < myRows) { AdvanceRowState(); }
                r++;
            }
        }
    }

    // 载入一行的最后维数据：标量广播（lastStride==0）只取 1 个元素并展开；
    // 连续（lastStride!=0）直接整块搬运 chunk 个元素。
    __aicore__ inline void LoadOperand(TQue<TPosition::VECIN, 2> &q,
                                       const GlobalTensor<DT_X1> &gm,
                                       int32_t baseIdx, int32_t lastStride,
                                       uint32_t c, uint32_t chunk, uint32_t Lc,
                                       LocalTensor<DT_X1> &out) {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (lastStride == 0) {
            // 整行标量广播：只搬 1 个元素，并把标量展开成整 tile 常量
            // （int8 例外：在 ComputeTile 的 int8 分支中转 half 展开）。
            DataCopyPad(loc, gm[static_cast<uint32_t>(baseIdx)],
                        DataCopyExtParams{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
            if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                Duplicate(loc, loc.GetValue(0), Lc);
            }
        } else {
            // 连续段：从 baseIdx + c 搬 chunk 个元素
            DataCopyPad(loc, gm[static_cast<uint32_t>(baseIdx) + c],
                        DataCopyExtParams{1, chunk * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
        }
        out = loc;
    }

    // 多行连续打包时使用：isScalar=true 表示该操作数是全局标量（rowStride==0），
    // 只搬 1 个元素并展开到 Lc；否则从 baseIdx 连续搬运 realLen 个元素。
    // 注意：q 为 BCAST 专用队列 qX1B/qX2B，depth=1（与 FAST/SCALAR 的 depth=2 队列区分），
    // 故此处参数模板 depth 必须写 1，否则类型不匹配无法绑定。
    __aicore__ inline void LoadOperandPacked(TQue<TPosition::VECIN, 1> &q,
                                             const GlobalTensor<DT_X1> &gm,
                                             int32_t baseIdx, int32_t lastStride,
                                             uint32_t realLen, uint32_t Lc,
                                             LocalTensor<DT_X1> &out,
                                             bool isScalar) {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (isScalar) {
            // 全局标量（rowStride==0）：只搬 1 个元素并展开到整 tile 常量
            // （int8 例外：在 ComputeTile 的 int8 分支中转 half 展开）。
            DataCopyPad(loc, gm[static_cast<uint32_t>(baseIdx)],
                        DataCopyExtParams{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
            if constexpr (!std::is_same_v<DT_X1, int8_t>) {
                Duplicate(loc, loc.GetValue(0), Lc);
            }
        } else {
            DataCopyPad(loc, gm[static_cast<uint32_t>(baseIdx)],
                        DataCopyExtParams{1, realLen * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
        }
        out = loc;
    }

    // 队列 / 缓冲 / Pipe
    TPipe pipe;
    TQue<TPosition::VECIN, 2> qX1;
    TQue<TPosition::VECIN, 2> qX2;
    TQue<TPosition::VECOUT, 2> qOut;
    // BCAST 多行打包专用队列（上限 PACK_TILE=8192），与 FAST/SCALAR 的 4096 队列隔离。
    // 注意：这里使用 depth=1，因为 BCAST 打包路径每个 tile 内顺序完成 Alloc/Load/Compute/Free，
    // 双缓冲槽并未被利用；depth=1 可显著降低 UB 占用，使 float32/int32 也保持在安全范围内。
    TQue<TPosition::VECIN, 1> qX1B;
    TQue<TPosition::VECIN, 1> qX2B;
    TQue<TPosition::VECOUT, 1> qOutB;
    TBuf<TPosition::VECCALC> bufMask;
    TBuf<TPosition::VECCALC> bufMinI32;
    TBuf<TPosition::VECCALC> bufOnesHalf;
    TBuf<TPosition::VECCALC> bufZerosHalf;
    TBuf<TPosition::VECCALC> bufOutHalf;
    TBuf<TPosition::VECCALC> bufAHalf;
    TBuf<TPosition::VECCALC> bufBHalf;

    GlobalTensor<DT_X1> x1Gm;
    GlobalTensor<DT_X1> x2Gm;
    GlobalTensor<int8_t> yGm;

    // Tiling 数据
    uint32_t mode;
    uint32_t totalElements;
    uint32_t tileSize;
    uint32_t perCore;
    uint32_t ndim;
    uint32_t lastDimLen;
    uint32_t totalRows;
    uint32_t scalarInput;
    uint32_t outShape[LE_MAX_DIM];
    int32_t  x1StrideArr[LE_MAX_DIM];
    int32_t  x2StrideArr[LE_MAX_DIM];

    // BCAST 行状态
    int32_t  x1Base;
    int32_t  x2Base;
    uint32_t idx[LE_MAX_DIM];
};

template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, tiling_data);
    op.Process();
}
