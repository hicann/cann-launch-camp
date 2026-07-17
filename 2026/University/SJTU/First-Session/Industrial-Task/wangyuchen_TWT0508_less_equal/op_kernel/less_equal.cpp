#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

// ==================== KernelLessEqual ====================
// Element-wise comparison y = (x1 <= x2) with broadcast support.
//
// Two execution paths:
//   Direct    — both inputs have the same shape, contiguous DMA works
//   Broadcast — shapes differ; host has collapsed dims so the last dim
//               is always contiguous (or scalar), kernel works row by row.
//
// Per-type comparison strategy:
//   half/float — hardware Compare with CMPMODE::LE
//   int8       — cast to half then compare (all int8 values fit in half exactly)
//   int32      — use min(a,b)==a instead of a<=b (avoids float conversion precision loss)

template <class DT_X1>
class KernelLessEqual {
    static constexpr bool kHalf  = std::is_same_v<DT_X1, half>;
    static constexpr bool kFloat = std::is_same_v<DT_X1, float>;
    static constexpr bool kInt8  = std::is_same_v<DT_X1, int8_t>;
    static constexpr bool kInt32 = std::is_same_v<DT_X1, int32_t>;

public:
    __aicore__ inline KernelLessEqual() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const LessEqualTilingData &td)
    {
        count_  = td.count;
        tile_   = td.tile;
        perCore_= td.perCore;
        mode_   = td.mode;
        ndim_   = td.ndim;
        tail_   = td.tail;
        rows_   = td.rows;

        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            shape_[i] = td.shape[i];
            s1_[i]    = td.s1[i];
            s2_[i]    = td.s2[i];
        }

        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y));

        if (count_ == 0) return;

        // input/output queues — depth 2 for double buffering
        pipe_.InitBuffer(qX1_, 2, tile_ * sizeof(DT_X1));
        pipe_.InitBuffer(qX2_, 2, tile_ * sizeof(DT_X1));
        pipe_.InitBuffer(qY_,  2, tile_ * sizeof(int8_t));

        // compute buffers
        pipe_.InitBuffer(mask_,   Up32((tile_ + 7) / 8));
        pipe_.InitBuffer(ones_,   tile_ * sizeof(half));
        pipe_.InitBuffer(zeros_,  tile_ * sizeof(half));
        pipe_.InitBuffer(outH_,   tile_ * sizeof(half));

        if constexpr (kInt8) {
            pipe_.InitBuffer(buf1_, tile_ * sizeof(half));
            pipe_.InitBuffer(buf2_, tile_ * sizeof(half));
        }
        if constexpr (kInt32) {
            pipe_.InitBuffer(bufI_, tile_ * sizeof(int32_t));
        }

        // pre-fill ones/zeros once, reused across all tiles
        LocalTensor<half> o = ones_.Get<half>();
        LocalTensor<half> z = zeros_.Get<half>();
        Duplicate(o, static_cast<half>(1.0), tile_);
        Duplicate(z, static_cast<half>(0.0), tile_);
    }

    __aicore__ inline void Process() {
        if (count_ == 0) return;
        if (mode_ == LE_MODE_DIRECT)
            ProcessDirect();
        else
            ProcessBroadcast();
    }

private:
    // ---- helpers ----
    __aicore__ inline uint32_t Up32(uint32_t v) const {
        return (v + 31) / 32 * 32;
    }

    __aicore__ inline uint32_t Up256(uint32_t v) const {
        return (v + 255) / 256 * 256;
    }

    // ---- compare + mask-to-int8 conversion ----
    __aicore__ inline void DoCompare(
        const LocalTensor<DT_X1> &a,
        const LocalTensor<DT_X1> &b,
        const LocalTensor<int8_t> &y,
        uint32_t n,
        bool aScl, bool bScl)
    {
        LocalTensor<uint8_t> mk = mask_.Get<uint8_t>();
        LocalTensor<half>    o  = ones_.Get<half>();
        LocalTensor<half>    z  = zeros_.Get<half>();
        LocalTensor<half>    oh = outH_.Get<half>();

        if constexpr (kHalf || kFloat) {
            Compare(mk, a, b, CMPMODE::LE, n);
        } else if constexpr (kInt8) {
            LocalTensor<half> ah = buf1_.Get<half>();
            LocalTensor<half> bh = buf2_.Get<half>();
            if (aScl) {
                int8_t va = a.GetValue(0);
                Duplicate(ah, static_cast<half>(static_cast<float>(static_cast<int32_t>(va))), n);
            } else {
                Cast(ah, a, RoundMode::CAST_NONE, n);
            }
            if (bScl) {
                int8_t vb = b.GetValue(0);
                Duplicate(bh, static_cast<half>(static_cast<float>(static_cast<int32_t>(vb))), n);
            } else {
                Cast(bh, b, RoundMode::CAST_NONE, n);
            }
            Compare(mk, ah, bh, CMPMODE::LE, n);
        } else {
            // int32: min(a,b) == a  is equivalent to  a <= b
            LocalTensor<int32_t> mv = bufI_.Get<int32_t>();
            Min(mv, a, b, n);
            Compare(mk, mv, a, CMPMODE::EQ, n);
        }

        // bitmask -> half (1.0 / 0.0) -> int8 (1 / 0)
        Select(oh, mk, o, z, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
        Cast(y, oh, RoundMode::CAST_RINT, n);
    }

    // ============================================================
    //  Direct mode: same shape, contiguous DMA throughout
    // ============================================================
    __aicore__ inline void ProcessDirect() {
        uint32_t bid = GetBlockIdx();
        uint32_t off = bid * perCore_;
        if (off >= count_) return;
        uint32_t len = perCore_;
        if (off + len > count_) len = count_ - off;

        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};

        for (uint32_t t = 0; t < len; t += tile_) {
            uint32_t L  = tile_;
            if (t + L > len) L = len - t;
            uint32_t Lc = Up256(L);
            uint32_t base = off + t;

            // DMA x1 (async)
            LocalTensor<DT_X1> a = qX1_.AllocTensor<DT_X1>();
            DataCopyPad(a, x1Gm_[base],
                DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            qX1_.EnQue(a);

            // DMA x2 (async, runs in parallel with x1 DMA)
            LocalTensor<DT_X1> b = qX2_.AllocTensor<DT_X1>();
            DataCopyPad(b, x2Gm_[base],
                DataCopyExtParams{1, L * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            qX2_.EnQue(b);

            // wait for DMA, compute, enqueue output
            a = qX1_.DeQue<DT_X1>();
            b = qX2_.DeQue<DT_X1>();
            LocalTensor<int8_t> y = qY_.AllocTensor<int8_t>();
            DoCompare(a, b, y, Lc, false, false);
            qX1_.FreeTensor(a);
            qX2_.FreeTensor(b);
            qY_.EnQue(y);

            // wait for output DMA, write to GM
            y = qY_.DeQue<int8_t>();
            DataCopyPad(yGm_[base], y, DataCopyExtParams{1, L, 0, 0, 0});
            qY_.FreeTensor(y);
        }
    }

    // ============================================================
    //  Broadcast mode: row-based processing
    // ============================================================

    // compute base offset in x1 and x2 for row r
    __aicore__ inline void RowStart(uint32_t r, uint32_t &b1, uint32_t &b2) {
        b1 = 0; b2 = 0;
        uint32_t rem = r;
        for (uint32_t d = 0; d + 1 < ndim_; ++d) {
            uint32_t mul = 1;
            for (uint32_t e = d + 1; e + 1 < ndim_; ++e) mul *= shape_[e];
            uint32_t id = (mul == 0) ? 0 : (rem / mul);
            rem = (mul == 0) ? rem : (rem % mul);
            idx_[d] = id;
            b1 += id * s1_[d];
            b2 += id * s2_[d];
        }
    }

    // advance to the next row (mixed-radix increment)
    __aicore__ inline void RowNext(uint32_t &b1, uint32_t &b2) {
        if (ndim_ < 2) return;
        int32_t d = static_cast<int32_t>(ndim_) - 2;
        idx_[d]++;
        b1 += s1_[d]; b2 += s2_[d];
        while (d > 0 && idx_[d] >= shape_[d]) {
            b1 -= shape_[d] * s1_[d];
            b2 -= shape_[d] * s2_[d];
            idx_[d] = 0;
            --d;
            idx_[d]++;
            b1 += s1_[d]; b2 += s2_[d];
        }
    }

    // load one operand into UB:
    //   step != 0 — contiguous, DMA chunk elements
    //   step == 0 — scalar broadcast, load 1 element then duplicate
    __aicore__ inline void LoadOperand(
        TQue<TPosition::VECIN, 2> &q,
        GlobalTensor<DT_X1> &gm,
        uint32_t base, uint32_t step,
        uint32_t col, uint32_t chunk, uint32_t Lc,
        LocalTensor<DT_X1> &out)
    {
        DataCopyPadExtParams<DT_X1> pad{false, 0, 0, static_cast<DT_X1>(0)};
        LocalTensor<DT_X1> loc = q.AllocTensor<DT_X1>();
        if (step == 0) {
            DataCopyPad(loc, gm[base],
                DataCopyExtParams{1, static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
            // Duplicate<int8_t> not supported on A2 — handled in DoCompare instead
            if constexpr (!kInt8) {
                Duplicate(loc, loc.GetValue(0), Lc);
            }
        } else {
            DataCopyPad(loc, gm[base + col],
                DataCopyExtParams{1, chunk * static_cast<uint32_t>(sizeof(DT_X1)), 0, 0, 0}, pad);
            q.EnQue(loc);
            loc = q.DeQue<DT_X1>();
        }
        out = loc;
    }

    __aicore__ inline void ProcessBroadcast() {
        uint32_t bid = GetBlockIdx();
        uint32_t r0  = bid * perCore_;
        if (r0 >= rows_) return;
        uint32_t nr = perCore_;
        if (r0 + nr > rows_) nr = rows_ - r0;

        uint32_t L     = tail_;
        uint32_t last1 = s1_[ndim_ - 1];
        uint32_t last2 = s2_[ndim_ - 1];

        uint32_t b1, b2;
        RowStart(r0, b1, b2);

        for (uint32_t ri = 0; ri < nr; ++ri) {
            uint32_t outRowOff = (r0 + ri) * L;
            for (uint32_t c = 0; c < L; c += tile_) {
                uint32_t chunk = tile_;
                if (c + chunk > L) chunk = L - c;
                uint32_t Lc = Up256(chunk);

                LocalTensor<DT_X1> a, b;
                LoadOperand(qX1_, x1Gm_, b1, last1, c, chunk, Lc, a);
                LoadOperand(qX2_, x2Gm_, b2, last2, c, chunk, Lc, b);

                LocalTensor<int8_t> y = qY_.AllocTensor<int8_t>();
                DoCompare(a, b, y, Lc, last1 == 0, last2 == 0);
                qX1_.FreeTensor(a);
                qX2_.FreeTensor(b);
                qY_.EnQue(y);

                y = qY_.DeQue<int8_t>();
                DataCopyPad(yGm_[outRowOff + c], y,
                    DataCopyExtParams{1, chunk, 0, 0, 0});
                qY_.FreeTensor(y);
            }
            if (ri + 1 < nr) RowNext(b1, b2);
        }
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN,  2> qX1_, qX2_;
    TQue<TPosition::VECOUT, 2> qY_;
    TBuf<TPosition::VECCALC> mask_, ones_, zeros_, outH_;
    TBuf<TPosition::VECCALC> buf1_, buf2_;
    TBuf<TPosition::VECCALC> bufI_;

    GlobalTensor<DT_X1> x1Gm_, x2Gm_;
    GlobalTensor<int8_t> yGm_;

    uint32_t count_  = 0;
    uint32_t tile_   = 0;
    uint32_t perCore_= 0;
    uint32_t mode_   = 0;
    uint32_t ndim_   = 0;
    uint32_t tail_   = 0;
    uint32_t rows_   = 0;
    uint32_t shape_[LESS_EQUAL_MAX_DIMS] = {};
    uint32_t s1_[LESS_EQUAL_MAX_DIMS]    = {};
    uint32_t s2_[LESS_EQUAL_MAX_DIMS]    = {};

    uint32_t idx_[LESS_EQUAL_MAX_DIMS] = {};
};

// ==================== kernel entry point ====================

template <typename DT_X1>
__global__ __aicore__ void less_equal(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, td, tiling);
    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y, td);
    op.Process();
}
