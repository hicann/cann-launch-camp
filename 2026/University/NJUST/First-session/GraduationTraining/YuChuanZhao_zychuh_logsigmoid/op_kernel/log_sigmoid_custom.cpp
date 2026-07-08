#include "kernel_operator.h"
#include "log_sigmoid_custom_tiling.h"
extern "C" __global__ __aicore__ void log_sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSigmoidCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 请完成Kernel侧代码实现
    using namespace AscendC;
    const uint32_t tile = 1024;
    uint32_t n = tilingData.size;
    uint32_t blk = GetBlockNum();
    uint32_t id = GetBlockIdx();
    uint32_t step = (n + blk - 1) / blk;
    uint32_t off = id * step;
    uint32_t len = off < n ? ((off + step <= n) ? step : n - off) : 0;
    if (TILING_KEY_IS(1)) {
        GlobalTensor<float> xGm;
        GlobalTensor<float> yGm;
        xGm.SetGlobalBuffer((__gm__ float*)x + off, len);
        yGm.SetGlobalBuffer((__gm__ float*)y + off, len);
        TPipe pipe;
        TQue<TPosition::VECIN, 1> qx;
        TQue<TPosition::VECOUT, 1> qy;
        TBuf<TPosition::VECCALC> buf;
        pipe.InitBuffer(qx, 1, tile * sizeof(float));
        pipe.InitBuffer(qy, 1, tile * sizeof(float));
        pipe.InitBuffer(buf, tile * sizeof(float));
        for (uint32_t p = 0; p < len; p += tile) {
            uint32_t c = len - p > tile ? tile : len - p;
            DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            LocalTensor<float> a = qx.AllocTensor<float>();
            DataCopyPad(a, xGm[p], cp, pad);
            qx.EnQue(a);
            LocalTensor<float> b = qx.DeQue<float>();
            LocalTensor<float> d = qy.AllocTensor<float>();
            LocalTensor<float> t = buf.Get<float>();
            Mins(d, b, 0.0f, c);
            Abs(t, b, c);
            Muls(t, t, -1.0f, c);
            Exp(t, t, c);
            Adds(t, t, 1.0f, c);
            Ln(t, t, c);
            Sub(d, d, t, c);
            qy.EnQue(d);
            qx.FreeTensor(b);
            LocalTensor<float> e = qy.DeQue<float>();
            DataCopyPad(yGm[p], e, cp);
            qy.FreeTensor(e);
        }
    } else if (TILING_KEY_IS(2)) {
        GlobalTensor<half> xGm;
        GlobalTensor<half> yGm;
        xGm.SetGlobalBuffer((__gm__ half*)x + off, len);
        yGm.SetGlobalBuffer((__gm__ half*)y + off, len);
        TPipe pipe;
        TQue<TPosition::VECIN, 1> qx;
        TQue<TPosition::VECOUT, 1> qy;
        TBuf<TPosition::VECCALC> buf1;
        TBuf<TPosition::VECCALC> buf2;
        pipe.InitBuffer(qx, 1, tile * sizeof(half));
        pipe.InitBuffer(qy, 1, tile * sizeof(half));
        pipe.InitBuffer(buf1, tile * sizeof(float));
        pipe.InitBuffer(buf2, tile * sizeof(float));
        for (uint32_t p = 0; p < len; p += tile) {
            uint32_t c = len - p > tile ? tile : len - p;
            DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> pad{false, 0, 0, 0};
            LocalTensor<half> a = qx.AllocTensor<half>();
            DataCopyPad(a, xGm[p], cp, pad);
            qx.EnQue(a);
            LocalTensor<half> b = qx.DeQue<half>();
            LocalTensor<half> d = qy.AllocTensor<half>();
            LocalTensor<float> t = buf1.Get<float>();
            LocalTensor<float> m = buf2.Get<float>();
            Cast(t, b, RoundMode::CAST_NONE, c);
            Mins(m, t, 0.0f, c);
            Abs(t, t, c);
            Muls(t, t, -1.0f, c);
            Exp(t, t, c);
            Adds(t, t, 1.0f, c);
            Ln(t, t, c);
            Sub(t, m, t, c);
            Cast(d, t, RoundMode::CAST_RINT, c);
            qy.EnQue(d);
            qx.FreeTensor(b);
            LocalTensor<half> e = qy.DeQue<half>();
            DataCopyPad(yGm[p], e, cp);
            qy.FreeTensor(e);
        }
    } else if (TILING_KEY_IS(3)) {
        GlobalTensor<bfloat16_t> xGm;
        GlobalTensor<bfloat16_t> yGm;
        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x + off, len);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y + off, len);
        TPipe pipe;
        TQue<TPosition::VECIN, 1> qx;
        TQue<TPosition::VECOUT, 1> qy;
        TBuf<TPosition::VECCALC> buf1;
        TBuf<TPosition::VECCALC> buf2;
        pipe.InitBuffer(qx, 1, tile * sizeof(bfloat16_t));
        pipe.InitBuffer(qy, 1, tile * sizeof(bfloat16_t));
        pipe.InitBuffer(buf1, tile * sizeof(float));
        pipe.InitBuffer(buf2, tile * sizeof(float));
        for (uint32_t p = 0; p < len; p += tile) {
            uint32_t c = len - p > tile ? tile : len - p;
            DataCopyExtParams cp{1, static_cast<uint32_t>(c * sizeof(bfloat16_t)), 0, 0, 0};
            DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
            LocalTensor<bfloat16_t> a = qx.AllocTensor<bfloat16_t>();
            DataCopyPad(a, xGm[p], cp, pad);
            qx.EnQue(a);
            LocalTensor<bfloat16_t> b = qx.DeQue<bfloat16_t>();
            LocalTensor<bfloat16_t> d = qy.AllocTensor<bfloat16_t>();
            LocalTensor<float> t = buf1.Get<float>();
            LocalTensor<float> m = buf2.Get<float>();
            Cast(t, b, RoundMode::CAST_NONE, c);
            Mins(m, t, 0.0f, c);
            Abs(t, t, c);
            Muls(t, t, -1.0f, c);
            Exp(t, t, c);
            Adds(t, t, 1.0f, c);
            Ln(t, t, c);
            Sub(t, m, t, c);
            Cast(d, t, RoundMode::CAST_RINT, c);
            qy.EnQue(d);
            qx.FreeTensor(b);
            LocalTensor<bfloat16_t> e = qy.DeQue<bfloat16_t>();
            DataCopyPad(yGm[p], e, cp);
            qy.FreeTensor(e);
        }
    }
}
