/**
 * ============================================================================
 * @file less_equal.cpp
 * @brief LessEqual算子的AscendC内核实现
 *
 * 支持：half / float / int32_t / int8_t + NumPy广播（x1和x2均可广播）
 * ============================================================================
 */

#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template <class DT_X1>
class KernelLessEqual {
 public:
  __aicore__ inline KernelLessEqual() {}

  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                              uint32_t blockLength, uint32_t tileNum,
                              uint32_t tileLength, uint32_t lasttileLength,
                              uint32_t x1Length, uint32_t x2Length,
                              uint32_t totalLength) {
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

    this->blockLength = blockLength;
    this->tileNum =
        tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
    this->tileLength = tileLength / BUFFER_NUM;
    this->lasttileLength = lasttileLength;
    this->x1Length = x1Length;
    this->x2Length = x2Length;
    this->totalLength = totalLength;
    this->x1Cast = (x1Length != 0);
    this->x2Cast = (x2Length != 0);

    // x1：广播时从起始位置读取；非广播时按Block偏移
    if (this->x1Cast) {
      x1Gm.SetGlobalBuffer((__gm__ DT_X1*)x1, this->x1Length);
    } else {
      x1Gm.SetGlobalBuffer((__gm__ DT_X1*)x1 + this->blockLength * GetBlockIdx(),
                          this->blockLength);
    }
    // x2：同理
    if (this->x2Cast) {
      x2Gm.SetGlobalBuffer((__gm__ DT_X1*)x2, this->x2Length);
    } else {
      x2Gm.SetGlobalBuffer((__gm__ DT_X1*)x2 + this->blockLength * GetBlockIdx(),
                          this->blockLength);
    }
    yGm.SetGlobalBuffer(
        (__gm__ uint8_t *)y + this->blockLength * GetBlockIdx(),
        this->blockLength);

    this->val = (DT_X1)0;

    pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 2 * sizeof(DT_X1));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->tileLength * sizeof(uint8_t));

    Init_temp_pip(val);
  }

  /* ================ 临时缓冲区初始化（原始逻辑） ================ */
  __aicore__ inline void Init_temp_pip(half flag) {
    pipe.InitBuffer(tempBuf, this->tileLength * sizeof(half));
    pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(float));
    pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(float));
  }
  __aicore__ inline void Init_temp_pip(float flag) {
    pipe.InitBuffer(tempBuf, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(tempBuf3, this->tileLength * sizeof(half));
  }
  __aicore__ inline void Init_temp_pip(int32_t flag) {
    pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(float));
    pipe.InitBuffer(tempBuf, this->tileLength * sizeof(uint8_t));
    pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(half));
  }
  __aicore__ inline void Init_temp_pip(int8_t flag) {
    pipe.InitBuffer(tempBuf1, this->tileLength * sizeof(half));
    pipe.InitBuffer(tempBuf2, this->tileLength * sizeof(half));
    pipe.InitBuffer(tempBuf, this->tileLength * sizeof(uint8_t));
  }

  /* ==================== 主处理流程 ==================== */
  __aicore__ inline void Process() {
    if (this->blockLength == 0) return;
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      // 尾部边界检查：跳过超出totalLength的progress
      uint32_t outStart = ComputeOutStart(i);
      uint32_t globalStart = this->blockLength * GetBlockIdx() + outStart;
      if (globalStart >= this->totalLength) break;
      CopyIn(i);
      Compute(i, val);
      CopyOut(i);
    }
  }

 private:
  /* ===== 模运算读取广播数据 ===== */
  __aicore__ inline void CopyInModular(LocalTensor<DT_X1> dst,
                                       GlobalTensor<DT_X1> gm,
                                       uint32_t dataLen,
                                       uint32_t globalOutStart) {
    uint32_t off = globalOutStart % dataLen;
    uint32_t remaining = dataLen - off;
    if (this->tileLength <= remaining) {
      DataCopy(dst, gm[off], this->tileLength);
    } else {
      DataCopy(dst, gm[off], remaining);
      uint32_t left = this->tileLength - remaining;
      uint32_t dstOff = remaining;
      while (left > 0) {
        uint32_t copyLen = (left > dataLen) ? dataLen : left;
        DataCopy(dst[dstOff], gm[0], copyLen);
        left -= copyLen;
        dstOff += copyLen;
      }
    }
  }

  /* ===== 计算Tile在输出中的起始位置 ===== */
  __aicore__ inline uint32_t ComputeOutStart(int32_t progress) {
    if (BUFFER_NUM == 2) {
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {
        // 修正：当lasttileLength != tileLength*2时，
        // 最后一tile的起始 = (tileNum-1)*tileLength*BUFFER_NUM
        uint32_t lastTileBase =
            (this->tileNum - 1) * this->tileLength * BUFFER_NUM;
        uint32_t chunkInLastTile =
            progress - (this->tileNum * BUFFER_NUM - BUFFER_NUM);
        return lastTileBase + chunkInLastTile * this->tileLength;
      }
      return progress * this->tileLength;
    } else {
      if (progress == this->tileNum - 1) {
        return (progress == 0) ? 0
                               : (progress - 1) * this->tileLength +
                                     this->lasttileLength;
      }
      return progress * this->tileLength;
    }
  }

  /* ==================== CopyIn ==================== */
  __aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<DT_X1> inLocal = inQueueIN.AllocTensor<DT_X1>();
    uint32_t outStart = ComputeOutStart(progress);
    uint32_t globalOutStart = this->blockLength * GetBlockIdx() + outStart;

    // ---- 读取x1 ----
    if (this->x1Cast) {
      CopyInModular(inLocal, x1Gm, this->x1Length, globalOutStart);
    } else {
      if (BUFFER_NUM == 2) {
        if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
            (progress == (this->tileNum * BUFFER_NUM - 1))) {
          uint32_t lastTileBase =
              (this->tileNum - 1) * this->tileLength * BUFFER_NUM;
          uint32_t chunkInLastTile =
              progress - (this->tileNum * BUFFER_NUM - BUFFER_NUM);
          DataCopy(inLocal[0],
                   x1Gm[lastTileBase + chunkInLastTile * this->tileLength],
                   (this->tileLength));
        } else {
          DataCopy(inLocal[0], x1Gm[progress * (this->tileLength)],
                   (this->tileLength));
        }
      } else {
        if (progress == this->tileNum - 1) {
          if (progress == 0) {
            DataCopy(inLocal[0], x1Gm[0], this->tileLength);
          } else {
            DataCopy(inLocal[0],
                     x1Gm[(progress - 1) * this->tileLength + this->lasttileLength],
                     this->tileLength);
          }
        } else {
          DataCopy(inLocal[0], x1Gm[progress * this->tileLength], this->tileLength);
        }
      }
    }

    // ---- 读取x2 ----
    if (this->x2Cast) {
      CopyInModular(inLocal[this->tileLength], x2Gm, this->x2Length, globalOutStart);
    } else {
      if (BUFFER_NUM == 2) {
        if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
            (progress == (this->tileNum * BUFFER_NUM - 1))) {
          uint32_t lastTileBase =
              (this->tileNum - 1) * this->tileLength * BUFFER_NUM;
          uint32_t chunkInLastTile =
              progress - (this->tileNum * BUFFER_NUM - BUFFER_NUM);
          DataCopy(inLocal[this->tileLength],
                   x2Gm[lastTileBase + chunkInLastTile * this->tileLength],
                   (this->tileLength));
        } else {
          DataCopy(inLocal[this->tileLength], x2Gm[progress * (this->tileLength)],
                   (this->tileLength));
        }
      } else {
        if (progress == this->tileNum - 1) {
          if (progress == 0) {
            DataCopy(inLocal[this->tileLength], x2Gm[0], this->tileLength);
          } else {
            DataCopy(inLocal[this->tileLength],
                     x2Gm[(progress - 1) * this->tileLength + this->lasttileLength],
                     this->tileLength);
          }
        } else {
          DataCopy(inLocal[this->tileLength], x2Gm[progress * this->tileLength],
                   this->tileLength);
        }
      }
    }

    inQueueIN.EnQue(inLocal);
  }

  /* ==================== Compute（原始逻辑，已验证正确） ==================== */

  // ── half：half→float→比较→Select→Cast（原始逻辑） ──
  __aicore__ inline void Compute(int32_t progress, half flag) {
    LocalTensor<half> inLocal = inQueueIN.DeQue<half>();
    LocalTensor<half> x1Local = inLocal;
    LocalTensor<half> x2Local = inLocal[this->tileLength];
    LocalTensor<float> temp1 = tempBuf1.Get<float>();
    LocalTensor<float> temp2 = tempBuf2.Get<float>();
    LocalTensor<uint8_t> outLocal = outQueueOUT.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> temp = tempBuf.Get<uint8_t>();

    Cast(temp1, x1Local, RoundMode::CAST_NONE, this->tileLength);
    Cast(temp2, x2Local, RoundMode::CAST_NONE, this->tileLength);
    Compare(temp, temp1, temp2, CMPMODE::LE, this->tileLength);
    Sub(temp1, temp1, temp1, this->tileLength);
    Adds(temp1, temp1, (float)1, this->tileLength);
    Select(temp2, temp, temp1, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
    Cast(x1Local, temp2, RoundMode::CAST_NONE, this->tileLength);
    Cast(outLocal, x1Local, RoundMode::CAST_NONE, this->tileLength);

    outQueueOUT.EnQue<uint8_t>(outLocal);
    inQueueIN.FreeTensor(inLocal);
  }

  // ── float：直接float比较（原始逻辑） ──
  __aicore__ inline void Compute(int32_t progress, float flag) {
    LocalTensor<float> inLocal = inQueueIN.DeQue<float>();
    LocalTensor<float> x1Local = inLocal;
    LocalTensor<float> x2Local = inLocal[this->tileLength];
    LocalTensor<uint8_t> outLocal = outQueueOUT.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> temp = tempBuf.Get<uint8_t>();
    LocalTensor<half> temp1 = tempBuf3.Get<half>();

    Compare(temp, x1Local, x2Local, CMPMODE::LE, this->tileLength);
    Sub(x1Local, x1Local, x1Local, this->tileLength);
    Adds(x1Local, x1Local, (float)1, this->tileLength);
    Select(x1Local, temp, x1Local, static_cast<float>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
    Cast(temp1, x1Local, RoundMode::CAST_NONE, this->tileLength);
    Cast(outLocal, temp1, RoundMode::CAST_NONE, this->tileLength);

    outQueueOUT.EnQue<uint8_t>(outLocal);
    inQueueIN.FreeTensor(inLocal);
  }

  // ── int32_t：直接int32比较，避免float32精度丢失 ──
  __aicore__ inline void Compute(int32_t progress, int32_t flag) {
    LocalTensor<int32_t> inLocal = inQueueIN.DeQue<int32_t>();
    LocalTensor<int32_t> x1Local = inLocal;
    LocalTensor<int32_t> x2Local = inLocal[this->tileLength];
    LocalTensor<float> tempOneLocal = tempBuf1.Get<float>();
    LocalTensor<uint8_t> temp = tempBuf.Get<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQueueOUT.AllocTensor<uint8_t>();
    LocalTensor<half> tempHalf = tempBuf2.Get<half>();

    // 直接int32比较，无精度损失 — Compare输出uint8_t mask (0xFF/0x00)
    Compare(temp, x1Local, x2Local, CMPMODE::LE, this->tileLength);
    // 通过Select将mask转换为float 1.0f/0.0f
    Sub(tempOneLocal, tempOneLocal, tempOneLocal, this->tileLength);
    Adds(tempOneLocal, tempOneLocal, (float)1, this->tileLength);
    Select(tempOneLocal, temp, tempOneLocal, static_cast<float>(0),
           SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
    Cast(tempHalf, tempOneLocal, RoundMode::CAST_NONE, this->tileLength);
    Cast(outLocal, tempHalf, RoundMode::CAST_NONE, this->tileLength);

    outQueueOUT.EnQue<uint8_t>(outLocal);
    inQueueIN.FreeTensor(inLocal);
  }

  // ── int8_t：int8→half→比较（原始逻辑） ──
  __aicore__ inline void Compute(int32_t progress, int8_t flag) {
    LocalTensor<int8_t> inLocal = inQueueIN.DeQue<int8_t>();
    LocalTensor<int8_t> x1Local = inLocal;
    LocalTensor<int8_t> x2Local = inLocal[this->tileLength];
    LocalTensor<uint8_t> temp = tempBuf.Get<uint8_t>();
    LocalTensor<half> tempOneLocal = tempBuf1.Get<half>();
    LocalTensor<half> tempTwoLocal = tempBuf2.Get<half>();
    LocalTensor<uint8_t> outLocal = outQueueOUT.AllocTensor<uint8_t>();

    Cast(tempOneLocal, x1Local, RoundMode::CAST_NONE, this->tileLength);
    Cast(tempTwoLocal, x2Local, RoundMode::CAST_NONE, this->tileLength);
    Compare(temp, tempOneLocal, tempTwoLocal, CMPMODE::LE, this->tileLength);
    Sub(tempOneLocal, tempOneLocal, tempOneLocal, this->tileLength);
    Adds(tempOneLocal, tempOneLocal, (half)1, this->tileLength);
    Select(tempOneLocal, temp, tempOneLocal, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
    Cast(outLocal, tempOneLocal, RoundMode::CAST_NONE, this->tileLength);

    outQueueOUT.EnQue<uint8_t>(outLocal);
    inQueueIN.FreeTensor(inLocal);
  }

  /* ==================== CopyOut（原始逻辑） ==================== */
  __aicore__ inline void CopyOut(int32_t progress) {
    LocalTensor<uint8_t> outLocal = outQueueOUT.DeQue<uint8_t>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {
        if (progress == 0) {
          DataCopy(yGm[0], outLocal, this->tileLength);
        } else {
          DataCopy(yGm[(progress - 1) * this->tileLength + this->lasttileLength],
                   outLocal, this->tileLength);
        }
      } else {
        DataCopy(yGm[progress * this->tileLength], outLocal, this->tileLength);
      }
    }

    if (BUFFER_NUM == 2) {
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {
        uint32_t lastTileBase =
            (this->tileNum - 1) * this->tileLength * BUFFER_NUM;
        uint32_t chunkInLastTile =
            progress - (this->tileNum * BUFFER_NUM - BUFFER_NUM);
        DataCopy(yGm[lastTileBase + chunkInLastTile * this->tileLength],
                 outLocal, (this->tileLength));
      } else {
        DataCopy(yGm[progress * (this->tileLength)], outLocal,
                 (this->tileLength));
      }
    }

    outQueueOUT.FreeTensor(outLocal);
  }

 private:
  TPipe pipe;
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  GlobalTensor<DT_X1> x1Gm;
  GlobalTensor<DT_X1> x2Gm;
  GlobalTensor<uint8_t> yGm;
  TBuf<> tempBuf;
  TBuf<> tempBuf1;
  TBuf<> tempBuf2;
  TBuf<> tempBuf3;
  TBuf<> tempBuf4;

  DT_X1 val;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lasttileLength;
  uint32_t x1Length;
  uint32_t x2Length;
  uint32_t totalLength;
  bool x1Cast;
  bool x2Cast;
};

/* ======================== Kernel入口 ======================== */
/* ======================== Kernel入口 ======================== */
template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                      GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LessEqualTilingData);
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, tiling_data, tiling);

    // ---> 核心改动：在最外层拦截空闲核，强行控为 0，触发算子的安全快速返回 <---
    uint32_t current_core_start = tiling_data.blockLength * GetBlockIdx();
    uint32_t actual_block_length = tiling_data.blockLength;

    if (current_core_start >= tiling_data.totalLength) {
        // 如果当前核的起始位置已经超过了总长度，说明是完全无用的越界核
        actual_block_length = 0; 
    }

    KernelLessEqual<DT_X1> op;
    op.Init(x1, x2, y,
            actual_block_length, tiling_data.tileNum, // 传入修正后的长度
            tiling_data.tileLength, tiling_data.lasttileLength,
            tiling_data.x1Length, tiling_data.x2Length,
            tiling_data.totalLength);
    op.Process();
}