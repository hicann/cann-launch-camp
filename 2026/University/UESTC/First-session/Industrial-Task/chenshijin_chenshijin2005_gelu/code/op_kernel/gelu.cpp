// Kernel侧核函数实现
#include "kernel_operator.h"


#include "gelu_tiling.h"

#include "tiling_key_gelu.h"
constexpr int32_t BUFFER_NUM = 2;
template <class T_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t smallCoreDataNum,
        uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
        uint32_t finalSmallTileNum, uint32_t tileDataNum, 
        uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
        uint32_t tailBlockNum) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * blockIdx;
        this->tileDataNum = tileDataNum;
        if (blockIdx < tailBlockNum) { 
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ T_INPUT_X*)x + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ T_INPUT_X*)z + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(T_INPUT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T_INPUT_X));

            // half 需要额外的 float32 临时 buffer
        
            pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));
            //pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));


    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        uint32_t Dprogress=0;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            Dprogress=i * this->tileDataNum;
            CopyIn(Dprogress);
            Compute(Dprogress);
            CopyOut(Dprogress);
        }
    }
    
        __aicore__ inline void Process2() {
        int32_t loopCount = this->tileNum;
        uint32_t Dprogress=0;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            Dprogress=i * this->tileDataNum;
            CopyIn(Dprogress);
            Compute2(Dprogress);
            CopyOut(Dprogress);
        }

    }
private:
__aicore__ inline void CopyIn(int32_t progress)
{
  AscendC::LocalTensor<T_INPUT_X> xLocal = inQueueX.AllocTensor<T_INPUT_X>();
  AscendC::DataCopy(xLocal, xGm[progress], this->processDataNum);
  inQueueX.EnQue(xLocal);
}

__aicore__ inline void Compute(int32_t progress)
{
  AscendC::LocalTensor<T_INPUT_X> xLocal = inQueueX.DeQue<T_INPUT_X>();
  AscendC::LocalTensor<T_INPUT_X> zLocal = outQueueZ.AllocTensor<T_INPUT_X>();
  
           AscendC::LocalTensor<float> tmp = tmpBuf0.Get<float>();
 	         AscendC::Muls(zLocal,xLocal,static_cast<T_INPUT_X>(0.7071067811865476),this->processDataNum);
 	         AscendC::Erf(tmp,zLocal,this->processDataNum);
 	         AscendC::Adds(tmp,tmp,static_cast<T_INPUT_X>(1.0),this->processDataNum);
 	         AscendC::Muls(tmp,tmp,static_cast<T_INPUT_X>(0.5),this->processDataNum);
 	         AscendC::Mul(zLocal,xLocal,tmp,this->processDataNum);
  

  //AscendC::Gelu(zLocal, xLocal, this->processDataNum);
  outQueueZ.EnQue<T_INPUT_X>(zLocal);
  inQueueX.FreeTensor(xLocal);
}

__aicore__ inline void Compute2(int32_t progress)
{
  AscendC::LocalTensor<T_INPUT_X> xLocal = inQueueX.DeQue<T_INPUT_X>();
  AscendC::LocalTensor<T_INPUT_X> zLocal = outQueueZ.AllocTensor<T_INPUT_X>();


        //AscendC::LocalTensor<float> xFloat = tmpBuf0.Get<float>();
        //AscendC::LocalTensor<float> zFloat = tmpBuf1.Get<float>();
        //AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        //AscendC::Gelu(zFloat, xFloat, this->processDataNum);
        //AscendC::Cast(zLocal, zFloat, AscendC::RoundMode::CAST_NONE, this->processDataNum);

  //AscendC::Gelu<T_INPUT_X,false,true>(zLocal, xLocal, this->processDataNum);
AscendC::Gelu(zLocal, xLocal, this->processDataNum);
  outQueueZ.EnQue<T_INPUT_X>(zLocal);
  inQueueX.FreeTensor(xLocal);
}

__aicore__ inline void CopyOut(int32_t progress)
{
  AscendC::LocalTensor<T_INPUT_X> zLocal = outQueueZ.DeQue<T_INPUT_X>();  
  AscendC::DataCopy(zGm[progress], zLocal, this->processDataNum);
  outQueueZ.FreeTensor(zLocal);
}


private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;

    AscendC::GlobalTensor<T_INPUT_X> xGm;
    AscendC::GlobalTensor<T_INPUT_X> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DTYPE_INPUT_X> op;
    op.Init(x, z, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);
    if constexpr (std::is_same_v<DTYPE_INPUT_X, float>) {
        op.Process();
    } else {
        op.Process2();
    }
    //op.Process();
}

