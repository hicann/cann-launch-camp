// Kernel侧核函数实现
#include "kernel_operator.h"


#include "gelu_tiling.h"

#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;
using namespace AscendC;

template <typename DT_INPUT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum)  {
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
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
        inGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)input_x + globalBufferIndex, this->coreDataNum);
        outGm.SetGlobalBuffer((__gm__ DT_INPUT_X*)output + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileDataNum * sizeof(DT_INPUT_X));    
        if constexpr (std::is_same_v<DT_INPUT_X, float>) {
            pipe.InitBuffer(tBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(calcBuf, this->tileDataNum * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> inLocal = inQueue.AllocTensor<DT_INPUT_X>();
        AscendC::DataCopy(inLocal, inGm[progress * this->tileDataNum], this->processDataNum);
        inQueue.EnQue(inLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DT_INPUT_X> outLocal = outQueue.AllocTensor<DT_INPUT_X>();//y
        LocalTensor<DT_INPUT_X> inLocal = inQueue.DeQue<DT_INPUT_X>();//x

        uint32_t n = this->processDataNum;
        if constexpr (std::is_same_v<DT_INPUT_X, half>) {
            const DT_INPUT_X NEG_SQRT_EIGHT_OVER_PI = -1.595769121 * 0.044715;
            const DT_INPUT_X TANH_APPROX_FACTOR = 1 / 0.044715;
            // Calculate x^2
            Mul(outLocal, inLocal, inLocal, this->processDataNum);
            // Calculate x^2 + 22.363860002236
            Adds(outLocal, outLocal, TANH_APPROX_FACTOR, this->processDataNum);
            // Calculate (x^2 + 22.363860002236)* x
            Mul(outLocal, outLocal, inLocal, this->processDataNum);
            // Calculate (x^2 + 22.363860002236)* x - 0.0713548162726
            Muls(outLocal, outLocal, NEG_SQRT_EIGHT_OVER_PI, this->processDataNum);
            Exp(outLocal,outLocal,this->processDataNum);
            Adds(outLocal, outLocal, (DT_INPUT_X)1, this->processDataNum);        
            Div(outLocal,inLocal,outLocal,this->processDataNum);
            inQueue.FreeTensor(inLocal);
        } 
        else {
            
            LocalTensor<DT_INPUT_X> t = tBuf.Get<DT_INPUT_X>();
            LocalTensor<DT_INPUT_X> calc = calcBuf.Get<DT_INPUT_X>();
            // 预算常量
            const DT_INPUT_X P_TIMES_INV_SQRT2 = static_cast<DT_INPUT_X>(0.47047 * 0.707106781f);
            const DT_INPUT_X NEG_A1 = static_cast<DT_INPUT_X>(-0.3480242);
            const DT_INPUT_X NEG_A2 = static_cast<DT_INPUT_X>(0.0958798);
            const DT_INPUT_X NEG_A3 = static_cast<DT_INPUT_X>(-0.7478556);

            AscendC::Abs(outLocal, inLocal, n);                        // outLocal = |x|  
    
            // t = 1/(1 + p·z) ---
            AscendC::Muls(t, outLocal, P_TIMES_INV_SQRT2, n);         // t = p·|x|/√2 = p·z  ★ 常量融合
            AscendC::Adds(t, t, static_cast<DT_INPUT_X>(1.0f), n);    // t = 1 + p·z
            Duplicate(calc ,static_cast<DT_INPUT_X>(1) ,this->processDataNum);
            Div(t, calc, t, this->processDataNum);                              // t = 1/(1+p·z)  
    
            AscendC::Muls(calc, t, NEG_A3, n);                         // -a3·t
            AscendC::Adds(calc, calc, NEG_A2, n);                      // -a3·t + (-a2)
            AscendC::Mul(calc, calc, t, n);                             // (-a3·t - a2)·t
            AscendC::Adds(calc, calc, NEG_A1, n);                      // ...+ (-a1)
            AscendC::Mul(calc, calc, t, n);                             // calc = -poly(t)   吸收了 Muls(-1)
    
            // exp(-z²) = exp(-|x|²/2) ---
            AscendC::Mul(t, outLocal, outLocal, n);                    // t = |x|²
            AscendC::Muls(t, t, static_cast<DT_INPUT_X>(-0.5f), n);   // t = -|x|²/2 = -z²   -0.5 替代 Muls(-1)   
            AscendC::Exp(t, t, n);                                      // t = exp(-z²)
    
            // erf(z) = 1 - poly·exp(-z²) ---
            AscendC::Mul(calc, calc, t, n);                             // -poly·exp(-z²)
            AscendC::Adds(calc, calc, static_cast<DT_INPUT_X>(1.0f), n);// erf(z) ✓
    
            // GELU(x) = 0.5·(x + |x|·erf(z))
            AscendC::Mul(calc, calc, outLocal, n);                     // |x|·erf(z)
            AscendC::Add(calc, inLocal, calc, n);                      // x + |x|·erf(z)
            AscendC::Muls(outLocal, calc, static_cast<DT_INPUT_X>(0.5f), n); // GELU(x) ✓
            inQueue.FreeTensor(inLocal);
        }
        outQueue.EnQue<DT_INPUT_X>(outLocal);
        
    }


    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DT_INPUT_X> outLocal = outQueue.DeQue<DT_INPUT_X>();  
        AscendC::DataCopy(outGm[progress * this->tileDataNum], outLocal, this->processDataNum);
        outQueue.FreeTensor(outLocal);
    }


private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tBuf;
    AscendC::GlobalTensor<DT_INPUT_X> inGm;
    AscendC::GlobalTensor<DT_INPUT_X> outGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

template <typename DT_INPUT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);
    
    op.Process();

}

