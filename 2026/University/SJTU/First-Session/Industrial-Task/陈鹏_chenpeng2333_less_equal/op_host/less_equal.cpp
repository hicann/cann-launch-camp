/**
 * ============================================================================
 * @file less_equal.cpp (Host侧)
 * @brief LessEqual算子的Host端实现
 * ============================================================================
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/less_equal_tiling.h"
#include "../op_kernel/tiling_key_less_equal.h"


namespace optiling {
const uint32_t BLOCK_SIZE = 32;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {

  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_size;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
  auto aivNum = ascendcPlatform.GetCoreNumAiv();

  const gert::Tensor *tensor_x1 = context->GetRequiredInputTensor(0);
  const gert::Tensor *tensor_x2 = context->GetRequiredInputTensor(1);
  ge::DataType dtype_x1 = tensor_x1->GetDataType();

  uint32_t DT_X1 = static_cast<uint32_t>(dtype_x1);
  ASCENDC_TPL_SEL_PARAM(context, DT_X1);

  uint32_t x1Length = tensor_x1->GetShapeSize();
  uint32_t x2Length = tensor_x2->GetShapeSize();

  // 计算广播后的真实输出长度
  auto x1_shape = context->GetInputShape(0)->GetOriginShape();
  auto x2_shape = context->GetInputShape(1)->GetOriginShape();
  size_t x1_dims = x1_shape.GetDimNum();
  size_t x2_dims = x2_shape.GetDimNum();
  size_t out_dims = (x1_dims > x2_dims) ? x1_dims : x2_dims;
  uint32_t totalLength = 1;
  for (size_t i = 0; i < out_dims; i++) {
    int64_t d1 = (i < out_dims - x1_dims) ? 1 : x1_shape.GetDim(i - (out_dims - x1_dims));
    int64_t d2 = (i < out_dims - x2_dims) ? 1 : x2_shape.GetDim(i - (out_dims - x2_dims));
    if (d1 == d2) totalLength *= (uint32_t)d1;
    else if (d1 == 1) totalLength *= (uint32_t)d2;
    else if (d2 == 1) totalLength *= (uint32_t)d1;
  }

  // 空张量快速返回
  if (totalLength == 0) {
    context->SetBlockDim(1);
    LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
    tiling->blockLength = 0;
    tiling->tileNum = 0;
    tiling->tileLength = 0;
    tiling->lasttileLength = 0;
    tiling->x1Length = 0;
    tiling->x2Length = 0;
    tiling->totalLength = 0;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
  }

  // 广播标志// 广播标志
  uint32_t broadcastX1Len = (x1Length == totalLength) ? 0 : x1Length;
  uint32_t broadcastX2Len = (x2Length == totalLength) ? 0 : x2Length;

  uint32_t ALIGN_NUM = BLOCK_SIZE;
  uint32_t ub_block_num = 256;
  if (ub_block_num % 2 != 0) ub_block_num--;

  // 1. 计算总对齐长度
  uint32_t totalLengthAligned;
  if (totalLength % ALIGN_NUM != 0) {
    totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
  } else {
    totalLengthAligned = totalLength;
  }

  // 2. 根据对齐后的基本块（blocks）数，动态决定 BlockDim（真实用核数）
  uint32_t total_blocks = totalLengthAligned / ALIGN_NUM;
  uint32_t max_blocks_per_core = ub_block_num; 
  uint32_t needed_cores = (total_blocks + max_blocks_per_core - 1) / max_blocks_per_core;
  
  if (needed_cores < aivNum && needed_cores > 0) {
      context->SetBlockDim(needed_cores);
  } else {
      context->SetBlockDim(aivNum);
  }
  
  // 3. 获取准确的核数并反推【天然对齐】的 blockLength
  auto block_dim = context->GetBlockDim();
  uint32_t block_blocks = (total_blocks + block_dim - 1) / block_dim;
  uint32_t blockLength = block_blocks * ALIGN_NUM;

  // 4. 进行安全的单核内 Tile 切分
  uint32_t tile_num = blockLength / ALIGN_NUM / ub_block_num;
  uint32_t tileLength = 0;
  uint32_t lasttileLength = 0;

  if ((blockLength / ALIGN_NUM) % ub_block_num == 0 || tile_num == 0) {
    if (tile_num == 0) tile_num = 1;
    if (blockLength < ub_block_num * ALIGN_NUM) {
      // 规整小数据量下的双缓冲单块长度，保证是 2 的倍数（BUFFER_NUM）且对齐
      tileLength = ((blockLength / ALIGN_NUM + 1) / 2) * 2 * ALIGN_NUM;
      if (tileLength > blockLength) tileLength = blockLength;
      lasttileLength = tileLength;
    } else {
      tileLength = ub_block_num * ALIGN_NUM;
      lasttileLength = tileLength;
    }
  } else {
    tile_num = tile_num + 1;
    tileLength = ub_block_num * ALIGN_NUM;
    lasttileLength = blockLength - (tile_num - 1) * tileLength;
  }
  
  LessEqualTilingData *tiling = context->GetTilingData<LessEqualTilingData>();
  tiling->blockLength = blockLength;
  tiling->tileNum = tile_num;
  tiling->tileLength = tileLength;
  tiling->lasttileLength = lasttileLength;
  tiling->x1Length = broadcastX1Len;
  tiling->x2Length = broadcastX2Len;
  tiling->totalLength = totalLength; 

  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {

/** @brief 形状推理 —— 支持NumPy广播语义 */
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *x1_shape = context->GetInputShape(0);
    const gert::Shape *x2_shape = context->GetInputShape(1);
    gert::Shape *y_shape = context->GetOutputShape(0);

    // 形状完全相同时直接复制
    if (x1_shape->GetDimNum() == x2_shape->GetDimNum()) {
        bool same = true;
        for (size_t i = 0; i < x1_shape->GetDimNum(); i++) {
            if (x1_shape->GetDim(i) != x2_shape->GetDim(i)) { same = false; break; }
        }
        if (same) { *y_shape = *x1_shape; return GRAPH_SUCCESS; }
    }

    // NumPy广播：右侧对齐，逐维取较大值
    size_t x1_dims = x1_shape->GetDimNum();
    size_t x2_dims = x2_shape->GetDimNum();
    size_t out_dims = (x1_dims > x2_dims) ? x1_dims : x2_dims;
    y_shape->SetDimNum(out_dims);
    for (size_t i = 0; i < out_dims; i++) {
        int64_t d1 = (i < out_dims - x1_dims) ? 1 : x1_shape->GetDim(i - (out_dims - x1_dims));
        int64_t d2 = (i < out_dims - x2_dims) ? 1 : x2_shape->GetDim(i - (out_dims - x2_dims));
        if (d1 == d2) y_shape->SetDim(i, d1);
        else if (d1 == 1) y_shape->SetDim(i, d2);
        else if (d2 == 1) y_shape->SetDim(i, d1);
        else return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

/** @brief 类型推理 —— 输出固定为BOOL */
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge


namespace ops {
class LessEqual : public OpDef {
public:
    explicit LessEqual(const char *name) : OpDef(name) {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(LessEqual);
}  // namespace ops
