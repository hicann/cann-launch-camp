import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HOST = ROOT / "op_host" / "fast_gelu.cpp"
KERNEL = ROOT / "op_kernel" / "fast_gelu.cpp"
TILING = ROOT / "op_kernel" / "fast_gelu_tiling.h"
KERNEL_KEY = ROOT / "op_kernel" / "tiling_key_fast_gelu.h"


def read(path):
    return path.read_text(encoding="utf-8")


class FastGeluStaticTests(unittest.TestCase):
    def test_host_infers_output_shape_and_dtype_from_input(self):
        host = read(HOST)
        self.assertIn("GetInputShape", host)
        self.assertIn("GetOutputShape", host)
        self.assertIn("*y_shape = *x_shape", host)
        self.assertIn('Input("x")', host)
        self.assertIn('Output("y")', host)
        self.assertGreaterEqual(host.count("ge::DT_FLOAT16"), 2)
        self.assertGreaterEqual(host.count("ge::DT_FLOAT"), 2)

    def test_tiling_keeps_total_length_for_dynamic_kernel(self):
        tiling = read(TILING)
        self.assertIn("uint32_t totalLength", tiling)
        self.assertIn("uint32_t smallCoreDataNum", tiling)
        self.assertIn("uint32_t bigCoreDataNum", tiling)
        self.assertIn("uint32_t tileDataNum", tiling)
        self.assertIn("uint32_t tailBlockNum", tiling)
        self.assertIn("FastGeluTilingData", tiling)

    def test_host_uses_compute_cost_segmented_core_policy(self):
        host = read(HOST)
        self.assertIn("VECTOR_CORE_CAP = 48", host)
        self.assertIn("GetTargetCoreNum", host)
        self.assertIn("SMALL_SHAPE_BASE_THRESHOLD = CORE_SPLIT_ELEM_NUM * 24U", host)
        self.assertIn("GetSmallShapeThreshold", host)
        self.assertIn(
            "dtype == ge::DT_FLOAT16 ? SMALL_SHAPE_BASE_THRESHOLD * 2U : SMALL_SHAPE_BASE_THRESHOLD",
            host,
        )
        self.assertIn("length_x <= small_shape_threshold", host)
        self.assertIn("ASCENDC_TPL_SEL_PARAM(context, DT_X, IS_SMALL_SHAPE)", host)
        self.assertIn("length_x <= CORE_SPLIT_ELEM_NUM * 3U", host)
        self.assertIn("length_x <= tile_elem_num * 4U", host)
        self.assertIn("length_x <= tile_elem_num * 8U", host)
        self.assertIn("length_x <= tile_elem_num * 16U", host)
        self.assertIn("length_x <= tile_elem_num * 48U", host)
        self.assertIn("block_dim = GetTargetCoreNum(length_x, dtype_x)", host)
        self.assertIn("tile_elem_num = total_block_num * BLOCK_SIZE / type_length", host)
        self.assertIn("total_block_num > 0 && IS_SMALL_SHAPE == 0U", host)

    def test_tiling_key_has_small_shape_specialization(self):
        key = read(KERNEL_KEY)
        self.assertIn("ASCENDC_TPL_BOOL_DECL(IS_SMALL_SHAPE, 0, 1)", key)
        self.assertIn("ASCENDC_TPL_BOOL_SEL(IS_SMALL_SHAPE, 0, 1)", key)

    def test_kernel_has_small_direct_tbuf_path(self):
        kernel = read(KERNEL)
        self.assertIn("class KernelFastGeluSmall", kernel)
        self.assertIn("TBuf<AscendC::TPosition::VECCALC> xBuf_", kernel)
        self.assertIn("TBuf<AscendC::TPosition::VECCALC> yBuf_", kernel)
        self.assertIn("template <typename DT_X, int IS_SMALL_SHAPE>", kernel)
        self.assertIn("if (IS_SMALL_SHAPE == 1)", kernel)
        self.assertIn("PipeBarrier<PIPE_ALL>()", kernel)

    def test_kernel_splits_work_by_core_and_handles_empty_cores(self):
        kernel = read(KERNEL)
        self.assertIn("GetBlockIdx()", kernel)
        self.assertIn("coreDataNum_", kernel)
        self.assertIn("realCoreDataNum_", kernel)
        self.assertIn("tileNum_", kernel)
        self.assertIn("== 0", kernel)
        self.assertIn("return;", kernel)

    def test_kernel_uses_tail_safe_padded_copies(self):
        kernel = read(KERNEL)
        self.assertIn("DataCopyPad", kernel)
        self.assertIn("DataCopyExtParams", kernel)
        self.assertIn("DataCopyPadExtParams", kernel)

    def test_kernel_implements_fast_gelu_vector_formula(self):
        kernel = read(KERNEL)
        for token in [
            "Muls(",
            "Sigmoid(",
            "Mul(",
            "1.702",
        ]:
            self.assertIn(token, kernel)

    def test_kernel_uses_balanced_queues_and_local_buffers(self):
        kernel = read(KERNEL)
        for token in [
            "TPipe",
            "TQue",
            "AllocTensor",
            "EnQue",
            "DeQue",
            "FreeTensor",
            "TBuf",
        ]:
            self.assertIn(token, kernel)


if __name__ == "__main__":
    unittest.main()
