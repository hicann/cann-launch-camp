#!/usr/bin/env python3
"""生成 GELU 算子的 tiling 数据二进制文件"""
import struct
import os

def gen_tiling(total_length, dtype_is_fp16):
    """根据 gelu.cpp 中的 TilingFunc 逻辑计算 tiling 数据"""
    elem_size = 2 if dtype_is_fp16 else 4
    total_bytes = total_length * elem_size

    # 常量设置（与 op_host/gelu.cpp 中的 TilingFunc 保持一致）
    BLOCK_ALIGN_BYTES = 256   # 多核对齐粒度（字节）
    UB_ALIGN_BYTES = 256      # UB 对齐粒度（字节）
    MIN_CORE_BYTES = 8192     # 每核最少处理字节数（v29: 4096→8192）
    MAX_CORE_NUM = 24
    UB_SIZE = 192 * 1024

    align_elements = BLOCK_ALIGN_BYTES // elem_size
    ub_align_elements = UB_ALIGN_BYTES // elem_size

    # 动态核数: ceil(total_bytes / MIN_CORE_BYTES)
    core_num = (total_bytes + MIN_CORE_BYTES - 1) // MIN_CORE_BYTES
    if core_num < 1: core_num = 1
    if core_num > MAX_CORE_NUM: core_num = MAX_CORE_NUM

    # 每核基础元素数，512 元素对齐
    block_former = ((total_length + core_num - 1) // core_num + align_elements - 1) // align_elements * align_elements
    if block_former < align_elements: block_former = align_elements

    # 总 block 数
    block_num = (total_length + block_former - 1) // block_former
    if block_num < 1: block_num = 1

    # UB 切分
    buffer_divisor = 20 if dtype_is_fp16 else 24
    max_ub_elements = UB_SIZE * 95 // 100 // buffer_divisor
    ub_former = (max_ub_elements // ub_align_elements) * ub_align_elements
    if ub_former < ub_align_elements: ub_former = ub_align_elements
    if ub_former > block_former: ub_former = block_former
    if ub_former < 1: ub_former = 1

    # 首 block 的 UB 循环与尾段
    ub_loop_of_former_block = block_former // ub_former
    ub_tail_of_former_block = block_former % ub_former

    # 末 block 的 UB 循环与尾段
    tail_block_elements = total_length - (block_num - 1) * block_former
    ub_loop_of_tail_block = tail_block_elements // ub_former
    ub_tail_of_tail_block = tail_block_elements % ub_former

    # GeluTilingData struct: 9 x uint32_t
    tiling = struct.pack('<IIIIIIIII',
        total_length,
        core_num,
        block_former,
        block_num,
        ub_former,
        ub_loop_of_former_block,
        ub_tail_of_former_block,
        ub_loop_of_tail_block,
        ub_tail_of_tail_block,
    )

    print(f"  totalLength={total_length}, dtype={'fp16' if dtype_is_fp16 else 'fp32'}")
    print(f"  coreNum={core_num}, blockFormer={block_former}, blockNum={block_num}")
    print(f"  ubFormer={ub_former}, ubLoopFormer={ub_loop_of_former_block}, ubTailFormer={ub_tail_of_former_block}")
    print(f"  ubLoopTail={ub_loop_of_tail_block}, ubTailTail={ub_tail_of_tail_block}")
    print(f"  Tiling data size: {len(tiling)} bytes")

    return tiling


def main():
    out_dir = "/opt/atomgit/GELU/prof_data"
    os.makedirs(out_dir, exist_ok=True)

    # 生成 float32 65536 的 tiling 数据
    print("=== float32 [65536] ===")
    t32 = gen_tiling(65536, dtype_is_fp16=False)
    with open(f"{out_dir}/tiling_f32_65536.bin", "wb") as f:
        f.write(t32)
    # 补齐到 44 bytes (opParaSize)
    with open(f"{out_dir}/tiling_f32_65536.bin", "ab") as f:
        f.write(b'\x00' * (44 - len(t32)))
    print(f"  Written ({44} bytes)")

    # 生成 float16 8192 的 tiling 数据
    print("\n=== float16 [8192] ===")
    t16 = gen_tiling(8192, dtype_is_fp16=True)
    with open(f"{out_dir}/tiling_f16_8192.bin", "wb") as f:
        f.write(t16)
    with open(f"{out_dir}/tiling_f16_8192.bin", "ab") as f:
        f.write(b'\x00' * (44 - len(t16)))
    print(f"  Written ({44} bytes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
