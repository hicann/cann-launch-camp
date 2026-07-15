<<<<<<< HEAD
=======
# kernel源码
>>>>>>> 6b4669c59779fb679eb65471a6bc86b0e6fd17d5
npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

<<<<<<< HEAD
=======
# kernel动态库
>>>>>>> 6b4669c59779fb679eb65471a6bc86b0e6fd17d5
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

<<<<<<< HEAD
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
=======
# 添加到package中
npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
>>>>>>> 6b4669c59779fb679eb65471a6bc86b0e6fd17d5
