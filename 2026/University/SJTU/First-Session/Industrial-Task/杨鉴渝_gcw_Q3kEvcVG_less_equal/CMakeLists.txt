cmake_minimum_required(VERSION 3.16)

# 查找 AscendC 编译工具链
find_package(ASC REQUIRED)

# ASC = Ascend C 语言支持, CXX = 标准C++
project(less_equal_custom LANGUAGES ASC CXX)

add_executable(demo
    less_equal_custom.asc
)

# 指定 NPU 架构 (910B)
target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
