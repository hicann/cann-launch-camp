cmake_minimum_required(VERSION 3.16)
project(less_equal_cpu_debug)

# 查找 CPU调试库 tikicpulib
find_package(tikicpulib REQUIRED)

add_executable(less_equal_test
    ${CMAKE_CURRENT_SOURCE_DIR}/less_equal_test.cpp
)

target_compile_options(less_equal_test PRIVATE
    -g -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_link_libraries(less_equal_test PRIVATE
    ascendcl
    tikicpulib::${soc_version}
)

install(TARGETS less_equal_test
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
