# ClipByValue 算子课题

## 课题说明

本项目基于 Ascend C 实现 ClipByValue 算子，将输入张量中的元素限制在给定的最小值和最大值之间。

计算规则：

- 当输入值小于 `clip_value_min` 时，输出 `clip_value_min`；
- 当输入值大于 `clip_value_max` 时，输出 `clip_value_max`；
- 其他情况下保持输入值不变。

本项目面向 Ascend 910B 平台，源码位于 `src` 目录。

## 目录结构

```text
src/
├── CMakeLists.txt
├── op_host/
│   ├── CMakeLists.txt
│   └── clip_by_value.cpp
└── op_kernel/
    ├── CMakeLists.txt
    ├── clip_by_value.cpp
    ├── clip_by_value_tiling.h
    └── tiling_key_clip_by_value.h