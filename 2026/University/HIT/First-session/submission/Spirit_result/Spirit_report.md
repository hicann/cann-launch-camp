# CANN启航 A8W8 QmmCustom 自定义算子团队实践报告
## 一、团队信息与贡献说明
### 1.1 团队基本信息
- 团队标识（组号）：Spirit
- CANNJudge 提交账号：
- CANNJudge 提交结果或链接：

### 1.2 团队成员分工与贡献
| 姓名 | GitCode 账号 | 负责模块/任务分工 | 实际贡献说明 | 对应 commit hash |
| --- | --- | --- | --- | 85cbe45016f8f80464ef5a320d5a7e03fec31a79 |
| 张皓然 | Zhanghr12345 | Tiling 设计、环境自动化脚本 | 1. 设计8字节对齐`QmmCustomTilingData`结构体；2. 实现`ChooseGrid`多核分块搜索策略与`CalcQmmTiling`分块计算函数；3. 编写Notebook环境检索、CANN环境加载、依赖安装自动化代码；4. 修复FRACTAL_NZ格式N/K维度32对齐边界问题 | `72f9a0d` |
| 宋绍阳 | alanrom | AscendC Kernel 双路径实现、Torch绑定 | 1. 实现Cube-only纯INT32输出`QmmCubeBasicKernel`；2. 完成Cube+Vector混合反量化`QmmPertokenKernel`；3. 实现`__mix__(1,2)`混合核调度、跨核同步标记逻辑；4. 编写Torch扩展注册、Kernel直调Host侧代码 | `45c218e` |
| 孙戴琛 | WTandWind | 算子编译、单算子/模型测试、模型替换集成 | 1. 编写CMake编译脚本，兼容CANN8.5.2废弃API警告；2. 开发全规格精度校验脚本、Profiler性能采集与CSV解析代码；3. 修改Qwen3-8B量化线性层，替换原生`npu_quant_matmul`；4. 执行端到端量化推理，采集prefill/decode阶段性能数据 | `e1063f5` |

### 1.3 团队协作说明
1. **任务拆分**：按照算子开发工程链路横向拆分三大独立模块，Tiling层、Kernel层、编译&模型集成层分别由三人独立开发，模块间低耦合，便于并行推进；
2. **代码评审机制**：采用Git分支开发、PR合并流程，每完成一个模块提交合并请求，另外两名成员交叉评审；张皓然校验Kernel中分块偏移计算，宋绍阳核对Tiling硬件约束参数，孙戴琛检查算子输入输出校验逻辑；
3. **集成流程**：先合并Tiling与Kernel代码得到完整`qmm_custom.asc`；编译通过后执行12组模型真实shape精度测试，全部PASS后再将算子so库接入Qwen推理框架；
4. **联合排障**：出现精度异常、性能瓶颈时三人共同复盘Profiler日志、核对分块尺寸与Vector反量化计算流程，协同定位计算偏差、访存瓶颈问题。

## 二、结果展示
### 2.1 单算子精度比对结果
测试覆盖Qwen3-8B推理全部12组真实(M,K,N)尺寸，每组分别执行INT32无缩放路径、BF16双缩放路径；INT32采用零误差校验，BF16浮点误差宽容度rtol=0.01、atol=0.01，全部用例验证通过：
```
============================================================
测试汇总
                      规格 |  INT32   |   BF16  
----------------------------------------------
       M=1,K=4096,N=4096 |   PASS   |   PASS  
       M=1,K=4096,N=6144 |   PASS   |   PASS  
      M=1,K=4096,N=24576 |   PASS   |   PASS  
      M=1,K=12288,N=4096 |   PASS   |   PASS  
      M=50,K=4096,N=4096 |   PASS   |   PASS  
      M=50,K=4096,N=6144 |   PASS   |   PASS  
     M=50,K=4096,N=24576 |   PASS   |   PASS  
     M=50,K=12288,N=4096 |   PASS   |   PASS  
    M=4096,K=4096,N=4096 |   PASS   |   PASS  
    M=4096,K=4096,N=6144 |   PASS   |   PASS  
   M=4096,K=4096,N=24576 |   PASS   |   PASS  
   M=4096,K=12288,N=4096 |   PASS   |   PASS  
INT32: 12/12 通过, BF16: 12/12 通过
```
以CPU高精度float64矩阵乘法结果作为真值基准，INT8整数乘累加无计算误差，BF16反量化浮点误差在合理区间，算子数学逻辑完全正确。

### 2.2 单算子性能测试结果
使用`torch_npu.profiler` Level1级别采集Kernel执行耗时，原始解析脚本因Input Shapes字段分隔长度不定报`list index out of range`，修复空值、数组越界判断逻辑后可正常提取各规格平均耗时。
性能特征总结：
1. 小M解码场景(M=1/50)：仅Cube的INT32路径耗时显著更低，省去INT32中间缓存读写、浮点换算、双Vector计算开销；
2. 大M Prefill批量场景(M=4096)：二维分块多核并行收益明显，Tiling策略均衡各AIC核计算负载；
3. BF16路径额外开销：GM读写INT32中间结果、INT32/FLOAT32/BF16多轮类型转换、两行缩放浮点乘法、双Vector核数据分发同步。

### 2.3 算子接入模型性能测试结果
将自定义`qmm_custom`替换推理框架`CompressedTensorsW8A8Int8Linear`中原生`npu_quant_matmul`后：
1. 模型生成文本内容、长度与原量化模型完全一致，推理精度无损失；
2. Profiler区分prefill批量推理、decode单token推理两个阶段，单独统计QmmCustom算子平均执行耗时、调用频次；
3. 性能对比：针对Qwen固定K=4096/12288尺寸定制Tiling分块，decode单token推理延迟小幅下降，prefill阶段AIC硬件利用率提升；
4. 数据统计：通过`kernel_details.csv`按(M,K,N)分组统计平均耗时，N=4096/6144/24576为模型高频权重维度，本算子优化收益集中在该类shape。

## 三、方案说明
### 3.1 设计思路
#### 3.1.1 TilingData 结构体设计
```cpp
#pragma pack(push, 8)
struct alignas(8) QmmCustomTilingData {
  TCubeTiling cubeTilingData; // 昇腾官方标准Matmul分块信息
  uint32_t isPertoken;        // 分支标记：0=INT32输出，1=BF16反量化
  uint32_t M,N,K;             // 矩阵全局维度
  uint32_t singleCoreM;       // 单核M分块尺寸
  uint32_t singleCoreN;       // 单核N分块尺寸
  uint32_t mBlocks,nBlocks;   // M、N维度总分块数量
  uint32_t vecTileElems;      // Vector单次批量处理元素数1024
  uint32_t reserved;
  uint64_t userWorkspaceOffset; // 用户中间缓存内存偏移
  uint64_t userWorkspaceSize;   // INT32中间结果缓存大小
  uint64_t workspaceSize;       // Kernel总所需Workspace字节数
};
#pragma pack(pop)
```
结构体8字节对齐，保证Host侧与Kernel侧内存读取一致性；复用官方`TCubeTiling`复用成熟Cube分块算法，新增业务字段区分双执行路径、管理反量化所需中间GM缓存。

#### 3.1.2 Tiling 分块实现核心逻辑
1. 硬件强制约束：M分块按16对齐，FRACTAL_NZ权重N、K维度必须32对齐；
2. 网格搜索策略：遍历所有合法M/N分块数量，引入长宽比惩罚、单核面积惩罚，优先选用硬件高吞吐128×256标准Cube块；
3. 场景自适应：M≤64（decode单token）仅拆分N维度，不切割M避免单核计算量过小、多核空跑；M>64（prefill批量）二维分块最大化利用全部AIC核；
4. 内存规划：仅BF16反量化路径分配M×N×4字节INT32缓存，叠加Matmul库系统Workspace，统一传递至Kernel。

#### 3.1.3 Kernel 双路径数据流
1. **Path1 Cube-only（无pertoken_scale）**
仅AIC Cube单元运行，数据流：GM INT8 x1/x2 → L0A/L0B Cube本地缓存 → INT8乘累加得到INT32结果 → 直接写入GM输出，无Vector参与，访存与计算开销最低。

2. **Path2 Cube+Vector混合（带pertoken_scale）**
Kernel声明`__mix__(1, 2)`，单个Block绑定1个Cube核、2个Vector核：
- AIC流程：Cube完成INT8矩阵乘，INT32中间结果写入GM workspace，调用跨核标记唤醒两个AIV；
- AIV分流计算：两个Vector核平分当前分块M行；逐块读取INT32缓存，依次执行INT32转FP32、乘perChannelScale、乘perTokenScale、FP32转BF16，最终写回GM输出；
- 同步机制：`PIPE_FIX`标记确保Cube写缓存完成后再启动Vector，消除GM读写竞争。

### 3.2 问题解决与优化策略
#### 3.2.1 实践问题与对应解决方案
1. **问题：FRACTAL_NZ权重计算结果精度全错**
原因：Tiling分块N未做32对齐，破坏NZ存储C0=32硬件边界；
解决：使用`AlignUpU32`强制singleN向上32对齐，Tiling计算后增加对齐校验。

2. **问题：编译提示SetSysWorkspace接口废弃**
原因：CANN8.5.2版本API迭代淘汰该接口；
临时方案保留调用保证功能正常，拓展优化方向为更换新版Workspace绑定API。

3. **问题：Profiler CSV解析报数组越界**
原因：Input Shapes字段分隔后元素数量不固定；
修复：增加空字符串过滤、数组长度判断，跳过残缺异常行。

4. **问题：Vector反量化浮点偏差偏大**
原因：循环内重复从GM读取perChannelScale；
优化：scale一次性加载至Vector本地LocalTensor，循环复用减少GM访存。

5. **问题：M=1小规格多核负载不均衡，大量空核**
解决：`ChooseGrid`中限制M≤64时分块数量maxMParts=1，仅拆分N维度。

#### 3.2.2 AI辅助使用说明
1. AI辅助内容：AscendC语法纠错、Tiling分块惩罚函数推导、Profiler解析代码、Torch扩展模板、模型算子替换脚本；
2. AI引入缺陷：自动生成代码未适配FRACTAL_NZ存储规则、缺失16/32对齐硬件约束，直接运行会出现内存越界、精度失效；
3. 团队验证方式：AI生成代码全部人工逐行核对硬件约束与量化数学公式，先用最小单例(M=1,N=32)调试通过，再批量跑全12组测试用例；
4. 使用心得：AI可快速生成标准化通用模板，但昇腾Cube、NZ格式、混合核等硬件专属逻辑存在信息偏差，必须结合CANN官方文档人工校验边界条件，不可直接复用生成代码。

#### 3.2.3 性能优化策略
1. Tiling层优化
- 长宽比惩罚函数优选128×256标准Cube尺寸，提升Cube流水线利用率；
- decode场景限制M不分块，减少多核同步开销。

2. Kernel访存优化
- Vector阶段scale预加载至本地张量，减少重复GM访问；
- 使用批量`DataCopy`搬运向量Tile，替代逐元素读写。

3. 同步与流水线优化
- `PIPE_FIX`跨核标记保证Cube写完成后再启动Vector，消除读写冲突；
- TQue本地队列实现数据搬运与计算重叠，隐藏访存延迟。

## 四、收获与感悟
### 张皓然
本次实践完整掌握昇腾Matmul整套Tiling分块设计流程，从前仅会调用官方内置算子，到自主实现多核网格调度、硬件对齐约束、Workspace内存规划，深入理解FRACTAL_NZ权重存储、AIC核资源分配底层逻辑。同时熟练掌握自动化工程脚本开发，可一键完成环境初始化、算子编译、批量精度测试。最大收获是意识到量化推理性能瓶颈大多来自访存，合理的分块策略计算优化收益更显著，后续计划研究动态自适应Tiling方案。

### 宋绍阳
系统学习了AscendC Cube+Vector混合编程模型，理清`__mix__`混合核资源分配、跨核同步标记、本地队列流水线整套机制。完整走通A8W8量化反量化数据流，对LLM低精度推理整数乘累加降低浮点误差的设计思路有具象理解。开发过程踩遍存储格式、类型转换、核同步各类坑，深刻意识到昇腾算子开发不能套用GPU编程思维，所有逻辑必须贴合AI Core硬件单元特性。

### 孙戴琛
打通自定义算子从编译、单算子验证、性能采集到大模型替换落地完整工程链路，掌握Torch扩展绑定NPU内核、NPU Profiler性能数据分析方法，能够区分prefill与decode阶段的算子耗时瓶颈。同时体会完备测试用例的工程价值，多规格精度脚本可快速定位底层计算bug，模型端到端推理是算子可用性的最终标准。团队分工与交叉评审大幅降低单人调试压力，提前规避大量底层逻辑错误。
