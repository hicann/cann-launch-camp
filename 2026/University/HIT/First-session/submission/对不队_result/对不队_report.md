团队实践报告
一、团队信息与贡献说明
1.1 团队基本信息
团队标识（组号）：对不队
CANNJudge 提交账号：chenming06
CANNJudge 提交结果或链接：https://cannjudge.cn/hit/20260721/qmmcustom/submission/6a61c8311336c465ba87b74d
1.2 团队成员分工与贡献
表格
姓名	GitCode 账号	负责模块/任务分工	实际贡献说明	对应 commit hash
张绍箕	chenming06	完善自定义算子实现与Notebook全流程验证	完成三项核心实现的集成与代码规范化，将最终ASC实现同步至Notebook并补充双Kernel执行流程说明；修复Notebook中Profiler结果解析崩溃与结果选择逻辑，补全Qwen3实际调用的shape信息，修复BF16与bias路径问题；运行并保留算子编译、INT32/BF16功能与性能测试以及接入Qwen3-8B模型测试的结果。	8c5da26
郑志成	yolo_mzzz	任务三Cube与Vector Kernel及验证	实现带perTokenScale的反量化路径：将Cube Matmul产生的INT32中间结果按行、按Vector核划分，从GM搬入后转换为FP32，依次融合perChannelScale与perTokenScale，最终转换并写回BF16；补充独立AIV Vector Kernel、同Stream顺序启动及Host侧双Kernel调度逻辑。	dd9d0cf
徐祥宇	gcw_s0J9fvim	实践总结与报告撰写	基于团队实验过程整理实践报告，汇总成员分工与协作过程、精度和性能测试结果、模型接入结果、方案设计、问题解决与优化策略、AI辅助使用及验证情况和成员学习总结。	c1e1a8b
盛冠华	2401_89825452	任务一Tiling设计与验证	设计并补全QmmCustomTilingData，记录Matmul Tiling、M/N/K、单核M分块、Cube/Vector核数及workspace信息；基于MultiCoreMatmulTiling配置INT8 ND/NZ输入和INT32 ND输出，按M维完成多核划分与尾块计算，动态查询系统workspace，并处理M=1等小M场景下的Tiling生成。	d908d37
1.3 团队协作说明
团队将实践任务按完整开发链路拆分为Tiling设计、Cube-only Kernel实现、Cube+Vector反量化实现、Notebook与模型集成验证、实验结果整理和报告撰写五部分。各成员在个人分支中完成修改并形成独立commit，队长通过git fetch获取成员分支，使用git diff审查后通过非squash merge汇总。任务一补充的M/N/K、单核分块和Vector核数字段为任务二、三提供统一接口；任务二生成INT32结果；任务三在其基础上完成双Scale反量化。队长在合并后统一处理头文件、workspace、Kernel启动和Torch接口，确保各模块组合成完整ASC实现，并将源码同步到Notebook。
集成完成后，团队共同检查算子编译日志、12组Qwen3-8B实际shape下的INT32与BF16精度结果、单算子Profiler数据以及模型接入结果。功能测试共包含12组INT32路径和12组BF16路径，全部通过；模型接入后生成文本能够正确描述attention输出，256次decode的平均推理耗时为94.245ms。
二、结果展示
2.1 单算子精度比对结果
单算子精度比对结果如下：
表格
M	K	N	INT32结果	BF16结果
1	4096	4096	PASS	PASS
1	4096	6144	PASS	PASS
1	4096	24576	PASS	PASS
1	12288	4096	PASS	PASS
50	4096	4096	PASS	PASS
50	4096	6144	PASS	PASS
50	4096	24576	PASS	PASS
50	12288	4096	PASS	PASS
4096	4096	4096	PASS	PASS
4096	4096	6144	PASS	PASS
4096	4096	24576	PASS	PASS
4096	12288	4096	PASS	PASS
2.2 单算子性能测试结果
单算子性能测试结果如下：
表格
M	K	N	INT32 Duration(us)	BF16 Duration(us)
1	4096	4096	141.417	2.220
1	4096	6144	209.055	2.980
1	4096	24576	902.782	7.719
1	12288	4096	410.752	2.680
50	4096	4096	148.637	6.440
50	4096	6144	216.356	7.200
50	4096	24576	923.501	16.240
50	12288	4096	421.291	7.500
4096	4096	4096	411.112	166.257
4096	4096	6144	636.947	187.416
4096	4096	24576	3021.020	774.764
4096	12288	4096	1445.112	167.437
2.3 算子接入模型性能测试结果
未开启Profiling的模型推理中，单次prefill耗时为104.510 ms；日志报告的decode平均推理耗时为93.820 ms，256次decode的观测范围为91.460～109.340 ms；模型输出文本与参考结果一致，且平均decode耗时低于100 ms。
开启Profiling后，单次prefill耗时为105.940 ms；日志报告的decode平均推理耗时为93.100 ms，256次decode的观测范围为91.160～108.110 ms。
三、方案说明
3.1 设计思路
本算子面向Qwen3-8B的A8W8量化矩阵乘场景，输入激活x1和权重x2均为INT8，矩阵乘累加结果为INT32。无perTokenScale时直接输出INT32；存在perTokenScale时，按照BF16((INT32结果)×(perChannelScale×perTokenScale))完成反量化并输出BF16。权重采用FRACTAL_NZ格式以适配Cube计算单元，激活和输出采用ND格式。
QmmCustomTilingData由TCubeTiling、路径标记、workspace信息、M/N/K、单核M分块和Vector核数构成。Host侧根据平台AIC核数和M计算usedCoreNum=min(AIC核数,M)，再通过向上取整得到singleCoreM，使每个Cube核负责一段连续行。随后使用MultiCoreMatmulTiling设置A矩阵为INT8 ND、B矩阵为INT8 NZ、C矩阵为INT32 ND，并设置原始shape、计算shape、使用核数和单核shape。Tiling API自动选择baseM/baseN/baseK，避免在M=1等小shape下强制固定baseM导致Tiling失败。系统库workspace通过平台接口动态查询；Vector核数取平台AIV核数与M的较小值。
Cube路径由QmmCubeBasicKernel完成。Init根据TilingData建立INT8输入和INT32输出的GM Tensor映射；Process根据核索引计算当前核的起始行和实际行数，对末核调用SetTail处理尾块，再通过AscendC Matmul对象设置A/B矩阵并执行IterateAll，将结果写回INT32输出或中间张量。
反量化路径采用两个独立Kernel在同一NPU Stream中顺序执行。qmm_cube_kernel先完成INT8矩阵乘并产生INT32中间结果；qmm_pertoken_kernel随后由多个Vector核按M维分行，每次按2048个元素从GM搬入INT32结果和perChannelScale，将INT32转换为FP32，先计算perChannelScale×perTokenScale，再与累加结果相乘，最后以CAST_RINT转换为BF16并写回输出。相同Stream保证Vector Kernel在Cube Kernel完成后执行，从而避免在单个混合Kernel中维护复杂且脆弱的跨核同步标志。
Host侧Torch接口负责检查输入维度、数据类型和K维匹配关系，根据是否传入perTokenScale选择INT32或BF16输出，计算Tiling并申请workspace。BF16路径额外申请INT32中间张量，先启动Cube Kernel，再启动Vector Kernel；INT32路径只启动Cube Kernel。最终通过TORCH_LIBRARY注册ascendc_ops::qmm_custom，供Qwen3-8B量化线性层调用。
3.2 问题解决与优化策略
3.2.1 问题及其解决过程
小M场景Tiling失败的问题：最初若固定较大的baseM，M=1时GetTiling可能失败。团队改为让Tiling API根据实际shape自动选择基础分块，并使用min(AIC核数,M)限制有效核数。
Cube与Vector同步复杂的问题：在单个混合Kernel中使用跨核flag容易因核编号、执行时序和内存可见性产生错误。最终将矩阵乘和反量化拆成两个Kernel，在同一NPU Stream中顺序启动，以明确的Kernel边界保证执行顺序。
BF16结果存在舍入敏感性的问题：若依次将INT32结果分别乘两个Scale，浮点运算顺序可能造成一个BF16 ULP的差异，并进一步影响贪心解码token。实现中先计算perChannelScale×perTokenScale，再与FP32累加结果相乘，最后使用CAST_RINT转换为BF16。
Profiler解析不稳定的问题：Profiler目录名包含随机标识，按目录名称排序可能选错结果，部分BF16记录还可能缺失shape。Notebook改为按文件修改时间选择最新CSV，并结合测试顺序或Qwen3固定线性层顺序补全缺失元数据，同时对空结果和异常字段进行检查。
模型接入路径不完整的问题：替换时需要同时覆盖INT32和BF16分支，并恢复量化线性层原有输出shape。团队统一通过qmm_custom选择输出类型，保存并恢复输入前缀维度；对于当前算子不支持的bias路径显式抛出异常，避免静默产生错误结果。
3.2.2 AI辅助
团队使用了AI工具辅助理解AscendC Matmul接口、梳理Cube与Vector的数据流、分析跨核同步风险、检查BF16运算顺序，并协助定位Profiler结果选择、CSV字段缺失和模型接入shape恢复等问题。AI提供的建议并不直接视为正确答案：过程中曾出现接口假设、同步方案和性能解释需要结合实际环境重新核对的情况，也存在建议可编译但未必满足精度或模型语义的风险。
团队采用"代码审查—编译—单算子精度—Profiler—模型输出"的顺序验证AI建议。所有建议先由成员结合CANN接口和现有代码检查，再通过12组INT32、12组BF16测试验证数值正确性，通过Profiler确认Kernel实际执行，最后以Qwen3-8B生成文本和推理耗时验证端到端行为。此次实践的体会是AI适合快速提供排查方向、解释接口和补充边界场景，但不能替代真实NPU环境中的编译、精度与性能验证；清晰描述问题、保留原始日志并逐项验证，比直接接受生成代码更可靠。
3.2.3 性能优化策略及效果
用到的性能优化策略主要包含如下：
按M维进行多核划分，并针对末核设置尾块，减少核间负载不均；
使用FRACTAL_NZ权重格式和AscendC Matmul高阶接口，发挥Cube单元的INT8矩阵乘能力；
根据实际平台动态选择AIC/AIV核数和workspace，避免固定参数在不同shape或环境下造成资源浪费或运行失败；
Vector反量化按行分配到多个AIV核，并以2048个元素为一块完成搬运、Cast、Scale融合和BF16写回，提升连续访存和向量计算效率；
无perTokenScale时不启动Vector Kernel，直接返回INT32结果；有perTokenScale时才申请中间结果并执行反量化，避免无关开销；
将两个Scale先融合再参与结果计算，在保持向量化处理的同时减少精度敏感的运算次序差异。
通过上述策略，成功使得平均推理耗时低于100ms。

四、收获与感悟
张绍箕
通过负责代码集成、Notebook完善和端到端验证，我对自定义算子从Tiling、Kernel、Torch注册到模型替换的完整链路有了系统认识。实践也让我认识到，能够单独通过测试的模块不代表集成后一定正确，接口统一、精度回归、Profiler复核和模型语义验证同样重要。

郑志成
通过Tiling设计，我理解了shape、核数、单核分块和workspace之间的联系，也认识到M=1等边界场景会直接影响Tiling策略的可用性。相比固定参数，根据实际平台和输入动态生成Tiling能够获得更好的兼容性。

徐祥宇
通过实现Cube-only Kernel，我掌握了GM Tensor映射、AscendC Matmul对象配置、多核行划分和尾块处理方法。此次实践让我更直观地理解了INT8矩阵乘如何利用Cube单元完成INT32累加和结果写回。

盛冠华
通过实现Cube+Vector路径，我加深了对perChannelScale、perTokenScale广播和BF16舍入的理解，也认识到Kernel间同步和浮点运算顺序可能影响最终模型token。单算子误差很小并不意味着可以忽略，仍需通过模型输出进行验证。