# FastGelu v54 native-half + aligned DataCopy top5 candidate

在 v53 基础上继续针对 3/4/5 做安全提速：
- 保留 v52 的 Host blockLength / 32B 对齐 offset 架构；
- half 直接半精度计算，减少 Cast；
- TILE_LENGTH = 8192；
- 对 32B 对齐 tile 走 DataCopy，只有尾块走 DataCopyPad；
- 不改数学公式，不用 FasterGelu/FasterGeluV2/Tanh。
