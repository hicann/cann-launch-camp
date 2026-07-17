#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

namespace {

constexpr uint32_t BLOCK_BYTES = 32;               // < 一个 Block 的字节数（256 bit）
constexpr uint32_t VECTOR_BYTES = 256;             // < Vector 单次处理的字节数
constexpr uint32_t MIN_BROADCAST_VECTOR_RUN = 16;  // < 广播路径启用向量化 DataCopy 的最小连续元素数

/**
 * @brief 返回两个 uint64_t 中较小的值。
 * @param a 第一个值
 * @param b 第二个值
 * @return 较小的值
 */
__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

/**
 * @brief 将值 v 向上对齐到 a 的倍数。
 * @param v 待对齐的值
 * @param a 对齐基数
 * @return 对齐后的值
 */
__aicore__ inline uint32_t AlignUpU32(uint32_t v, uint32_t a)
{
    return ((v + a - 1) / a) * a;
}

}  // namespace

/**
 * @brief LessEqual 算子的 Kernel 计算类模板。
 * @tparam DT_X1 输入数据类型，支持 half / float / int8_t / int32_t。
 */
template <class DT_X1>
class KernelLessEqual {
    // 编译期类型判定：是否为 half
    static constexpr bool kHalf = std::is_same_v<DT_X1, half>;
    // 编译期类型判定：是否为 float
    static constexpr bool kFloat = std::is_same_v<DT_X1, float>;
    // 编译期类型判定：是否为 int8_t
    static constexpr bool kInt8 = std::is_same_v<DT_X1, int8_t>;
    // 编译期类型判定：是否为 int32_t
    static constexpr bool kInt32 = std::is_same_v<DT_X1, int32_t>;

public:
    // 默认构造函数
    __aicore__ inline KernelLessEqual()
    {}

    /**
     * @brief 初始化 Kernel，从 TilingData 中提取参数并设置 GlobalBuffer 和 LocalBuffer。
     * @param x1     第一个输入操作数的 Global Memory 地址
     * @param x2     第二个输入操作数的 Global Memory 地址
     * @param y      输出操作数的 Global Memory 地址
     * @param tiling Host 侧下发的 TilingData 结构体引用，包含分块、形状、广播等全部信息
     */
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const LessEqualTilingData &tiling)
    {
        // ---- 从 TilingData 中解析参数 ----
        totalLength_ = tiling.totalLength;          // < 输出元素总数
        blockLength_ = tiling.blockLength;          // < 每个核处理的元素数
        vectorRunLength_ = tiling.vectorRunLength;  // < 广播模式下连续可向量化的最大段长度
        tileLength_ = tiling.tileLength;            // < 每次循环处理的 tile 大小（元素数）
        rank_ = tiling.rank;                        // < 输出张量的维度数
        mode_ = tiling.mode;                        // < 计算模式（非广播/标量广播/通用广播）
        x1RunMode_ = tiling.x1RunMode;              // < x1 在广播路径下的运行模式（连续/标量）
        x2RunMode_ = tiling.x2RunMode;              // < x2 在广播路径下的运行模式（连续/标量）

        // ---- 拷贝形状和步长信息 ----
        for (uint32_t i = 0; i < LESS_EQUAL_MAX_DIMS; ++i) {
            outDims_[i] = tiling.outDims[i];      // < 输出各维度大小
            x1Strides_[i] = tiling.x1Strides[i];  // < x1 各维度步长
            x2Strides_[i] = tiling.x2Strides[i];  // < x2 各维度步长
        }

        // ---- 设置 GlobalTensor 缓冲 ----
        x1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x1));
        x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X1 *>(x2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(y));

        // ---- 初始化 LocalTensor 队列和临时 Buffer ----
        pipe_.InitBuffer(x1Queue_, 1, tileLength_ * sizeof(DT_X1));                  // < x1 输入队列
        pipe_.InitBuffer(x2Queue_, 1, tileLength_ * sizeof(DT_X1));                  // < x2 输入队列
        pipe_.InitBuffer(yQueue_, 1, tileLength_ * sizeof(uint8_t));                 // < y 输出队列
        pipe_.InitBuffer(maskBuf_, AlignUpU32((tileLength_ + 7) / 8, BLOCK_BYTES));  // < Compare 结果 mask 缓冲
        pipe_.InitBuffer(resultBuf_, tileLength_ * sizeof(half));                    // < Select 中间结果缓冲

        // int8_t 需要额外的 half 中间缓冲用于类型转换
        if constexpr (kInt8) {
            pipe_.InitBuffer(cvtBuf1_, tileLength_ * sizeof(half));
            pipe_.InitBuffer(cvtBuf2_, tileLength_ * sizeof(half));
        }
        // int32_t 需要额外的 Min 结果缓冲
        if constexpr (kInt32) {
            pipe_.InitBuffer(minBuf_, tileLength_ * sizeof(int32_t));
        }
    }

    /**
     * @brief 算子主处理入口。根据当前核 ID 计算负责的数据段，
     *        并依据计算模式（广播/非广播）分发到不同的处理函数。
     */
    __aicore__ inline void Process()
    {
        if (totalLength_ == 0 || blockLength_ == 0)
            return;  // 空数据直接返回

        // 计算当前核负责的起始偏移和处理长度
        const uint64_t coreStart = static_cast<uint64_t>(GetBlockIdx()) * blockLength_;
        if (coreStart >= totalLength_)
            return;  // 超出总长度的核直接退出
        const uint64_t coreLength = MinU64(blockLength_, totalLength_ - coreStart);

        // 根据计算模式分发
        if (mode_ == LESS_EQUAL_GENERAL_BROADCAST) {
            // 广播模式：若连续段足够长则用向量化 DataCopy，否则逐元素标量读取
            if (vectorRunLength_ >= MIN_BROADCAST_VECTOR_RUN)
                ProcessBroadcastVector(coreStart, coreLength);
            else
                ProcessGeneralBroadcast(coreStart, coreLength);
        } else {
            // 非广播模式（含标量广播）
            ProcessVector(coreStart, coreLength);
        }
    }

private:
    /**
     * @brief 计算当前数据类型下一次向量操作的对齐元素数。
     * @return 对齐元素数（VECTOR_BYTES / sizeof(DT_X1)）
     */
    __aicore__ inline uint32_t VectorAlignElements() const
    {
        return VECTOR_BYTES / sizeof(DT_X1);
    }

    /**
     * @brief 用标量值填充 LocalTensor。
     *        对于 int8_t 类型，先填充 half 中间缓冲再 Cast 转换，因为 Duplicate 不直接支持 int8_t。
     * @param dst   目标 LocalTensor
     * @param value 要填充的标量值
     * @param n     要填充的元素个数
     */
    __aicore__ inline void FillLocalTensor(LocalTensor<DT_X1> dst, DT_X1 value, uint32_t n)
    {
        if constexpr (kInt8) {
            LocalTensor<half> t = cvtBuf1_.Get<half>();
            Duplicate(t, static_cast<half>(value), n);
            Cast(dst, t, RoundMode::CAST_NONE, n);
        } else {
            Duplicate(dst, value, n);
        }
    }

    /**
     * @brief 从 GlobalTensor 拷贝数据到 LocalTensor，支持非对齐场景（使用 DataCopyPad）。
     * @param dst     目标 LocalTensor
     * @param src     源 GlobalTensor 引用
     * @param off     源 GlobalTensor 中的起始偏移（元素索引）
     * @param count   实际需要拷贝的元素数
     * @param compCnt 计算（对齐）所需的元素数，不足部分补零
     */
    __aicore__ inline void CopyVectorInput(
        LocalTensor<DT_X1> dst, GlobalTensor<DT_X1> &src, uint64_t off, uint32_t count, uint32_t compCnt)
    {
        const uint32_t bytes = count * sizeof(DT_X1);
        if (count == compCnt && (bytes % BLOCK_BYTES) == 0) {
            // 数据完全对齐，直接使用 DataCopy
            DataCopy(dst, src[off], count);
            return;
        }
        // 非对齐场景使用 DataCopyPad，并手动补零至 compCnt
        DataCopyExtParams cp{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<DT_X1> pp{false, 0, 0, static_cast<DT_X1>(0)};
        DataCopyPad(dst, src[off], cp, pp);
        for (uint32_t i = count; i < compCnt; ++i)
            dst.SetValue(i, static_cast<DT_X1>(0));
    }

    /**
     * @brief 将计算结果从 LocalTensor 拷贝到 GlobalTensor，支持非对齐输出。
     * @param src   源 LocalTensor（uint8_t 类型的输出）
     * @param off   目标 GlobalTensor 中的起始偏移（字节索引）
     * @param count 实际需要拷贝的字节数
     */
    __aicore__ inline void CopyVectorOutput(const LocalTensor<uint8_t> &src, uint64_t off, uint32_t count)
    {
        if ((count % BLOCK_BYTES) == 0)
            DataCopy(yGm_[off], src, count);
        else {
            // 非对齐场景使用 DataCopyPad
            DataCopyExtParams p{1, count, 0, 0, 0};
            DataCopyPad(yGm_[off], src, p);
        }
    }

    /**
     * @brief 将 Compare 生成的 mask（uint8_t）转换为实际的 bool 输出值。
     *        mask 中每个 byte 的 bit 位表示比较结果（1 表示 true，0 表示 false）。
     *        通过 Select 操作将 mask 转换为 half 类型的 1.0/0.0，再 Cast 为 uint8_t。
     * @param mask   输入 mask，来自 Compare API
     * @param yLocal 输出 LocalTensor，存放转换后的 uint8_t 结果
     * @param n      处理元素个数
     */
    __aicore__ inline void MaterializeMask(LocalTensor<uint8_t> mask, LocalTensor<uint8_t> yLocal, uint32_t n)
    {
        LocalTensor<half> r = resultBuf_.Get<half>();
        Duplicate(r, static_cast<half>(1.0), n);  // 先全填 1.0
        // Select: mask 为 true 的位置保留 1.0，否则设为 0.0
        Select(r, mask, r, static_cast<half>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
        Cast(yLocal, r, RoundMode::CAST_NONE, n);  // 转换为 uint8_t
    }

    /**
     * @brief 非广播路径的数据搬入。根据计算模式处理标量广播或逐元素拷贝。
     * @param off     当前 tile 在输出中的全局偏移
     * @param count   实际元素数
     * @param aligned 对齐后的元素数（用于计算）
     */
    __aicore__ inline void CopyInVector(uint64_t off, uint32_t count, uint32_t aligned)
    {
        LocalTensor<DT_X1> x1L = x1Queue_.AllocTensor<DT_X1>();
        LocalTensor<DT_X1> x2L = x2Queue_.AllocTensor<DT_X1>();
        if (mode_ == LESS_EQUAL_X1_SCALAR) {
            // x1 是标量：从 GM 读取标量值后填充整个 LocalTensor
            FillLocalTensor(x1L, x1Gm_.GetValue(0), aligned);
            CopyVectorInput(x2L, x2Gm_, off, count, aligned);
        } else if (mode_ == LESS_EQUAL_X2_SCALAR) {
            // x2 是标量
            CopyVectorInput(x1L, x1Gm_, off, count, aligned);
            FillLocalTensor(x2L, x2Gm_.GetValue(0), aligned);
        } else {
            // 两个输入均连续
            CopyVectorInput(x1L, x1Gm_, off, count, aligned);
            CopyVectorInput(x2L, x2Gm_, off, count, aligned);
        }
        x1Queue_.EnQue(x1L);
        x2Queue_.EnQue(x2L);
    }

    /**
     * @brief 核心计算逻辑。从队列取出输入数据，执行 x1 <= x2 比较，
     *        将 mask 转换为 bool 输出后送入输出队列。
     * @param n 处理元素个数（已对齐）
     */
    __aicore__ inline void ComputeVector(uint32_t n)
    {
        LocalTensor<DT_X1> x1 = x1Queue_.DeQue<DT_X1>();
        LocalTensor<DT_X1> x2 = x2Queue_.DeQue<DT_X1>();
        LocalTensor<uint8_t> y = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> m = maskBuf_.Get<uint8_t>();

        if constexpr (kHalf || kFloat) {
            // half/float 直接使用 Compare
            Compare(m, x1, x2, CMPMODE::LE, n);
        } else if constexpr (kInt8) {
            // int8_t 需要先 Cast 到 half 再比较
            LocalTensor<half> h1 = cvtBuf1_.Get<half>();
            LocalTensor<half> h2 = cvtBuf2_.Get<half>();
            Cast(h1, x1, RoundMode::CAST_NONE, n);
            Cast(h2, x2, RoundMode::CAST_NONE, n);
            Compare(m, h1, h2, CMPMODE::LE, n);
        } else {
            // int32_t：利用 Min(x1, x2) == x1 等价于 x1 <= x2
            LocalTensor<int32_t> mv = minBuf_.Get<int32_t>();
            Min(mv, x1, x2, n);
            Compare(m, mv, x1, CMPMODE::EQ, n);
        }
        MaterializeMask(m, y, n);  // mask → uint8_t
        yQueue_.EnQue(y);
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }

    /**
     * @brief 非广播路径的数据搬出。从输出队列取出结果并拷贝到 GlobalMemory。
     * @param off   输出全局偏移（字节索引）
     * @param count 实际输出的字节数
     */
    __aicore__ inline void CopyOutVector(uint64_t off, uint32_t count)
    {
        LocalTensor<uint8_t> y = yQueue_.DeQue<uint8_t>();
        CopyVectorOutput(y, off, count);
        yQueue_.FreeTensor(y);
    }

    // ========== 非广播路径 ==========

    /**
     * @brief 非广播模式的主处理循环。将当前核负责的数据段按 tileLength 分块，
     *        依次执行 CopyIn → Compute → CopyOut 流水。
     * @param coreStart  当前核数据段的起始偏移
     * @param coreLength 当前核数据段的长度
     */
    __aicore__ inline void ProcessVector(uint64_t coreStart, uint64_t coreLength)
    {
        const uint32_t align = VectorAlignElements();
        uint64_t p = 0;
        while (p < coreLength) {
            // 计算当前 tile 的实际元素数和对齐元素数
            uint32_t n = static_cast<uint32_t>(MinU64(tileLength_, coreLength - p));
            uint32_t an = AlignUpU32(n, align);
            uint64_t off = coreStart + p;
            CopyInVector(off, n, an);
            ComputeVector(an);
            CopyOutVector(off, n);
            p += n;
        }
    }

    // ========== 广播偏移解析 ==========

    /**
     * @brief 将一维扁平索引 flat 解析为多维坐标，并计算 x1 和 x2 在各自 GM 中的偏移。
     * @param flat  输出的扁平索引
     * @param coord 输出的多维坐标数组（从最低维开始填充）
     * @param x1Off 输出：x1 在 GM 中的偏移
     * @param x2Off 输出：x2 在 GM 中的偏移
     */
    __aicore__ inline void ResolveOffsets(
        uint64_t flat, uint64_t coord[LESS_EQUAL_MAX_DIMS], uint64_t &x1Off, uint64_t &x2Off) const
    {
        x1Off = 0;
        x2Off = 0;
        uint64_t r = flat;
        for (int32_t ax = static_cast<int32_t>(rank_) - 1; ax >= 0; --ax) {
            uint64_t d = outDims_[ax];
            uint64_t c = (d == 0) ? 0 : (r % d);  // 当前维度的坐标
            coord[ax] = c;
            r = (d == 0) ? 0 : (r / d);
            x1Off += c * x1Strides_[ax];
            x2Off += c * x2Strides_[ax];
        }
    }

    /**
     * @brief 将多维坐标前进一步（类似多维计数器加 1），并增量更新 x1/x2 的偏移。
     *        用于逐元素广播场景下避免每次都做完整的偏移重算。
     * @param coord 当前多维坐标（会被修改）
     * @param x1Off x1 的当前偏移（会被增量更新）
     * @param x2Off x2 的当前偏移（会被增量更新）
     */
    __aicore__ inline void AdvanceCoord(uint64_t coord[LESS_EQUAL_MAX_DIMS], uint64_t &x1Off, uint64_t &x2Off) const
    {
        for (int32_t ax = static_cast<int32_t>(rank_) - 1; ax >= 0; --ax) {
            ++coord[ax];
            x1Off += x1Strides_[ax];
            x2Off += x2Strides_[ax];
            if (coord[ax] < outDims_[ax])
                return;  // 未进位，直接返回
            // 进位：回退当前维度并继续向高维进位
            coord[ax] = 0;
            x1Off -= x1Strides_[ax] * outDims_[ax];
            x2Off -= x2Strides_[ax] * outDims_[ax];
        }
    }

    /**
     * @brief 广播路径的数据搬入。根据 x1/x2 的运行模式决定是连续拷贝还是标量填充。
     * @param x1Off x1 在 GM 中的偏移
     * @param x2Off x2 在 GM 中的偏移
     * @param n     实际元素数
     * @param an    对齐后的元素数
     */
    __aicore__ inline void CopyInBroadcastVector(uint64_t x1Off, uint64_t x2Off, uint32_t n, uint32_t an)
    {
        LocalTensor<DT_X1> x1L = x1Queue_.AllocTensor<DT_X1>();
        LocalTensor<DT_X1> x2L = x2Queue_.AllocTensor<DT_X1>();
        // 根据运行模式选择连续 DataCopy 或标量填充
        if (x1RunMode_ == LESS_EQUAL_RUN_CONTIGUOUS)
            CopyVectorInput(x1L, x1Gm_, x1Off, n, an);
        else
            FillLocalTensor(x1L, x1Gm_.GetValue(x1Off), an);
        if (x2RunMode_ == LESS_EQUAL_RUN_CONTIGUOUS)
            CopyVectorInput(x2L, x2Gm_, x2Off, n, an);
        else
            FillLocalTensor(x2L, x2Gm_.GetValue(x2Off), an);
        x1Queue_.EnQue(x1L);
        x2Queue_.EnQue(x2L);
    }

    /**
     * @brief 广播向量化处理主循环。当广播维度产生的连续段较长时，
     *        利用 DataCopy 批量搬运，并按 vectorRunLength 对齐切分 tile。
     * @param coreStart  当前核数据段的起始偏移
     * @param coreLength 当前核数据段的长度
     */
    __aicore__ inline void ProcessBroadcastVector(uint64_t coreStart, uint64_t coreLength)
    {
        const uint32_t align = VectorAlignElements();
        uint64_t p = 0;
        while (p < coreLength) {
            uint64_t off = coreStart + p;
            uint64_t ro = off % vectorRunLength_;  // 当前连续段内偏移
            uint64_t rem = vectorRunLength_ - ro;  // 当前连续段剩余长度
            // tile 大小不超过 tileLength_、剩余核数据、当前连续段剩余长度三者最小值
            uint32_t n = static_cast<uint32_t>(MinU64(tileLength_, MinU64(coreLength - p, rem)));
            uint32_t an = AlignUpU32(n, align);
            uint64_t coord[LESS_EQUAL_MAX_DIMS] = {0};
            uint64_t x1Off = 0, x2Off = 0;
            ResolveOffsets(off, coord, x1Off, x2Off);
            CopyInBroadcastVector(x1Off, x2Off, n, an);
            ComputeVector(an);
            CopyOutVector(off, n);
            p += n;
        }
    }

    /**
     * @brief 通用逐元素广播处理主循环。当连续段太短不适合向量化 DataCopy 时，
     *        逐个元素通过 SetValue 从 GM 读取并写入 LocalTensor。
     * @param coreStart  当前核数据段的起始偏移
     * @param coreLength 当前核数据段的长度
     */
    __aicore__ inline void ProcessGeneralBroadcast(uint64_t coreStart, uint64_t coreLength)
    {
        uint64_t p = 0;
        while (p < coreLength) {
            uint32_t n = static_cast<uint32_t>(MinU64(tileLength_, coreLength - p));
            uint32_t an = AlignUpU32(n, VectorAlignElements());
            uint64_t off = coreStart + p;
            LocalTensor<DT_X1> x1L = x1Queue_.AllocTensor<DT_X1>();
            LocalTensor<DT_X1> x2L = x2Queue_.AllocTensor<DT_X1>();
            uint64_t coord[LESS_EQUAL_MAX_DIMS] = {0};
            uint64_t x1Off = 0, x2Off = 0;
            ResolveOffsets(off, coord, x1Off, x2Off);
            // 逐元素从 GM 读取，通过 AdvanceCoord 增量推进坐标
            for (uint32_t i = 0; i < n; ++i) {
                x1L.SetValue(i, x1Gm_.GetValue(x1Off));
                x2L.SetValue(i, x2Gm_.GetValue(x2Off));
                if (i + 1 < n)
                    AdvanceCoord(coord, x1Off, x2Off);
            }
            // 对齐部分补零
            for (uint32_t i = n; i < an; ++i) {
                x1L.SetValue(i, static_cast<DT_X1>(0));
                x2L.SetValue(i, static_cast<DT_X1>(0));
            }
            x1Queue_.EnQue(x1L);
            x2Queue_.EnQue(x2L);
            ComputeVector(an);
            CopyOutVector(off, n);
            p += n;
        }
    }

private:
    // ---- 管理和队列 ----
    TPipe pipe_;                         // < 管理队列和 Buffer 的 Pipe 对象
    TQue<TPosition::VECIN, 1> x1Queue_;  // < x1 输入队列（VECIN，深度 1）
    TQue<TPosition::VECIN, 1> x2Queue_;  // < x2 输入队列（VECIN，深度 1）
    TQue<TPosition::VECOUT, 1> yQueue_;  // < y 输出队列（VECOUT，深度 1）

    // ---- 临时计算 Buffer ----
    TBuf<TPosition::VECCALC> maskBuf_;    // < Compare 结果 mask 缓冲
    TBuf<TPosition::VECCALC> resultBuf_;  // < Select 中间结果缓冲（half）
    TBuf<TPosition::VECCALC> cvtBuf1_;    // < int8 类型转换中间缓冲 1（half）
    TBuf<TPosition::VECCALC> cvtBuf2_;    // < int8 类型转换中间缓冲 2（half）
    TBuf<TPosition::VECCALC> minBuf_;     // < int32 Min 中间结果缓冲

    // ---- GlobalTensor ----
    GlobalTensor<DT_X1> x1Gm_;   // < x1 全局内存
    GlobalTensor<DT_X1> x2Gm_;   // < x2 全局内存
    GlobalTensor<uint8_t> yGm_;  // < y 全局内存（bool 输出）

    // ---- Tiling 参数 ----
    uint64_t totalLength_ = 0;                       // < 输出元素总数
    uint64_t blockLength_ = 0;                       // < 每个核处理的元素数
    uint64_t vectorRunLength_ = 1;                   // < 广播模式下连续可向量化的最大段长度
    uint32_t tileLength_ = 0;                        // < 每个 tile 的元素数
    uint32_t rank_ = 0;                              // < 输出张量维度数
    uint32_t mode_ = 0;                              // < 计算模式枚举值
    uint32_t x1RunMode_ = 0;                         // < x1 广播运行模式（连续/标量）
    uint32_t x2RunMode_ = 0;                         // < x2 广播运行模式（连续/标量）
    uint64_t outDims_[LESS_EQUAL_MAX_DIMS] = {0};    // < 输出各维度大小
    uint64_t x1Strides_[LESS_EQUAL_MAX_DIMS] = {0};  // < x1 各维度步长
    uint64_t x2Strides_[LESS_EQUAL_MAX_DIMS] = {0};  // < x2 各维度步长
};

/**
 * @brief LessEqual 算子的 Kernel 入口函数。
 * @tparam DT_X1 输入数据类型
 * @param x1        第一个输入操作数的 GM 地址
 * @param x2        第二个输入操作数的 GM 地址
 * @param y         输出操作数的 GM 地址
 * @param workspace 工作空间 GM 地址（本算子未使用）
 * @param tiling     TilingData 的 GM 地址
 */
template <typename DT_X1>
__global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);                // 指定为 AIV（Vector）类型任务
    REGISTER_TILING_DEFAULT(LessEqualTilingData);                  // 注册 TilingData 类型
    GET_TILING_DATA_WITH_STRUCT(LessEqualTilingData, td, tiling);  // 获取 TilingData
    KernelLessEqual<DT_X1> op;                                     // 实例化算子对象
    op.Init(x1, x2, y, td);                                        // 初始化
    op.Process();                                                  // 执行计算
}
