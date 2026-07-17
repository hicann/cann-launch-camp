#include "kernel_operator.h"
#include "less_equal_tiling.h"
#include "tiling_key_less_equal.h"

using namespace AscendC;

/**
 * @file less_equal.cpp
 * @brief LessEqual算子Kernel侧实现
 * 
 * 该文件实现了LessEqual算子在AI Core上的核心计算逻辑，
 * 支持float16、float32、int32、int8四种数据类型，
 * 并实现了同形计算(fast模式)和广播计算(broadcast模式)两种处理策略。
 */

/**
 * @brief LessEqual算子Kernel核心类
 * 
 * 模板参数DT_X1表示输入数据类型，支持float16、float32、int32、int8。
 * 该类封装了算子在AI Core上的所有计算逻辑，包括：
 * - 初始化配置参数和UB缓冲区
 * - 快速模式处理（输入形状相同时）
 * - 广播模式处理（输入形状不同时）
 * - 逐Tile的比较计算
 */
template <class DT_X1>
class KernelLessEqual {
public:
    /**
     * @brief 默认构造函数
     */
    __aicore__ inline KernelLessEqual() {}

    /**
     * @brief 初始化算子配置和UB缓冲区
     * 
     * @param x1 输入张量x1的全局内存地址
     * @param x2 输入张量x2的全局内存地址
     * @param y 输出张量y的全局内存地址
     * @param td Host侧传递的Tiling配置数据
     */
    __aicore__ inline void Init(
        GM_ADDR x1,
        GM_ADDR x2,
        GM_ADDR y,
        const LessEqualTilingData& td)
    {
        // 从Tiling数据中提取配置参数
        mode = td.mode;
        totalLen = td.totalLen;
        tileLen = td.tileLen;
        blockDim = td.blockDim;
        perCore = td.perCore;
        ndim = td.ndim;
        lastDimLen = td.lastDimLen;
        totalRows = td.totalRows;

        // 复制形状和步长信息
        for (uint32_t i = 0; i < LE_MAX_DIM; ++i) {
            outShape[i] = td.outShape[i];
            x1StrideArr[i] = td.x1Stride[i];
            x2StrideArr[i] = td.x2Stride[i];
        }

        // 绑定全局内存缓冲区
        x1Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X1*>(x1));

        x2Gm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X1*>(x2));

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(y));

        // 空张量直接返回，无需初始化缓冲区
        if (totalLen == 0) {
            return;
        }

        // 初始化输入队列缓冲区（双缓冲）
        pipe.InitBuffer(
            qX1,
            2,
            tileLen * static_cast<uint32_t>(sizeof(DT_X1)));

        pipe.InitBuffer(
            qX2,
            2,
            tileLen * static_cast<uint32_t>(sizeof(DT_X1)));

        // 初始化输出队列缓冲区（双缓冲）
        pipe.InitBuffer(
            qOut,
            2,
            tileLen * static_cast<uint32_t>(sizeof(int8_t)));

        // 初始化比较掩码缓冲区（按字节对齐）
        pipe.InitBuffer(
            bufMask,
            ((tileLen / 8) + 31) / 32 * 32);

        // 初始化常量缓冲区（用于生成True值）
        pipe.InitBuffer(
            bufOnes,
            tileLen * static_cast<uint32_t>(sizeof(half)));

        // 初始化中间结果缓冲区（用于类型转换）
        pipe.InitBuffer(
            bufOutHalf,
            tileLen * static_cast<uint32_t>(sizeof(half)));

        // int8类型需要额外的转换缓冲区
        if constexpr (std::is_same_v<DT_X1, int8_t>) {
            pipe.InitBuffer(
                bufAHalf,
                tileLen * static_cast<uint32_t>(sizeof(half)));

            pipe.InitBuffer(
                bufBHalf,
                tileLen * static_cast<uint32_t>(sizeof(half)));
        }

        // int32类型需要额外的最小值缓冲区
        if constexpr (std::is_same_v<DT_X1, int32_t>) {
            pipe.InitBuffer(
                bufI32,
                tileLen * static_cast<uint32_t>(sizeof(int32_t)));
        }

        // 预填充ones缓冲区为1.0（用于生成True结果）
        LocalTensor<half> ones =
            bufOnes.Get<half>();

        Duplicate(
            ones,
            static_cast<half>(1.0),
            tileLen);
    }

    /**
     * @brief 主处理入口函数
     * 
     * 根据mode参数选择处理模式：
     * - mode=0: fast模式，输入形状相同，直接逐元素比较
     * - mode=1: broadcast模式，输入形状不同，需要处理广播逻辑
     */
    __aicore__ inline void Process()
    {
        if (totalLen == 0) {
            return;
        }

        if (mode == 0) {
            ProcessFast();
        } else {
            ProcessBcast();
        }
    }

private:
    /**
     * @brief 将数值向上对齐到256的倍数
     * 
     * AI Core向量指令要求数据长度必须是256的倍数，
     * 该函数用于将实际数据长度对齐到符合要求的长度。
     * 
     * @param value 原始数据长度
     * @return 对齐后的数据长度
     */
    __aicore__ inline uint32_t RoundUp256(
        uint32_t value)
    {
        return
            (value + 255U) /
            256U *
            256U;
    }

    /**
     * @brief 将比较掩码转换为最终输出格式
     * 
     * Compare指令返回的是uint8掩码（每bit表示一个比较结果），
     * 该函数将掩码转换为int8格式的输出（1表示True，0表示False）。
     * 
     * @param mask 比较指令输出的掩码张量
     * @param output 最终输出张量（int8格式）
     * @param alignedLen 对齐后的数据长度
     */
    __aicore__ inline void PostProcessMask(
        const LocalTensor<uint8_t>& mask,
        const LocalTensor<int8_t>& output,
        uint32_t alignedLen)
    {
        LocalTensor<half> ones =
            bufOnes.Get<half>();

        LocalTensor<half> outputHalf =
            bufOutHalf.Get<half>();

        // 根据掩码选择输出1.0(True)或0.0(False)
        Select(
            outputHalf,
            mask,
            ones,
            static_cast<half>(0.0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE,
            alignedLen);

        // 将half类型转换为int8类型
        Cast(
            output,
            outputHalf,
            RoundMode::CAST_RINT,
            alignedLen);
    }

    /**
     * @brief 执行单个Tile的比较计算
     * 
     * 根据输入数据类型采用不同的比较策略：
     * - int32: 使用Min+EQ组合实现LE比较（因为Compare不支持int32 LE）
     * - int8: 先转换为half类型再进行比较
     * - float16/float32: 直接使用Compare指令进行LE比较
     * 
     * @param inputA 输入张量A的本地Tile
     * @param inputB 输入张量B的本地Tile
     * @param output 输出张量的本地Tile
     * @param alignedLen 对齐后的数据长度
     * @param inputAScalar 输入A是否为标量广播
     * @param inputBScalar 输入B是否为标量广播
     */
    __aicore__ inline void ComputeTile(
        const LocalTensor<DT_X1>& inputA,
        const LocalTensor<DT_X1>& inputB,
        const LocalTensor<int8_t>& output,
        uint32_t alignedLen,
        bool inputAScalar,
        bool inputBScalar)
    {
        LocalTensor<uint8_t> mask =
            bufMask.Get<uint8_t>();

        // int32类型：使用Min+EQ组合实现LE比较
        if constexpr (
            std::is_same_v<DT_X1, int32_t>) {
            LocalTensor<int32_t> minimum =
                bufI32.Get<int32_t>();

            // 计算两个输入的最小值
            Min(
                minimum,
                inputA,
                inputB,
                static_cast<int32_t>(alignedLen));

            // 如果最小值等于inputA，则说明inputA <= inputB
            Compare(
                mask,
                minimum,
                inputA,
                CMPMODE::EQ,
                alignedLen);
        }
        // int8类型：先转换为half再比较
        else if constexpr (
            std::is_same_v<DT_X1, int8_t>) {
            LocalTensor<half> inputAHalf =
                bufAHalf.Get<half>();

            LocalTensor<half> inputBHalf =
                bufBHalf.Get<half>();

            // 处理标量广播情况
            if (inputAScalar) {
                int8_t value =
                    inputA.GetValue(0);

                Duplicate(
                    inputAHalf,
                    static_cast<half>(
                        static_cast<float>(
                            static_cast<int32_t>(value))),
                    alignedLen);
            } else {
                Cast(
                    inputAHalf,
                    inputA,
                    RoundMode::CAST_NONE,
                    alignedLen);
            }

            // 处理标量广播情况
            if (inputBScalar) {
                int8_t value =
                    inputB.GetValue(0);

                Duplicate(
                    inputBHalf,
                    static_cast<half>(
                        static_cast<float>(
                            static_cast<int32_t>(value))),
                    alignedLen);
            } else {
                Cast(
                    inputBHalf,
                    inputB,
                    RoundMode::CAST_NONE,
                    alignedLen);
            }

            // 执行LE比较
            Compare(
                mask,
                inputAHalf,
                inputBHalf,
                CMPMODE::LE,
                alignedLen);
        }
        // float16/float32类型：直接比较
        else {
            Compare(
                mask,
                inputA,
                inputB,
                CMPMODE::LE,
                alignedLen);
        }

        // 将掩码转换为最终输出格式
        PostProcessMask(
            mask,
            output,
            alignedLen);
    }

    /**
     * @brief Fast模式处理函数（输入形状相同时）
     * 
     * 当两个输入张量形状完全相同时，采用简单的逐元素比较策略：
     * 1. 根据blockIndex计算当前核心处理的起始偏移
     * 2. 按tileLen大小分批从GM加载数据到UB
     * 3. 执行比较计算
     * 4. 将结果写回GM
     * 
     * 该模式下x1和x2具有相同的内存布局，可以直接按相同偏移访问。
     */
    __aicore__ inline void ProcessFast()
    {
        // 获取当前核心的block索引
        uint32_t blockIndex =
            GetBlockIdx();

        // 计算当前核心处理数据的起始偏移
        uint64_t offset =
            static_cast<uint64_t>(blockIndex) *
            perCore;

        // 超出范围直接返回
        if (offset >= totalLen) {
            return;
        }

        // 计算当前核心需要处理的元素数量
        uint32_t currentCoreLen =
            perCore;

        if (offset + currentCoreLen > totalLen) {
            currentCoreLen =
                static_cast<uint32_t>(
                    totalLen - offset);
        }

        // 初始化数据拷贝参数（无padding）
        DataCopyPadExtParams<DT_X1> pad {
            false,
            0,
            0,
            static_cast<DT_X1>(0)
        };

        // 按Tile分批处理数据
        for (uint32_t processed = 0;
             processed < currentCoreLen;
             processed += tileLen) {
            uint32_t currentLen =
                tileLen;

            // 处理最后一个Tile，长度可能不足tileLen
            if (processed + currentLen >
                currentCoreLen) {
                currentLen =
                    currentCoreLen - processed;
            }

            // 对齐到256的倍数
            uint32_t alignedLen =
                RoundUp256(currentLen);

            // 当前Tile在全局内存中的起始位置
            uint64_t base =
                offset + processed;

            // 从GM加载x1的当前Tile到UB
            LocalTensor<DT_X1> inputA =
                qX1.AllocTensor<DT_X1>();

            DataCopyPad(
                inputA,
                x1Gm[base],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            qX1.EnQue(inputA);

            // 从GM加载x2的当前Tile到UB
            LocalTensor<DT_X1> inputB =
                qX2.AllocTensor<DT_X1>();

            DataCopyPad(
                inputB,
                x2Gm[base],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            qX2.EnQue(inputB);

            // 等待数据加载完成
            inputA =
                qX1.DeQue<DT_X1>();

            inputB =
                qX2.DeQue<DT_X1>();

            // 执行比较计算
            LocalTensor<int8_t> output =
                qOut.AllocTensor<int8_t>();

            ComputeTile(
                inputA,
                inputB,
                output,
                alignedLen,
                false,
                false);

            // 释放输入缓冲区，提交输出
            qX1.FreeTensor(inputA);
            qX2.FreeTensor(inputB);
            qOut.EnQue(output);

            // 等待计算完成
            output =
                qOut.DeQue<int8_t>();

            // 将结果写回GM
            DataCopyPad(
                yGm[base],
                output,
                DataCopyExtParams {
                    1,
                    currentLen,
                    0,
                    0,
                    0
                });

            qOut.FreeTensor(output);
        }
    }

    /**
     * @brief 初始化广播模式下的行状态
     * 
     * 根据firstRow计算当前行在各维度上的坐标，
     * 并据此计算x1和x2在全局内存中的起始偏移。
     * 
     * @param firstRow 当前核心处理的第一行索引
     */
    __aicore__ inline void InitRowState(
        uint32_t firstRow)
    {
        x1Base = 0;
        x2Base = 0;

        uint32_t remaining =
            firstRow;

        // 计算各维度坐标（除最后一维外）
        for (uint32_t dim = 0;
             dim + 1 < ndim;
             ++dim) {
            uint32_t multiplier = 1;

            // 计算当前维度的步长乘数
            for (uint32_t inner = dim + 1;
                 inner + 1 < ndim;
                 ++inner) {
                multiplier *=
                    outShape[inner];
            }

            // 计算当前维度的坐标
            uint32_t coordinate =
                multiplier == 0
                    ? 0
                    : remaining / multiplier;

            remaining =
                multiplier == 0
                    ? remaining
                    : remaining % multiplier;

            indices[dim] =
                coordinate;

            // 根据步长计算x1和x2的内存偏移
            x1Base +=
                static_cast<int32_t>(coordinate) *
                x1StrideArr[dim];

            x2Base +=
                static_cast<int32_t>(coordinate) *
                x2StrideArr[dim];
        }
    }

    /**
     * @brief 推进到下一行的状态
     * 
     * 模拟进位机制，更新各维度坐标和内存偏移，
     * 支持多维度的行索引递增。
     */
    __aicore__ inline void AdvanceRowState()
    {
        if (ndim < 2) {
            return;
        }

        // 从倒数第二维开始递增
        int32_t dim =
            static_cast<int32_t>(ndim) - 2;

        indices[dim]++;

        // 更新内存偏移
        x1Base +=
            x1StrideArr[dim];

        x2Base +=
            x2StrideArr[dim];

        // 处理进位
        while (dim > 0 &&
               indices[dim] >=
                   outShape[dim]) {
            // 回退当前维度的偏移
            x1Base -=
                static_cast<int32_t>(
                    outShape[dim]) *
                x1StrideArr[dim];

            x2Base -=
                static_cast<int32_t>(
                    outShape[dim]) *
                x2StrideArr[dim];

            // 重置当前维度坐标
            indices[dim] = 0;
            --dim;
            indices[dim]++;

            // 更新更高维度的偏移
            x1Base +=
                x1StrideArr[dim];

            x2Base +=
                x2StrideArr[dim];
        }
    }

    /**
     * @brief 加载广播模式下的操作数
     * 
     * 根据lastStride判断是否为标量广播：
     * - lastStride=0: 标量广播，加载单个元素并复制到整个Tile
     * - lastStride!=0: 正常加载，按列偏移读取数据
     * 
     * @param queue 输入队列
     * @param global 全局张量
     * @param baseIndex 当前行的基址索引
     * @param lastStride 最后一维的步长（0表示标量广播）
     * @param column 当前列偏移
     * @param currentLen 当前Tile的实际长度
     * @param alignedLen 对齐后的长度
     * @param output 输出的本地张量
     */
    __aicore__ inline void LoadOperand(
        TQue<TPosition::VECIN, 2>& queue,
        const GlobalTensor<DT_X1>& global,
        int32_t baseIndex,
        int32_t lastStride,
        uint32_t column,
        uint32_t currentLen,
        uint32_t alignedLen,
        LocalTensor<DT_X1>& output)
    {
        DataCopyPadExtParams<DT_X1> pad {
            false,
            0,
            0,
            static_cast<DT_X1>(0)
        };

        LocalTensor<DT_X1> local =
            queue.AllocTensor<DT_X1>();

        // 标量广播情况：只加载一个元素，然后复制到整个Tile
        if (lastStride == 0) {
            DataCopyPad(
                local,
                global[
                    static_cast<uint32_t>(
                        baseIndex)],
                DataCopyExtParams {
                    1,
                    static_cast<uint32_t>(
                        sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            queue.EnQue(local);

            local =
                queue.DeQue<DT_X1>();

            // int8类型特殊处理：打包两个int8为一个uint16进行复制
            if constexpr (
                std::is_same_v<DT_X1, int8_t>) {
                int8_t value =
                    local.GetValue(0);

                const uint16_t byteValue =
                    static_cast<uint16_t>(
                        static_cast<uint8_t>(
                            value));

                const uint16_t packedValue =
                    byteValue |
                    static_cast<uint16_t>(
                        byteValue << 8U);

                LocalTensor<uint16_t> packed =
                    local
                        .template ReinterpretCast<
                            uint16_t>();

                Duplicate<uint16_t>(
                    packed,
                    packedValue,
                    static_cast<int32_t>(
                        (alignedLen + 1U) /
                        2U));
            } else {
                // 其他类型直接复制
                DT_X1 value =
                    local.GetValue(0);

                Duplicate(
                    local,
                    value,
                    alignedLen);
            }
        }
        // 正常加载：按列偏移从GM读取数据
        else {
            DataCopyPad(
                local,
                global[
                    static_cast<uint32_t>(
                        baseIndex) +
                    column],
                DataCopyExtParams {
                    1,
                    currentLen *
                        static_cast<uint32_t>(
                            sizeof(DT_X1)),
                    0,
                    0,
                    0
                },
                pad);

            queue.EnQue(local);

            local =
                queue.DeQue<DT_X1>();
        }

        output = local;
    }

    /**
     * @brief Broadcast模式处理函数（输入形状不同时）
     * 
     * 当两个输入张量形状不同但满足广播条件时，采用广播比较策略：
     * 1. 将张量按行划分（除最后一维外的所有维度合并为行）
     * 2. 每行内按tileLen分批处理
     * 3. 根据步长信息处理广播逻辑（stride=0表示标量广播）
     * 4. 支持多种广播场景：标量与张量、向量与矩阵等
     */
    __aicore__ inline void ProcessBcast()
    {
        // 获取当前核心的block索引
        uint32_t blockIndex =
            GetBlockIdx();

        // 计算当前核心处理的起始行
        uint32_t firstRow =
            blockIndex * perCore;

        // 超出范围直接返回
        if (firstRow >= totalRows) {
            return;
        }

        // 计算当前核心需要处理的行数
        uint32_t currentCoreRows =
            perCore;

        if (firstRow + currentCoreRows >
            totalRows) {
            currentCoreRows =
                totalRows - firstRow;
        }

        // 每行的长度（最后一维的长度）
        uint32_t rowLength =
            lastDimLen;

        // 获取最后一维的步长（用于判断是否为标量广播）
        int32_t x1LastStride =
            x1StrideArr[ndim - 1];

        int32_t x2LastStride =
            x2StrideArr[ndim - 1];

        // 初始化第一行的状态
        InitRowState(firstRow);

        // 逐行处理
        for (uint32_t row = 0;
             row < currentCoreRows;
             ++row) {
            // 计算当前行在输出中的起始偏移
            uint32_t outputRowBase =
                (firstRow + row) *
                rowLength;

            // 每行内按Tile分批处理
            for (uint32_t column = 0;
                 column < rowLength;
                 column += tileLen) {
                uint32_t currentLen =
                    tileLen;

                // 处理最后一个Tile
                if (column + currentLen >
                    rowLength) {
                    currentLen =
                        rowLength - column;
                }

                // 对齐到256的倍数
                uint32_t alignedLen =
                    RoundUp256(currentLen);

                LocalTensor<DT_X1> inputA;
                LocalTensor<DT_X1> inputB;

                // 加载x1的当前Tile（处理广播）
                LoadOperand(
                    qX1,
                    x1Gm,
                    x1Base,
                    x1LastStride,
                    column,
                    currentLen,
                    alignedLen,
                    inputA);

                // 加载x2的当前Tile（处理广播）
                LoadOperand(
                    qX2,
                    x2Gm,
                    x2Base,
                    x2LastStride,
                    column,
                    currentLen,
                    alignedLen,
                    inputB);

                // 判断是否为标量广播
                bool inputAScalar =
                    x1LastStride == 0;

                bool inputBScalar =
                    x2LastStride == 0;

                // 执行比较计算
                LocalTensor<int8_t> output =
                    qOut.AllocTensor<int8_t>();

                ComputeTile(
                    inputA,
                    inputB,
                    output,
                    alignedLen,
                    inputAScalar,
                    inputBScalar);

                // 释放输入缓冲区，提交输出
                qX1.FreeTensor(inputA);
                qX2.FreeTensor(inputB);
                qOut.EnQue(output);

                // 等待计算完成
                output =
                    qOut.DeQue<int8_t>();

                // 将结果写回GM
                DataCopyPad(
                    yGm[
                        outputRowBase +
                        column],
                    output,
                    DataCopyExtParams {
                        1,
                        currentLen,
                        0,
                        0,
                        0
                    });

                qOut.FreeTensor(output);
            }

            // 推进到下一行（最后一行不需要推进）
            if (row + 1 <
                currentCoreRows) {
                AdvanceRowState();
            }
        }
    }

private:
    TPipe pipe;                            /**< AI Core管道对象，用于管理UB缓冲区和队列 */

    TQue<TPosition::VECIN, 2> qX1;        /**< x1输入队列，双缓冲 */
    TQue<TPosition::VECIN, 2> qX2;        /**< x2输入队列，双缓冲 */
    TQue<TPosition::VECOUT, 2> qOut;      /**< 输出队列，双缓冲 */

    TBuf<TPosition::VECCALC> bufMask;     /**< 比较掩码缓冲区 */
    TBuf<TPosition::VECCALC> bufOnes;     /**< 常量1.0缓冲区（用于生成True值） */
    TBuf<TPosition::VECCALC> bufOutHalf;  /**< 中间结果缓冲区（half类型） */
    TBuf<TPosition::VECCALC> bufAHalf;    /**< int8转换缓冲区A */
    TBuf<TPosition::VECCALC> bufBHalf;    /**< int8转换缓冲区B */
    TBuf<TPosition::VECCALC> bufI32;      /**< int32最小值计算缓冲区 */

    GlobalTensor<DT_X1> x1Gm;             /**< x1全局内存张量 */
    GlobalTensor<DT_X1> x2Gm;             /**< x2全局内存张量 */
    GlobalTensor<int8_t> yGm;             /**< 输出全局内存张量（int8表示bool） */

    uint32_t mode;                        /**< 处理模式：0=fast，1=broadcast */
    uint64_t totalLen;                    /**< 总元素数 */
    uint32_t tileLen;                     /**< 每Tile处理的元素数 */
    uint32_t blockDim;                    /**< 核心数 */
    uint32_t perCore;                     /**< 每核心处理元素数/行数 */
    uint32_t ndim;                        /**< 维度数 */
    uint32_t lastDimLen;                  /**< 最后一维长度 */
    uint32_t totalRows;                   /**< 总行数 */

    uint32_t outShape[LE_MAX_DIM];        /**< 输出形状 */
    int32_t x1StrideArr[LE_MAX_DIM];      /**< x1各维度步长 */
    int32_t x2StrideArr[LE_MAX_DIM];      /**< x2各维度步长 */

    int32_t x1Base;                       /**< 当前行x1基址索引 */
    int32_t x2Base;                       /**< 当前行x2基址索引 */
    uint32_t indices[LE_MAX_DIM];         /**< 当前行各维度坐标 */
};

/**
 * @brief LessEqual算子Kernel入口函数
 * 
 * 这是AI Core上执行的全局函数，由Ascend C框架调用。
 * 函数负责：
 * 1. 注册Tiling数据结构
 * 2. 从GM获取Tiling配置数据
 * 3. 创建KernelLessEqual实例并初始化
 * 4. 执行核心计算逻辑
 * 
 * @param x1 输入张量x1的全局内存地址
 * @param x2 输入张量x2的全局内存地址
 * @param y 输出张量y的全局内存地址
 * @param workspace 工作空间地址（本算子未使用）
 * @param tiling Tiling配置数据的全局内存地址
 */
template <typename DT_X1>
__global__ __aicore__ void less_equal(
    GM_ADDR x1,
    GM_ADDR x2,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    // 注册Tiling数据结构类型
    REGISTER_TILING_DEFAULT(
        LessEqualTilingData);

    // 从GM加载Tiling配置数据
    GET_TILING_DATA_WITH_STRUCT(
        LessEqualTilingData,
        tilingData,
        tiling);

    // 创建算子实例
    KernelLessEqual<DT_X1> op;

    // 初始化算子（配置参数和UB缓冲区）
    op.Init(
        x1,
        x2,
        y,
        tilingData);

    // 执行核心计算
    op.Process();
}