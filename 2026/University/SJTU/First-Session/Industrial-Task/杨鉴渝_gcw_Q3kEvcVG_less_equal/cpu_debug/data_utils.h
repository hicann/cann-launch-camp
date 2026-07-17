/**
 * @file data_utils.h
 * @brief 数据读写工具函数 (CPU/NPU 通用)
 */

#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ---- 日志宏 ----
#define INFO_LOG(fmt, args...)  fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...)  fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

// ---- 文件读写 ----
static inline bool ReadBinaryFile(
    const std::string &path, void *buffer, size_t bufferSize,
    size_t &bytesRead)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        ERROR_LOG("Cannot open file: %s", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    if (size > bufferSize) {
        ERROR_LOG("File size %zu > buffer size %zu", size, bufferSize);
        return false;
    }
    f.read((char*)buffer, size);
    bytesRead = f.gcount();
    f.close();
f.read((char*)buffer, size);
    bytesRead = f.gcount();
    if (f.fail() || bytesRead != size) {
        ERROR_LOG("Read incomplete: expected %zu bytes, got %zu", size, bytesRead);
        f.close();
        return false;
    }
    f.close();
    return true;
}

static inline bool WriteBinaryFile(
    const std::string &path, const void *buffer, size_t size)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        ERROR_LOG("Cannot open file for write: %s", path.c_str());
        return false;
    }
    f.write((const char*)buffer, size);
    f.close();
    return true;
}

// ---- 打印数组 ----
template<typename T>
static void PrintData(const T *data, size_t count, size_t perRow = 10)
{
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(8) << +data[i];  // + for uint8_t promotion
        if ((i + 1) % perRow == 0) std::cout << "\n";
    }
    if (count % perRow != 0) std::cout << "\n";
}

#endif // DATA_UTILS_H
