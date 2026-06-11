#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace viz {

using Pixel = std::array<uint8_t, 3>;

// ───────── PPM (P6 二进制格式) ─────────
// 最简图片格式：文件头 + RGB 像素，任何图片查看器都能打开

inline void write_ppm(const std::string& filename, int width, int height,
                      const std::vector<Pixel>& pixels) {
    std::ofstream f(filename, std::ios::binary);
    f << "P6\n" << width << " " << height << "\n255\n";
    for (const auto& p : pixels) {
        f.put(static_cast<char>(p[0]));
        f.put(static_cast<char>(p[1]));
        f.put(static_cast<char>(p[2]));
    }
}

// ───────── BMP (24 位，Windows 原生) ─────────
// 双击即可用 Windows 画图打开

inline void write_bmp(const std::string& filename, int width, int height,
                      const std::vector<Pixel>& pixels) {
    int row_bytes = width * 3;
    int padding = (4 - row_bytes % 4) % 4;
    int padded_row = row_bytes + padding;
    int data_size = padded_row * height;
    int file_size = 54 + data_size;

    std::ofstream f(filename, std::ios::binary);

    // ── 14 字节文件头 ──
    f.put('B'); f.put('M');
    auto write32 = [&f](int v) {
        f.put(v & 0xFF); f.put((v >> 8) & 0xFF);
        f.put((v >> 16) & 0xFF); f.put((v >> 24) & 0xFF);
    };
    auto write16 = [&f](int v) {
        f.put(v & 0xFF); f.put((v >> 8) & 0xFF);
    };
    write32(file_size);       // 文件大小
    write16(0); write16(0);   // 保留
    write32(54);              // 像素数据偏移

    // ── 40 字节 DIB 头 (BITMAPINFOHEADER) ──
    write32(40);              // DIB 头大小
    write32(width);
    write32(height);
    write16(1);               // 颜色平面
    write16(24);              // 位深度
    write32(0);               // 无压缩
    write32(data_size);       // 像素数据大小
    write32(2835);            // 水平分辨率 (72 DPI)
    write32(2835);            // 垂直分辨率
    write32(0);               // 调色板颜色数
    write32(0);               // 重要颜色数

    // ── 像素数据 (BGR 顺序，底行在前) ──
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            const auto& p = pixels[y * width + x];
            f.put(static_cast<char>(p[2]));  // B
            f.put(static_cast<char>(p[1]));  // G
            f.put(static_cast<char>(p[0]));  // R
        }
        for (int i = 0; i < padding; ++i) f.put(0);
    }
}

}  // namespace viz
