#pragma once

#include "image_writer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace viz {

// ───────── 颜色工具 ─────────

inline Pixel color_lerp(Pixel a, Pixel b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    return {static_cast<uint8_t>(a[0] + (b[0] - a[0]) * t),
            static_cast<uint8_t>(a[1] + (b[1] - a[1]) * t),
            static_cast<uint8_t>(a[2] + (b[2] - a[2]) * t)};
}

// 蓝(0) → 白(0.5) → 红(1)
inline Pixel heatmap_color(double value) {
    value = std::clamp(value, 0.0, 1.0);
    constexpr Pixel blue  = {59, 130, 246};
    constexpr Pixel white = {255, 255, 255};
    constexpr Pixel red   = {239, 68, 68};
    if (value < 0.5) return color_lerp(blue, white, value * 2.0);
    return color_lerp(white, red, (value - 0.5) * 2.0);
}

// ───────── 坐标 → 像素映射 ─────────

struct CoordMapper {
    double x_min, x_max, y_min, y_max;
    int width, height;

    // 数学坐标 → 像素坐标 (注意 y 轴翻转)
    int to_pixel_x(double x) const {
        return static_cast<int>((x - x_min) / (x_max - x_min) * (width - 1));
    }
    int to_pixel_y(double y) const {
        // y 轴翻转：数学 y 向上，图片 y 向下
        return height - 1 - static_cast<int>((y - y_min) / (y_max - y_min) * (height - 1));
    }
};

// ───────── 决策边界热力图 ─────────
// model_fn: 输入 (x,y)，返回 [0,1] 的预测值

inline void plot_decision_boundary(
    const std::function<double(double, double)>& model_fn,
    double x_min, double x_max, double y_min, double y_max,
    int width, int height,
    const std::string& ppm_file, const std::string& bmp_file)
{
    std::vector<Pixel> pixels(width * height);

    for (int py = 0; py < height; ++py) {
        double y = y_min + (y_max - y_min) * (height - 1 - py) / (height - 1);
        for (int px = 0; px < width; ++px) {
            double x = x_min + (x_max - x_min) * px / (width - 1);
            double pred = model_fn(x, y);
            pixels[py * width + px] = heatmap_color(pred);
        }
    }

    write_ppm(ppm_file, width, height, pixels);
    write_bmp(bmp_file, width, height, pixels);
}

// ───────── 在已有图片上叠加散点 ─────────

inline void draw_point(std::vector<Pixel>& pixels, int width, int height,
                       int cx, int cy, int radius, Pixel color) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius) continue;
            int px = cx + dx;
            int py = cy + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                pixels[py * width + px] = color;
            }
        }
    }
}

inline void draw_ring(std::vector<Pixel>& pixels, int width, int height,
                      int cx, int cy, int radius, int thickness, Pixel color) {
    for (int dy = -radius - thickness; dy <= radius + thickness; ++dy) {
        for (int dx = -radius - thickness; dx <= radius + thickness; ++dx) {
            int dist2 = dx * dx + dy * dy;
            if (dist2 < (radius - thickness) * (radius - thickness)) continue;
            if (dist2 > (radius + thickness) * (radius + thickness)) continue;
            int px = cx + dx;
            int py = cy + dy;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                pixels[py * width + px] = color;
            }
        }
    }
}

// ───────── 完整流程：热力图 + 数据散点 + 导出 ─────────

struct DataPoint {
    double x, y;
    int label;  // 0 或 1
};

inline void plot_classification(
    const std::function<double(double, double)>& model_fn,
    const std::vector<DataPoint>& data,
    double x_min, double x_max, double y_min, double y_max,
    int width, int height,
    const std::string& ppm_file, const std::string& bmp_file)
{
    // 1) 生成热力图底图
    std::vector<Pixel> pixels(width * height);
    for (int py = 0; py < height; ++py) {
        double y = y_min + (y_max - y_min) * (height - 1 - py) / (height - 1);
        for (int px = 0; px < width; ++px) {
            double x = x_min + (x_max - x_min) * px / (width - 1);
            double pred = model_fn(x, y);
            pixels[py * width + px] = heatmap_color(pred);
        }
    }

    // 2) 叠加训练数据散点
    CoordMapper mapper{x_min, x_max, y_min, y_max, width, height};
    constexpr Pixel white = {255, 255, 255};
    constexpr Pixel dark_blue = {30, 64, 175};
    constexpr Pixel dark_red = {185, 28, 28};

    for (const auto& d : data) {
        int cx = mapper.to_pixel_x(d.x);
        int cy = mapper.to_pixel_y(d.y);
        Pixel dot_color = (d.label == 0) ? dark_blue : dark_red;
        draw_ring(pixels, width, height, cx, cy, 4, 2, white);   // 白色边框
        draw_point(pixels, width, height, cx, cy, 3, dot_color); // 彩色填充
    }

    // 3) 导出
    write_ppm(ppm_file, width, height, pixels);
    write_bmp(bmp_file, width, height, pixels);
}

// ───────── Loss 曲线图 ─────────

inline void plot_loss_curve(
    const std::vector<double>& losses,
    int width, int height,
    const std::string& ppm_file, const std::string& bmp_file)
{
    constexpr Pixel bg = {255, 255, 255};
    constexpr Pixel line_color = {59, 130, 246};
    constexpr Pixel grid_color = {230, 230, 230};
    constexpr Pixel axis_color = {100, 100, 100};

    std::vector<Pixel> pixels(width * height, bg);

    if (losses.empty()) {
        write_ppm(ppm_file, width, height, pixels);
        write_bmp(bmp_file, width, height, pixels);
        return;
    }

    // 绘图区域留边距
    int margin_l = 60, margin_r = 20, margin_t = 20, margin_b = 40;
    int plot_w = width - margin_l - margin_r;
    int plot_h = height - margin_t - margin_b;

    double max_loss = *std::max_element(losses.begin(), losses.end());
    if (max_loss < 1e-10) max_loss = 1.0;

    auto set_pixel = [&](int x, int y, Pixel c) {
        if (x >= 0 && x < width && y >= 0 && y < height)
            pixels[y * width + x] = c;
    };

    // 画水平网格线
    for (int i = 0; i <= 4; ++i) {
        int y = margin_t + plot_h * i / 4;
        for (int x = margin_l; x < margin_l + plot_w; ++x)
            set_pixel(x, y, grid_color);
    }

    // 画坐标轴
    for (int x = margin_l; x < margin_l + plot_w; ++x)
        set_pixel(x, margin_t + plot_h, axis_color);
    for (int y = margin_t; y < margin_t + plot_h; ++y)
        set_pixel(margin_l, y, axis_color);

    // 画 loss 折线
    int n = static_cast<int>(losses.size());
    auto to_px = [&](int i) { return margin_l + i * plot_w / (n - 1); };
    auto to_py = [&](double v) { return margin_t + plot_h - static_cast<int>(v / max_loss * plot_h); };

    int prev_x = to_px(0), prev_y = to_py(losses[0]);
    for (int i = 1; i < n; ++i) {
        int cx = to_px(i), cy = to_py(losses[i]);
        // 简单线段绘制 (Bresenham)
        int dx = std::abs(cx - prev_x), sx = prev_x < cx ? 1 : -1;
        int dy = -std::abs(cy - prev_y), sy = prev_y < cy ? 1 : -1;
        int err = dx + dy;
        int x = prev_x, y = prev_y;
        while (true) {
            set_pixel(x, y, line_color);
            if (x == cx && y == cy) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x += sx; }
            if (e2 <= dx) { err += dx; y += sy; }
        }
        prev_x = cx;
        prev_y = cy;
    }

    write_ppm(ppm_file, width, height, pixels);
    write_bmp(bmp_file, width, height, pixels);
}

}  // namespace viz
