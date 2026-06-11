#pragma once
// Win32 实时可视化窗口 —— 零依赖，纯 Windows API

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace viz {

using Pixel = std::array<uint8_t, 3>;

// ───────── 颜色工具 ─────────

inline Pixel heatmap_color(double value) {
    value = value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
    auto lerp = [](uint8_t a, uint8_t b, double t) -> uint8_t {
        return static_cast<uint8_t>(a + (b - a) * t);
    };
    if (value < 0.5) {
        double t = value * 2.0;
        return {lerp(59, 255, t), lerp(130, 255, t), lerp(246, 255, t)};
    }
    double t = (value - 0.5) * 2.0;
    return {lerp(255, 239, t), lerp(255, 68, t), lerp(255, 68, t)};
}

// ───────── Win32 实时可视化窗口 ─────────

class LiveWindow {
public:
    LiveWindow() = default;
    ~LiveWindow() {
        if (mem_dc_) {
            SelectObject(mem_dc_, old_bmp_);
            DeleteObject(bmp_);
            DeleteDC(mem_dc_);
        }
        if (hwnd_) DestroyWindow(hwnd_);
    }

    LiveWindow(const LiveWindow&) = delete;
    LiveWindow& operator=(const LiveWindow&) = delete;

    bool create(const std::string& title, int w, int h) {
        width_ = w; height_ = h;

        WNDCLASSEXA wc{};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = wnd_proc;
        wc.hInstance = GetModuleHandleA(nullptr);
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = "LiveVizWindow";
        RegisterClassExA(&wc);

        // 计算窗口大小（使客户区恰好 w×h）
        RECT rc = {0, 0, w, h};
        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

        hwnd_ = CreateWindowExA(
            0, "LiveVizWindow", title.c_str(),
            WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rc.right - rc.left, rc.bottom - rc.top,
            nullptr, nullptr, GetModuleHandleA(nullptr), this);

        if (!hwnd_) return false;

        // 创建 DIB section（像素缓冲）
        HDC hdc = GetDC(hwnd_);
        mem_dc_ = CreateCompatibleDC(hdc);

        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = w;
        bi.bmiHeader.biHeight = -h;  // 顶行在前
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        bmp_ = CreateDIBSection(mem_dc_, &bi, DIB_RGB_COLORS, &bits_, nullptr, 0);
        old_bmp_ = (HBITMAP)SelectObject(mem_dc_, bmp_);

        ReleaseDC(hwnd_, hdc);
        ShowWindow(hwnd_, SW_SHOW);
        process_messages();
        return true;
    }

    // 直接访问像素缓冲（BGRA 格式，每像素 4 字节）
    uint32_t* pixels() { return static_cast<uint32_t*>(bits_); }
    int width() const { return width_; }
    int height() const { return height_; }

    // 设置单个像素的颜色
    void set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b) {
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            pixels()[y * width_ + x] = (b << 16) | (g << 8) | r;  // BGRA
        }
    }

    // 用热力图颜色设置像素
    void set_heatmap(int x, int y, double value) {
        auto [r, g, b] = heatmap_color(value);
        set_pixel(x, y, r, g, b);
    }

    // 填充整个窗口为纯色
    void fill(uint8_t r, uint8_t g, uint8_t b) {
        uint32_t c = (b << 16) | (g << 8) | r;
        auto* p = pixels();
        for (int i = 0; i < width_ * height_; ++i) p[i] = c;
    }

    // 画实心圆
    void fill_circle(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b) {
        for (int dy = -radius; dy <= radius; ++dy)
            for (int dx = -radius; dx <= radius; ++dx)
                if (dx * dx + dy * dy <= radius * radius)
                    set_pixel(cx + dx, cy + dy, r, g, b);
    }

    // 画圆环（边框）
    void draw_ring(int cx, int cy, int radius, int thickness,
                   uint8_t r, uint8_t g, uint8_t b) {
        int inner2 = (radius - thickness) * (radius - thickness);
        int outer2 = (radius + thickness) * (radius + thickness);
        for (int dy = -radius - thickness; dy <= radius + thickness; ++dy)
            for (int dx = -radius - thickness; dx <= radius + thickness; ++dx) {
                int d2 = dx * dx + dy * dy;
                if (d2 >= inner2 && d2 <= outer2)
                    set_pixel(cx + dx, cy + dy, r, g, b);
            }
    }

    // 画折线（用于 loss 曲线）
    void draw_line(int x0, int y0, int x1, int y1,
                   uint8_t r, uint8_t g, uint8_t b, int thickness = 1) {
        int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true) {
            for (int t = -thickness / 2; t <= thickness / 2; ++t) {
                set_pixel(x0, y0 + t, r, g, b);
                set_pixel(x0 + t, y0, r, g, b);
            }
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    // 画矩形框
    void draw_rect(int x0, int y0, int x1, int y1,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (int x = x0; x <= x1; ++x) { set_pixel(x, y0, r, g, b); set_pixel(x, y1, r, g, b); }
        for (int y = y0; y <= y1; ++y) { set_pixel(x0, y, r, g, b); set_pixel(x1, y, r, g, b); }
    }

    // 填充矩形
    void fill_rect(int x0, int y0, int x1, int y1,
                   uint8_t r, uint8_t g, uint8_t b) {
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x)
                set_pixel(x, y, r, g, b);
    }

    // 用 Windows GDI 绘制文字（使用系统字体）
    void draw_text(int x, int y, const std::string& text,
                   uint8_t r, uint8_t g, uint8_t b, int size = 16) {
        if (!hwnd_) return;
        HDC hdc = GetDC(hwnd_);
        HFONT font = CreateFontA(-size, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                 DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                 CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 FIXED_PITCH | FF_MODERN, "Consolas");
        HFONT old = (HFONT)SelectObject(hdc, font);
        SetTextColor(hdc, RGB(r, g, b));
        SetBkMode(hdc, TRANSPARENT);
        TextOutA(hdc, x, y, text.c_str(), static_cast<int>(text.size()));
        SelectObject(hdc, old);
        DeleteObject(font);
        ReleaseDC(hwnd_, hdc);
    }

    // 刷新窗口显示
    void refresh() {
        if (!hwnd_) return;
        InvalidateRect(hwnd_, nullptr, FALSE);
        UpdateWindow(hwnd_);
        process_messages();
    }

    // 处理窗口消息（非阻塞）
    void process_messages() {
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                hwnd_ = nullptr;
                return;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    // 让出 CPU 时间 + 处理窗口消息，防止界面卡死
    void yield() {
        process_messages();
        Sleep(0);  // 让出当前时间片，不消耗 CPU
    }

    bool is_open() const { return hwnd_ != nullptr; }

private:
    static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        if (msg == WM_CREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCT*>(lp);
            SetWindowLongPtrA(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
            return 0;
        }
        auto* self = reinterpret_cast<LiveWindow*>(
            GetWindowLongPtrA(hwnd, GWLP_USERDATA));
        switch (msg) {
        case WM_PAINT: {
            if (!self || !self->bits_) break;
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            BitBlt(hdc, 0, 0, self->width_, self->height_,
                   self->mem_dc_, 0, 0, SRCCOPY);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_DESTROY:
            if (self) self->hwnd_ = nullptr;
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcA(hwnd, msg, wp, lp);
    }

    HWND hwnd_ = nullptr;
    HDC mem_dc_ = nullptr;
    HBITMAP bmp_ = nullptr;
    HBITMAP old_bmp_ = nullptr;
    void* bits_ = nullptr;
    int width_ = 0, height_ = 0;
};

}  // namespace viz
