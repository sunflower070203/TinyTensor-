#include "tensor.hpp"
#include "live_window.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using autodiff::Linear;
using autodiff::SGD;
using autodiff::Tensor;

// ───────── 格式化工具 ─────────

static std::string fmt(const char* label, double value, int prec = 4) {
    std::ostringstream ss;
    ss << label << std::fixed << std::setprecision(prec) << value;
    return ss.str();
}

static std::string fmt_int(const char* label, int value) {
    return label + std::to_string(value);
}

// ───────── 决策边界绘制（带 yield，每画几行让出 CPU） ─────────

static void render_boundary(viz::LiveWindow& win,
                            const std::function<double(double, double)>& model,
                            double x_min, double x_max,
                            double y_min, double y_max,
                            int ox, int oy, int size) {
    for (int py = 0; py < size; ++py) {
        if (!win.is_open()) return;
        double y = y_max - (y_max - y_min) * py / (size - 1);
        for (int px = 0; px < size; ++px) {
            double x = x_min + (x_max - x_min) * px / (size - 1);
            double pred = model(x, y);
            win.set_heatmap(ox + px, oy + py, pred);
        }
        // 每 20 行让出一次 CPU，保持窗口响应
        if (py % 20 == 0) win.yield();
    }
}

// ───────── Loss 曲线绘制 ─────────

static void render_loss_curve(viz::LiveWindow& win,
                              const std::vector<double>& losses,
                              int ox, int oy, int w, int h,
                              uint8_t lr, uint8_t lg, uint8_t lb) {
    if (losses.size() < 2) return;

    win.fill_rect(ox, oy, ox + w, oy + h, 30, 30, 40);

    double max_loss = 0;
    for (double l : losses) if (l > max_loss) max_loss = l;
    if (max_loss < 1e-10) max_loss = 1.0;

    int n = static_cast<int>(losses.size());
    auto to_x = [&](int i) { return ox + i * w / (n - 1); };
    auto to_y = [&](double v) { return oy + h - static_cast<int>(v / max_loss * h); };

    int prev_x = to_x(0), prev_y = to_y(losses[0]);
    for (int i = 1; i < n; ++i) {
        int cx = to_x(i), cy = to_y(losses[i]);
        win.draw_line(prev_x, prev_y, cx, cy, lr, lg, lb, 2);
        prev_x = cx;
        prev_y = cy;
    }
    win.draw_rect(ox, oy, ox + w, oy + h, 100, 100, 120);
}

// ───────── 散点绘制 ─────────

struct Pt { double x, y; int label; };

static void render_data_points(viz::LiveWindow& win,
                               const std::vector<Pt>& data,
                               double x_min, double x_max,
                               double y_min, double y_max,
                               int ox, int oy, int size) {
    for (const auto& d : data) {
        int px = ox + static_cast<int>((d.x - x_min) / (x_max - x_min) * (size - 1));
        int py = oy + static_cast<int>((y_max - d.y) / (y_max - y_min) * (size - 1));
        win.draw_ring(px, py, 5, 2, 255, 255, 255);
        if (d.label == 0)
            win.fill_circle(px, py, 4, 30, 64, 175);
        else
            win.fill_circle(px, py, 4, 185, 28, 28);
    }
}

// ───────── 绘制信息面板 ─────────

static void render_info(viz::LiveWindow& win,
                        const char* title, int epoch, int total_epochs,
                        double loss, double accuracy,
                        int ox, int oy) {
    win.fill_rect(ox, oy, ox + 280, oy + 90, 25, 25, 35);
    win.draw_rect(ox, oy, ox + 280, oy + 90, 80, 80, 100);
    win.draw_text(ox + 10, oy + 6, title, 255, 255, 255, 18);
    win.draw_text(ox + 10, oy + 28,
                  fmt_int("Epoch: ", epoch) + " / " + std::to_string(total_epochs),
                  200, 200, 220, 14);
    win.draw_text(ox + 10, oy + 46, fmt("Loss: ", loss, 6), 200, 200, 220, 14);
    if (accuracy >= 0) {
        win.draw_text(ox + 10, oy + 64, fmt("Acc: ", accuracy * 100, 1) + "%",
                      100, 255, 100, 14);
    }
}

// ───────── XOR 实时训练 ─────────

static void train_xor_live(viz::LiveWindow& win) {
    autodiff::set_seed(7);
    auto l1 = Linear(2, 8);
    auto l2 = Linear(8, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.5);

    auto x = Tensor::from({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}});
    auto y = Tensor::from({{0.0}, {1.0}, {1.0}, {0.0}});

    std::vector<Pt> data = {{0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}};
    std::vector<double> losses;

    auto model_fn = [&l1, &l2](double x1, double x2) -> double {
        auto input = Tensor::from({{x1, x2}});
        return l2.forward(l1.forward(input).tanh()).sigmoid().at(0, 0);
    };

    const int boundary_size = 200;  // 200×200 = 40000 点，比 500×500 快 6 倍
    const int epochs = 3000;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        if (!win.is_open()) return;

        // 训练一步
        if (epoch > 0) {
            auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
            auto loss = autodiff::mse_loss(pred, y);
            optim.zero_grad();
            loss.backward();
            optim.step();
        }

        // 每 50 epoch 刷新一次
        if (epoch % 50 == 0 || epoch == epochs) {
            auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
            double loss = autodiff::mse_loss(pred, y).item();
            losses.push_back(loss);

            int correct = 0;
            for (size_t i = 0; i < x.rows(); ++i) {
                int got = pred.at(i, 0) > 0.5 ? 1 : 0;
                correct += (got == data[i].label);
            }
            double acc = static_cast<double>(correct) / data.size();

            win.fill(20, 20, 30);
            render_boundary(win, model_fn, -0.5, 1.5, -0.5, 1.5,
                            40, 50, boundary_size);
            render_data_points(win, data, -0.5, 1.5, -0.5, 1.5,
                               40, 50, boundary_size);
            render_loss_curve(win, losses, 570, 300, 200, 150, 59, 130, 246);
            render_info(win, "XOR", epoch, epochs, loss, acc, 570, 80);
            win.refresh();
        }

        // 每 5 步让出 CPU，防止窗口无响应
        if (epoch % 5 == 0) win.yield();
    }

    while (win.is_open()) { win.yield(); Sleep(50); }
}

// ───────── 同心圆实时训练 ─────────

static void train_circle_live(viz::LiveWindow& win) {
    autodiff::set_seed(11);
    auto l1 = Linear(2, 12);
    auto l2 = Linear(12, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.35);

    std::vector<Pt> data;
    std::vector<std::vector<double>> xs;
    std::vector<std::vector<double>> ys;
    for (double xv = -1.0; xv <= 1.001; xv += 0.1) {
        for (double yv = -1.0; yv <= 1.001; yv += 0.1) {
            double r2 = xv * xv + yv * yv;
            int label = r2 < 0.35 ? 1 : 0;
            data.push_back({xv, yv, label});
            xs.push_back({xv, yv});
            ys.push_back({static_cast<double>(label)});
        }
    }
    auto xt = Tensor::from(xs);
    auto yt = Tensor::from(ys);

    std::vector<double> losses;

    auto model_fn = [&l1, &l2](double x1, double x2) -> double {
        auto input = Tensor::from({{x1, x2}});
        return l2.forward(l1.forward(input).tanh()).sigmoid().at(0, 0);
    };

    const int boundary_size = 200;
    const int epochs = 2500;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        if (!win.is_open()) return;

        if (epoch > 0) {
            auto pred = l2.forward(l1.forward(xt).tanh()).sigmoid();
            auto loss = autodiff::mse_loss(pred, yt);
            optim.zero_grad();
            loss.backward();
            optim.step();
        }

        if (epoch % 50 == 0 || epoch == epochs) {
            auto pred = l2.forward(l1.forward(xt).tanh()).sigmoid();
            double loss = autodiff::mse_loss(pred, yt).item();
            losses.push_back(loss);

            int correct = 0;
            for (size_t i = 0; i < xs.size(); ++i) {
                int got = pred.at(i, 0) > 0.5 ? 1 : 0;
                correct += (got == data[i].label);
            }
            double acc = static_cast<double>(correct) / data.size();

            win.fill(20, 20, 30);
            render_boundary(win, model_fn, -1.5, 1.5, -1.5, 1.5,
                            40, 50, boundary_size);
            render_data_points(win, data, -1.5, 1.5, -1.5, 1.5,
                               40, 50, boundary_size);
            render_loss_curve(win, losses, 570, 300, 200, 150, 59, 130, 246);
            render_info(win, "Circle", epoch, epochs, loss, acc, 570, 80);
            win.refresh();
        }

        if (epoch % 5 == 0) win.yield();
    }

    while (win.is_open()) { win.yield(); Sleep(50); }
}

// ───────── 主函数 ─────────

int main() {
    viz::LiveWindow win;

    std::cout << "启动 XOR 实时训练可视化...\n";
    if (!win.create("XOR - Live Training", 800, 600)) {
        std::cerr << "窗口创建失败\n";
        return 1;
    }
    train_xor_live(win);

    std::cout << "启动同心圆实时训练可视化...\n";
    viz::LiveWindow win2;
    if (!win2.create("Circle - Live Training", 800, 600)) {
        std::cerr << "窗口创建失败\n";
        return 1;
    }
    train_circle_live(win2);

    std::cout << "训练完成!\n";
    return 0;
}
