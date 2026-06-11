#include "tensor.hpp"
#include "plot.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using autodiff::Linear;
using autodiff::SGD;
using autodiff::Tensor;

// ───────────────────────── 数据结构 ─────────────────────────

struct TrainingRecord {
    int epoch;
    double loss;
};

// ───────────────────────── 工具函数 ─────────────────────────

static void ensure_output_dir() {
    std::filesystem::create_directories("output");
}

static void save_loss_csv(const std::string& filename,
                          const std::vector<TrainingRecord>& records) {
    std::ofstream f(filename);
    f << "epoch,loss\n";
    for (auto& r : records)
        f << r.epoch << "," << r.loss << "\n";
}

static void save_data_csv(const std::string& filename,
                          const std::vector<viz::DataPoint>& data) {
    std::ofstream f(filename);
    f << "x1,x2,label\n";
    for (auto& d : data)
        f << d.x << "," << d.y << "," << d.label << "\n";
}

// ───────────────────────── XOR 训练 ─────────────────────────

static void run_xor() {
    std::cout << "=== XOR 分类 ===\n";
    autodiff::set_seed(7);

    auto l1 = Linear(2, 8);
    auto l2 = Linear(8, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.5);

    auto x = Tensor::from({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}});
    auto y = Tensor::from({{0.0}, {1.0}, {1.0}, {0.0}});

    std::vector<TrainingRecord> loss_history;
    for (int epoch = 0; epoch < 2500; ++epoch) {
        auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
        auto loss = autodiff::mse_loss(pred, y);
        optim.zero_grad();
        loss.backward();
        optim.step();
        if (epoch % 10 == 0)
            loss_history.push_back({epoch, loss.item()});
    }

    // 打印结果
    auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
    for (size_t i = 0; i < x.rows(); ++i) {
        std::cout << "  [" << x.at(i, 0) << ", " << x.at(i, 1) << "] -> "
                  << pred.at(i, 0) << "\n";
    }

    // 模型函数
    auto model = [&l1, &l2](double x1, double x2) -> double {
        auto input = Tensor::from({{x1, x2}});
        return l2.forward(l1.forward(input).tanh()).sigmoid().at(0, 0);
    };

    // 训练数据
    std::vector<viz::DataPoint> data = {
        {0, 0, 0}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}
    };

    // 导出
    save_loss_csv("output/xor_loss.csv", loss_history);
    save_data_csv("output/xor_data.csv", data);

    viz::plot_classification(model, data, -0.5, 1.5, -0.5, 1.5,
                             400, 400,
                             "output/xor_boundary.ppm",
                             "output/xor_boundary.bmp");

    // loss 曲线
    std::vector<double> losses;
    for (auto& r : loss_history) losses.push_back(r.loss);
    viz::plot_loss_curve(losses, 600, 300,
                         "output/xor_loss_curve.ppm",
                         "output/xor_loss_curve.bmp");

    std::cout << "  -> output/xor_boundary.bmp, xor_loss_curve.bmp\n\n";
}

// ───────────────────────── 同心圆训练 ─────────────────────────

static void run_circle() {
    std::cout << "=== 同心圆分类 ===\n";
    autodiff::set_seed(11);

    auto l1 = Linear(2, 12);
    auto l2 = Linear(12, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.35);

    // 生成训练数据
    std::vector<viz::DataPoint> data;
    std::vector<std::vector<double>> xs;
    std::vector<std::vector<double>> ys;
    for (double x = -1.0; x <= 1.001; x += 0.1) {
        for (double y = -1.0; y <= 1.001; y += 0.1) {
            double r2 = x * x + y * y;
            int label = r2 < 0.35 ? 1 : 0;
            data.push_back({x, y, label});
            xs.push_back({x, y});
            ys.push_back({static_cast<double>(label)});
        }
    }
    auto xt = Tensor::from(xs);
    auto yt = Tensor::from(ys);

    // 训练
    std::vector<TrainingRecord> loss_history;
    for (int epoch = 0; epoch < 1800; ++epoch) {
        auto pred = l2.forward(l1.forward(xt).tanh()).sigmoid();
        auto loss = autodiff::mse_loss(pred, yt);
        optim.zero_grad();
        loss.backward();
        optim.step();
        if (epoch % 10 == 0)
            loss_history.push_back({epoch, loss.item()});
    }

    // 准确率
    auto pred = l2.forward(l1.forward(xt).tanh()).sigmoid();
    int correct = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        int got = pred.at(i, 0) > 0.5 ? 1 : 0;
        correct += (got == data[i].label);
    }
    std::cout << "  准确率: " << 100.0 * correct / data.size() << "%\n";

    // 模型函数
    auto model = [&l1, &l2](double x1, double x2) -> double {
        auto input = Tensor::from({{x1, x2}});
        return l2.forward(l1.forward(input).tanh()).sigmoid().at(0, 0);
    };

    // 导出
    save_loss_csv("output/circle_loss.csv", loss_history);
    save_data_csv("output/circle_data.csv", data);

    viz::plot_classification(model, data, -1.5, 1.5, -1.5, 1.5,
                             400, 400,
                             "output/circle_boundary.ppm",
                             "output/circle_boundary.bmp");

    std::vector<double> losses;
    for (auto& r : loss_history) losses.push_back(r.loss);
    viz::plot_loss_curve(losses, 600, 300,
                         "output/circle_loss_curve.ppm",
                         "output/circle_loss_curve.bmp");

    std::cout << "  -> output/circle_boundary.bmp, circle_loss_curve.bmp\n\n";
}

// ───────────────────────── 主函数 ─────────────────────────

int main() {
    ensure_output_dir();

    std::cout << "====================================\n";
    std::cout << "  自动求导系统 - 训练与可视化导出\n";
    std::cout << "====================================\n\n";

    run_xor();
    run_circle();

    std::cout << "====================================\n";
    std::cout << "  完成! 用画图打开 output/*.bmp 查看\n";
    std::cout << "====================================\n";

    return 0;
}
