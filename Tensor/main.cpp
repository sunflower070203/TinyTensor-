#include "tensor.hpp"

#include <iostream>
#include <vector>

using autodiff::Linear;
using autodiff::SGD;
using autodiff::Tensor;

static void train_linear_regression() {
    auto w = Tensor::scalar(0.0);
    auto b = Tensor::scalar(0.0);
    SGD optim({w, b}, 0.05);
    auto x = Tensor::from({{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}});
    auto y = Tensor::from({{-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}});

    for (int epoch = 0; epoch < 200; ++epoch) {
        auto loss = autodiff::mse_loss(x * w + b, y);
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    std::cout << "Linear regression y = wx + b\n";
    std::cout << "  w ~= " << w.item() << ", b ~= " << b.item() << "\n\n";
}

static void train_xor() {
    autodiff::set_seed(7);
    auto l1 = Linear(2, 8);
    auto l2 = Linear(8, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.5);

    auto x = Tensor::from({{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}});
    auto y = Tensor::from({{0.0}, {1.0}, {1.0}, {0.0}});

    for (int epoch = 0; epoch < 2500; ++epoch) {
        auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
        auto loss = autodiff::mse_loss(pred, y);
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
    std::cout << "XOR predictions\n";
    for (size_t i = 0; i < x.rows(); ++i) {
        std::cout << "  " << x.at(i, 0) << " xor " << x.at(i, 1) << " -> " << pred.at(i, 0) << "\n";
    }
    std::cout << "\n";
}

static void train_circle_classifier() {
    autodiff::set_seed(11);
    auto l1 = Linear(2, 12);
    auto l2 = Linear(12, 1);
    auto params = l1.parameters();
    auto p2 = l2.parameters();
    params.insert(params.end(), p2.begin(), p2.end());
    SGD optim(params, 0.35);

    std::vector<std::vector<double>> xs;
    std::vector<std::vector<double>> ys;
    for (double x = -1.0; x <= 1.001; x += 0.25) {
        for (double y = -1.0; y <= 1.001; y += 0.25) {
            double r2 = x * x + y * y;
            xs.push_back({x, y});
            ys.push_back({r2 < 0.35 ? 1.0 : 0.0});
        }
    }
    auto x = Tensor::from(xs);
    auto y = Tensor::from(ys);

    for (int epoch = 0; epoch < 1800; ++epoch) {
        auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
        auto loss = autodiff::mse_loss(pred, y);
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
    int correct = 0;
    for (size_t i = 0; i < xs.size(); ++i) {
        int got = pred.at(i, 0) > 0.5 ? 1 : 0;
        int want = ys[i][0] > 0.5 ? 1 : 0;
        correct += (got == want);
    }
    std::cout << "Concentric circle classifier accuracy: "
              << static_cast<double>(correct) / xs.size() << "\n";
}

int main() {
    train_linear_regression();
    train_xor();
    train_circle_classifier();
}
