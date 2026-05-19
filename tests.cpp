#include "tensor.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using autodiff::Linear;
using autodiff::SGD;
using autodiff::Tensor;

static void expect_close(double actual, double expected, double eps = 1e-4) {
    if (std::fabs(actual - expected) > eps) {
        std::cerr << "expected " << expected << ", got " << actual << "\n";
        std::exit(1);
    }
}

static void test_scalar_autograd() {
    auto x = Tensor::scalar(2.0);
    auto y = Tensor::scalar(3.0);
    auto z = ((x * y) + x.pow(3.0)) / y;
    z.backward();

    expect_close(z.item(), (2.0 * 3.0 + 8.0) / 3.0);
    expect_close(x.grad_at(0, 0), (3.0 + 3.0 * 4.0) / 3.0);
    expect_close(y.grad_at(0, 0), -8.0 / 9.0);
}

static void test_matrix_broadcast_and_matmul() {
    auto x = Tensor::from({{1.0, 2.0}, {3.0, 4.0}});
    auto w = Tensor::from({{2.0}, {-1.0}});
    auto b = Tensor::from({{0.5}});
    auto y = (x.matmul(w) + b).sum();
    y.backward();

    expect_close(y.item(), (1.0 * 2.0 + 2.0 * -1.0 + 0.5) +
                             (3.0 * 2.0 + 4.0 * -1.0 + 0.5));
    expect_close(w.grad_at(0, 0), 4.0);
    expect_close(w.grad_at(1, 0), 6.0);
    expect_close(b.grad_at(0, 0), 2.0);
}

static void test_activations_and_softmax() {
    auto x = Tensor::from({{-1.0, 0.0, 2.0}});
    auto r = x.relu();
    auto s = x.sigmoid();
    auto t = x.tanh();
    auto p = x.softmax();

    expect_close(r.at(0, 0), 0.0);
    expect_close(r.at(0, 2), 2.0);
    expect_close(s.at(0, 1), 0.5);
    expect_close(t.at(0, 1), 0.0);
    expect_close(p.sum().item(), 1.0);
}

static void test_linear_regression_converges() {
    auto w = Tensor::scalar(0.0);
    auto b = Tensor::scalar(0.0);
    SGD optim({w, b}, 0.05);

    for (int epoch = 0; epoch < 200; ++epoch) {
        auto x = Tensor::from({{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}});
        auto y = Tensor::from({{-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}});
        auto pred = x * w + b;
        auto loss = (pred - y).pow(2.0).mean();
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    expect_close(w.item(), 2.0, 0.05);
    expect_close(b.item(), 1.0, 0.05);
}

static void test_xor_training() {
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
        auto loss = (pred - y).pow(2.0).mean();
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    auto pred = l2.forward(l1.forward(x).tanh()).sigmoid();
    assert(pred.at(0, 0) < 0.25);
    assert(pred.at(1, 0) > 0.75);
    assert(pred.at(2, 0) > 0.75);
    assert(pred.at(3, 0) < 0.25);
}

static void test_circle_training() {
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
        auto loss = (pred - y).pow(2.0).mean();
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
    double acc = static_cast<double>(correct) / xs.size();
    if (acc < 0.90) {
        std::cerr << "circle accuracy too low: " << acc << "\n";
        std::exit(1);
    }
}

int main() {
    test_scalar_autograd();
    test_matrix_broadcast_and_matmul();
    test_activations_and_softmax();
    test_linear_regression_converges();
    test_xor_training();
    test_circle_training();
    std::cout << "All Tensor autodiff tests passed.\n";
}
