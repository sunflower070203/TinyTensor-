#include "tensor.hpp"

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using autodiff::Linear;
using autodiff::SGD;
using autodiff::Tensor;

static void print_help() {
    std::cout
        << "Commands:\n"
        << "  scalar <name> <value>\n"
        << "  matrix <name> <rows> <cols> <values...>\n"
        << "  print <name>\n"
        << "  grad <name>\n"
        << "  add|sub|mul|div <out> <a> <b>\n"
        << "  pow <out> <a> <exponent>\n"
        << "  matmul <out> <a> <b>\n"
        << "  transpose|relu|sigmoid|tanh|softmax|sum|mean <out> <a>\n"
        << "  backward <loss>\n"
        << "  zero <name>\n"
        << "  zero_all\n"
        << "  step <learning_rate> <param_count> <param1> ... <paramN>\n"
        << "  fit_linear <epochs> <learning_rate> <n> <x1> <y1> ... <xn> <yn>\n"
        << "  xor\n"
        << "  circle\n"
        << "  list\n"
        << "  help\n"
        << "  quit\n";
}

static bool has_var(const std::map<std::string, Tensor>& vars, const std::string& name) {
    if (vars.count(name)) return true;
    std::cout << "unknown tensor: " << name << "\n";
    return false;
}

static void print_tensor(const Tensor& t, bool gradients) {
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < t.rows(); ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < t.cols(); ++j) {
            std::cout << (gradients ? t.grad_at(i, j) : t.at(i, j)) << " ";
        }
        std::cout << "]\n";
    }
}

static void run_fit_linear(std::istringstream& in) {
    int epochs = 0;
    double lr = 0.0;
    int n = 0;
    in >> epochs >> lr >> n;
    if (!in || epochs <= 0 || n <= 0) {
        std::cout << "usage: fit_linear <epochs> <learning_rate> <n> <x1> <y1> ... <xn> <yn>\n";
        return;
    }

    std::vector<std::vector<double>> xs;
    std::vector<std::vector<double>> ys;
    for (int i = 0; i < n; ++i) {
        double x = 0.0;
        double y = 0.0;
        in >> x >> y;
        if (!in) {
            std::cout << "not enough x/y values\n";
            return;
        }
        xs.push_back({x});
        ys.push_back({y});
    }

    auto w = Tensor::scalar(0.0);
    auto b = Tensor::scalar(0.0);
    SGD optim({w, b}, lr);
    auto x = Tensor::from(xs);
    auto y = Tensor::from(ys);
    Tensor loss;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto pred = x * w + b;
        loss = autodiff::mse_loss(pred, y);
        optim.zero_grad();
        loss.backward();
        optim.step();
    }

    std::cout << "trained linear model: w=" << w.item() << ", b=" << b.item()
              << ", loss=" << loss.item() << "\n";
}

static void run_xor_demo() {
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
    for (size_t i = 0; i < x.rows(); ++i) {
        std::cout << x.at(i, 0) << " xor " << x.at(i, 1) << " -> " << pred.at(i, 0) << "\n";
    }
}

static void run_circle_demo() {
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
    std::cout << "circle accuracy: " << static_cast<double>(correct) / xs.size() << "\n";
}

int main() {
    std::map<std::string, Tensor> vars;
    std::cout << "Tensor autodiff interactive shell. Type 'help' for commands.\n";

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string cmd;
        in >> cmd;
        if (cmd.empty()) continue;

        try {
            if (cmd == "quit" || cmd == "exit") {
                break;
            } else if (cmd == "help") {
                print_help();
            } else if (cmd == "scalar") {
                std::string name;
                double value = 0.0;
                in >> name >> value;
                if (!in) {
                    std::cout << "usage: scalar <name> <value>\n";
                } else {
                    vars[name] = Tensor::scalar(value);
                    std::cout << "created " << name << "\n";
                }
            } else if (cmd == "matrix") {
                std::string name;
                size_t rows = 0;
                size_t cols = 0;
                in >> name >> rows >> cols;
                std::vector<std::vector<double>> values(rows, std::vector<double>(cols));
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) in >> values[i][j];
                }
                if (!in || rows == 0 || cols == 0) {
                    std::cout << "usage: matrix <name> <rows> <cols> <values...>\n";
                } else {
                    vars[name] = Tensor::from(values);
                    std::cout << "created " << name << "\n";
                }
            } else if (cmd == "print" || cmd == "grad" || cmd == "zero") {
                std::string name;
                in >> name;
                if (!has_var(vars, name)) continue;
                if (cmd == "print") print_tensor(vars[name], false);
                if (cmd == "grad") print_tensor(vars[name], true);
                if (cmd == "zero") vars[name].zero_grad();
            } else if (cmd == "list") {
                for (const auto& [name, t] : vars) {
                    std::cout << name << " (" << t.rows() << "x" << t.cols() << ")\n";
                }
            } else if (cmd == "zero_all") {
                for (auto& [name, t] : vars) t.zero_grad();
            } else if (cmd == "add" || cmd == "sub" || cmd == "mul" || cmd == "div" || cmd == "matmul") {
                std::string out, a, b;
                in >> out >> a >> b;
                if (!has_var(vars, a) || !has_var(vars, b)) continue;
                if (cmd == "add") vars[out] = vars[a] + vars[b];
                if (cmd == "sub") vars[out] = vars[a] - vars[b];
                if (cmd == "mul") vars[out] = vars[a] * vars[b];
                if (cmd == "div") vars[out] = vars[a] / vars[b];
                if (cmd == "matmul") vars[out] = vars[a].matmul(vars[b]);
                std::cout << "created " << out << "\n";
            } else if (cmd == "pow") {
                std::string out, a;
                double exponent = 0.0;
                in >> out >> a >> exponent;
                if (!has_var(vars, a)) continue;
                vars[out] = vars[a].pow(exponent);
                std::cout << "created " << out << "\n";
            } else if (cmd == "transpose" || cmd == "relu" || cmd == "sigmoid" || cmd == "tanh" ||
                       cmd == "softmax" || cmd == "sum" || cmd == "mean") {
                std::string out, a;
                in >> out >> a;
                if (!has_var(vars, a)) continue;
                if (cmd == "transpose") vars[out] = vars[a].transpose();
                if (cmd == "relu") vars[out] = vars[a].relu();
                if (cmd == "sigmoid") vars[out] = vars[a].sigmoid();
                if (cmd == "tanh") vars[out] = vars[a].tanh();
                if (cmd == "softmax") vars[out] = vars[a].softmax();
                if (cmd == "sum") vars[out] = vars[a].sum();
                if (cmd == "mean") vars[out] = vars[a].mean();
                std::cout << "created " << out << "\n";
            } else if (cmd == "backward") {
                std::string loss;
                in >> loss;
                if (!has_var(vars, loss)) continue;
                vars[loss].backward();
                std::cout << "backward finished\n";
            } else if (cmd == "step") {
                double lr = 0.0;
                int count = 0;
                in >> lr >> count;
                std::vector<Tensor> params;
                for (int i = 0; i < count; ++i) {
                    std::string name;
                    in >> name;
                    if (!has_var(vars, name)) {
                        params.clear();
                        break;
                    }
                    params.push_back(vars[name]);
                }
                if (params.empty() || count <= 0) {
                    std::cout << "usage: step <learning_rate> <param_count> <param1> ... <paramN>\n";
                } else {
                    SGD(params, lr).step();
                    std::cout << "parameters updated\n";
                }
            } else if (cmd == "fit_linear") {
                run_fit_linear(in);
            } else if (cmd == "xor") {
                run_xor_demo();
            } else if (cmd == "circle") {
                run_circle_demo();
            } else {
                std::cout << "unknown command. Type 'help'.\n";
            }
        } catch (const std::exception& e) {
            std::cout << "error: " << e.what() << "\n";
        }
    }
}
