#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace autodiff {

struct Node {
    size_t rows = 1;
    size_t cols = 1;
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<std::shared_ptr<Node>> prev;
    std::function<void()> backward = [] {};
};

class Tensor {
public:
    Tensor() : node_(std::make_shared<Node>()) {
        node_->data = {0.0};
        node_->grad = {0.0};
    }

    Tensor(size_t rows, size_t cols, double value = 0.0) : node_(std::make_shared<Node>()) {
        node_->rows = rows;
        node_->cols = cols;
        node_->data.assign(rows * cols, value);
        node_->grad.assign(rows * cols, 0.0);
    }

    static Tensor scalar(double value) {
        return Tensor(1, 1, value);
    }

    static Tensor from(const std::vector<std::vector<double>>& values) {
        if (values.empty() || values[0].empty()) {
            throw std::invalid_argument("Tensor::from needs a non-empty matrix");
        }
        Tensor out(values.size(), values[0].size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (values[i].size() != values[0].size()) {
                throw std::invalid_argument("ragged matrix is not supported");
            }
            for (size_t j = 0; j < values[0].size(); ++j) {
                out.node_->data[out.offset(i, j)] = values[i][j];
            }
        }
        return out;
    }

    size_t rows() const { return node_->rows; }
    size_t cols() const { return node_->cols; }
    size_t size() const { return node_->data.size(); }
    double item() const { return at(0, 0); }
    double at(size_t row, size_t col) const { return node_->data[offset(row, col)]; }
    double grad_at(size_t row, size_t col) const { return node_->grad[offset(row, col)]; }
    void set(size_t row, size_t col, double value) { node_->data[offset(row, col)] = value; }
    void zero_grad() { std::fill(node_->grad.begin(), node_->grad.end(), 0.0); }

    Tensor operator+(const Tensor& other) const {
        auto [r, c] = broadcast_shape(*this, other);
        Tensor out(r, c);
        auto a = node_;
        auto b = other.node_;
        out.node_->prev = {a, b};
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                out.node_->data[out.offset(i, j)] = value_at(a, i, j) + value_at(b, i, j);
            }
        }
        auto o = out.node_;
        out.node_->backward = [a, b, o] {
            add_broadcast_grad(a, o->grad, o->rows, o->cols);
            add_broadcast_grad(b, o->grad, o->rows, o->cols);
        };
        return out;
    }

    Tensor operator-() const {
        Tensor out(rows(), cols());
        auto a = node_;
        out.node_->prev = {a};
        for (size_t i = 0; i < size(); ++i) out.node_->data[i] = -a->data[i];
        auto o = out.node_;
        out.node_->backward = [a, o] {
            for (size_t i = 0; i < a->grad.size(); ++i) a->grad[i] -= o->grad[i];
        };
        return out;
    }

    Tensor operator-(const Tensor& other) const { return *this + (-other); }

    Tensor operator*(const Tensor& other) const {
        auto [r, c] = broadcast_shape(*this, other);
        Tensor out(r, c);
        auto a = node_;
        auto b = other.node_;
        out.node_->prev = {a, b};
        std::vector<double> aval(r * c), bval(r * c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                size_t k = i * c + j;
                aval[k] = value_at(a, i, j);
                bval[k] = value_at(b, i, j);
                out.node_->data[k] = aval[k] * bval[k];
            }
        }
        auto o = out.node_;
        out.node_->backward = [a, b, o, aval, bval] {
            std::vector<double> ga(o->grad.size()), gb(o->grad.size());
            for (size_t i = 0; i < o->grad.size(); ++i) {
                ga[i] = o->grad[i] * bval[i];
                gb[i] = o->grad[i] * aval[i];
            }
            add_broadcast_grad(a, ga, o->rows, o->cols);
            add_broadcast_grad(b, gb, o->rows, o->cols);
        };
        return out;
    }

    Tensor operator/(const Tensor& other) const {
        auto [r, c] = broadcast_shape(*this, other);
        Tensor out(r, c);
        auto a = node_;
        auto b = other.node_;
        out.node_->prev = {a, b};
        std::vector<double> aval(r * c), bval(r * c);
        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < c; ++j) {
                size_t k = i * c + j;
                aval[k] = value_at(a, i, j);
                bval[k] = value_at(b, i, j);
                out.node_->data[k] = aval[k] / bval[k];
            }
        }
        auto o = out.node_;
        out.node_->backward = [a, b, o, aval, bval] {
            std::vector<double> ga(o->grad.size()), gb(o->grad.size());
            for (size_t i = 0; i < o->grad.size(); ++i) {
                ga[i] = o->grad[i] / bval[i];
                gb[i] = -o->grad[i] * aval[i] / (bval[i] * bval[i]);
            }
            add_broadcast_grad(a, ga, o->rows, o->cols);
            add_broadcast_grad(b, gb, o->rows, o->cols);
        };
        return out;
    }

    Tensor pow(double exponent) const {
        Tensor out(rows(), cols());
        auto a = node_;
        out.node_->prev = {a};
        for (size_t i = 0; i < size(); ++i) {
            out.node_->data[i] = std::pow(a->data[i], exponent);
        }
        auto o = out.node_;
        out.node_->backward = [a, o, exponent] {
            for (size_t i = 0; i < a->data.size(); ++i) {
                a->grad[i] += o->grad[i] * exponent * std::pow(a->data[i], exponent - 1.0);
            }
        };
        return out;
    }

    Tensor matmul(const Tensor& other) const {
        if (cols() != other.rows()) throw std::invalid_argument("matmul shape mismatch");
        Tensor out(rows(), other.cols());
        auto a = node_;
        auto b = other.node_;
        out.node_->prev = {a, b};
        for (size_t i = 0; i < a->rows; ++i) {
            for (size_t j = 0; j < b->cols; ++j) {
                double v = 0.0;
                for (size_t k = 0; k < a->cols; ++k) v += a->data[i * a->cols + k] * b->data[k * b->cols + j];
                out.node_->data[i * out.cols() + j] = v;
            }
        }
        auto o = out.node_;
        out.node_->backward = [a, b, o] {
            for (size_t i = 0; i < a->rows; ++i) {
                for (size_t k = 0; k < a->cols; ++k) {
                    double g = 0.0;
                    for (size_t j = 0; j < b->cols; ++j) g += o->grad[i * o->cols + j] * b->data[k * b->cols + j];
                    a->grad[i * a->cols + k] += g;
                }
            }
            for (size_t k = 0; k < b->rows; ++k) {
                for (size_t j = 0; j < b->cols; ++j) {
                    double g = 0.0;
                    for (size_t i = 0; i < a->rows; ++i) g += a->data[i * a->cols + k] * o->grad[i * o->cols + j];
                    b->grad[k * b->cols + j] += g;
                }
            }
        };
        return out;
    }

    Tensor transpose() const {
        Tensor out(cols(), rows());
        auto a = node_;
        out.node_->prev = {a};
        for (size_t i = 0; i < rows(); ++i) {
            for (size_t j = 0; j < cols(); ++j) out.node_->data[j * rows() + i] = at(i, j);
        }
        auto o = out.node_;
        out.node_->backward = [a, o] {
            for (size_t i = 0; i < a->rows; ++i) {
                for (size_t j = 0; j < a->cols; ++j) a->grad[i * a->cols + j] += o->grad[j * a->rows + i];
            }
        };
        return out;
    }

    Tensor sum() const {
        Tensor out(1, 1);
        auto a = node_;
        out.node_->data[0] = std::accumulate(a->data.begin(), a->data.end(), 0.0);
        out.node_->prev = {a};
        auto o = out.node_;
        out.node_->backward = [a, o] {
            for (double& g : a->grad) g += o->grad[0];
        };
        return out;
    }

    Tensor mean() const { return sum() / Tensor::scalar(static_cast<double>(size())); }

    Tensor relu() const {
        return unary(
            [](double x) { return x > 0.0 ? x : 0.0; },
            [](double x, double) { return x > 0.0 ? 1.0 : 0.0; });
    }

    Tensor sigmoid() const {
        return unary(
            [](double x) { return 1.0 / (1.0 + std::exp(-x)); },
            [](double, double y) { return y * (1.0 - y); });
    }

    Tensor tanh() const {
        return unary(
            [](double x) { return std::tanh(x); },
            [](double, double y) { return 1.0 - y * y; });
    }

    Tensor softmax() const {
        Tensor out(rows(), cols());
        auto a = node_;
        out.node_->prev = {a};
        for (size_t i = 0; i < rows(); ++i) {
            double m = -1e100;
            for (size_t j = 0; j < cols(); ++j) m = std::max(m, at(i, j));
            double total = 0.0;
            for (size_t j = 0; j < cols(); ++j) {
                double e = std::exp(at(i, j) - m);
                out.node_->data[i * cols() + j] = e;
                total += e;
            }
            for (size_t j = 0; j < cols(); ++j) out.node_->data[i * cols() + j] /= total;
        }
        auto o = out.node_;
        out.node_->backward = [a, o] {
            for (size_t i = 0; i < o->rows; ++i) {
                double dot = 0.0;
                for (size_t j = 0; j < o->cols; ++j) dot += o->grad[i * o->cols + j] * o->data[i * o->cols + j];
                for (size_t j = 0; j < o->cols; ++j) {
                    size_t k = i * o->cols + j;
                    a->grad[k] += o->data[k] * (o->grad[k] - dot);
                }
            }
        };
        return out;
    }

    void backward() {
        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<Node*> seen;
        build_topo(node_, seen, topo);
        std::fill(node_->grad.begin(), node_->grad.end(), 1.0);
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }

private:
    explicit Tensor(std::shared_ptr<Node> node) : node_(std::move(node)) {}

    size_t offset(size_t row, size_t col) const { return row * node_->cols + col; }

    Tensor unary(const std::function<double(double)>& f,
                 const std::function<double(double, double)>& df) const {
        Tensor out(rows(), cols());
        auto a = node_;
        out.node_->prev = {a};
        for (size_t i = 0; i < size(); ++i) out.node_->data[i] = f(a->data[i]);
        auto o = out.node_;
        out.node_->backward = [a, o, df] {
            for (size_t i = 0; i < a->data.size(); ++i) a->grad[i] += o->grad[i] * df(a->data[i], o->data[i]);
        };
        return out;
    }

    static void build_topo(const std::shared_ptr<Node>& n, std::unordered_set<Node*>& seen,
                           std::vector<std::shared_ptr<Node>>& topo) {
        if (seen.count(n.get())) return;
        seen.insert(n.get());
        for (const auto& p : n->prev) build_topo(p, seen, topo);
        topo.push_back(n);
    }

    static std::pair<size_t, size_t> broadcast_shape(const Tensor& a, const Tensor& b) {
        size_t r = std::max(a.rows(), b.rows());
        size_t c = std::max(a.cols(), b.cols());
        if ((a.rows() != r && a.rows() != 1) || (b.rows() != r && b.rows() != 1) ||
            (a.cols() != c && a.cols() != 1) || (b.cols() != c && b.cols() != 1)) {
            throw std::invalid_argument("broadcast shape mismatch");
        }
        return {r, c};
    }

    static double value_at(const std::shared_ptr<Node>& n, size_t row, size_t col) {
        size_t r = n->rows == 1 ? 0 : row;
        size_t c = n->cols == 1 ? 0 : col;
        return n->data[r * n->cols + c];
    }

    static void add_broadcast_grad(const std::shared_ptr<Node>& n, const std::vector<double>& g,
                                   size_t out_rows, size_t out_cols) {
        for (size_t i = 0; i < out_rows; ++i) {
            for (size_t j = 0; j < out_cols; ++j) {
                size_t r = n->rows == 1 ? 0 : i;
                size_t c = n->cols == 1 ? 0 : j;
                n->grad[r * n->cols + c] += g[i * out_cols + j];
            }
        }
    }

    std::shared_ptr<Node> node_;

    friend class SGD;
    friend class Linear;
};

inline Tensor operator+(double a, const Tensor& b) { return Tensor::scalar(a) + b; }
inline Tensor operator+(const Tensor& a, double b) { return a + Tensor::scalar(b); }
inline Tensor operator-(double a, const Tensor& b) { return Tensor::scalar(a) - b; }
inline Tensor operator-(const Tensor& a, double b) { return a - Tensor::scalar(b); }
inline Tensor operator*(double a, const Tensor& b) { return Tensor::scalar(a) * b; }
inline Tensor operator*(const Tensor& a, double b) { return a * Tensor::scalar(b); }
inline Tensor operator/(double a, const Tensor& b) { return Tensor::scalar(a) / b; }
inline Tensor operator/(const Tensor& a, double b) { return a / Tensor::scalar(b); }
inline Tensor mse_loss(const Tensor& pred, const Tensor& target) { return (pred - target).pow(2.0).mean(); }

inline std::mt19937& rng() {
    static std::mt19937 gen(1);
    return gen;
}

inline void set_seed(unsigned seed) {
    rng().seed(seed);
}

class Linear {
public:
    Linear(size_t in_features, size_t out_features)
        : weight_(in_features, out_features), bias_(1, out_features) {
        double limit = std::sqrt(6.0 / static_cast<double>(in_features + out_features));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (double& v : weight_.node_->data) v = dist(rng());
        std::fill(bias_.node_->data.begin(), bias_.node_->data.end(), 0.0);
    }

    Tensor forward(const Tensor& x) const { return x.matmul(weight_) + bias_; }
    std::vector<Tensor> parameters() const { return {weight_, bias_}; }

private:
    Tensor weight_;
    Tensor bias_;
};

class SGD {
public:
    SGD(std::vector<Tensor> params, double learning_rate)
        : params_(std::move(params)), learning_rate_(learning_rate) {}

    void zero_grad() {
        for (auto& p : params_) p.zero_grad();
    }

    void step() {
        for (auto& p : params_) {
            for (size_t i = 0; i < p.node_->data.size(); ++i) {
                p.node_->data[i] -= learning_rate_ * p.node_->grad[i];
            }
        }
    }

private:
    std::vector<Tensor> params_;
    double learning_rate_;
};

}  // namespace autodiff
