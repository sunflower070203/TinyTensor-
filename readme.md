# Tensor: 极简 C++ 自动求导系统

这是一个只使用 C++17 实现的教学版自动求导系统，目标是覆盖从标量 Tensor 到简单神经网络训练的完整闭环。

## 功能

- 标量、向量、矩阵 `Tensor`
- 前向计算自动构建计算图
- `backward()` 使用拓扑排序和链式法则反向传播梯度
- 标量/矩阵广播运算：`+`、`-`、`*`、`/`
- 幂运算：`pow(exponent)`
- 矩阵运算：`matmul()`、`transpose()`、`sum()`、`mean()`
- 激活函数：`relu()`、`sigmoid()`、`tanh()`、`softmax()`
- 损失函数：`mse_loss(pred, target)`
- 神经网络组件：`Linear`
- 优化器：`SGD`
- 示例训练：
  - 线性回归拟合 `y = 2x + 1`
  - 多层神经网络解决 XOR
  - 多层神经网络解决同心圆二分类

## 文件结构

```text
Tensor/
  CMakeLists.txt   CMake 构建配置
  tensor.hpp       自动求导核心、Linear、SGD
  main.cpp         训练示例
  interactive.cpp  用户自定义输入的交互式终端
  tests.cpp        功能验证测试
  readme.md        项目说明
```

## 快速运行

如果使用当前 MSYS2 MinGW 环境，先确保编译器动态库目录在 `PATH` 中：

```powershell
$env:PATH='C:\msys64\mingw64\bin;' + $env:PATH
```

直接编译测试：

```powershell
g++ -std=c++17 tests.cpp -O2 -o tensor_tests.exe
.\tensor_tests.exe
```

直接编译示例：

```powershell
g++ -std=c++17 main.cpp -O2 -o tensor_demo.exe
.\tensor_demo.exe
```

直接编译交互式终端：

```powershell
g++ -std=c++17 interactive.cpp -O2 -o tensor_cli.exe
.\tensor_cli.exe
```

也可以用 CMake：

```powershell
cmake -S . -B build
cmake --build build
.\build\tensor_tests.exe
.\build\tensor_demo.exe
.\build\tensor_cli.exe
```

## 用户自定义输入

`interactive.cpp` 提供命令行入口，用户可以在终端创建变量、执行运算、反向传播并更新参数。

启动：

```powershell
g++ -std=c++17 interactive.cpp -O2 -o tensor_cli.exe
.\tensor_cli.exe
```

常用命令：

```text
scalar <name> <value>
matrix <name> <rows> <cols> <values...>
add|sub|mul|div <out> <a> <b>
pow <out> <a> <exponent>
matmul <out> <a> <b>
transpose|relu|sigmoid|tanh|softmax|sum|mean <out> <a>
backward <loss>
grad <name>
print <name>
step <learning_rate> <param_count> <param1> ... <paramN>
fit_linear <epochs> <learning_rate> <n> <x1> <y1> ... <xn> <yn>
xor
circle
```

示例：用户创建变量并求导。

```text
scalar x 2
scalar y 3
mul xy x y
pow x3 x 3
add z xy x3
div loss z y
backward loss
print loss
grad x
grad y
```

示例：用户输入数据并用梯度下降拟合线性回归。

```text
fit_linear 200 0.05 5 -2 -3 -1 -1 0 1 1 3 2 5
```

示例：直接运行挑战版神经网络任务。

```text
xor
circle
```

## 使用示例

```cpp
#include "tensor.hpp"

using autodiff::Tensor;

int main() {
    auto x = Tensor::scalar(2.0);
    auto y = Tensor::scalar(3.0);
    auto z = ((x * y) + x.pow(3.0)) / y;

    z.backward();

    double dz_dx = x.grad_at(0, 0);
    double dz_dy = y.grad_at(0, 0);
}
```

批量矩阵训练：

```cpp
auto x = Tensor::from({{-2.0}, {-1.0}, {0.0}, {1.0}, {2.0}});
auto y = Tensor::from({{-3.0}, {-1.0}, {1.0}, {3.0}, {5.0}});
auto w = Tensor::scalar(0.0);
auto b = Tensor::scalar(0.0);
autodiff::SGD optim({w, b}, 0.05);

for (int epoch = 0; epoch < 200; ++epoch) {
    auto pred = x * w + b;
    auto loss = autodiff::mse_loss(pred, y);
    optim.zero_grad();
    loss.backward();
    optim.step();
}
```

## 实现机制

每个 `Tensor` 内部持有一个共享的计算节点。节点保存：

- 数据 `data`
- 梯度 `grad`
- 父节点 `prev`
- 当前操作对应的局部反向传播函数 `backward`

前向运算会生成新节点并记录父节点。调用最终 loss 的 `backward()` 时，系统先从 loss 节点做拓扑排序，再逆序执行每个节点的反向函数，把梯度累加到父节点中。

广播反向传播会把输出梯度按原始 Tensor 形状折叠求和。例如标量 bias 被加到一个 batch 输出上时，bias 的梯度等于整个 batch 对应梯度之和。

## 验证范围

`tests.cpp` 覆盖：

- 标量链式求导
- 矩阵乘法和广播梯度
- ReLU、Sigmoid、Tanh、Softmax
- 线性回归 200 轮收敛到 `w ~= 2, b ~= 1`
- 两层神经网络解决 XOR
- 两层神经网络解决同心圆二分类，准确率不低于 90%

当前实现面向教学和小规模实验，不追求工业级性能、GPU 支持或复杂张量维度。
