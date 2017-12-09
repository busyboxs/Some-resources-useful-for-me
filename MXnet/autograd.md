# Automatic differentiation with autograd

在机器学习中，我们训练模型以随着经验而变得越来越好。通常情况下，越来越好意味着最大限度地减少损失函数，即回答“ 我们的模型有多糟糕 ”的分数。使用神经网络，我们选择损失函数在参数方面是可微的。简而言之，这意味着对于模型的每个参数，我们可以确定增加或减少多少可能会影响损失。虽然计算简单明了，但对于复杂的模型，手工操作可能会很痛苦。

MXNet 的 autograd 包通过自动计算导数来加速这项工作。虽然大多数其他库要求我们编译一个 symbolic graph 来自动生成导数，` mxnet.autograd`，就像 PyTorch 一样，它允许您在编写普通的命令式代码时使用导数。每当你通过你的模型，`autograd`建立一个 graph，通过它可以立即反向传播梯度。

让我们一步一步的实现。对于本教程，我们只需要导入 `mxnet.ndarray` 和 `mxnet.autograd` 。

```python
import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)
```

## Attaching gradients

作为一个玩具的例子，假设我们有兴趣区分函数 `f = 2 * (x ** 2)x` 和参数 x 。我们可以从分配一个初始值开始。

```python
x = nd.array([[1, 2], [3, 4]])
```

一旦我们计算了 `f` 相对于 `x` 的导数，我们需要一个地方来存储它。在 MXNet 中，我们可以通过调用其 `attach_grad()` 方法来告诉 NDArray 我们计划存储一个导数。

```python
x.attach_grad()
```

现在我们要定义这个函数 `f`，MXNet 将在运行中生成一个计算图。就好像 MXNet 打开一个记录设备，并捕获每个变量生成的确切路径。

请注意，构建计算图需要非常多的计算。因此，MXNet 只会在明确告知时才会构建图表。我们可以指示 MXNet 通过在代码块内部放置代码来开始记录。`with autograd.record()`:

```python
with autograd.record():
    y = x * 2
    z = y * x
```

让我们通过调用 `z.backward()` 反向传播。当 `z` 有多个条目时，`z.backward()` 相当于`mx.nd.sum(z).backward()`

```python
z.backward()
```

现在，让我们看看这是否是预期的输出。请记住 ，`y = x * 2` 和 `z = x * y`，所以 `z` 应该等于 `2 * x * x`。之后，`z.backward()` 做 backprop ，我们期望得到梯度dz/dx如下：dy/dx = `2`, dz/dx = `4 * x` 。所以，如果一切按计划进行，`x.grad` 应该包含一个具有值 `[[4, 8],[12, 16]]` 的NDArray 。

```python
print(x.grad)
```

```
[[  4.   8.]
 [ 12.  16.]]
<NDArray 2x2 @cpu(0)>
```

## Head gradients and the chain rule

警告：这部分是棘手的，但不需要理解后续部分。

有时，当我们调用反向传播中的NDArray，例如 `y.backward()`，其中`y`是`x`的函数，我们只对`y`对`x`的导数感兴趣。数学家把这写成$\frac{dy(x)}{dx}$。在其他时候，我们可能感兴趣的是 `z` 相对于`x`的梯度，其中`z`是`y`的函数，反过来，`y`是`x`的函数 。那就是，我们对$\frac{d}{dx} z(y(x))$感兴趣。回想一下，由链式法则 $\frac{d}{dx} z(y(x)) = \frac{dz(y)}{dy} \frac{dy(x)}{dx}$。所以，当`y`是一个更大的函数的`z`一部分时，我们要`x.grad`存储$\frac{dz}{dx}$，我们可以传入 head gradient $\frac{dz}{dy}$ 给 `backward()`。默认参数是`nd.ones_like(y)`。请参阅 [维基百科](https://en.wikipedia.org/wiki/Chain_rule)了解更多详情。

```python
with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)
```

```
[[ 40.           8.        ]
 [  1.20000005   0.16      ]]
<NDArray 2x2 @cpu(0)>
```

现在我们知道了基础知识，我们可以用`autograd`做一些狂野的事情，包括使用Pythonic控制流来构建可微函数。

```python
a = nd.random_normal(shape=3)
a.attach_grad()

with autograd.record():
    b = a * 2
    while (nd.norm(b) < 1000).asscalar():
        b = b * 2

    if (mx.nd.sum(b) > 0).asscalar():
        c = b
    else:
        c = 100 * b
```


```python
head_gradient = nd.array([0.01, 1.0, .1])
c.backward(head_gradient)
```

```python
print(a.grad)
```

```
[   2048.  204800.   20480.]
<NDArray 3 @cpu(0)>
```
