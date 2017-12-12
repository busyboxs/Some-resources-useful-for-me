# Linear regression from scratch(从头开始线性回归)

功能强大的ML库可以消除重复的工作，但是如果过分依赖抽象，可能永远也不会学习神经网络是如何在真正的工作环境下工作的。所以对于第一个例子，让我们从零开始构建一切，只依靠 `autograd` 和 `NDArray`。首先，我们将导入与`autograd` 章节相同的依赖包。我们也将导入强大的`gluon`包，但在本章中，我们只会用它来加载数据。

```python
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)
```

## Set the context

我们还要指定计算应该发生的上下文。本教程非常简单，您可以在计算器手表上运行它。但是，为了养成良好的习惯，我们要指定两个上下文：一个用于数据，另一个用于我们的模型。

```python
data_ctx = mx.cpu()
model_ctx = mx.cpu()
```

## Linear regression

我们首先看回归的问题。这个任务是给出一个数据点$x$预测真实价值目标 $y$。在线性回归中，最简单，也许是最有用的方法，我们假设预测可以表示为输入特征的线性组合（因此给出名称线性回归）：

$$
\hat{y} = w_1 \cdot x_1 + ... + w_d \cdot x_d + b
$$

给定数据点的集合$X$和相应的目标值$y$，我们将尝试找到权重向量 $w$和偏移向量$b$（也称为偏移 或截距），使得数据点$x_{i}$与对应的标签$y_{i}$大致关联。使用稍微更高级的数学符号，我们可以通过矩阵向量乘积来表示

$$
\boldsymbol{\hat{y}} = X \boldsymbol{w} + b
$$

在我们开始之前，我们还需要两件事

* 一些方法来衡量当前模型的质量
* 一些操纵模型以提高质量的方法

### Square loss

为了说明我们是否做得好，我们需要一些方法来衡量一个模型的质量。一般来说，我们将定义一个损失函数，说明我们对正确答案的预测有多远。对于线性回归的经典情况，我们通常关注平方误差。具体来说，我们在所有例子中，对平方误差$(y_i-\hat{y})^2$求和：

$$
\ell(y, \hat{y}) = \sum_{i=1}^n (\hat{y}_i-y_i)^2.
$$

对于一维数据，我们可以很容易地看到我们的单一特征和目标变量之间的关系。可视化线性预测器和每个示例的误判也很容易。请注意，平方损失严重惩罚异常值。对于下面的可视化预测变量，孤立的离群值会造成大部分的损失。

![](http://gluon.mxnet.io/_images/linear-regression.png)


### Manipulating the model

为了使误差最小化，我们需要一些机制来改变模型。我们通过选择参数 $w$和 $b$ 的值来做到这一点。这是学习算法的唯一工作。以训练数据$(X,y)$和模型$\hat{y} = X\boldsymbol{w} + b$。然后学习选择最好的$w$和$b$根据现有的数据。

### Historical note

您可能会合理地指出，线性回归是一个经典的统计模型。根据维基百科的记载，勒布朗于1805年首先开发了最小二乘回归方法，1809年高斯重新发现了这种方法。据推测，勒布朗已经多次在论文中发表了几篇文章，但据说高斯没有引用他的arXiv预印本。

抛开物源的问题，你可能会怀疑 - 如果勒让德和高斯在线性回归上工作，这是否意味着有原始的深度学习研究者？如果线性回归不完全属于深度学习，那么为什么我们将线性模型作为神经网络教程系列中的第一个例子？那么事实证明，我们可以将线性回归表示为最简单的（有用的）神经网络。神经网络只是由有向边连接的节点（又称神经元）的集合。在大多数网络中，我们将节点排列成层，每层都将输出输出到上面的层中。要计算任何节点的值，我们首先执行输入的权重总和（根据权重w），然后应用激活函数。对于线性回归，我们只有两个层次，一个对应于输入（用橙色表示）和一个单层节点（用绿色表示）对应于输出。对于输出节点，激活函数只是标识函数。

![](http://gluon.mxnet.io/_images/onelayer.png)

当然你不必从深度学习的角度看线性回归，你可以（而且我们会的！）。为了使我们在代码中讨论的概念成为可能，我们实际上编写了一个从头开始进行线性回归的神经网络。

开始，我们将通过以下方式对随机数据点`X[i]`和相应的标签`y[i]`进行采样来生成一个简单的合成数据集。输入输出将分别从均值为$0$,方差为$1$的随机正态分布中采样。我们的功能将是独立的。另一种说法是，他们将有对角线协方差。根据真正的标记函数`y[i] = 2 * X[i][0]- 3.4 * X[i][1] + 4.2 + noise`生成标签 ，其中噪声是从具有均值``0`和方差`.01`的随机高斯中提取的。我们可以用数学符号表示标注函数：
$$
y = X \cdot w + b + \eta, \quad \text{for } \eta \sim \mathcal{N}(0,\sigma^2)
$$

```python
num_inputs = 2
num_outputs = 1
num_examples = 10000

def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs), ctx=data_ctx)
noise = .1 * nd.random_normal(shape=(num_examples,), ctx=data_ctx)
y = real_fn(X) + noise
```

请注意，$X$每行为一个二维数据点，$Y$每一行都由一维目标值组成。

```python
print(X[0])
print(y[0])
```

```
[-1.22338355  2.39233518]
<NDArray 2 @cpu(0)>

[-6.09602737]
<NDArray 1 @cpu(0)>
```

需要注意的是，因为我们的合成特征在`data_ctx`上，噪音也在`data_ctx`，标签`y`由`X`和`noise`组合产生，也在 `data_ctx`上。我们可以证实，对于任意随机选择的点，与（已知的）最优参数的线性组合产生确实接近目标值的预测

```python
print(2 * X[0, 0] - 3.4 * X[0, 1] + 4.2)
```

```
[-6.38070679]
<NDArray 1 @cpu(0)>
```

我们可以通过用Python绘图`matplotlib`包生成一个散点图来可视化我们的第二个特征（`X[:, 1]`）和目标值`Y`之间的对应关系。确保`matplotlib` 已安装。否则，可以通过在命令`pip2 install matplotlib`（对于Python 2）或 `pip3 install matplotlib`（对于Python 3）来安装它 。

为了用`matplotlib`绘制，我们只需要用`.asnumpy()`转换`X` 和`y`为 NumPy 数组。

```python
import matplotlib.pyplot as plt
plt.scatter(X[:, 1].asnumpy(),y.asnumpy())
plt.show()
```

![](http://gluon.mxnet.io/_images/chapter02_supervised-learning_linear-regression-scratch_11_0.png)

## Data iterators

一旦我们开始使用神经网络，我们将需要快速迭代我们的数据点。我们也希望能够一次抓取批量的`k`个数据点来洗牌我们的数据。在MXNet中，数据迭代器为我们提供了一组用于获取和操作数据的实用程序。具体来说，我们将使用简单的 `DataLoader`类，它提供了一种直观的方式来使用 `ArrayDataset`训练模型。

我们可以通过调用`gluon.data.ArrayDataset(X, y)`加载`X`和`y`到 ArrayDataset 。对于多维的输入`X`和一维的标签`y`是ok的。一个要求是它们在第一个轴上具有相等的长度，即`len(X) == len(y)`。

给定一个`ArrayDataset`，我们可以创建一个 DataLoader ，它将从`ArrayDataset`中随机抓取批量的数据。我们将要指定两个参数。首先，我们需要说的是`batch_size`，我们一次想要抓多少个示例。其次，我们要指定是否在迭代之间通过数据集混洗数据。

```python
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)
```

一旦我们初始化了 DataLoader（`train_data`），我们可以通过在`train_data`上迭代来轻松地获取批处理，就像它是一个Python列表一样。您可以使用喜欢的迭代技术，如 foreach 循环：`for data, label in train_data`或者 enumerations：`for i, (data, label) in enumerate(train_data)`。首先，让我们抓住一个批次并跳出循环。

```python
for i, (data, label) in enumerate(train_data):
    print(data, label)
    break
```

```
[[-0.14732301 -1.32803488]
 [-0.56128627  0.48301753]
 [ 0.75564283 -0.12659997]
 [-0.96057719 -0.96254188]]
<NDArray 4x2 @cpu(0)>
[ 8.25711536  1.30587864  6.15542459  5.48825312]
<NDArray 4 @cpu(0)>
```

如果我们再次运行相同的代码，您会注意到我们获得了不同的批次。这是因为我们的指定`DataLoader`中 `shuffle=True`。

```python
for i, (data, label) in enumerate(train_data):
    print(data, label)
    break
```

```
[[-0.59027743 -1.52694809]
 [-0.00750104  2.68466949]
 [ 1.50308061  0.54902577]
 [ 1.69129586  0.32308948]]
<NDArray 4x2 @cpu(0)>
[ 8.28844357 -5.07566643  5.3666563   6.52408457]
<NDArray 4 @cpu(0)>
```

最后，如果我们实际传递整个数据集，并计算批次数，我们会发现有2500个批次。我们得到这个，是因为我们的数据集有10,000个例子，我们 设置`DataLoader`的批量大小为4。

```python
counter = 0
for i, (data, label) in enumerate(train_data):
    pass
print(i+1)
```

```
2500
```

## Model parameters

现在让我们为我们的参数分配一些内存并设置它们的初始值。我们想要在`model_ctx`上初始化这些参数。

```python
w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
params = [w, b]
```

在接下来的单元格中，我们将更新这些参数以更好地适合我们的数据。这将涉及到对参数取一些损失函数的梯度（多维导数）。我们将在减少损失的方向更新每个参数。但首先，让我们为每个渐变分配一些内存。

```python
for param in params:
    param.attach_grad()
```

## Neural networks

接下来我们要定义我们的模型。在这种情况下，我们将使用线性模型，最简单的有用的神经网络。要计算线性模型的输出，我们只需将给定的输入与模型的权重（`w`）相乘，然后添加偏移量`b`。

```python
def net(X):
    return mx.nd.dot(X, w) + b
```

好的，那很简单。

## Loss function

训练一个模型意味着在一段训练过程中变得越来越好。但是为了使这个目标有意义，我们首先需要首先定义更好的方法。在这种情况下，我们将使用我们的预测和真实值之间的平方距离。

```python
def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)
```

## Optimizer

事实证明，线性回归实际上有一个封闭的解决方案。但是，我们关心的最有趣的模型不能通过分析来解决。所以我们将通过随机梯度下降来解决这个问题。在每一步中，我们将使用从我们的数据集中随机抽取的一批来估计相对于我们的权重的损失的梯度。然后，我们将在损失减少的方向更新我们的参数。步的大小由学习速率`lr`决定。

```python
def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
```

## Execute training loop

现在我们有了所有的东西，我们只需要通过编写一个训练循环来连接它们。首先，我们将定义`epochs`，数据集的通过次数。然后，对于每一遍，我们将遍历`train_data`，抓取批量的实例和相应的标签。

对于每一批，我们将通过以下过程：

* 通过网络执行正向传递，生成预测（`yhat`）和损失（`loss`）。
* 通过网络向后传递来计算梯度（`loss.backward()`）。
* 通过调用我们的SGD优化器来更新模型参数。

```python
epochs = 10
learning_rate = .0001
num_batches = num_examples/batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()
    print(cumulative_loss / num_batches)
```

```
24.6606138554
9.09776815639
3.36058844271
1.24549788469
0.465710770596
0.178157229481
0.0721970594548
0.0331197250206
0.0186954441286
0.0133724625537
```

## Visualizing our training progess

在接下来的章节中，我们将介绍更加真实的数据，发烧友模型，更复杂的损失函数等等。但核心思想是相同的，训练循环看起来非常熟悉。因为这些教程是独立的，你会很好地了解这个流程。除了更新模型之外，我们还经常要做一些簿记工作。除此之外，我们可能需要跟踪训练进度并以图形方式对其进行可视化。我们在下面演示一个稍微复杂的训练循环。

```python
############################################
#    Re-initialize parameters because they
#    were already trained in the first loop
############################################
w[:] = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
b[:] = nd.random_normal(shape=num_outputs, ctx=model_ctx)

############################################
#    Script to plot the losses over time
############################################
def plot(losses, X, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title('Loss during training')
    fg1.plot(xs, losses, '-r')
    fg2.set_title('Estimated vs real function')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             net(X[:sample_size, :]).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(),
             real_fn(X[:sample_size, :]).asnumpy(), '*g', label='Real')
    fg2.legend()

    plt.show()

learning_rate = .0001
losses = []
plot(losses, X)

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += loss.asscalar()

    print("Epoch %s, batch %s. Mean loss: %s" % (e, i, cumulative_loss/num_batches))
    losses.append(cumulative_loss/num_batches)

plot(losses, X)
```

![](http://gluon.mxnet.io/_images/chapter02_supervised-learning_linear-regression-scratch_34_0.png)

```
Epoch 0, batch 2499. Mean loss: 16.9325145943
Epoch 1, batch 2499. Mean loss: 6.24987681103
Epoch 2, batch 2499. Mean loss: 2.31109857569
Epoch 3, batch 2499. Mean loss: 0.858666448605
Epoch 4, batch 2499. Mean loss: 0.323071002489
Epoch 5, batch 2499. Mean loss: 0.125603744188
Epoch 6, batch 2499. Mean loss: 0.0527891687471
Epoch 7, batch 2499. Mean loss: 0.0259436405713
Epoch 8, batch 2499. Mean loss: 0.0160523827007
Epoch 9, batch 2499. Mean loss: 0.0124009371101
```

![](http://gluon.mxnet.io/_images/chapter02_supervised-learning_linear-regression-scratch_34_2.png)

## Conclusion

你已经看到只使用 mxnet.ndarray 和 mxnet.autograd ，我们可以从头开始建立统计模型。在下面的教程中，我们将在此基础上，介绍现代神经网络背后的基本思想，并展示 MXNet `gluon`包中用于构建复杂模型的强大抽象 。

