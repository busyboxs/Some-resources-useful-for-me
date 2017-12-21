# Linear regression with gluon

现在，我们已经实现了一个从无到有的整个神经网络，使用`mx.ndarray`和`mxnet.autograd`，让我们看看我们如何能够做出同样的模式，而少做很多工作。

再次，我们导入一些包，这次添加`mxnet.gluon`到依赖关系列表中。

```python
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon
```

## Set the context

我们也想设置一个上下文来告诉 gluon 在哪里做大部分的计算。

```python
data_ctx = mx.cpu()
model_ctx = mx.cpu()
```

## Build the dataset

我们再来看看线性回归的问题，并坚持使用相同的合成数据。

```python
num_inputs = 2
num_outputs = 1
num_examples = 10000 

def real_fn(X):
	return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_imputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise
```

## Load the data iterator

我们将坚持`DataLoader`处理数据批处理。

```python
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)
```

## Define the model

当我们从头开始实施时，我们必须单独分配参数，然后将它们组合在一起作为模型。虽然知道如何从头开始做是件好事，但是用`gluon`，我们可以从预定义的图层组成一个网络。对于线性模型，调用适当的图层`Dense`。它被称为密集层，因为输入中的每个节点都连接到后续层中的每个节点。这个描述看起来太多了，因为我们在这里只有一个（非输入）层，而这个层只包含一个节点！但是在后面的章节中，我们通常会使用具有多个输出的网络，所以我们不妨从节点层面开始思考。因为一个线性模型只包含一个`Dense`图层，我们可以用一行来实例化它。

我们具有2维的输入和1维的输出。最直接的方式来实例化一个`Dense`与这些尺寸层是指定的输入的数目和输出的数目。

```python
net = gluon.nn.Dense(1, in_units=2)
```

我们已经有了一个神经网络。就像我们之前的手工制作的模型一样，这个模型有一个权重矩阵和偏向量。

```python
print(net.weight)
print(net.bias)
```

```
Parameter dense4_weight (shape=(1, 2), dtype=None)
Parameter dense4_bias (shape=(1,), dtype=None)
```

在这里，`net.weight`和`net.bias`实际上不是 NDArrays。他们是这个`Parameter`类的实例。我们使用`Parameter`而不是直接访问 NDAarrays 有几个原因。例如，它们为初始化值提供了方便的抽象。与 NDArrays 不同，参数可以同时与多个上下文关联。当我们开始考虑跨多个GPU的分布式学习时，这将在未来的章节中派上用场。

在`gluon` 中，所有的神经网络都是由 Blocks（`gluon.Block`）组成的。块是只需要输入和生成输出的单元。块还包含我们可以更新的参数。在这里，我们的网络只有一层，所以直接访问我们的参数是很方便的。当我们的网络由10个层组成时，这不会太好玩。无论我们的网络多么复杂，我们都可以通过调用`collect_params()`如下来获取它的所有参数：

```python
net.collect_params()
```

```
dense4_ (
  Parameter dense4_weight (shape=(1, 2), dtype=None)
  Parameter dense4_bias (shape=(1,), dtype=None)
)
```
返回的对象是一个`gluon.parameter.ParameterDict`。这对于检索和操作 Parameter 对象组是一个方便的抽象。大多数情况下，我们要检索神经网络中的所有参数：

```python
type(net.collect_params())
```

```
mxnet.gluon.parameter.ParameterDict
```

## Initialize parameters

一旦我们初始化了参数，我们就可以访问它们的底层数据和上下文，并且还可以通过神经网络提供数据来生成输出。但是，我们还不能走。如果我们尝试通过调用`net(nd.array([[0,1]]))`来调用模型，我们将面对以下可怕的错误消息：

`RuntimeError: Parameter dense1_weight has not been initialized...`

那是因为我们还没有告诉 我们参数`gluon`的初始值应该是什么！我们通过调用`ParameterDict` 的`.initialize()`方法初始化参数 。我们需要传递两个参数。

* 一个初始化程序，其中许多都在`mx.init`模块中。
* 参数应该存在的上下文。在这种情况下，我们会通过`model_ctx`。大多数情况下，这可能是GPU或GPU列表。

MXNet提供了各种常用的初始化器`mxnet.init`。为了保持与我们手工建立的模型一致，我们将使用一个标准的正态分布进行采样来初始化每个参数 `mx.init.Normal(sigma=1.)`。

```python
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
```

## Deferred Initialization

当我们调用`initialize`时，`gluon`将每个参数与一个初始化器相关联。然而，实际的初始化被推迟，直到我们进行第一次正向传递。换句话说，参数只在需要的时候被初始化。如果我们尝试调用`net.weight.data()` 我们会得到以下错误：

`DeferredInitializationError: Parameter dense2_weight has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters.`

通过`gluon`模型传递数据很容易。我们只是采样一批适当的形状，并调用`net`，就好像它是一个函数。这将调用网络的`forward()`方法。

```python
example_data = nd.array([[4,7]])
net(example_data)
```

```
[[-1.33219385]]
<NDArray 1x1 @cpu(0)>
```

现在net已经初始化了，我们可以访问它的每个参数。

```python
print(net.weight.data())
print(net.bias.data())
```

```
[[-0.25217363 -0.04621419]]
<NDArray 1x2 @cpu(0)>

[ 0.]
<NDArray 1 @cpu(0)>
```

## Shape inference

回想一下，以前，我们通过`gluon.nn.Dense(1, in_units=2)`实例化了我们的网络 。gluon中我们可以利用的一个光滑的特征是参数的形状推断。因为在我们通过网络传递数据之前，我们的参数从不起作用，所以我们实际上并不需要声明输入维（in_units）。让我们再试一次，让`gluon`做更多的工作：

```python
net = gluon.nn.Dense(1)
net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
```

在接下来的章节中，我们将详细阐述这个以及更多`gluon`的内部工作。

## Define loss

我们不是写自己的损失函数，而是通过`gluon.loss.L2Loss`实例化来访问平方误差。就像层和整个网络，gluon 的损失只是一个 `Block`。

square_loss = gluon.loss.L2Loss()

## Optimizer

不是每次从头开始编写随机梯度下降，我们可以通过`gluon.Trainer`实例化一个，传递一个参数字典。请注意，`gluon`中的`sgd`优化器实际上使用了动量和削减的 SGD（如果需要，可以关闭），因为这些修改使得它更好地收敛。稍后我们将详细讨论一系列优化算法。

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})
```

## Execute training loop

你可能已经注意到，用`gluon`表达我们的模型比较简单一些。例如，我们不必单独分配参数，定义我们的损失函数，或者实现随机梯度下降。一旦我们开始使用更复杂的模型，依赖`gluon`的抽象的好处将会大大增加。但是，一旦我们掌握了所有的基础知识，训练循环本身与从零开始实施所有工作时所做的工作非常相似。

刷新你的记忆。对于某些数字`epochs`，我们将完整地传递数据集（`train_data`），一次抓取一个小批量的输入和相应的真值标签。

然后，对于每一批，我们将通过下面的仪式。为了使这个过程成为最大的仪式，我们将逐字重复：

* 通过网络执行正向传递，生成预测（`yhat`）和损失（`loss`）。
* ``loss.backward()`通过网络向后传递来计算梯度。
* 通过调用我们的 SGD 优化器更新模型参数（注意，我们不需要告诉`trainer.step`哪些参数，而只是数据量，因为我们已经在初始化中执行了`trainer`）。

```python
epochs = 10
loss_sequence = []
num_batches = num_examples / batch_size

for e in range(epochs):
    cumulative_loss = 0
    # inner loop
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.mean(loss).asscalar()
    print("Epoch %s, loss: %s" % (e, cumulative_loss / num_examples))
    loss_sequence.append(cumulative_loss)

```

```
Epoch 0, loss: 3.44980202263
Epoch 1, loss: 2.10364257665
Epoch 2, loss: 1.28279426137
Epoch 3, loss: 0.782256319318
Epoch 4, loss: 0.477034088909
Epoch 5, loss: 0.290909814427
Epoch 6, loss: 0.177411796283
Epoch 7, loss: 0.108197494675
Epoch 8, loss: 0.0659899789031
Epoch 9, loss: 0.040249745576
```

## Visualizing the learning curve

现在我们来看看SGD如何通过绘制学习曲线来学习线性回归模型。

```python
# plot the convergence of the estimated loss function
%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt

plt.figure(num=None,figsize=(8, 6))
plt.plot(loss_sequence)

# Adding some bells and whistles to the plot
plt.grid(True, which="both")
plt.xlabel('epoch',fontsize=14)
plt.ylabel('average loss',fontsize=14)
```

```
<matplotlib.text.Text at 0x7efc87a7f0f0>
```

![](http://gluon.mxnet.io/_images/chapter02_supervised-learning_linear-regression-gluon_32_1.png)

我们可以看到，损失函数迅速收敛到最优解。

## Getting the learned model parameters

作为一个额外的完整性检查，因为我们从高斯线性回归模型生成的数据，我们希望确保学习者设法恢复模型参数，分别设置权重为 2 ，- 3.4， 偏移量为4.2。

```python
params = net.collect_params() # this returns a ParameterDict

print('The type of "params" is a ',type(params))

# A ParameterDict is a dictionary of Parameter class objects
# therefore, here is how we can read off the parameters from it.

for param in params.values():
    print(param.name,param.data())
```

```
The type of "params" is a  <class 'mxnet.gluon.parameter.ParameterDict'>
dense5_weight
[[ 1.7913872  -3.10427046]]
<NDArray 1x2 @cpu(0)>
dense5_bias
[ 3.85259581]
<NDArray 1 @cpu(0)>
```

## Conclusion

正如你所看到的，即使是像线性回归这样一个简单的例子， `gluon`也可以帮助你写出快速而干净的代码。接下来，我们将重复这个练习来获得多层感知器，将这些教训扩展到深层神经网络和（相对）真实的数据集。
