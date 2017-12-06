# Manipulate data the MXNet way with ndarray

如果我们不能处理数据，就不可能完成任何事情。一般来说，我们需要做两件重要的事情：（i）获得它！（ii）在计算机内部处理它。如果我们不知道如何存储数据，试图获取数据是没有意义的，所以让我们先用合成数据来练手。

我们将首先介绍NDArrays，MXNet用于存储和转换数据的主要工具。如果你之前使用过NumPy，你会注意到NDArray在设计上类似于NumPy的多维数组。但是，它们具有一些关键优势。首先，NDArrays支持CPU，GPU和分布式云架构的异步计算。其次，他们提供支持自动微分。这些特性使NDArray成为研究人员和工程师推出生产系统的机器学习理想库。

## Getting started

在这一章中，我们将介绍基本功能。不要担心，如果你不明白任何基本的数学，如逐元素操作或正态分布。在接下来的两章中，我们将再次通过NDArray，教授你需要的数学和如何在代码中实现它。

在开始，让我们导入`mxnet`。为了方便，我们还将从`mxnet`导入`ndarray`。我们习惯设置一个随机种子，以便总是得到和我们一样的结果。

```python
import mxnet as mx
from mxnet import nd
mx.random.seed(1)
```

接下来，让我们看看如何创建一个NDArray，没有任何值初始化。具体来说，我们将创建一个3行4列的二维数组（也称为矩阵）。

```python
x = nd.empty((3, 4))
print(x)
```

```
[[  0.00000000e+00   0.00000000e+00   2.26995938e-20   4.57734143e-41]
 [  1.38654559e-38   0.00000000e+00   1.07958838e-15   4.57720130e-41]
 [  6.48255647e-37   0.00000000e+00   4.70016266e-18   4.57734143e-41]]
<NDArray 3x4 @cpu(0)>
```

`empty`方法只是抓住一些内存，并返回一个矩阵，而不设置它的任何条目的值。这意味着条目可以有任何形式的值，包括非常大的值！但通常情况下，我们希望我们的矩阵初始化。通常，我们需要一个全零的矩阵。

```python
x = nd.zeros((3, 5))
x
```

```
[[0 0. 0. 0. 0.] 
 [0 0. 0. 0. 0.] 
 [0 0. 0. 0. 0.]] 
<3×5 NDArray @cpu（0）>
```

同样，ndarray有一个函数来创建一个全1矩阵。

```python
x = nd.ones((3, 4))
x
```

```
[[1. 1. 1. 1.] 
 [1. 1. 1. 1.] 
 [1. 1. 1. 1.]] 
<NDArray 3x4 @cpu（0）>
```

通常，我们需要创建数值随机抽样的数组。当我们打算使用数组作为神经网络中的参数时，这是特别常见的。在这个片段中，我们用从零均值和单位方差的标准正态分布中得出的值进行初始化。

```python
y = nd.random_normal(0, 1, shape=(3, 4))
y
```

```
[[0.11287736 -1.30644417 -0.10713575 -2.63099265] 
 [-0.05735848 0.31348416 -0.57651091 -1.11059952] 
 [0.57960719 -0.22899596 1.04484284 0.81243682]] 
<NDArray 3x4 @cpu（0）>
```

和NumPy一样，每个NDArray的维度都可以通过`.shape`属性访问 。

```python
y.shape
```
```
（3,4）
```

我们也可以查询它的大小，它等于形状组件的乘积。加上存储值的精度，这告诉我们阵列占用了多少内存。

```python
y.size
```

```
12
```

## Operations

NDArray支持大量标准的数学运算。如逐元素

相加：

```python
x + y
```

```
[[1.11287737 -0.30644417 0.89286423 -1.63099265] 
 [0.9426415 1.31348419 0.42348909 -0.11059952] 
 [1.57960725 0.77100402 2.04484272 1.81243682]] 
<NDArray 3x4 @cpu（0）>
```

相乘：

```python
x * y
```

```
[[0.11287736 -1.30644417 -0.10713575 -2.63099265] 
 [-0.05735848 0.31348416 -0.57651091 -1.11059952] 
 [0.57960719 -0.22899596 1.04484284 0.81243682]] 
<NDArray 3x4 @cpu（0）>
```

和指数：

```python
nd.exp(y)
```

```
[[1.11949468 0.27078119 0.8984037 0.07200695] 
 [0.94425553 1.36818385 0.56185532 0.32936144] 
 [1.78533697 0.79533172 2.84295177 2.25339246]] 
<NDArray 3x4 @cpu（0）>
```

我们也可以抓住一个矩阵的转置来计算一个适当的矩阵-矩阵点乘。

```
nd.dot(x, y.T)
```

```
[[-3.93169522 -1.43098474 2.20789099] 
 [-3.93169522 -1.43098474 2.20789099] 
 [-3.93169522 -1.43098474 2.20789099]] 
<NDArray 3x3 @cpu（0）>
```

我们将解释这些操作，并在线性代数章节介绍更多的操作符。但现在，我们将坚持使用NDArray的机制。

## In-place operations（就地操做）

在前面的例子中，每次我们运行一个操作，我们都分配了新的内存来存放它的结果。For example, if we write y = x + y, we will dereference the matrix that y used to point to and instead point it at the newly allocated memory. 在下面的例子中，我们用Python的`id()`函数来演示这个，它给了我们在内存中被引用的对象的确切地址。运行`y = y + x` ，我们会发现`id(y)`指向不同的位置。这是因为Python首先运算`y + x`，为结果分配新内存，然后重定向`y`到指向内存中的这个新位置。

```python
print('id(y):', id(y))
y = y + x
print('id(y):', id(y))
```

```
id（y）：140291459787296 
id（y）：140295515324600
```

这可能是不可取的，原因有两个。首先，我们不想乱跑分配内存。在机器学习中，我们可能有数百兆字节的参数，并且每秒更新所有参数。通常，我们将要执行这些更新。其次，我们可能会指出来自多个变量的相同参数。如果我们没有更新，这可能会导致内存泄漏，并可能导致我们无意中引用陈旧的参数。

幸运的是，在MXNet中执行就地操作非常简单。我们可以将一个操作的结果分配给一个以前分配的数组，例如，`y[:] = <expression>`。

```python
print('id(y):', id(y))
y[:] = x + y
print('id(y):', id(y))
```

```
id（y）：140295515324600 
id（y）：140295515324600
```

虽然这很好，但`x+y`在复制之前，仍然会分配一个临时缓冲区来存储结果`y[:]`。为了更好地利用内存，我们可以直接调用底层`ndarray`操作，在这种情况下`elemwise_add`，避免临时缓冲区。我们通过指定`out`关键字参数（每个`ndarray`运算符支持的）来执行此操作：

```python
nd.elemwise_add(x, y, out=y)
```

```
[[3.11287737 1.69355583 2.89286423 0.36900735] 
 [2.9426415 3.31348419 2.42348909 1.88940048] 
 [3.57960725 2.77100396 4.04484272 3.81243682]] 
<NDArray 3x4 @cpu（0）>
```

如果我们不打算重新使用`x`，那么我们可以将结果分配给`x`自己。有两种方法可以在MXNet中执行此操作。1.使用切片符号x [：] = x op y 2.通过使用op-equals运算符`+=`

```python
print('id(x):', id(x))
x += y
x
print('id(x):', id(x))
```

```
id（x）：140291459564992 
id（x）：140291459564992
```

## Slicing

MXNet NDArrays支持所有您可能想象的访问数据的荒谬方式。这是一个从`x`第二和第三行读取的例子。

```python
x[1:3]
```

```
[[3.9426415 4.31348419 3.42348909 2.88940048] 
 [4.57960701 3.77100396 5.04484272 4.81243706]] 
<NDArray 2x4 @cpu（0）>
```

现在我们来尝试写一个特定的元素。

```
x[1,2] = 9.0
x
```

```
[[4.11287737 2.69355583 3.89286423 1.36900735] 
 [3.9426415 4.31348419 9. 2.88940048] 
 [4.57960701 3.77100396 5.04484272 4.81243706]] 
<NDArray 3x4 @cpu（0）>
```

多维切片也被支持。

```
x[1:2,1:3]
```

```
[[4.31348419 9.]] 
<NDArray 1x2 @cpu（0）>
```

```
x[1:2,1:3] = 5.0
x
```

```
[[4.11287737 2.69355583 3.89286423 1.36900735] 
 [3.9426415 5. 5.88940048] 
 [4.57960701 3.77100396 5.04484272 4.81243706]] 
<NDArray 3x4 @cpu（0）>
```

## Broadcasting

你可能想知道，如果你添加一个向量`y`矩阵到`x`会发生什么？这些操作，我们组合一个高维数组`x`和低维数组`y`将调用一个称为广播的功能。这里，低维数组沿着具有维度1的任何轴重复以匹配高维数组的形状。考虑下面的例子。

```python
x = nd.ones(shape=(3,3))
print('x = ', x)
y = nd.arange(3)
print('y = ', y)
print('x + y = ', x + y)
```

```
x =
[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
<NDArray 3x3 @cpu(0)>
y =
[ 0.  1.  2.]
<NDArray 3 @cpu(0)>
x + y =
[[ 1.  2.  3.]
 [ 1.  2.  3.]
 [ 1.  2.  3.]]
<NDArray 3x3 @cpu(0)>
```

虽然y最初是形状（3），MXNet推断其形状为（1,3），然后沿行广播形成（3,3）矩阵。您可能会想，为什么MXNet选择解释y为（1,3）矩阵而不是（3,1）。这是因为广播喜欢沿着最左边的轴重复。我们可以通过明确给出y的2D形状来改变这种行为 。

```python
y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)
```

```
y = 
[[0.] 
 [1.] 
 [2.]] 
<NDArray 3x1 @cpu（0）> 
x + y = 
[[1. 1. 1] 
 [2. 2. 2.] 
 [3.3 。3]] 
<NDArray 3x3 @cpu（0）>
```

## Converting from MXNet NDArray to NumPy

MXNet NDArrays与NumPy的转换很简单。转换后的阵列不共享内存。

```python
a = x.asnumpy()
type(a)
```
```
numpy.ndarray
```


```python
y = nd.array(a)
y
```

```
[[1. 1. 1.] 
 [1. 1. 1.] 
 [1. 1. 1.]] 
<NDArray 3x3 @cpu（0）>
```

## Managing context

您可能已经注意到MXNet NDArray与NumPy几乎相同。但是有一些重要的区别。MXNet与NumPy不同之处在于它支持各种硬件设备。

在MXNet中，每个数组都有一个上下文。一个上下文可以是CPU。其他上下文可能是各种GPU。当我们在多台服务器上部署作业时，事情会变得更加复杂。通过智能地将数组分配给上下文，我们可以最小化在设备之间传输数据的时间。例如，当在带有GPU的服务器上训练神经网络时，我们通常喜欢模型的参数在GPU上。首先，让我们尝试在第一个GPU上初始化一个数组。

```python
z = nd.ones(shape=(3,3), ctx=mx.gpu(0))
z
```

```
[[1. 1. 1.] 
 [1. 1. 1.] 
 [1. 1. 1.]] 
<NDArray 3x3 @gpu（0）>
```

给定上下文的NDArray，我们可以使用copyto()方法将其复制到另一个上下文中。

```python
x_gpu = x.copyto(mx.gpu(0))
print(x_gpu)
```

```
[[1. 1. 1.] 
 [1. 1. 1.] 
 [1. 1. 1.]] 
<NDArray 3x3 @gpu（0）>
```

运算符的结果与输入具有相同的上下文。

```python
x_gpu  +  z
```

```
[[2. 2. 2.] 
 [2. 2. 2.] 
 [2. 2. 2.]] 
<NDArray 3x3 @gpu（0）>
```

如果我们想要检查一个NDArray程序的上下文，我们可以调用它的`.context`属性。

```python
print(x_gpu.context)
print(z.context)
```

```
gpu（0）
gpu（0）
```

为了在`x1`和`x2`ndarrays上执行操作，我们需要他们都处在同一个环境。如果他们还没有，我们可能需要明确地将数据从一个上下文复制到另一个上下文。你可能会觉得这很烦人。毕竟，我们只是展示了MXNet知道每个NDArray的存在。那么，为什么MXNet不能自动复制`x1`到`x2.context`，然后相加呢？

简而言之，人们使用MXNet来进行机器学习，因为他们期望它很快。但是在不同的上下文之间传递变量是很慢的。所以我们希望你在我们让你做这件事之前，要100％确定你想做些什么。如果MXNet只是在没有崩溃的情况下自动完成复制，那么您可能不会意识到您已经写了一些慢速代码。我们不希望你在StackOverflow上花费你的整个生命，所以我们犯了一些错误是不可能的。

![](http://gluon.mxnet.io/_images/operator-context.png)

## Watch out!

想象一下你的变量z已经存在于你的第二个GPU（`gpu(0)`）上。如果我们调用`z.copyto(gpu(0))`会怎么样？它将复制并分配新的内存，即使该变量已经存在于所需的设备上！

有时候，根据我们的代码运行的环境，两个变量可能已经存在于同一个设备上。所以我们只想在变量目前存在于不同的上下文时才复制。在这些情况下，我们可以=调用`as_in_context()`。如果变量已经是指定的上下文，那么这是一个无操作。

```python
print('id(z):', id(z))
z = z.copyto(mx.gpu(0))
print('id(z):', id(z))
z = z.as_in_context(mx.gpu(0))
print('id(z):', id(z))
print(z)
```

```
id(z): 140291459785224
id(z): 140291460485072
id(z): 140291460485072

[[ 1.  1.  1.]
 [ 1.  1.  1.]
 [ 1.  1.  1.]]
<NDArray 3x3 @gpu(0)>
```