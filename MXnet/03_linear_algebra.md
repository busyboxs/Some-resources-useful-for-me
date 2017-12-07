# Linear algebra

现在您可以存储和操作数据，让我们简要回顾一下您需要了解大多数模型的基本线性代数的子集。我们将在一个地方介绍所有的基本概念，相应的数学符号，以及它们在代码中的实现。如果你已经对自己的基本线性代数有信心，可以随意浏览或跳过本章。

```python
from mxnet import nd
```

### Scalars

如果你从来没有学过线性代数或机器学习，那么你可能曾经习惯于一次使用一个数字。而且知道如何做基本的事情，比如把它们加在一起或者把它们相乘。例如，在帕洛阿尔托，温度是52华氏度。在形式上，我们称这些值scalars。如果您想将此值转换为摄氏温度（使用公制系统的温度测量更明智的单位），您可以评估表达式$c = (f - 32) * 5/9$设置$f$为$52$。在这个等式中，$32$，$5$，和$9$都是一个标量值。占位符$c$和$f$称为变量，它们代表未知的标量值。

在数学表示法中，我们使用小写字母表示标量($x$,$y$,$z$)。我们也将所有标量的空间表示$\mathcal{R}$。为了方便起见，我们打算详细地讨论一个空间是什么，但是现在记住，如果你想说x是一个标量，你可以简单地说$x \in \mathcal{R}$。符号$\in$可以发音为“in”，只是表示一组中的成员。

在MXNet中，我们使用一个元素创建NDArrays来处理标量。在这个片段中，我们实例化两个标量并且用它们执行一些熟悉的算术运算。

```python
##########################
# Instantiate two scalars
##########################
x = nd.array([3.0])
y = nd.array([2.0])

##########################
# Add them
##########################
print('x + y = ', x + y)

##########################
# Multiply them
##########################
print('x * y = ', x * y)

##########################
# Divide x by y
##########################
print('x / y = ', x / y)

##########################
# Raise x to the power y.
##########################
print('x ** y = ', nd.power(x,y))
```

```
x + y =
[5.]
<NDArray 1 @cpu（0）>
x * y =
[6.]
<NDArray 1 @cpu（0）>
x / y =
[1.5]
<NDArray 1 @cpu（0）>
x ** y =
[9.]
<NDArray 1 @cpu（0）>
```

我们可以通过调用它的`asscalar`方法将任何NDArray转换为Python浮点数

```python

x.asscalar()
```

```
3.0
```

### Vectors

例如，您可以将矢量视为简单的数字列表 `[1.0,3.0,4.0,2.0]`。向量中的每个数字都由一个标量值组成。我们称这些值为矢量的entries或components。通常，我们感兴趣的是那些价值观具有真实世界意义的向量。例如，如果我们正在研究贷款违约的风险，我们可能会将每个申请人与一个向量的组成部分对应于他们的收入，就业时间，以前的违约数量等等联系起来。如果我们正在研究心脏病发作的风险医院病人，我们可以用每个病人的组成部分来捕捉他们最近的生命体征，胆固醇水平，每天运动的分钟数等等。在数学符号中，我们通常将向量表示为粗体，小写字母（$\mathbf{u}$， $\mathbf{v}$，$\mathbf{w}$）。在MXNet中，我们通过带有任意数量组件的1D NDArray来处理矢量。

```python
u = nd.arange(4)
print('u = ', u)
```

```
u =
[ 0.  1.  2.  3.]
<NDArray 4 @cpu(0)>
```

我们可以通过使用下标引用矢量的任何元素。例如，我们可以用$u_4$表示$\mathbf{u}$第4个元素。请注意，元素$u_4$是一个标量，所以当引用它时，我们不会加粗字体。在代码中，我们访问任何元素i通过索引到NDArray。

```python
u[3]
```

```
[ 3.]
<NDArray 1 @cpu(0)>
```

### Length, dimensionality, and, shape

矢量只是一个数字的数组。就像每个数组都有一个长度一样，每个矢量也是如此。在数学符号中，如果我们想要说一个向量x由n个实数表量组成，我们可以表达这种$\mathbf{x} \in \mathcal{R}^n$。矢量的长度通常称为$dimension$。和普通的Python数组一样，我们可以通过调用Python的内置`len()`函数来访问NDArray的长度。

```python
len(u)
```

```
4
```

我们也可以通过它的`.shape`属性来访问矢量的长度。形状是一个元组，它列出NDArray沿其每个轴的维度。因为矢量只能沿一个轴索引，所以它的形状只有一个元素。

```
u.shape
```

```
(4,)
```

请注意，维度是超载的，这往往会混淆人。有些使用矢量的维数来指代其长度（组件数量）。然而，有些使用维度这个词来指代一个数组所具有的轴的数量。在这个意义上，一个标量将会有0维度和向量将有1维度。为了避免混淆，当我们说**2D**数组或**3D**数组时，我们是指具有2或3个轴的数组。但是，如果我们说：`n`维向量，我们是指一个长度为`n`的向量。

```python
a = 2
x = nd.array([1,2,3])
y = nd.array([10,20,30])
print(a * x)
print(a * x + y)
```

### Matrices

正如矢量概括标量从order $0$ 到 order $1$ 一样，矩阵概括矢量从$1D$到$2D$。矩阵，我们用大写字母（$A$，$B$，$C$）在代码中表示为2轴的数组。通常，我们可以绘制一个矩阵作为表，其中每个条目$a_{ij}$表示第$i$行和第$j$列。

$$
\begin{split}A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix}\end{split}
$$

我们可以调用MXnet中任何我们最喜欢的函数来实例化一个`ndarray`,如`ones`，或者`zeros`指定维度`(n,m)`的n行m列矩阵。

```python
A = nd.zeros((5,4))
A
```

```
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]
<NDArray 5x4 @cpu(0)>
```

我们还可以通过调用`ndarray`的``reshape`方法并传递所需的形状来将任何一维数组重塑为二维状态。请注意，形状`n * m`的乘积必须等于原始矢量的长度。

```python
x = nd.arange(20)
A = x.reshape((5, 4))
A
```

```
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]
 [ 16.  17.  18.  19.]]
<NDArray 5x4 @cpu(0)>
```

矩阵是有用的数据结构：它们允许我们组织具有不同变化形式的数据。例如，回到医学数据的例子，矩阵中的行可能对应于不同的患者，而列可能对应于不同的属性。

我们可以通过分别指定行$i$和列$j$来访问矩阵$A$中的标量元素$a_{ij}$。

```python
print('A[2, 3] = ', A[2, 3])
```

```
A[2, 3] =
[ 11.]
<NDArray 1 @cpu(0)>
```

我们也可以抓取整个行$\mathbf{a}_{i,:}$或列$\mathbf{a}_{:,j}$。

```python
print('row 2', A[2, :])
print('column 3', A[:, 3])
```

```
row 2
[  8.   9.  10.  11.]
<NDArray 4 @cpu(0)>
column 3
[  3.   7.  11.  15.  19.]
<NDArray 5 @cpu(0)>
```

我们可以通过`T`转置矩阵。也就是说，如果$B = A^T$，那么对于任意的$i$和$j$，$b_{ij} = a_{ji}$。

```python
A.T
```

```
[[  0.   4.   8.  12.  16.]
 [  1.   5.   9.  13.  17.]
 [  2.   6.  10.  14.  18.]
 [  3.   7.  11.  15.  19.]]
<NDArray 4x5 @cpu(0)>
```

### Tensors

正如向量泛化标量和矩阵推广向量一样，我们实际上可以构建更多轴的数据结构。张量给我们一个讨论具有任意数量的轴的数组的通用方法。例如，矢量是一阶张量，矩阵是二阶张量。

当我们开始处理以$3D$数据结构形式出现的图像时，使用张量将变得更加重要，轴的高度，宽度以及三个（RGB）颜色通道对应。但是在本章中，我们将跳过，确保你知道基础知识。

```python
X = nd.arange(24).reshape((2, 3, 4))
print('X.shape =', X.shape)
print('X =', X)
```

```
X.shape = (2, 3, 4)
X =
[[[  0.   1.   2.   3.]
  [  4.   5.   6.   7.]
  [  8.   9.  10.  11.]]

 [[ 12.  13.  14.  15.]
  [ 16.  17.  18.  19.]
  [ 20.  21.  22.  23.]]]
<NDArray 2x3x4 @cpu(0)>
```

### Element-wise operations

通常情况下，我们想要将函数应用于数组。一些最简单和最有用的功能是element-wise功能。这些操作通过对两个数组的相应元素执行单个标量操作来进行。我们可以从任何从标量映射到标量的函数创建一个element-wise函数。在数学符号中，我们将表示这样的函数为$f: \mathcal{R} \rightarrow \mathcal{R}$。给定任何两个同样的形状的向量$u$，$v$和函数$f$，我们可以通过对所有$i$设置$c_i \gets f(u_i, v_i)$产生一个向量 $\mathbf{c} = F(\mathbf{u},\mathbf{v})$。在这里，我们通过将标量函数提升为按元素的矢量操作产生了矢量值$F: \mathcal{R}^d \rightarrow \mathcal{R}^d$。在MXNet中，常见的标准算术运算符（+， - ，/，\*，**）都被提升为任意形状的相同形状张量的元素运算。

```python
u = nd.array([1, 2, 4, 8])
v = nd.ones_like(u) * 2
print('v =', v)
print('u + v', u + v)
print('u - v', u - v)
print('u * v', u * v)
print('u / v', u / v)
```

```
v =
[ 2.  2.  2.  2.]
<NDArray 4 @cpu(0)>
u + v
[  3.   4.   6.  10.]
<NDArray 4 @cpu(0)>
u - v
[-1.  0.  2.  6.]
<NDArray 4 @cpu(0)>
u * v
[  2.   4.   8.  16.]
<NDArray 4 @cpu(0)>
u / v
[ 0.5  1.   2.   4. ]
<NDArray 4 @cpu(0)>
```

我们可以在任何两个相同形状的张量上调用元素操作，包括矩阵。

```python
B = nd.ones_like(A) * 3
print('B =', B)
print('A + B =', A + B)
print('A * B =', A * B)
```

```
B =
[[ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]
 [ 3.  3.  3.  3.]]
<NDArray 5x4 @cpu(0)>
A + B =
[[  3.   4.   5.   6.]
 [  7.   8.   9.  10.]
 [ 11.  12.  13.  14.]
 [ 15.  16.  17.  18.]
 [ 19.  20.  21.  22.]]
<NDArray 5x4 @cpu(0)>
A * B =
[[  0.   3.   6.   9.]
 [ 12.  15.  18.  21.]
 [ 24.  27.  30.  33.]
 [ 36.  39.  42.  45.]
 [ 48.  51.  54.  57.]]
<NDArray 5x4 @cpu(0)>
```

### Basic properties of tensor arithmetic

任何顺序的标量，矢量，矩阵和张量都有一些我们经常依赖的好的属性。例如，您可能已经注意到了元素操作的定义，给定具有相同形状的操作数，任何元素操作的结果都是相同形状的张量。另一个方便的性质是，对于所有张量，乘以标量产生相同形状的张量。在数学中，给出两个张量$X$和$Y$具有相同形状，$\alpha X + Y$具有相同的形状。（数学家把这称为AXPY操作）。

```python
a = 2
x = nd.ones(3)
y = nd.zeros(3)
print(x.shape)
print(y.shape)
print((a * x).shape)
print((a * x + y).shape)
```

```
(3,)
(3,)
(3,)
(3,)
```

形状不是在标量的加法和乘法下保存的唯一属性。这些操作也保留向量空间的成员资格。但是我们将在本章的后半部分推迟讨论，因为启动并运行第一个模型并不重要。

### Sums and means

下一个更复杂的东西，我们可以对任意张量计算他们的元素的总和。在数学符号中，我们使用$\sum$表示。表达矢量$\mathbf{u}$中$d$个元素的总和，我们可以写为$\sum_{i=1}^d u_i$。在代码中，我们可以调用`nd.sum()`。

```python
nd.sum(u)
```

我们也可以类似地对任意形状的张量元素进行求和。例如，一个$m \times n$矩阵$A$的元素的总和可以写成 $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。

```python
nd.sum(A)
```

相关量是平均值，也称为平均值。我们通过将总和除以元素的总数来计算均值。用数学符号，我们可以写出一个向量$\mathbf{u}$的平均值为$\frac{1}{d} \sum_{i=1}^{d} u_i$和矩阵A上的平均值为$\frac{1}{n \cdot m} \sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。在代码中，我们可以调用`nd.mean()`任意形状的张量：

```python
print(nd.mean(A))
print(nd.sum(A) / A.size)
```

### Dot products

最基本的操作之一是点积。给定两个向量$\mathbf{u}$和$\mathbf{v}$，点积 $\mathbf{u}^T\mathbf{v}$是相应元素点积的总和：$\mathbf{u}^T\mathbf{v} = \sum_{i=1}^{d} u_i \cdot v_i $。

```python
nd.dot(u, v)
```

请注意，我们可以通过执行元素乘法然后求和来等价地表达两个向量的点积：`nd.dot(u, v)`

```python
nd.sum(u * v)
```

点积在各种情况下都很有用。例如，给定一组权重$\mathbf{w}$，某些值$u$的加权和可以表示为点积 $\mathbf{u}^T\mathbf{w}$。当权重是非负的，总和为1（$\sum_{i=1}^{d} {w_i} = 1$ ），点积表示加权平均值。当两个向量各有一个长度（我们将在规范部分讨论下面的长度意味着什么）时，点积也可以捕获它们之间角度的余弦。

### Matrix-vector products

现在我们知道如何计算点积，我们可以开始理解矩阵向量积。我们首先看一个矩阵$A$一个和一个列向量$x$。

$$
\begin{split}A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1m} \\
 a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nm} \\
\end{pmatrix},\quad\mathbf{x}=\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix}\end{split}
$$

我们可以根据行矢量来显示矩阵

$$
\begin{split}A=
\begin{pmatrix}
\cdots & \mathbf{a}^T_{1} &...  \\
\cdots & \mathbf{a}^T_{2} & \cdots \\
 & \vdots &  \\
 \cdots &\mathbf{a}^T_n & \cdots \\
\end{pmatrix},\end{split}
$$

每一个$\mathbf{a}^T_{i} \in \mathbb{R}^{m}$是表示矩阵$A$的第$i$行。

那么矩阵向量乘积$\mathbf{y} = A\mathbf{x}$是一个简单的列向量$\mathbf{y} \in \mathbb{R}^n$每个条目$y_i$是点积$\mathbf{a}^T_i \mathbf{x}$。

$$
\begin{split}A\mathbf{x}=
\begin{pmatrix}
\cdots & \mathbf{a}^T_{1} &...  \\
\cdots & \mathbf{a}^T_{2} & \cdots \\
 & \vdots &  \\
 \cdots &\mathbf{a}^T_n & \cdots \\
\end{pmatrix}
\begin{pmatrix}
 x_{1}  \\
 x_{2} \\
\vdots\\
 x_{m}\\
\end{pmatrix}
= \begin{pmatrix}
 \mathbf{a}^T_{1} \mathbf{x}  \\
 \mathbf{a}^T_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^T_{n} \mathbf{x}\\
\end{pmatrix}\end{split}
$$

所以，你可以通过矩阵乘法考虑的$A\in \mathbb{R}^{m \times n}$作为从$\mathbb{R}^{m}$到$\mathbb{R}^{n}$的投影变换。

这些转变变得相当有用。例如，我们可以将旋转表示为乘以方阵。我们将在后面的章节中看到，我们也可以使用矩阵向量产品来描述神经网络中每一层的计算。

用`ndarray`代码表示矩阵矢量乘积，我们使用与`nd.dot()`相同的函数实现点积。当我们对一个矩阵`A`和一个向量`a`调用nd.dot(A, x)时，`MXNet`知道要执行一个矩阵向量乘积。请注意，列维度`A`必须与`x`维度相同。

```python
nd.dot(A, u)
```

### Matrix-matrix multiplication

如果你已经掌握了点积和矩阵向量乘法，那么矩阵 - 矩阵乘法应该是非常简单的。

假设我们有两个矩阵，$A \in \mathbb{R}^{n \times k}$ 和 $B \in \mathbb{R}^{k \times m}$：

$$
\begin{split}A=\begin{pmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{pmatrix},\quad
B=\begin{pmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{pmatrix}\end{split}
$$

矩阵乘积$C = AB$，最简单的就是从行向量的角度考虑$A$，从列向量角度考虑$B$：

$$
\begin{split}A=
\begin{pmatrix}
\cdots & \mathbf{a}^T_{1} &...  \\
\cdots & \mathbf{a}^T_{2} & \cdots \\
 & \vdots &  \\
 \cdots &\mathbf{a}^T_n & \cdots \\
\end{pmatrix},
\quad B=\begin{pmatrix}
\vdots & \vdots &  & \vdots \\
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
 \vdots & \vdots &  &\vdots\\
\end{pmatrix}.\end{split}
$$

这里请注意，每一行矢量$\mathbf{a}^T_{i}$在$\mathbb{R}^k$并且每个列向量$\mathbf{b}_j$还在于$\mathbb{R}^k$。

然后生成矩阵乘积$C \in \mathbb{R}^{n \times m}$ 我们简单地计算每个条目$c_{ij}$作为点积$\mathbf{a}^T_i \mathbf{b}_j$。

$$
\begin{split}C = AB = \begin{pmatrix}
\cdots & \mathbf{a}^T_{1} &...  \\
\cdots & \mathbf{a}^T_{2} & \cdots \\
 & \vdots &  \\
 \cdots &\mathbf{a}^T_n & \cdots \\
\end{pmatrix}
\begin{pmatrix}
\vdots & \vdots &  & \vdots \\
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
 \vdots & \vdots &  &\vdots\\
\end{pmatrix}
= \begin{pmatrix}
\mathbf{a}^T_{1} \mathbf{b}_1 & \mathbf{a}^T_{1}\mathbf{b}_2& \cdots & \mathbf{a}^T_{1} \mathbf{b}_m \\
 \mathbf{a}^T_{2}\mathbf{b}_1 & \mathbf{a}^T_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^T_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^T_{n} \mathbf{b}_1 & \mathbf{a}^T_{n}\mathbf{b}_2& \cdots& \mathbf{a}^T_{n} \mathbf{b}_m
\end{pmatrix}\end{split}
$$

你可以想到矩阵相乘$AB$作为简单地执行$m$矩阵向量乘积并将结果拼接在一起形成$n \times m$矩阵。就像普通的点积和矩阵向量乘积一样，我们可以`MXNet`通过使用`nd.dot()`来计算矩阵相乘。

```python
A = nd.ones(shape=(3, 4))
B = nd.ones(shape=(4, 5))
nd.dot(A, B)
```

```
[[ 4.  4.  4.  4.  4.]
 [ 4.  4.  4.  4.  4.]
 [ 4.  4.  4.  4.  4.]]
<NDArray 3x5 @cpu(0)>
```

### Norms

在我们开始实施模型之前，我们将介绍最后一个概念。线性代数中一些最有用的运算符是范数。非正式地，他们告诉我们矢量或矩阵有多大。我们代表与符号规范$\|\cdot\|$。该$\cdot$ 在这个表达式中只是一个占位符。例如，我们将表示向量$\mathbf{x}$或矩阵$A$的范数分别表示为$\|\mathbf{x}\|$，$\|A\|$ 。

所有准则必须满足性质的少数：
*  $\|\alpha A\| = |\alpha| \|A\|$
*  $\|A + B\| \leq \|A\| + \|B\|$
*  $\|A\| \geq 0$
*  if $\forall {i,j}, a_{ij} = 0$，then $\|A\|=0$

换句话说，第一条规则说如果我们用常数因子α来缩放矩阵或向量的所有分量，α其范数也通过相同常数因子的绝对值来缩放。第二个规则是熟悉的三角形不等式。第三条规则简单地说，范数必须是非负的。这是有道理的，在大多数情况下，任何事物的最小尺寸都是0.最后的规则基本上说，最小的范数是由全零组成的矩阵或向量来实现的。可以定义一个给非零矩阵赋零范数的范数，但是不能给零矩阵赋予非零范数。这是一口，但如果你消化它，那么你可能已经在这里写下了重要的概念。

如果你还记得小学的欧几里德距离（think 毕达哥拉斯定理），那么非负面的和三角形的不平等就可能敲响了钟声。你可能会注意到，范数听起来很像距离的测量。

事实上，欧几里德距离$\sqrt{x_1^2 + \cdots + x_n^2}$是一种范数。具体来说就是$\ell_2$范数。在矩阵的条目上执行类似的计算，例如$\sqrt{\sum_{i,j} a_{ij}^2}$，被称为Frobenius范数。更多的时候，在机器学习中我们使用平方$\ell_2$范数（notated $\ell_2^2$）。我们也常用$\ell_1$范数。$\ell_1$范数只是绝对值的总和。它具有不太重视异常值的便利性。

计算$\ell_2$范数，我们可以调用`nd.norm()`。

```python
nd.norm(u)
```

为了计算$\ell_1$范数，我们可以简单地执行绝对值，然后求和元素。

```python
nd.sum(nd.abs(u))
```

### Norms and objectives

虽然我们不想太过于自己，但我们确实希望你能预见为什么这些概念是有用的。在机器学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率。最大限度地减少预测和地面实况观测之间的距离。将向量表示分配给项目（如文字，产品或新闻文章），以使相似项目之间的距离最小化，并使不相似项目之间的距离最大化。通常，这些目标可能是机器学习算法中最重要的组成部分（除了数据本身之外）被表达为范数。

### Intermediate linear algebra

如果你已经做到了这一点，并且理解我们所涵盖的一切，那么说实话，你已经准备好开始建模了。如果你感觉有点尴尬，这是一个非常合理的地方继续前进。你已经知道几乎所有的线性代数来实现许多实际上有用的模型，并且当你想了解更多时，你总是可以回头看看。

但是线性代数还有很多，即使是关于机器学习。在某个时候，如果你打算从事机器学习，那么到目前为止，你需要知道更多的知识。在本章的其余部分中，我们将介绍一些有用的，更高级的概念。

#### Basic vector properties

向量是有用的，除了数据结构来携带数字。除了读写矢量分量的值以及执行一些有用的数学运算之外，我们还可以用一些有趣的方式分析矢量。

一个重要的概念是向量空间的概念。以下是构成向量空间的条件：

* 加性公理（我们假设x，y，z都是矢量）： $x+y = y+x$ 和$(x+y)+z = x+(y+z)$和 $0+x = x+0 = x$ 和 $(-x) + x = x + (-x) = 0$。
* 乘法公理（我们假定x是一个向量，a和b是标量）：$0 \cdot x = 0$ 和 $1 \cdot x = x$ 和 $(a b) x = a (b x)$。
* 分配公理（我们假设x和y是矢量，a，b是标量）：$a(x+y) = ax + ay$ 和 $(a+b)x = ax +bx$。

#### Special matrices

本教程中将使用许多特殊的矩阵。我们来仔细看看它们：

*  **对称矩阵**  这些矩阵，其中对角线下方和上方的条目是相同的。换句话说，我们有那个$M^\top = M$。这种矩阵的例子是描述成对距离的例子，即$M_{ij} = \|x_i - x_j\|$。同样，Facebook的友谊图可以写成一个对称矩阵，其中$M_{ij} = 1$如果$i$和$j$是朋友和$M_{ij} = 0$如果他们不是。请注意， Twitter图形是不对称的 - $M_{ij} = 1$, i.e. $i$ following $j$ does not imply that $M_{ij} =1$, i.e.  $j$ following $i$。
* **反对称矩阵**  这些矩阵满足$M^\top = -M$。注意，任何任意的矩阵总是可以用$M = \frac{1}{2}(M + M^\top) + \frac{1}{2}(M - M^\top)$来分解成对称矩阵和反对称矩阵 。
* **对角占优矩阵**  这些矩阵是非对角元素相对于主对角线元素较小的矩阵。特别是我们有那$M_{ii} \geq \sum_{j \neq i} M_{ij}$ 和 $M_{ii} \geq \sum_{j \neq i} M_{ji}$。如果一个矩阵有这个属性，我们通常可以近似于$M$由其对角线。这通常表示为$\mathrm{diag}(M)$。
* **正定矩阵**  这些矩阵具有很好的属性，其中$x^\top M x > 0$每当$x \neq 0$。直观地说，他们是一个矢量的平方范的推广$\|x\|^2 = x^\top x$。每当$M = A^\top A$时很容易检查 $x^\top M x = x^\top A^\top A x = \|A x\|^2$。有一个更为深刻的定理，指出所有的正定矩阵都可以用这种形式写出来。

#### Conclusions

在几页（或一个Jupyter笔记本）中，我们已经教会了所有需要了解大量神经网络的线性代数。线性代数当然还有很多。很多数学是对机器学习有用。例如，矩阵可以分解为因子，这些分解可以揭示真实世界数据集中的低维结构。机器学习的整个子领域集中在使用矩阵分解及其对高阶张量的推广以发现数据集中的结构并解决预测问题。但是这本书着重于深度学习。而且我们相信，一旦你弄脏了在实际数据集上部署有用的机器学习模型，你就会更倾向于学习更多的数学。所以，虽然我们保留稍后介绍更多数学的权利，但我们将在这里结束这一章。

如果您渴望更多地了解线性代数，请参阅我们的一些关于该主题的最喜爱的资源*有关基础知识的详细入门知识，请查阅Gilbert Strang的书“[ Introduction to Linear Algebra](http://math.mit.edu/~gs/linearalgebra/)” * Zico Kolter的 [Linear Algebra Reivew and Reference](http://cs229.stanford.edu/section/cs229-linalg.pdf)
