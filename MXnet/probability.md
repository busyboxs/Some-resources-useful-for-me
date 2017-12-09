# Probability and statistics

机器学习是以某种形式进行预测的。考虑到他们的临床病史，我们可能想要预测明年患者发生心脏病的可能性。在异常检测，我们可能要评估如何可能从飞机的喷气发动机一组读数会，是它运行正常。在强化学习中，我们希望代理人在一个环境中聪明地行动。这意味着我们需要考虑在每个可用行动下获得高回报的可能性。而当我们建立推荐系统时，我们也需要考虑概率。例如，如果我们假设曾在一家大型网上书店工作，如果得到提示，我们可能要估计某个特定用户购买特定书籍的可能性。为此，我们需要使用概率和统计的语言。整个课程，专业，论文，职业甚至部门都致力于概率。所以我们这里的目标不是要教导整个学科。相反，我们希望能让你脱颖而出，教给你足够的知识，让你知道开始建立你的第一个机器学习模型所需的一切，并且有足够的主题味道，你可以自己开始探索希望。

到目前为止，我们已经谈了很多关于概率的问题，却没有详细说明它们究竟是什么，或者给出了具体的例子。考虑到基于照片区分猫狗的问题，让我们更加认真。这听起来可能更简单，但这实际上是一个艰巨的挑战。首先，问题的难度可能取决于图像的分辨率。

|20 px|40 px|80 px|160 px|320 px|
|:----:|:----:|:----:|:----:|:----:|
|![](http://gluon.mxnet.io/_images/whitecat20.jpg)|![](http://gluon.mxnet.io/_images/whitecat40.jpg)|![](http://gluon.mxnet.io/_images/whitecat80.jpg)|![](http://gluon.mxnet.io/_images/whitecat160.jpg)|![](http://gluon.mxnet.io/_images/whitecat320.jpg)|
|![](http://gluon.mxnet.io/_images/whitedog20.jpg)|![](http://gluon.mxnet.io/_images/whitedog40.jpg)|![](http://gluon.mxnet.io/_images/whitedog80.jpg)|![](http://gluon.mxnet.io/_images/whitedog160.jpg)|![](http://gluon.mxnet.io/_images/whitedog320.jpg)|

虽然人们很容易识别320像素分辨率的猫和狗，但在40像素处变得具有挑战性，在20像素处变得不可能。换句话说，我们把猫和狗分开很远的距离（因此分辨率很低），辨别能力可能会导致不明智的猜测。概率给我们一个关于我们确定性水平的正式推理方法。如果我们完全确定图像描绘的是一只猫，我们说相应的标签$l$的概率是 $P(l=\mathrm{cat}) = 1$。如果我们没有证据表明 $l =\mathrm{cat}$ 或者说 $l = \mathrm{dog}$ ，那么我们可以说，这两种可能性相等，表示为 $P(l=\mathrm{cat}) = 0.5$。如果我们有合理的信心，但不确定图像是否描绘了一只猫，我们可能会给出一个概率 $.5 < P(l=\mathrm{cat}) < 1.0$。

现在考虑第二种情况：给出一些天气监测数据，我们要预测明天台北下雨的可能性。如果是夏天，下雨可能会以概率 $.5$ 在这两种情况下，我们都有一些利益价值。在这两种情况下，我们都不确定结果。但是这两种情况之间有一个关键的区别。在这第一种情况下，图像实际上是一只狗或一只猫，我们不知道是哪一只。在第二种情况下，如果你相信这样的事情（大多数物理学家），结果实际上可能是一个随机事件。所以概率是一个灵活的语言来推理我们的确定性水平，它可以在广泛的环境中有效地应用。

## Basic probability theory

我们掷骰子，想知道掷到1的机会是多少而不是另一个数字。如果掷骰子是公平的，所有六个结果 $\mathcal{X} = \{1, \ldots, 6\}$ 同样可能发生，所以我们会看到一个 1 有六分之一的机会。我们正式地说 1 以概率 $\frac{1}{6}$ 发生。

对于实际的情况，我们可能不知道这些比例，我们需要检查它是否满足。调查实际的唯一方法是多次实验并记录结果。对于每次掷骰子，我们将观察值 $\{1, 2, \ldots, 6\}$ 。鉴于这些结果，我们要调查观察每个结果的概率。

每个值的一个自然的方法是采取个人计数的价值，并除以总投掷数量。这给了我们一个给定事件的概率的估计。大数定律告诉我们，随着投掷次数的增加，这个估计将越来越接近真实的潜在概率。在讨论这里的细节之前，让我们试试看。

首先，我们导入必要的软件包：

```python
import mxnet as mx
from mxnet import nd
```
   
接下来，我们希望能够掷骰子。在统计中，我们把这个过程称为概率分布抽样的例子。将概率分配给多个离散选择的分布称为多项分布。之后我们会给出一个更为正式的分配定义，但是在较高的层面上，把它看作是对事件概率的分配。在MXNet中，我们可以通过适当命名的`nd.sample_multinomial`函数从多项分布中抽样 。这个函数可以用很多方法调用，但是我们将把重点放在最简单的方面。为了绘制一个样本，我们简单地给出一个概率向量的通过。

```python
probabilities = nd.ones(6) / 6
nd.sample_multinomial(probabilities)
```

```
[3]
<NDArray 1 @cpu(0)>
```

如果你运行这一行（`nd.sample_multinomial(probabilities)`）很多次，你会发现你每次都得到随机值。正如估计骰子的公平性一样，我们通常希望从同一分布中生成许多样本。用Python `for`循环做这个真的很慢，所以`sample_multinomial`支持一次绘制多个样本，返回任意形状的独立样本数组。

```python
print(nd.sample_multinomial(probabilities, shape=(10)))
print(nd.sample_multinomial(probabilities, shape=(5,10)))
```

```
[3 4 5 3 5 3 5 2 3 3]
<NDArray 10 @cpu(0)>

[[2 2 1 5 0 5 1 2 2 4]
 [4 3 2 3 2 5 5 0 2 0]
 [3 0 2 4 5 4 0 5 5 5]
 [2 4 4 2 3 4 4 0 4 3]
 [3 0 3 5 4 3 0 2 2 1]]
<NDArray 5x10 @cpu(0)>
```

现在我们知道如何对一个模具进行取样，我们可以模拟1000次。

```python
rolls = nd.sample_multinomial(probabilities, shape=(1000))
```

然后，我们可以在1000次中，每个数字出现多少次。

```python
counts = nd.zeros((6,1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals
```

首先，我们可以在1000次后检查最后的记录。

```python
totals / 1000
```

```
[ 0.167       0.168       0.175       0.15899999  0.15800001  0.17299999]
<NDArray 6 @cpu(0)>
```

正如你所看到的，任何数字的最低估计概率约为 $.15$ ，最高估计概率为 $0.188$。因为我们从公平骰子中产生的数据，我们知道，每个数字实际上具有的概率 $\frac{1}{6}$，大概 $.167$，所以这些估计是相当不错的。我们也可以想象这些概率如何随着时间的推移而趋于合理的估计。

开始让我们来看看`counts`有`(6, 10)`形状的数组 。对于每个时间步（1000次），计数，说明每个数字已经显示了多少次。所以我们可以规范每个计数向量的第 $j$列除以投掷次数以给出`current`估计概率。计数对象如下所示：

```
counts
```

```
[[   0.    0.    0. ...,  165.  166.  167.]
 [   1.    1.    1. ...,  168.  168.  168.]
 [   0.    0.    0. ...,  175.  175.  175.]
 [   0.    0.    0. ...,  159.  159.  159.]
 [   0.    1.    2. ...,  158.  158.  158.]
 [   0.    0.    0. ...,  173.  173.  173.]]
<NDArray 6x1000 @cpu(0)>
```

通过规范化抛掷次数，我们得到：

```python
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
```

```
[ 0.  1.  0.  0.  0.  0.]
<NDArray 6 @cpu(0)>

[ 0.   0.5  0.   0.   0.5  0. ]
<NDArray 6 @cpu(0)>

[ 0.1980198   0.15841584  0.17821783  0.18811882  0.12871288  0.14851485]
<NDArray 6 @cpu(0)>
```

正如你所看到的，在第一轮投掷之后，我们得到极端的估计，其中一个数字概率 $1.0$ 其他概率 $0$。后 100 次，事情已经看起来更合理。我们可以通过使用 `matplotlib` 绘图软件包来观察这种趋同。如果你没有安装它，现在是安装它的好时机。

```python
from matplotlib import pyplot as plt
plt.plot(estimates[0, :].asnumpy(), label="Estimated P(die=1)")
plt.plot(estimates[1, :].asnumpy(), label="Estimated P(die=2)")
plt.plot(estimates[2, :].asnumpy(), label="Estimated P(die=3)")
plt.plot(estimates[3, :].asnumpy(), label="Estimated P(die=4)")
plt.plot(estimates[4, :].asnumpy(), label="Estimated P(die=5)")
plt.plot(estimates[5, :].asnumpy(), label="Estimated P(die=6)")
plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()
```

![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_18_0.png)

每条实线对应于六个骰子值中的一个，并给出我们估计的可能性，即在1000匝中的每一匝之后评估该骰子达到该值。黑色的虚线给出了真正的潜在概率。随着我们获得更多的数据，实线曲线向真正的答案汇聚。

在我们投掷骰子的例子中，我们介绍了随机变量的概念。一个随机变量，我们在这里表示为 $X$，表示几乎任何都不确定数量。随机变量可以在一组可能性中取一个值。我们用括号来表示集合，例如 $\{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$ 。包含在集合中的项目被称为元素，我们可以说元素 $x$ 在 $S$ 中，写作$x \in S$。符号$∈$被读作“in”并且表示成员。例如，我们可以如实地说 $\mathrm{dog} \in \{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$。在处理骰子时，我们关心的是一个变量 $X \in \{1, 2, 3, 4, 5, 6\}$。

请注意，离散的随机变量（如骰子的值）和连续的变量（如人的体重和身高）之间存在细微的差异。询问两个人是否有完全一样的高度是没有意义的。如果我们采取足够精确的测量，你会发现地球上没有两个人有完全相同的高度。事实上，如果我们采取一个足够好的测量方法，当你醒来和睡觉的时候，你的身高就不会一样了。所以没有什么目的去质疑某个人的可能性是 $2.00139278291028719210196740527486202$米高。概率为$0$. 在这种情况下，更有意义的是询问某个人的身高是否落在给定的区间内，比如在$1.99$ 和 $2.01 $米之间。在这些情况下，我们量化我们将密度看作一个值的可能性。正好$2.0$米的高度没有概率，但非零密度。在任何两个不同的高度之间，我们有非零概率。

有几个重要的概率公理可以记住：

* 对于任何事件$z$，概率从不是负数，即 $\Pr(Z=z) \geq 0$。
* 对于任何两个事件 $Z=z$ 和 $X=x$ ，其并集的概率不超过单独事件的概率和，即 $\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)$。
* 对于任何随机变量，它所能取的所有值的概率总和必须为1，$\sum_{i=1}^n P(Z=z_i) = 1$。
* 对于任何两个互斥的事件 $Z= z$ 和 $X = x$ ，发生的概率等于它们各自概率的和，$\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=z)$。


## Dealing with multiple random variables

很多时候，我们要一次考虑多个随机变量。例如，我们可能要模拟疾病和症状之间的关系。鉴于疾病和症状，说'流感'和'咳嗽'，可能或不可能发生一个概率患者。虽然我们希望两者的概率接近于零，但我们可能要估计这些概率及其相互之间的关系，以便我们可以运用我们的推论来实现更好的医疗保健。

作为一个更复杂的例子，图像包含数百万像素，因此有数百万个随机变量。而且在许多情况下，图像将带有一个标签，用于识别图像中的物体。我们也可以把标签看作一个随机变量。我们甚至可以把所有的元数据想象成随机变量，比如位置，时间，光圈，焦距，ISO，焦距，相机类型等，这些都是随机变量。当我们处理多个随机变量时，有几个感兴趣的量。第一个叫做联合分配 $\Pr(A, B)$。给任何元素 $a$ 和 $b$ ，联合分布让我们回答，$A = a$ 同时 $B=b$ 的概率是多少？可能很清楚，对于任何值 $a$ 和 $b$ ， $\Pr(A,B) \leq \Pr(A=a)$。

这是因为对于 $A$ 和 $B$ 发生， $A$ 必须发生， $B$ 也必须发生（反之亦然）。因此 $A ,B$一不可能超过$A$ 或 $B$ 单独发生。这给我们带来了一个有趣的比率：$0 \leq \frac{\Pr(A,B)}{\Pr(A)} \leq 1$。我们称之为 **条件概率**，用 $\Pr(B|A)$ 表示，$A$ 发生的条件下 $B$ 发生的概率。

使用条件概率的定义，我们可以推导出统计学中最有用和最有名的方程之一 - 贝叶斯定理。其结果如下：通过构造，我们有 $\Pr(A, B) = \Pr(B|A) \Pr(A)$。由于对称性，这也适用于 $\Pr(A,B) = \Pr(A|B) \Pr(B)$。解决其中一个条件变量，我们得到：

$$
\Pr(A|B) = \frac{\Pr(B|A) \Pr(A)}{\Pr(B)}
$$ 

如果我们想从另一个方面推断一个事物，说出因果关系，但是我们只知道相反方向的属性，这是非常有用的。我们做这个工作所需要的一个重要的操作是 边缘化(marginalization)，即从 $\Pr(A,B)$ 确定 $Pr (A)$  和 $Pr (B)$ 。我们可以看到A的概率等于所有可能的B的联合概率，即

$$
\Pr(A) = \sum_{B'} \Pr(A,B') \text{ and } \Pr(B) = \sum_{A'} \Pr(A',B)
$$

一个真正有用的属性来检查依赖性和 独立性。独立是指一个事件的发生不影响另一个事件的发生。在这种情况下，$\Pr(B|A) = \Pr(B)$。统计学家通常使用 $A \perp\!\!\!\perp B$表达这一点。从贝叶斯定理立即可以看出$\Pr(A|B) = \Pr(A)$。在所有其他情况下，我们称之为$A$和$B$依赖。例如，一个骰子的两个连续投掷是独立的。另一方面，灯开关的位置和房间的亮度不是（它们不是完全确定性的，但是，因为我们总是会有一个破损的灯泡，电源故障或开关损坏）。

让我们把我们的技能进行测试。假设医生对患者进行艾滋病检测。这个测试是相当准确的，如果病人是健康的，报告病情发生的概率只有1％的概率，如果病人真的有艾滋病病毒的话，它也不会失败。我们用$D$以指示诊断和$H$来表示艾滋病毒状况。写成表格的结果$\Pr(D|H)$看起来如下：

| |Patient is HIV positive|	Patient is HIV negative
|:----:|:----:|:----:
|Test positive|	1|0.01
|Test negative|0|0.99

请注意，列总和都是$1$（但行数不是），因为条件概率需要总计为$1$，就像概率一样。如果测试回来，让我们计算患有艾滋病的概率。显然这要取决于疾病的常见情况，因为它会影响误报的数量。假设人口相当健康，如 $\Pr(\text{HIV positive}) = 0.0015$。要应用贝叶斯定理，我们需要确定

$$
\Pr(\text{Test positive}) = \Pr(D=1|H=0) \Pr(H=0) + \Pr(D=1|H=1) \Pr(H=1) = 0.01 \cdot 0.9985 + 1 \cdot 0.0015 = 0.011485
$$

因此，我们得到$\Pr(H = 1|D = 1) = \frac{\Pr(D=1|H=1) \Pr(H=1)}{\Pr(D=1)} = \frac{1 \cdot 0.0015}{0.011485} = 0.131$，换句话说，尽管使用了99％的准确性测试，患者实际上只有13.1％的机会患有艾滋病！正如我们所看到的，统计数据可能很不直观。

## Conditional independence

接受这样可怕的消息，病人应该做些什么？他/她可能会要求医生进行另一个测试以获得清晰度。第二个测试有不同的特点（不如第一个）。

| |Patient is HIV positive|Patient is HIV negative|
|:----:|:----:|:----:|
|Test positive|0.98|0.03|
|Test negative|0.02|0.97|

不幸的是，第二个测试也回来了。让我们找出调用贝叶斯定理的必要概率。

*  $\Pr(D_1 = 1 \text{ and } D_2 = 1|H = 0) = 0.01 \cdot 0.03 = 0.0001$
*  $\Pr(D_1 = 1 \text{ and } D_2 = 1|H = 1) = 1 \cdot 0.98 = 0.98$
*  $\Pr(D_1 = 1 \text{ and } D_2 = 1) = 0.0001 \cdot 0.9985 + 0.98 \cdot 0.0015 = 0.00156985$
*  $\Pr(H = 1|D_1 = 1 \text{ and } D_2 = 1) = \frac{0.98 \cdot 0.0015}{0.00156985} = 0.936$

也就是说，第二次测试使我们获得了更高的信心，认为不是一切都好。尽管第二项测试比第一项测试精确得多，但它还是提高了我们的估计。 为什么我们不能再次进行第一次测试呢？毕竟，第一个测试更准确。原因是我们需要第二个测试，确实独立于第一个测试确认事情是可怕的。换句话说，我们默认了 $\Pr(D_1, D_2|H) = \Pr(D_1|H) \Pr(D_2|H)$。统计学家称这些随机变量是有条件独立的。这表示为 $D_1 \perp\!\!\!\perp D_2 | H$。

## Naive Bayes classification

在处理数据时，条件独立性很有用，因为它简化了很多方程。一种流行的算法是朴素贝叶斯分类器。其中的关键假设是属性是相互独立的，给定标签。换句话说，我们有：

$$
p(x|y) = \prod_i p(x_i|y)
$$

使用贝叶斯定理导致分类器 $p(y|x) = \frac{\prod_i p(x_i|y) p(y)}{p(x)}$。不幸的是，这仍然是棘手的，因为我们不知道$p(x)$ 。幸运的是，我们不需要它，因为我们知道$\sum_y p(y|x) = 1$，所以我们总是可以从$p(y|x) \propto \prod_i p(x_i|y) p(y)$恢复归一化。毕竟数学，是时候让一些代码展示如何使用Naive Bayes分类器来区分MNIST分类数据集上的数字。

问题是我们实际上并不知道$p(y)$和$p(x_i|y)$。所以我们首先要给出一些训练数据来估计它。这就是所谓的训练模式。在10个可能的类的情况下，我们简单地计算$n_y$，即类别$y$的出现次数，然后除以出现的总次数。例如，如果我们总共有6万张数字图片，4位数字出现5800次，我们估计它的概率为$\frac{5800}{60000}$ 。同样，要得到$p(x_i|y)$ 我们统计有多少次pixel $i$ 设置为数字$y$然后将其除以数字$y$的出现次数。这是该像素将被打开的概率。

```python
import numpy as np

# we go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the count statistics for p(y) and p(x_i|y)
# We initialize all numbers with a count of 1 to ensure that we don't get a
# division by zero.  Statisticians call this Laplace smoothing.
ycount = nd.ones(shape=(10))
xcount = nd.ones(shape=(784, 10))

# Aggregate count statistics of how frequently a pixel is on (or off) for
# zeros and ones.
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x

# normalize the probabilities p(x_i|y) (divide per pixel counts by total
# count)
for i in range(10):
    xcount[:, i] = xcount[:, i]/ycount[i]

# likewise, compute the probability p(y)
py = ycount / nd.sum(ycount)
```

现在我们计算了所有像素的每个像素发生次数，现在是时候看看我们的模型是如何工作的。是时候来绘制它。我们展示了观察开启像素的估计概率。这些是一些平均看数字。

```python
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print(py)
```

![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_25_0.png)

```
[ 0.09871688  0.11236461  0.09930012  0.10218297  0.09736711  0.09035161
  0.09863356  0.10441593  0.09751708  0.09915014]
<NDArray 10 @cpu(0)>
```

现在我们可以计算图像的可能性，给定模型。这是统计学讲$p(x|y)$，即在某些条件下（如标签）看到特定图像的可能性有多大。由于这在计算上很尴尬（如果许多像素发生小概率，我们可能必须乘以许多小数），所以我们最好是计算其对数。那就是$\log p(x|y) = \sum_i \log p(x_i|y)$，而不是 $p(x|y) = \prod_{i} p(x_i|y)$。

$$
l_y := \sum_i \log p(x_i|y) = \sum_i x_i \log p(x_i = 1|y) + (1-x_i) \log \left(1-p(x_i=1|y)\right)
$$

为了避免重新计算对数，我们预先计算所有像素。

```python
logxcount = nd.log(xcount)
logxcountneg = nd.log(1-xcount)
logpy = nd.log(py)

fig, figarr = plt.subplots(2, 10, figsize=(15, 3))

# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)

    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] += nd.dot(logxcount[:, i], x) + nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= nd.max(logpx)
    # and compute the softmax using logpx
    px = nd.exp(logpx).asnumpy()
    px /= np.sum(px)

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break

plt.show()
```

![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_27_0.png)

正如我们所看到的那样，这个分类器既不能胜任，也对其不正确的估计过于自信。也就是说，即使是可怕的错误，它也会产生接近于1或0的概率。现在不再是我们现在应该使用的分类器了。朴素贝叶斯分类器曾经在80年代和90年代流行，例如垃圾邮件过滤，他们的全盛时期已经结束。糟糕的表现是由于我们在模型中做出的统计假设不正确：我们假定每个像素都是独立生成的，仅取决于标签。这显然不是人类如何写数字，而这个错误的假设导致了我们过于幼稚的（贝叶斯）分类器的崩溃。

## Sampling

随机数只是随机变量的一种形式，而且由于计算机在数字方面特别好，所以在代码中几乎所有的东西最终都会被转换成数字。生成随机数所需的基本工具之一是从分布中抽样。让我们从使用随机数生成器时会发生什么开始。

```python
import random
for i in range(10):
    print(random.random())
```

```python
0.970844720223
0.11442244666
0.476145849846
0.154138063676
0.925771401913
0.347466944833
0.288795056587
0.855051122608
0.32666729925
0.932922304219
```

### Uniform Distribution

这些是一些相当随机的数字。正如我们所看到的，它们的范围在0到1之间，它们是均匀分布的。也就是说，（实际上，应该是，因为这不是一个真正的随机数发生器）没有数字比其他数字更可能的区间。换句话说，任何这些号码的机会落入间隔，说$[0.2,0.3)$在区间$[.593264, .693264)$。内部生成的方式是首先生成一个随机整数，然后将其除以最大范围。如果我们想要直接使用整数，请尝试下面的代码。它会生成0到100之间的随机数。

```python
for i in range(10):
    print(random.randint(1, 100))
```

```
75
23
34
85
99
66
13
42
19
14
```

如果我们想检查`randint`实际上是否一致，该怎么办？直观地说，最好的策略是运行它，比如说100万次，计算它产生每一个值的次数，并确保结果是一致的。

```python
import math

counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid

for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
plt.show()
```

![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_34_0.png)

从上图可以看出，最初的数字看起来很不平衡。如果我们从100个以上的结果中抽取少于100个抽签，这是非常值得期待的。但是即使对于1000个样本来说，抽签之间也有显着的变化。我们真正想要的是绘制概率为$p(x)$数字$x$。

### The categorical distribution
 
很明显，从统一的分布中抽取100个结果是非常简单的。但是，如果我们有不均匀的概率呢？让我们从一个简单的例子开始，一个有偏见的硬币，以0.35的概率出现正面，以0.65的概率出现反面。抽样的一个简单方法是生成$[ 0 ，1 ]$的均匀随机变量， 如果数字小于$0.35$，我们输出正面，否则我们输出反面。让我们试试这个。

```python
# number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

plt.figure(figsize=(15, 8))
plt.semilogx(x, p0)
plt.semilogx(x, p1)
plt.show()
```
![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_36_0.png)

正如我们所看到的，平均而言，这个采样器将产生35％的零和65％的1。现在如果我们有两个以上的可能结果呢？我们可以简单地概括如下。任何给定的概率分布，例如$p = [0.1, 0.2, 0.05, 0.3, 0.25, 0.1]$，我们可以计算它的累积分布（python `cumsum`会为你做这个） $F = [0.1, 0.3, 0.35, 0.65, 0.9, 1]$。一旦我们有了这个，我们绘制一个服从统一分布$U[0,1]$的随机变量$x$， 然后找到区间$F[i-1] \leq x < F[i]$。我们然后返回$i$作为样本。通过建设，击中区间$[F[i-1], F[i])$有概率$p(i)$。

请注意，有比上面更高效的抽样算法。例如，$F$上的二进制搜索对于$n$个随机变量将运行$O(\log n)$时间。还有更聪明的算法，例如恒定时间采样的Alias方法，$O(n)$的处理时间。

### The Normal distribution

正态分布（又称高斯分布）$p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2} x^2\right)$。让我们来绘制它来感受它。

```python
x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
plt.figure(figsize=(10, 5))
plt.plot(x, p)
plt.show()
```
![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_39_0.png)

从这个分布抽样是少得多的。首先，支持是无限的，也就是说，任何$x$密度$p (x)$是正的。其次，密度不均匀。从中抽样有很多技巧 - 所有算法的关键思想是 stratify $p(x)$ in such a way as to map it to the uniform distribution $U[0,1]$ 。一种做法是用概率积分变换。

由$F(x) = \int_{-\infty}^x p(z) dz$表示$p$的累积分布函数(CDF)。这是我们以前使用的累计和的连续版本。现在我们可以用相同的方式定义逆映射$F^{-1}(\xi)$，其中 $\xi$ is drawn uniformly.。与以前不同，我们需要为向量F找到正确的区间（即分段常数函数），我们现在反转函数$F(x)$。

在实践中，这是稍微棘手的，因为在高斯的情况下，反转CDF是困难的。事实证明，二维积分更容易处理，从而产生两个正态随机变量，尽管以两个均匀分布的价格为代价。现在，足以说有内置的算法来解决这个问题。

正态分布还有另一个理想的属性。从某种意义上说，如果我们只是从其他任何分配中抽取足够多的平均数，那么所有的分布都会收敛于这个分布。为了更详细地理解这一点，我们需要介绍三个重要的事情：预期值(expected values), 均值(means) 和方差()variances)。

* The expected value $\mathbb{E}_{x \sim p(x)}[f(x)]$ of a function $f$ under a distribution $p$ is given by the integral $\int_x p(x) f(x) dx$. That is, we average over all possible outcomes, as given by $p$.
* A particularly important expected value is that for the function $f(x)=x$, i.e. $\mu := \mathbb{E}_{x \sim p(x)}[x]$. It provides us with some idea about the typical values of $x$.
* Another important quantity is the variance, i.e. the typical deviation from the mean $\sigma^2 := \mathbb{E}_{x \sim p(x)}[(x-\mu)^2]$. Simple math shows (check it as an exercise) that $\sigma^2 = \mathbb{E}_{x \sim p(x)}[x^2] - \mathbb{E}^2_{x \sim p(x)}[x]$.

以上允许我们改变随机变量的均值和方差。很明显，对于一些均值为$\mu$的随机变量$x$，随机变量$x+ c$有均值$\mu + c$。而且，$\gamma x$有方差$\gamma^2\sigma^2$。将这个应用于正态分布，我们看到平均值为 $\mu$和方差σ为$\sigma^2$具有形式 $p(x) = \frac{1}{\sqrt{2 \sigma^2 \pi}} \exp\left(-\frac{1}{2 \sigma^2} (x-\mu)^2\right)$。注意比例因子$\frac{1}{\sigma}$- 这是由于如果我们用$\sigma$来拉伸分布，我们需要降低$\frac{1}{\sigma}$ 保持相同的概率质量（即分布下的权重总是需要整合到1）。

现在我们准备说出统计学中最基本的定理之一 - 中心极限定理。它指出，对于具有充分良好行为的随机变量，特别是具有良好定义的均值和方差的随机变量，总和趋于正态分布。为了得到一些想法，让我们重复开头所描述的实验，但现在用用整数值的随机变量$\{0, 1, 2\}$。

```python
# generate 10 random sequences of 10,000 random normal variables N(0,1)
tmp = np.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))
# cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()
```

![](http://gluon.mxnet.io/_images/chapter01_crashcourse_probability_41_1.png)

这与最初的例子看起来非常相似，至少在大量变量的平均值的极限内。理论证实了这一点。用随机变量的均值和方差来表示数量

$$
\mu[p] := \mathbf{E}_{x \sim p(x)}[x] \text{ and } \sigma^2[p] := \mathbf{E}_{x \sim p(x)}[(x - \mu[p])^2]
$$

然后我们有 $\lim_{n\to \infty} \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{x_i - \mu}{\sigma} \to \mathcal{N}(0, 1)$。换句话说，不管我们从什么开始，我们总是会聚到一个高斯。这就是高斯在统计中如此受欢迎的原因之一。

### More distributions

存在更多有用的分布。我们建议查阅一本统计书或者在维基百科上查看其中的一些以了解更多细节。

* **二项分布** 它用来描述从同一分布的多次抽签的分布，例如抛10次硬币时的正面数（即一个概率为$\pi$的硬币的正面）。概率为$p(x) = {n \choose x} \pi^x (1-\pi)^{n-x}$。
* **多项分布** 显然，我们可以有两个以上的结果，例如多次掷骰子。在这种情况下，分布为$p(x) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k \pi_i^{x_i}$。
* **泊松分布（Poisson Distribution）** 它被用来模拟以给定速率发生的点事件的发生，例如在一个区域内在给定的时间内到达的雨点的数量（奇怪的事实 - 被马踢死的普鲁士士兵的数量那分配）。给定一个速率$\lambda$，出现次数为$p(x) = \frac{1}{x!} \lambda^x e^{-\lambda}$。
* **Beta**，**Dirichlet**，**Gamma**和**Wishart分布** 它们是统计学家分别称为二项式，多项式，泊松和高斯共轭。没有详细说明，这些分布通常被用作后一组分布的系数的先验，例如Beta分布作为对二项式结果的概率进行建模的先验。
