# Introduction

在我们开始写作之前，这本书的作者和大部分劳动力一样，必须变成含咖啡因的。我们跳上车，开始开车。有一个Android，亚历克斯叫“Okay Google”，唤醒手机的语音识别系统。然后Mu命令“directions to Blue Bottle coffee shop”。电话很快显示了他的命令的转录。它也认识到我们正在寻求指导，并启动了地图应用程序来满足我们的要求。一旦启动，地图应用程序确定了一些路线。在每条路线旁边，电话显示预测的通行时间。虽然我们为了教学的方便而编写了这个故事，但是它展示了在几秒钟内，我们与智能手机的日常互动可以使用多种机器学习模式。

如果你以前从来没有使用过机器学习，那么你可能想知道我们到底在说什么。你可能会问，“这不仅仅是编程吗？”或“ 机器学习甚至意味着什么？”首先，要清楚的是，我们通过编写计算机程序来实现所有的机器学习算法。事实上，我们使用与计算机科学其他领域相同的语言和硬件，但并非所有计算机程序都涉及机器学习。针对第二个问题，精确定义一个像机器学习一样广泛的研究领域是很难的。这有点像回答：“数学是什么？”。但我们会尽力给你足够的直觉开始。

## A motivating example

我们每天所进行的大部分计算机程序都可以从最基本的原则进行编码。当您将商品添加到购物车时，您将触发电子商务应用程序，以将条目存储在购物车数据库表中，从而将您的用户标识与产品ID相关联。我们可以从最初的原则写出这样一个程序，在没有见过真正的客户的情况下启动。当写这个应用程序很容易，你不应该使用机器学习。

幸运的是（对于ML科学家的社区），但是，对于许多问题，解决方案并不是那么容易。回到我们获取咖啡的假故事，想象一下，写一个程序来回应“Alexa”，“Okay，Google”或“Siri”这样的 唤醒词。尝试使用计算机和代码编辑器自己编写实现。你将如何从最初的原则写出这样一个程序？想一想...这个问题很难。麦克风每秒会收集大约44,000个样本。什么规则可以可靠地从原始音频片段映射到对片段是否包含唤醒词的充满信心的预测{yes, no}？如果卡住了，别担心。我们不知道如何从头开始编写这样的程序。这就是为什么我们使用机器学习。

![](http://gluon.mxnet.io/_images/wake-word.png)

这是诀窍。通常，即使我们不知道如何明确地告诉计算机如何从输入映射到输出，我们仍然能够自己执行认知专长。换句话说，即使你不知道如何编程计算机来识别单词“Alexa”，你自己也可以识别单词“Alexa”。具有这种能力的武装，我们可以收集大量的数据集包含音频的样本，并标明标签，是否包含唤醒词。在机器学习方法中，我们没有明确地设计一个系统来识别唤醒词。相反，我们用一些参数来定义一个灵活的程序。这些是我们可以调节以改变程序行为的旋钮。我们把这个程序称为模型。一般来说，我们的模型只是一个将输入转化为输出的机器。在这种情况下，模型接收到一个音频片段作为输入，并产生一个答案{yes, no}，我们希望反映（或不）片段是否包含唤醒词作为输出。

如果我们选择正确的模型，那么应该存在一个旋钮的设置，使得模型每次听到“Alexa”这个词时都会触发yes。还应该有另一个旋钮设置能够对单词“Apricot” 触发yes。我们预计相同的模型应该适用于“Alexa”识别和“Apricot”识别，因为这些是类似的任务。但是，我们可能需要一个不同的模型来处理根本不同的投入或产出。例如，我们可以选择一种不同类型的机器将图像映射到标题，或者从英语句子到汉语句子。

正如你所猜测的，如果我们只是随机设置旋钮，模型可能不会识别“Alexa”，“Apricot”或任何其他英文单词。一般来说，在深度学习中，学习恰恰是指在训练期间更新模型的行为（通过扭动旋钮）。

训练过程通常如下所示：

1. 从不能做任何有用的事情随机初始化的模型开始。
2. 抓取一些标记的数据（例如音频片段和相应的 {yes,no}标签）
3. 调整旋钮，使模型相对于这些样本影响更少
4. 重复，直到模型很棒。

![](http://gluon.mxnet.io/_images/ml-loop.png)

总而言之，我们编写了一个程序，可以学习识别唤醒词，如果我们提供一个大的标记数据集，而不是编码唤醒词识别器。您可以通过将数据集与数据编程一起呈现来确定程序的行为。

我们可以通过为我们的机器学习系统提供许多猫和狗的例子来“编程”猫探测器，例如下面的图像：

<img src="http://gluon.mxnet.io/_images/death_cap.jpg" width = "200" height = "150" alt="图片名称" align=center />  <img src="http://gluon.mxnet.io/_images/cat2.jpg" width = "200" height = "150" alt="图片名称" align=center />  <img src="http://gluon.mxnet.io/_images/dog1.jpg" width = "200" height = "150" alt="图片名称" align=center />  <img src="http://gluon.mxnet.io/_images/dog2.jpg" width = "200" height = "150" alt="图片名称" align=center />

这样，如果它是一只猫，探测器最终会学会发出一个非常大的正数，如果它是一只狗，会学会发出一个非常大的负数，如果不确定，它会接近零，但这只是勉强表明什么机器学习可以做。

## The dizzying versatility of machine learning（机器学习令人眼花缭乱的多功能性）

这是机器学习背后的核心思想：我们不是设计具有固定行为的程序，而是设计能够随着他们获得更多经验而改进的程序。这个基本的想法可以采取多种形式。机器学习可以解决许多不同的应用领域，涉及许多不同类型的模型，并根据许多不同的学习算法进行更新。在这个特定的情况下，我们描述了一个监督学习实例应用于自动语音识别问题。

机器学习是一个多功能的工具集，让您可以在许多不同的情况下使用数据，在这些情况下，基于规则的简单系统会失败或者构建起来可能非常困难。由于其多功能性，机器学习对于新手来说可能相当混乱。例如，机器学习技术已经广泛应用于搜索引擎，自动驾驶，机器翻译，医疗诊断，垃圾邮件过滤，游戏（象棋，go），面部识别，数据匹配，计算保险费和添加过滤器照片等多种应用中。

尽管这些问题之间存在肤浅的差异，但它们中的许多都具有共同的结构，并且可以通过深入的学习工具来解决。它们大部分是相似的，因为它们是我们无法直接在代码中编写它们的行为的问题，但我们可以用数据编程它们。通常这些程序的最直接的语言是数学。在本书中，我们将介绍最少量的数学符号，但与机器学习和神经网络的其他书不同，我们将始终保持以真实示例和真实代码为基础的对话。

## Basics of machine learning

当我们考虑识别唤醒词的任务时，我们将由片段和标签组成的数据集放在一起。然后，我们描述（尽管抽象）如何训练一个机器学习模型来预测给定片段的标签。这个设置，从例子中预测标签，只是一个ML的过程，它被称为监督学习。即使在深入的学习中，还有很多其他的方法，我们将在随后的章节中讨论。要学习机器学习，我们需要四件事情：

1. 数据
2. 一个如何转换数据的模型
3. 衡量我们做得如何的损失函数
4. 调整模型参数的算法，使损失函数最小化

### 数据

一般来说，我们拥有的资料越多，工作就越容易。当我们有更多的数据时，我们可以训练更强大的模型。数据是深度学习再度兴起的核心，深度学习中许多最令人兴奋的模型在没有大数据集的情况下无法正常工作。下面是一些机器学习从业者经常使用的数据的例子：

* **图像**：通过智能手机拍摄的照片或从网上收集的照片，卫星图像，医疗条件照片，超声波和CT扫描和MRI等放射图像等。
* **文本**：电子邮件，高中作文，推特，新闻文章，医生的笔记，书本和翻译句子的语料库等。
* **音频**：发送到智能设备（如Amazon Echo，iPhone或Android手机），音频书籍，电话，音乐录音等的语音命令
* **视频**：电视节目和电影，YouTube视频，手机镜头，家庭监控，多摄像头追踪等
* **结构化数据**：网页，电子病历，汽车租赁记录，电费单等

### 模型

通常这些数据看起来与我们想要完成的有很大的不同。例如，我们可能会有人的照片，想知道他们是否看起来很高兴。我们可能需要一个能够摄取高分辨率图像并输出快乐分数的模型。虽然一些简单的问题可以通过简单的模型来解决，但是在这种情况下我们要问很多。为了做好自己的工作，我们的快乐探测器需要将成千上万的低级特征（像素值）转换成另一端（幸福分数）相当抽象的东西。选择正确的模型是困难的，不同的模型更适合不同的数据集。在本书中，我们将主要关注深度神经网络。这些模型由从上到下链接在一起的数据的许多连续转换组成，因此名称深度学习。在讨论深层网络的方法中，我们也将讨论一些更简单，更浅的模型。

### 损失函数

为了评估我们做得如何，我们需要将模型的输出与真值进行比较。损失函数为我们提供了一种测量产出有多糟糕的方法。例如，假设我们训练了一个模型来从图像中推断患者的心率。如果模型预测病人的心率是100bpm，那么当实际情况是60bpm时，我们需要一种与模型沟通的方式，那就是做一个糟糕的工作。

同样，如果模型为电子邮件分配评分，表明它们是垃圾邮件的概率，那么我们需要一种告诉模型何时预测不好的方法。通常，机器学习的学习部分包括最小化这种损失函数。通常，模型有许多参数。这些参数的最佳值是我们需要“学习”的，通常是通过最小化 观测数据的训练数据所带来的损失。不幸的是，在训练数据上做得好并不能保证我们能做好（看不见的）测试数据，所以我们需要跟踪两个量。

* **Training Error**：这是用于通过最小化训练集上的损失来训练我们的模型的数据集上的错误。这相当于在一个学生可能用来准备真正的考试的所有练习考试中做得很好。结果令人鼓舞，但决不能保证在期末考试中取得成功。

* **Test Error**：这是在看不见的测试集上发生的错误。这可能会偏离训练错误。这种情况，当一个模型不能概括为不可见的数据时，称为过度拟合。在现实生活中，尽管在实践考试中表现良好，但这相当于把考试搞砸了。

### 优化算法

最后，为了尽量减少损失，我们需要一些方法来获取模型及其损失函数，并搜索一组参数，以最大限度地减少损失。在神经网络上工作的最流行的优化算法遵循称为梯度下降的方法。简而言之，他们希望看到，如果你稍微改变一下参数，那么每个参数的训练集丢失将会移动。然后他们更新参数在减少损失的方向。

在下面的章节中，我们将更详细地讨论一些类型的机器学习。我们从目标列表开始，即机器学习可以做的事情列表。请注意，这些目标是由一系列如何完成它们的技巧，即训练，数据类型等等补充的。下面的列表实际上只是足以使读者的胃口变好，并且在我们谈论时给我们一个共同的语言问题。随着我们的进展，我们会引入更多的这类问题。


## Supervised learning

监督式学习解决了输入数据预测目标的任务。目标（通常也称为标签）通常表示为y。输入数据点（通常也称为examples or instances）通常用x表示。目标是产生一个模型fθ映射输入x预测fθ(x)

以一个具体的例子来说明这个描述，如果我们在医疗保健领域工作，那么我们可能要预测一个病人是否会有心脏病发作。这个观察，心脏病发作或没有心脏病发作，将是我们的标签y。输入数据x可能是心率，舒张压和收缩压等生命体征。

由于选择参数θ，监管起作用θ，我们（the supervisors）为模型提供了一系列带标签的例子(xi,yi)，其中每个例子xi匹配它正确的标签。

在概率条件下，我们通常有兴趣估计条件概率P(y|x)。虽然这只是机器学习的几种方法之一，但监督式学习在实践中占了机器学习的大部分。部分原因在于，许多重要的任务可以用一些可用的证据来估计一些未知的可能性：

* 给出CT图像，预测癌症与非癌症。
* 用英文预测正确的法语翻译。
* 根据本月的财务报告数据预测下个月股票的价格。

即使简单的描述“predict targets from inputs”监督学习可以采取很多形式，需要大量的建模决定，根据类型，大小和输入和输出的数量。例如，我们使用不同的模型来处理序列（如文本串或时间序列数据）和处理固定长度的向量表示。本书前9部分将深入探讨这些问题。

说白了，学习过程看起来像这样。抓取一大堆示例输入，随机选择它们。获取每个的真值标签。这些输入和相应的标签（所需的输出）一起构成训练集合。我们将训练数据集输入监督学习算法。所以这里的监督式学习算法是一个函数，它将一个数据集作为输入，并输出另一个函数，学习模型。然后，给定一个学习模型，我们可以采取一个新的前所未见的输入，并预测相应的标签。

![](http://gluon.mxnet.io/_images/supervised-learning.png)

### Regression

也许最简单的监督学习任务是回归。考虑一下，例如从一个房屋销售数据库收集的一组数据。我们可以构造一个表格，每一行对应一个不同的房屋，每一列对应一些相关的属性，例如房屋面积，卧室数量，卫生间数量和分钟数（步行）到镇中心。在形式上，我们把这个数据集中的一行称为一个 特征向量，并把这个对象（例如一个房子）与一个例子关联起来 。

如果您住在纽约或旧金山，而且您不是亚马逊，Google，微软或Facebook的首席执行官，那么您的家庭（平方英尺，卧室数量，浴室数量，步行距离）可能看起来像：[ 100 ，0 ，0.5 ，60 ][100，0，0.5，60]。但是，如果你住在匹兹堡，它可能看起来更像 [ 3000 ，4 ，3 ，10 ][3000，4，3，10]。像这样的特征向量对于所有经典的机器学习问题是必不可少的。我们通常将表示任何一个示例x的特征向量xi和我们所有例子X的特征向量集合X。

>什么使问题回归实际上是输出。假设你正在市场上建一个新房子，你可能要估计一个房子的公平市场价值，因为有这样的特点。目标价值，即销售价格，是一个实数。我们表示任何个人目标yi(对应于例子xi）和所有目标y的集合（对应于所有例子X）。当我们的目标在某个范围内具有任意的实际值时，我们称之为回归问题。我们模型的目标是产生与实际目标值非常接近的预测（在我们的例子中是价格的猜测）。
>我们表示这些预测ŷi,如果符号似乎不熟悉，那么现在就忽略它。在后面的章节中我们会更加彻底的解开它。

很多实际问题都是很好的回归问题。预测用户将分配给电影的评级是一个回归问题，如果您在2009年设计了一个伟大的算法来完成这一壮举，那么您可能赢得了100万美元的Netflix奖金。预测医院病人的住院时间也是一个回归问题。一个好的经验法则是任何多少？或多少？ 问题应该建议回归。*“这个手术需要多少小时？”... 回归 *“这张照片有多少只狗？”... 回归。但是，如果您可以轻松地将问题描述为“这是一个___？”，那么很可能是分类，这是我们下面将要介绍的一个不同的基本问题类型。

即使你以前从来没有使用机器学习，你也可能非正式地完成了一个回归问题。想象一下，例如，你的下水道已经修好，你的承包商花了x1 = 3小时从你的污水管道去除垃圾。然后，她寄给你一张账单y1= $350。现在想象一下你的朋友雇佣了的同一个承包商x2 = 2小时，她收到了一张账单y2= $250。如果有人问你对即将到来的垃圾清理发票有多少期待，你可能会做出一些合理的假设，比如更多的工作时间花费更多的美元。你也可以假设有一些基本收费，承包商每小时收费。如果这些假设成立，那么给出这两个数据点，就可以确定承包商的定价结构：每小时100美元，再加上50美元，在您的房子出现。如果你遵循了那么多，那么你已经理解了线性回归背后的高层次思想。

在这种情况下，我们可以生成完全符合承包商价格的参数。有时候这是不可能的，例如，如果除了你的两个特征之外，某些因素还有一些差异。在这些情况下，我们将尝试学习将我们的预测与观测值之间的距离最小化的模型。在我们的大多数章节中，我们将重点放在两个中的一个非常普遍的亏损，L1损失 ，其中l(y,y′)=∑i|yi−y′i|和L2的损失l(y,y′)=∑i(yi−y′i)2。正如我们稍后会看到的那样， L2损失对应于我们的数据被高斯噪声破坏的假设，而L1损失对应于拉普拉斯分布的噪声假设。

### Classification

虽然回归模型对解决多少问题很有帮助？问题，很多问题不能舒服地弯曲到这个模板。例如，一家银行希望将支票扫描添加到他们的移动应用程序。这将涉及到客户用智能手机的相机拍下支票的照片，机器学习模型需要能够自动理解图像中看到的文字。还需要理解手写文本才能更加强大。这种系统被称为光学字符识别（OCR），其解决的问题类型称为分类。它使用了一组不同于那些用于回归的算法。

在分类中，我们希望查看一个特征向量，就像图像中的像素值，然后在一些选项集合中，预测哪个这个实例属于哪个类别（通常称为类别）。对于手写数字，我们可能有10个类，对应于数字0到9.最简单的分类形式是当只有两个类时，我们称之为二元分类的问题。例如，我们的数据集X可以由动物的图像组成，我们的标签Y可能是类 {cat，dog}。在回归中，我们寻求一个回归者输出一个真实的价值ŷ，在分类中，我们寻找一个分类器，其输出ŷ是预测的班类别分配。

由于本书的技术含量越来越高，因此很难优化一个只能输出硬分类任务的模型，例如猫或狗。用概率的语言来表达模型要容易得多。给出一个例子 x，模型分配一个概率ŷķ到每个标签k。因为这些是概率，所以它们需要是正数，和为1。这意味着我们只需要 K- 1个数字来给出K个类别的概率。二元分类很容易看出来。如果一个不公平的硬币有0.6（60％）的概率出现正面，那么就有0.4（40％）的概率出现反面。回到我们的动物分类实例，分类器可能会看到一个图像，并输出图像是猫的概率Pr（y = cat|x）= 0.9。我们可以解释这个数字，说分类器是90％确定的图像描绘一只猫。预测阶级的概率的大小是一个信心的概念。这不是唯一的自信心，我们将在更高级的章节中讨论不确定性的不同概念。

当我们有两个以上的可能类时，我们称之为多分类。常见的例子包括手写字符[0, 1, 2, 3 ... 9, a, b, c, ...]识别。虽然我们通过尝试使L1或L2损失函数最小化来解决回归问题，但分类问题的常见损失函数被称为交叉熵。在`MXNet Gluon`中，相应的损失函数，可以在[这里](http://mxnet.io/api/python/gluon.html#mxnet.gluon.loss.SoftmaxCrossEntropyLoss)查看。

请注意，最有可能的类别并不一定是您要用来做决定的类别。假设你在你家后院找到这个美丽的蘑菇：

![Death cap - do not eat!](http://gluon.mxnet.io/_images/death_cap.jpg)

现在，假设你建立了一个分类器并训练它根据照片预测蘑菇是否有毒。说，我们的毒检测分类输出Pr(y=deathcap∣image)=0.2。换句话说，分类器是80％的信心，我们的蘑菇不是deathcap。不过，你必须是一个傻瓜吃它。那是因为美味晚餐的某些好处不值得有20％的机会死亡。换句话说，不确定风险的影响远远超过了收益。我们来看看数学。基本上，我们需要计算我们发生的预期风险，即我们需要将结果的概率乘以与之相关的收益（或损害）：

L(action∣x)=Ey∼p(y∣x)[loss(action,y)]

因此，吃蘑菇引起的损失L是L(a=eat∣x)=0.2∗∞+0.8∗0=∞，而丢弃它的成本是 L(a=discard∣x)=0.2∗0+0.8∗1=0.8。

我们很幸运，正如任何一位真菌学家告诉我们的，上面实际上是一个deathcap。分类可以变得比二元，多类甚至多标签分类复杂得多。例如，有一些处理层次结构的分类变体。层次结构假定在许多类之间存在一些关系。因此，并非所有的错误都是相同的 - 我们宁愿错误地分类到相关的类，而不是远程的类。通常这被称为分层分类。一个早期的例子是由于Linnaeus，谁在组织等级的动物。

![](http://gluon.mxnet.io/_images/taxonomy.jpg)

在动物分类的情况下，把poodle误认为schnauzer可能并不是一件坏事，但是如果把一只poodle认为dinosaur，我们的模型会付出巨大的代价。相关的层次结构可能取决于您打算如何使用模型。例如，rattle snakes和garter snakes可能在系统发育树上很接近，但把rattler误认为garter可能是致命的。

### Tagging

一些分类问题不适合二进制或多类分类设置。例如，我们可以训练一个普通的二元分类器来区分猫和狗。鉴于目前的计算机视觉状况，我们可以通过现成的工具轻松完成此任务。尽管如此，无论我们的模型如何准确，当分类器遇到像这样的图像时，我们可能会遇到麻烦：

![](http://gluon.mxnet.io/_images/catdog.jpg)

正如你所看到的，图中有一只猫。还有一条狗，一条轮胎，一些草，一扇门，混凝土，铁锈，单独的草叶等等。根据我们最终要对模型做什么，把它当作一个二元分类问题可能并不是很多感。相反，我们可能想给模型话说，形象地再现了一只猫的选项和一只狗，或者没有一只猫，也没有一只狗。

学习预测不相互排斥的类别的问题被称为多标签分类。自动标记问题通常被最好地描述为多标签分类问题。想想人们可能会应用到科技博客上的帖子，例如“机器学习”，“技术”，“小工具”，“编程语言”，“linux”，“云计算”，“AWS”。一篇典型的文章可能会应用5-10个标签，因为这些概念是相关的。有关“云计算”的帖子可能会提到“AWS”，关于“机器学习”的帖子也可能涉及“编程语言”。

在处理生物医学文献时，我们也必须处理这类问题，正确地标记文章是重要的，因为它允许研究人员对文献进行详尽的评论。在美国国家医学图书馆，许多专业的注释者遍历每篇被PubMed索引的文章，将每个与MeSH相关的术语联系起来，这些术语集合了大约28k个标签。这是一个耗时的过程，注释者通常在存档和标记之间有一年的时间滞后。机器学习可以在这里用来提供临时标签，直到每篇文章都可以有一个适当的手动审查。事实上，几年来，BioASQ组织已经举办了一个比赛来做到这一点。

### Search and ranking

有时候，我们不只是想把每个例子都分配给一个桶或一个真实的值。在信息检索领域，我们希望对一组项目进行排名。以网络搜索为例，目标不是确定某个特定的页面是否与查询相关，而是应该为用户显示过多的搜索结果中的哪一个。我们真正关心相关搜索结果的排序，我们的学习算法需要从一个更大的集合中产生有序的元素子集。换句话说，如果我们被要求产生字母表中的前5个字母，返回`A B C D E`和返回`C A B E D`之间是有区别的。即使结果集是相同的，集合内的排序仍然是重要的。

解决这个问题的一个可能的方法是对可能集合中的每个元素以及相应的相关性得分进行评分，然后检索评分最高的元素。 PageRank是这种相关性分数的早期例子。其中一个特点是它不依赖于实际的查询。相反，它只是帮助排序包含查询条件的结果。如今，搜索引擎使用机器学习和行为模型来获得依赖于查询的相关性分数。有很多专门讨论这个问题的会议。

### Recommender systems

推荐系统是与搜索和排名相关的另一个问题设置。就目标是向用户显示一组相关项目而言，这些问题是相似的。主要区别在于在推荐系统中强调对特定用户的个性化。例如，对于电影推荐，SciFi粉丝的结果页面和Woody Allen喜剧鉴赏家的结果页面可能会有很大的不同。

出现这样的问题，例如电影，产品或音乐推荐。在某些情况下，客户将提供他们喜欢产品的详细信息（例如亚马逊产品评论）。在其他一些情况下，如果他们不满意结果（跳过播放列表中的标题），他们可能会简单地提供反馈。通常，这样的系统努力估计一些分数，例如估计的评价或购买的概率，给定用户ui和产品 pj。

给定这样的模型，那么对于给定的用户，我们可以检索具有最大分数yij的一组对象然后用作推荐。生产系统相当先进，在计算这样的分数时要考虑详细的用户活动和项目特征。以下图片是亚马逊推荐的深度学习书籍的一个例子，它基于根据作者的偏好调整的个性化算法。

![](http://gluon.mxnet.io/_images/deeplearning_amazon.png)

### Sequence Learning

到目前为止，我们已经看到了我们有一些固定数量的输入和产生固定数量的输出的问题。在我们考虑从固定的一组特征来预测房价之前：平方英尺，卧室的数量，浴室的数量，到市中心的步行时间。我们还讨论了从固定尺寸的图像到属于固定数量类别的预测概率，或者采用用户ID和产品ID，以及预测星级。在这些情况下，一旦我们将固定长度的输入馈送到模型中以生成输出，模型立即会忘记刚刚看到的内容。

如果我们的投入真的都具有相同的规模，并且连续的投入真的没有任何关系，那么这可能是好的。但是，我们将如何处理视频片段？在这种情况下，每个片段可能由不同数量的帧组成。如果我们考虑前面或后面的帧，我们猜测每一帧中发生的事情可能会更强。语言也一样。一个流行的深度学习问题是机器翻译：用某种源语言读取句子，用另一种语言预测翻译的任务。

这些问题也出现在医学上。我们可能需要一个模型来监测重症监护病房的病人，如果他们在接下来的24小时内死亡的风险超过某个阈值，就会发出警报。我们绝对不希望这个模型每小时都把它所了解的关于患者病史的所有知识都扔掉，只是根据最近的测量结果做出预测。

这些问题是机器学习更令人兴奋的应用之一，它们是序列学习的例子。他们需要一个模型来摄入输入序列或发出输出序列（或两者！）。后面这些问题有时被称为 seq2seq问题。语言翻译是一个seq2seq问题。从口头讲话转录文本也是一个seq2seq问题。虽然不可能考虑所有类型的序列转换，但值得一提的是一些特殊情况：

### Tagging and Parsing

这涉及用属性注释文本序列。换句话说，输入和输出的数量基本相同。例如，我们可能想知道动词和主题在哪里。或者，我们可能想知道哪些词是命名实体。一般来说，目标是基于结构和语法假设来分解和注释文本以获得一些注释。这听起来比实际上更复杂。下面是一个非常简单的例子，用一个标签来标注一个句子，表明哪个单词指的是命名实体。

|Tom|
|----|
|Ent|

### Automatic Speech Recognition

用语音识别，输入序列x是扬声器的声音，输出y是发言者所说的文字抄本。挑战在于音频帧（音频通常是以8kHz或16kHz采样）比文本多得多，即音频和文本之间没有1：1的对应关系，因为数千个样本对应于单个口语单词。这些是seq2seq问题，输出比输入短得多。

----D----e----e-----p------- L----ea------r------ni-----ng---

![](http://gluon.mxnet.io/_images/speech.jpg)

### Text to Speech

文本到语音（TTS）是语音识别的逆向。换句话说，输入x是文本和输出y是一个音频文件。在这种情况下，输出比输入长得多。虽然人类很容易识别不好的音频文件，但对于计算机来说，这并不是那么微不足道。

### Machine Translation

与识别语音的情况不同，相应的输入和输出以相同的顺序（对齐之后）发生，在机器翻译中，顺序反转可能是至关重要的。换句话说，当我们仍然将一个序列转换成另一个序列时，输入和输出的数量以及相应的数据点的顺序都不被假定为相同的。考虑一下德国人厌恶倾向的例子（Alex写在这里），把动词放在句子结尾。

|German|Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?|
|----|----|
|English|Did you already check out this excellent tutorial?|
|Wrong alignmen t|Did you yourself already this excellent tutorial looked-at?|

存在一些相关的问题。例如，确定用户阅读网页的顺序是二维布局分析问题。同样，对于对话问题，我们需要考虑到世界知识和以前的状态。这是一个活跃的研究领域。

## Unsupervised learning

到目前为止，所有的例子都与监督式学习相关，也就是说，我们为模型提供了大量的例子和一堆相应的目标值。你可以把监督学习看作是一个非常专业化的工作和一个非常严格的老板。老板站在你的肩膀上，告诉你在各种情况下要做什么，直到你学习从情况到行动的映射。为这样的老板工作听起来很蹩脚。另一方面，这个老板很容易。你只要尽可能快地认出模式，模仿他们的行为。

用完全相反的方式，为一个不知道自己想要做什么的老板工作，可能会感到沮丧。但是，如果你打算成为一名数据科学家，你最好习惯它。老板可能只是给你一大堆数据，告诉你用它做一些数据科学！这听起来很模糊，因为它就是很模糊。我们把这类问题称为无监督学习，我们可以提出的问题的类型和数量仅受我们的创造力的限制。我们将在后面的章节中介绍一些无监督的学习技巧。为了现在激起你的胃口，我们描述一些你可能会问的问题：

>* 我们可以找到一些准确汇总数据的原型吗？给定一组照片，我们可以把它们分成风景照片，狗，宝宝，猫，山峰等图片？同样，给定用户浏览活动的集合，我们可以将他们分组到具有类似行为的用户吗？这个问题通常被称为集群。
>* 我们可以找到一小部分准确捕捉数据相关属性的参数吗？球的轨迹可以很好地描述为球的速度，直径和质量。裁缝们已经开发了很少的参数来描述人体的形状，以便于穿衣服。这些问题被称为子空间估计问题。如果依赖性是线性的，则称为主成分分析。
>* 在欧几里德空间中是否存在（任意结构化的）物体的表示（即ℝ中的向量空间）ñ[Rñ）这样的符号属性可以很好地匹配？这被称为 表示学习，用来描述实体及其关系，如罗马 - 意大利+法国=巴黎。
>* 有没有关于我们观察到的大部分数据的根本原因的描述？例如，如果我们有关于房价，污染，犯罪，地点，教育，工资等的人口统计数据，我们是否可以根据经验数据发现它们之间的关系？领域定向图形模型和因果关系这一交易。
>* 最近的一个重要和令人兴奋的发展是生成对抗网络。它们基本上是一种合成数据的程序化方法。基本的统计机制是检查真假数据是否相同的测试。我们会投入一些笔记本给他们。

## Interacting with an environment

到目前为止，我们还没有讨论数据实际来自哪里，或者当机器学习模型产生输出时究竟发生了什么。这是因为监督学习和无监督学习不能以非常复杂的方式解决这些问题。无论哪种情况，我们都会先抓一大堆数据，然后进行模式识别，而不再与环境进行交互。因为所有的学习都是在算法与环境断开之后进行的，所以这被称为离线学习。对于监督式学习，过程如下所示：

![](http://gluon.mxnet.io/_images/data-collection.png)

这种简单的离线学习有其魅力。好处是我们可以孤立地担心模式识别，而不需要处理这些其他问题，但是缺点是问题表达式是相当有限的。如果你更雄心勃勃，或者如果你长大了读阿西莫夫的机器人系列，那么你可能会想象人工智能机器人不仅能做出预测，而且能够在世界上采取行动。我们想要考虑智能代理，而不仅仅是预测模型。这意味着我们需要考虑选择 行动，而不是仅仅做出预测。而且，与预测不同，行动实际上影响着环境。如果我们想培养一个聪明的代理人，我们必须考虑其行为可能影响代理人未来观察的方式。

考虑到与环境的相互作用，可以打开一整套新的建模问题。环境：

* 记得我们以前做过什么？
* 想要帮助我们，例如用户将文本读入语音识别器？
* 想打败我们，即像垃圾邮件过滤（针对垃圾邮件发送者）或玩游戏（vs对手）这样的对抗性设置？
* 不在乎（在大多数情况下）？
* 有变化的动态（稳定与随着时间的推移）？

最后一个问题提出了协变量的问题（当训练和测试数据不同时）。这是我们大多数人在参加讲师写的考试时遇到的问题，而作业是由他的助教组成的。我们将简要介绍强化学习和敌对学习，这两个环境明确考虑与环境的交互。

## Reinforcement learning

如果您有兴趣使用机器学习来开发与环境相互作用并采取行动的代理，那么您可能会关注强化学习（RL）。这可能包括机器人应用，对话系统，甚至开发视频游戏AI。将深度神经网络应用于RL问题的深度强化学习（DRL）已经风靡一时。在Atari游戏中仅使用视觉输入击败人类的突破性深度Q网络，以及在棋盘游戏Go上淘汰世界冠军的AlphaGo计划是两个突出的例子。

强化学习给出了一个问题的非常一般的说法，即一个主体在一系列时间步骤中与一个环境进行交互。在每个时间步骤t，代理人从环境收到一些观察 ot，必须选择一个动作一个操作at然后传回环境。最后，代理人收到来自环境奖励rt。代理人然后收到后续的观察，并选择后续的操作，等等。RL代理的行为受政策的约束。简而言之，政策只是一个从（环境）观察到行动的功能。强化学习的目标是制定一个好的政策。

![](http://gluon.mxnet.io/_images/rl-environment.png)

很难夸大RL框架的一般性。例如，我们可以将任何监督学习问题视为RL问题。假设我们有分类问题。我们可以创建一个与 每个类对应的动作的RL代理。然后，我们可以创造一个环境，给予与原有监督问题完全相同的报酬。

这就是说，RL还可以解决监督学习不能解决的许多问题。例如，在监督学习中，我们总是期望训练输入与正确的标签相关联。但是在RL中，我们并不认为每个观察都会告诉我们最佳的行动。一般来说，我们只是得到一些奖励。而且，环境可能甚至不能告诉我们哪些行为导致了奖励。

考虑例如国际象棋的游戏。唯一真正的奖励信号是在游戏结束的时候，当我们赢了，我们可以分配奖励1，或者当我们输了，我们可以分配奖励-1。所以强化学习者必须处理信用分配问题。对于10月11日获得晋升的员工也是如此。该晋升可能反映了上一年的大量精心选择的行动。在未来获得更多的促销活动需要搞清楚促销过程中采取了哪些行动。

强化学习者也可能需要处理部分可观察性的问题。也就是说，目前的观察可能不会告诉你关于你当前状态的一切。说一个清洁机器人发现自己被困在一个房子的许多相同的壁橱之一。推断机器人的准确位置（以及状态）可能需要在进入衣柜之前考虑其先前的观察。

最后，在任何情况下，强化学习者可能知道一个好的策略，但是可能还有许多其他更好的策略，这个策略从来没有尝试过。强化学习者必须不断选择是否利用当前最好的策略作为策略，或者探索策略的空间，潜在地放弃一些短期奖励来交换知识。

## MDPs, bandits, and friends

一般强化学习问题是一个非常普遍的设置。行动影响后续的观察。奖励只对应于所选的行动。环境可能完全或部分被观察到。对这一切复杂性的一次性考虑可能会要求太多的研究人员。而且，并不是每个实际问题都表现出这种复杂性 因此，研究人员已经研究了一些强化学习问题的特殊情况。

当完全观察到环境时，我们将RL问题称为马尔可夫决策过程（MDP）。当国家不依赖于以前的行动时，我们称这个问题为一个背景性的土匪问题。当没有国家的时候，只有一套最初未知的奖励，这个问题就是经典的多手武装问题。

何时不使用机器学习
让我们仔细看看编程数据的思想，通过考虑乔尔·格鲁斯在工作面试中所经历的交流。面试官要求他给Fizz Buzz编码。这是一个儿童游戏，玩家数量从1到100， 每当数字可以被3整除时，将会说'嘶嘶'，只要它能被5整除，就会说'嗡嗡'，只要满足这两个标准，就会说'fizzbuzz'。否则，他们只会说明这个号码。它看起来像这样：

`1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 ...`

解决这个问题的传统方式非常简单。

```python
res = []
for i in range(1, 101):
    if i % 15 == 0:
        res.append('fizzbuzz')
    elif i % 3 == 0:
        res.append('fizz')
    elif i % 5 == 0:
        res.append('buzz')
    else:
        res.append(str(i))
print(' '.join(res))
```

`1 2 fizz 4 buzz fizz 7 8 fizz buzz 11 fizz 13 14 fizzbuzz 16 17 fizz 19 buzz fizz 22 23 fizz buzz 26 fizz 28 29 fizzbuzz 31 32 fizz 34 buzz fizz 37 38 fizz buzz 41 fizz 43 44 fizzbuzz 46 47 fizz 49 buzz fizz 52 53 fizz buzz 56 fizz 58 59 fizzbuzz 61 62 fizz 64 buzz fizz 67 68 fizz buzz 71 fizz 73 74 fizzbuzz 76 77 fizz 79 buzz fizz 82 83 fizz buzz 86 fizz 88 89 fizzbuzz 91 92 fizz 94 buzz fizz 97 98 fizz buzz`

如果你是一名优秀的程序员，这并不是很令人兴奋。乔尔继续在机器学习中“实现”这个问题。为了获得成功，他需要一些东西：

* 数据X 和标签Y [1, 2, 3, 4, ...]['fizz', 'buzz', 'fizzbuzz', identity
* 训练数据，即系统应该做什么的例子。如[(2, 2), (6, fizz), (15, fizzbuzz), (23, 23), (40, buzz)]
* 将数据映射到计算机可以更容易处理的功能，例如。这是可选的，但如果你有它的帮助很多。x -> [(x % 3), (x % 5), (x % 15)]

有了这个，Joel在TensorFlow（代码）中写了一个分类器。面试官不知所措，分类器没有完美的准确性。

很明显，这是愚蠢的。为什么要用更复杂和容易出错的东西来替代几行Python？然而，很多情况下，一个简单的Python脚本根本不存在，但一个3岁的孩子将完美地解决问题。幸运的是，这正是机器学习来拯救的地方。

## Conclusion

机器学习是巨大的。我们不可能全部覆盖。另一方面，神经网络很简单，只需要初等数学。那么我们开始吧。
