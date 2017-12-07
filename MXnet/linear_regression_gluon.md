# Linear regression with gluon

现在，我们已经实现了一个从无到有的整个神经网络，使用`mx.ndarray`和`mxnet.autograd`，让我们看看我们如何能够做出同样的模式，而少做很多工作。

再次，我们导入一些包，这次添加`mxnet.gluon`到依赖关系列表中。

在[32]中：
从 __future__  进口 print_function 
进口 mxnet  为 MX 
从 mxnet  进口 第二， autograd ， 胶子
设置上下文
我们也想设置一个上下文来告诉胶子在哪里做大部分的计算。

在[33]中：
data_ctx  =  mx 。cpu （）
model_ctx  =  mx 。cpu （）
构建数据集
我们再来看看线性回归的问题，并坚持使用相同的合成数据。

在[34]中：
num_inputs  =  2 
num_outputs  =  1 
num_examples  =  10000

def  real_fn （X ）：
    return  2  *  X [：， 0 ]  -  3.4  *  X [：， 1 ]  +  4.2

X  =  nd 。random_normal （shape = （num_examples ， num_inputs ））
noise  =  0.01  *  nd 。random_normal （shape = （num_examples ，））
y  =  real_fn （X ） +  noise
加载数据迭代器
我们将坚持DataLoader处理数据批处理。

在[35]中：
batch_size  =  4 
train_data  =  胶子。数据。的DataLoader （胶子。数据。ArrayDataset （X ， ÿ ）
                                      的batch_size = 的batch_size ， 洗牌= 真）
定义模型
当我们从头开始实施时，我们必须单独分配参数，然后将它们组合在一起作为模型。虽然知道如何从头开始做是件好事，但是gluon，我们可以从预定义的图层组成一个网络。对于线性模型，调用适当的图层Dense。它被称为密集层，因为输入中的每个节点都连接到后续层中的每个节点。这个描述看起来太多了，因为我们在这里只有一个（非输入）层，而这个层只包含一个节点！但是在后面的章节中，我们通常会使用具有多个输出的网络，所以我们不妨从节点层面开始思考。因为一个线性模型只包含一个Dense图层，我们可以用一行来实例化它。

正如在前面的笔记本，我们具有2 inputdimension和为1的输出尺寸的最直接的方式来实例化一个Dense与这些尺寸层是指定的输入的数目和输出的数目。

在[36]中：
净 =  胶子。nn 。密集（1 ， in_units = 2 ）
而已！我们已经有了一个神经网络。就像我们之前笔记本上的手工制作的模型一样，这个模型有一个权重矩阵和偏向量。

在[37]中：
打印（净。重量）
印刷（网。偏压）
出[37]：
参数dense4_weight（shape =（1，2），dtype = None）
参数dense4_bias（shape =（1，），dtype = None）

在这里，net.weight而net.bias实际上不是NDArrays。他们是这个Parameter班的实例。我们使用Parameter而不是直接访问NDAarrays有几个原因。例如，它们为初始化值提供了方便的抽象。与NDArrays不同，参数可以同时与多个上下文关联。当我们开始考虑跨多个GPU的分布式学习时，这将在未来的章节中派上用场。

在gluon所有的神经网络都是由Blocks（gluon.Block）组成的。块是只需要输入和生成输出的单元。块还包含我们可以更新的参数。在这里，我们的网络只有一层，所以直接访问我们的参数是很方便的。当我们的网络由10个层组成时，这不会太好玩。无论我们的网络多么复杂，我们都可以通过调用collect_params()如下来获取它的所有参数：

在[38]中：
净。collect_params （）
出[38]：
dense4_（
  参数dense4_weight（shape =（1，2），dtype = None）
  参数dense4_bias（shape =（1，），dtype = None）
）
返回的对象是一个gluon.parameter.ParameterDict。这对于检索和操作Parameter对象组是一个方便的抽象。大多数情况下，我们要检索神经网络中的所有参数：

在[39]中：
类型（净。collect_params （））
出[39]：
mxnet.gluon.parameter.ParameterDict
初始化参数
一旦我们初始化了参数，我们就可以访问它们的底层数据和上下文，并且还可以通过神经网络提供数据来生成输出。但是，我们还不能走。如果我们尝试通过调用来调用模型net(nd.array([[0,1]]))，我们将面对以下可怕的错误消息：

RuntimeError: Parameter dense1_weight has not been initialized...

那是因为我们还没有告诉 我们参数gluon的初始值应该是什么！我们通过调用.initialize()ParameterDict 的方法初始化参数 。我们需要传递两个参数。

一个初始化程序，其中许多都在mx.init模块中。
参数应该存在的上下文。在这种情况下，我们会通过model_ctx。大多数情况下，这可能是GPU或GPU列表。
MXNet提供了各种常用的初始化器mxnet.init。为了保持与我们手工建立的模型一致，我们将使用一个标准的正态分布进行采样来初始化每个参数 mx.init.Normal(sigma=1.)。

在[40]中：
净。collect_params （）。初始化（MX 。INIT 。普通（西格玛= 1 ）， CTX = model_ctx ）
延迟初始化
当我们调用时initialize，将gluon每个参数与一个初始化器相关联。然而，实际的初始化被推迟，直到我们进行第一次正向传递。换句话说，参数只在需要的时候被初始化。如果我们尝试打电话，net.weight.data() 我们会得到以下错误：

DeferredInitializationError: Parameter dense2_weight has not been initialized yet because initialization was deferred. Actual initialization happens during the first forward pass. Please pass one batch of data through the network before accessing Parameters.

通过gluon模型传递数据很容易。我们只是采样一批适当的形状，并调用net就好像它是一个函数。这将调用网络的forward()方法。

在[41]中：
example_data  =  nd 。阵列（[[ 4 ，7 ]]）
净（example_data ）
出[41]：

[-1.33219385]]
<NDArray 1x1 @cpu（0）>
现在net已经初始化了，我们可以访问它的每个参数。

在[42]中：
打印（净。重量。数据（））
打印（净。偏压。数据（））

[[-0.25217363 -0.04621419]]
<NDArray 1x2 @cpu（0）>

[0]
<NDArray 1 @cpu（0）>
形状推断
回想一下，以前，我们通过实例化了我们的网络 。我们可以利用的一个光滑的特征是参数的形状推断。因为在我们通过网络传递数据之前，我们的参数从不起作用，所以我们实际上并不需要声明输入维（）。让我们再试一次，但让更多的工作：gluon.nn.Dense(1, in_units=2)gluonin_unitsgluon

在[43]中：
净 =  胶子。nn 。密（1 ）
网。collect_params （）。初始化（MX 。INIT 。普通（西格玛= 1 ）， CTX = model_ctx ）
gluon在接下来的章节中，我们将详细阐述这个以及更多的内部工作。

定义损失
我们不是写自己的损失函数，而是通过实例化来访问平方误差gluon.loss.L2Loss。就像层和整个网络，胶子的损失只是一个Block。

在[44]中：
square_loss  =  胶子。损失。L2Loss （）
优化
每次从头开始编写随机梯度下降，我们可以实例化一个gluon.Trainer，传递一个参数字典。请注意，sgd优化器gluon实际上使用了动量和削减的SGD（如果需要，可以关闭），因为这些修改使得它更好地收敛。稍后我们将详细讨论一系列优化算法。

在[45]中：
教练 =  胶子。训练者（净。collect_params （）， 'SGD' ， { 'learning_rate' ： 0.0001 }）
执行训练循环
你可能已经注意到，表达我们的模型比较简单一些gluon。例如，我们不必单独分配参数，定义我们的损失函数，或者实现随机梯度下降。gluon一旦我们开始使用更复杂的模型，依赖抽象的好处将会大大增加。但是，一旦我们掌握了所有的基础知识，训练循环本身与从零开始实施所有工作时所做的工作非常相似。

刷新你的记忆。对于某些数字epochs，我们将完整地传递数据集（train_data），一次抓取一个小批量的输入和相应的地面实况标签。

然后，对于每一批，我们将通过下面的仪式。为了使这个过程成为最大的仪式，我们将逐字重复：

通过网络执行正向传递，生成预测（yhat）和损失（loss）。
通过网络向后传递来计算渐变loss.backward()。
通过调用我们的SGD优化器更新模型参数（注意，我们不需要告诉trainer.step哪些参数，而只是数据量，因为我们已经在初始化中执行了trainer）。
在[46]中：
历元 =  10 
loss_sequence  =  [] 
num_batches  =  num_examples  /  的batch_size

为 È  在 范围（信号出现时间）：
    cumulative_loss  =  0 
    ＃内环
    为 我， （数据， 标签） 在 枚举（train_data ）：
        数据 =  数据。as_in_context （model_ctx ）
        label  =  label 。as_in_context （model_ctx ）
        与 autograd 。record （）：
            output  =  net （data）
            损失 =  square_loss （输出， 标签）
        损失。落后（）
        教练。step （batch_size ）
        cumulative_loss  + =  nd 。意味着（损失）。asscalar （）
    print （“Epoch ％s ，loss：％s ”  ％ （e ， cumulative_loss  /  num_examples ））
    loss_sequence 。追加（cumulative_loss）

时代0，损失：3.44980202263
大纪元1，损失：2.10364257665
大纪元2，损失：1.28279426137
大纪元3，损失：0.782256319318
大纪元4，损失：0.477034088909
大纪元5，损失：0.290909814427
大纪元6，损失：0.177411796283
大纪元7，损失：0.108197494675
大纪元8，损失：0.0659899789031
大纪元9，损失：0.040249745576
可视化学习曲线
现在我们来看看SGD如何通过绘制学习曲线来学习线性回归模型。

在[47]中：
＃绘制
内联估计损失函数％matplotlib 的收敛性

import  matplotlib 
import  matplotlib.pyplot  as  plt

plt 。图（NUM = 无，figsize = （8 ， 6 ））
PLT 。plot （loss_sequence ）

＃添加一些花里胡哨的
plt 。网格（真， 其中= “两个” ）
plt 。xlabel （'epoch' ，fontsize = 14 ）
plt 。ylabel （'平均损失' ，fontsize = 14 ）
出[47]：
<matplotlib.text.Text at 0x7efc87a7f0f0>
../_images/chapter02_supervised-learning_linear-regression-gluon_32_1.png
我们可以看到，损失函数迅速收敛到最优解。

获取学习的模型参数
作为一个额外的完整性检查，因为我们从高斯线性回归模型生成的数据，我们希望确保学习者设法恢复模型参数，分别设置为重 2 ，- 3.42， - 3.4偏移量为4.24.2。

在[48]中：
params  =  net 。collect_params （） ＃这将返回一个ParameterDict

print （''params'的类型是' ，type （params ））

＃ParameterDict是Parameter类对象的字典
＃因此，我们可以从中读取参数。

对于 PARAM  在 PARAMS 。值（）：
    印刷（PARAM 。名称，PARAM 。数据（））
“params”的类型是一个<class“mxnet.gluon.parameter.ParameterDict'>
dense5_weight
[[1.7913872 -3.10427046]]
<NDArray 1x2 @cpu（0）>
dense5_bias
[3.85259581]
<NDArray 1 @cpu（0）>
结论
正如你所看到的，即使是像线性回归这样一个简单的例子， gluon也可以帮助你写出快速而干净的代码。接下来，我们将重复这个练习来获得多层感知器，将这些教训扩展到深层神经网络和（相对）真实的数据集。
