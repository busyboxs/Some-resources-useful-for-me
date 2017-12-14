# Improving Object Detection With One Line of Code 论文翻译

## Abstract

非最大抑制是目标检测流程中的一个组成部分。首先，根据分数排序所有检测框。选择具有最大分数的检测框M，并且抑制与M具有显着重叠（使用预定的阈值）的所有其他检测框。这个过程被递归地应用在其余的检测框上。按照算法的设计，如果一个目标位于预定义的重叠阈值内，就会导致漏掉。为此，我们提出Soft-NMS算法，该算法将所有其他目标的检测分数衰减为与M重叠的连续函数。因此，在这个过程中没有任何目标被消除。 Soft-NMS 只需更改NMS算法，不需要修改任何额外的超参数，就在 PASCAL VOC 2007（R-FCN和Faster-RCNN均为1.7%）和 MS-COCO（R-FCN为1.3％，Faster-RCN为1.1％）等标准数据集的coco-style mAP评判指标上得到一致的改进。通过使用Deformable-RFCN，Soft-NMS可以在单一模型中将顶尖的目标检测结果从39.8％提高到40.9％。而且，Soft-NMS的计算复杂度与传统的NMS相当，因此可以有效实施。由于Soft-NMS不需要额外的训练，而且易于实施，因此可以很容易地集成到任何物体检测流程中。 Soft-NMS代码在GitHub [http://bit.ly/2nJLNMu](http://bit.ly/2nJLNMu)上公开发布。

## 1. Introduction

目标检测是计算机视觉中的一个基本问题，其中算法为指定的目标类别生成边界框并为其分配类别分数。 它在自动驾驶[6，9]，视频/图像索引[28,22]，监视[2，11]等方面有很多实际应用。因此，任何为目标检测流流程提出的新组成部分都不应该产生计算瓶颈， 否则在实际应用中会被方便地“忽略”。 而且，如果引入一个复杂的模块，需要对模型进行重新训练，导致性能略有改善，也将被忽略。 但是，如果一个简单的模块可以在不需要对现有模型进行任何重新训练的情况下提高性能，就会被广泛采用。 为此，我们提出了一种 soft non-maximum suppression 算法，作为当前目标检测流程中传统 NMS 算法的替代方案。

传统的物体检测流程[4,8]采用了多尺度滑动窗口的方法，该方法根据每个窗口中计算的特征为每个类别分配前景/背景分数。然而，相邻的窗口通常具有相关的分数（这增加了false positives），所以非最大抑制被用作后处理步骤以获得最终检测。随着深度学习的出现，滑动窗口方法被替换为使用卷积神经网络生成的类别独立区域提议。在最先进的检测器中，这些提议被输入到一个分类子网络，该子网络将分配给这些区域提议特定类别的分数[16,24]。另一个平行的回归子网络重新定义这些区域提议的位置。这种细化过程改善了目标的定位，但也导致了混乱的检测，因为多个提议经常被归结到相同的感兴趣区域（RoI）。因此，即使在现有技术的检测器中，也可以使用非最大抑制来获得最终的检测集合，因为它显着减少了 false positives 的数量。

非最大抑制开始于具有分数$S$的检测框$B$的列表。在选择具有最大分数$M$的检测框之后，它将其从集合$B$中移除并附加到最终检测$D$的集合上。 在集合$B$中和$M$的重叠大于阈值$N_{t}$也被移除。针对$B$中剩余的框重复该过程。非最大抑制的主要问题是它将邻近检测的分数设置为零。 因此，如果一个对象实际上出现在这个重叠阈值中，那么它将被漏掉，这将导致平均精度的下降。 然而，如果我们将检测分数作为与$M$与重叠的函数来降低，则它仍然在排名列表中，尽管具有较低的置信度。 我们在图1中显示了这个问题的一个例子。

![figure1]()

使用这种直觉，我们提出了一种单线修改传统的贪婪网络算法，其中我们降低检测分数作为重叠函数的增加，而不是像在NMS中设置得分为零。 直观地说，如果一个边界框与M有很高的重叠，那么它应该被分配一个非常低的分数，而如果它有一个低的重叠，它可以保持它原来的检测分数。 Soft-NMS算法如图2所示。Soft-NMS在PASCAL VOC和MS-COCO等标准数据集上的物体检测器的多个重叠阈值上测量的平均精确度显着提高。 由于Soft-NMS不需要任何额外的培训，而且易于实施，因此可以很容易地将其集成到物体检测流程中。

![figure2]()

## 2. Related Work

近五十年来，NMS一直是计算机视觉中许多检测算法的一个组成部分。它首先被用于边缘检测技术[25]。随后，它被应用于多个任务，如特征点检测[19,12,20]，人脸检测[29]和物体检测[4,8,10]。在边缘检测中，NMS执行边缘细化来消除虚假响应[25，1，31]。在特征点检测器[12]中，NMS有效地执行局部阈值处理以获得独特的特征点检测。在人脸检测[29]中，NMS通过使用重叠准则将边界框划分为不相交的子集来执行。最后的检测是通过对检测框的坐标进行平均来获得的。对于人体检测，Dalal和Triggs [4]证明了一种贪心的NMS算法，其中选择了具有最大检测分数的边界框，并且使用预定义的重叠阈值抑制了其相邻框，改进了用于人脸检测的方法[29]。从那以后，贪婪的NMS一直是用于物体检测的事实上的算法[8,10,24,16]。

令人惊讶的是，检测的这个部分在十多年内一直保持不变。当平均精度（AP）被用作评估度量时，贪婪的NMS仍能获得最好的性能，因此被用于最先进的检测器[24,16]。已经提出了一些基于学习的方法来替代贪婪 NMS，它们在目标类别检测中获得了很好的性能[5,26,21]。例如，[26]首先计算每对检测框之间的重叠。然后执行亲和力传播聚类，为每个聚类选择代表最终检测框的样本。该算法的多类别版本在[21]中提出。然而，目标类别检测是一个不同的问题，其中所有类的对象实例在每个图像中同时被评估。因此，我们需要为所有类别选择一个阈值并生成一组固定的框。由于不同的阈值可能适用于不同的应用，所以在通用对象检测中，使用特定类别中的所有对象实例的排序列表来计算平均精度。因此，贪婪的NMS在通用对象检测度量上对这些算法有利。

在另一行工作中，为了检测显着对象，提出了一种提议子集优化算法[30]，作为贪婪 NMS 的替代方案。它执行基于MAP的子集优化来联合优化检测窗口的数量和位置。在显着的对象检测中，算法只能找到显着的对象而不是所有的对象。所以，这个问题也不同于一般的对象检测，当对物体检测度量的性能进行测量时，贪婪的NMS也会表现出色。对于特殊的情况下，像行人检测，a quadratic unconstrained binary optimization (QUBO) solution was proposed which uses detection scores as a unary potential and overlap between detections as a pairwise potential to obtain the optimal subset of detection boxes [27]。像贪婪的NMS一样，QUBO也采用硬阈值来抑制检测框，这与Soft-NMS不同。在另一个基于学习的行人检测框架中，将一个行列式过程与个人预测分数相结合，以最优地选择最终的检测结果[15]。就我们所知，对于通用目标检测，贪婪NMS仍然是在具有挑战性的目标检测数据集（如PASCAL VOC和MS-COCO）的最强基准。

## 3. Background

我们简要介绍本节介绍的最先进的物体检测器中使用的物体检测流程。在推断过程中，物体检测网络使用深度卷积神经网络（CNN）对图像执行一系列卷积运算。 网络在L层分叉成两个分支 - 一个分支产生区域提议，而另一个分支通过汇集提议网络产生的RoI内的卷积特征执行分类和回归。 提议网络为卷积特征图中每个像素处的多个比例和宽高比的锚生成分类分数和回归偏移[24]。 然后对这些锚点进行排序，并选择要添加边界框回归偏移的前K个（≈6000）锚点，以获取每个锚点的图像级别坐标。。 贪婪的非最大抑制适用于前K个锚点，最终生成区域建议。

分类网络为提案网络生成的每个提案生成类别和回归分数。 由于在网络中没有强制它为对象生成唯一的RoI的约束，所以多个提议可以对应于同一个对象。 因此，除了第一个正确的边界框之外，同一对象上的所有其他框会产生false positives。 为了缓解这个问题，在每个类别的检测框上独立地执行非最大抑制，并指定重叠阈值。 由于检测次数通常很小，并且可以通过修剪低于非常小的阈值的检测来进一步减少，所以在这个阶段应用非最大抑制在计算上并不昂贵。 在目标检测流程中，我们提出了这种非最大抑制算法的替代方法。 目标检测流程的概述如图3所示。

![figure3]()

## 4. Soft-NMS

目前的检测评估标准强调精确定位，并在多个重叠阈值（$0.5$ 到 $0.95$）内测量检测框的平均精度。 因此，如果将x像$0.3$的低阈值应用于NMS，而在评估真实的正值时，重叠准则为$0.7$（我们将检测评估阈值称为$O_{t}$），可能导致平均精度下降。 这是因为，可能有一个非常接近对象的检测框$b_{i}$（重叠在$0.7$以内），但是比$M$（$M$没有覆盖物体）的分数略低，因此$b_{i}$被低值$N_{t}$抑制。 这种情况的可能性会随着重叠阈值标准的增加而增加。 因此，用低值$N_{t}$抑制所有附近检测框会增加丢失率。

此外，使用像$0.7$这样的高$N_{t}$值会增加$O_{t}$值低时的 false positives，并因此降低在多个阈值上的平均精度。 在这种情况下，假阳性的增加将远高于真阳性的增加，因为对象的数量通常远小于由检测器产生的RoI的数量。 因此，使用高NMS阈值也不是最佳的。

为了克服这些困难，我们更详细地回顾了NMS算法。 NMS算法中的修剪步骤可以写成如下的重新评分函数，

$$
s_{i} = 
\begin{cases}
s_{i}, & iou(M,b_{i})<N_{t} \\
0, & iou(M,b_{i} \ge N_{t})
\end{cases}
$$

因此，NMS设置一个硬阈值，同时决定应该保留或从$M$的邻域中删除什么。相反，我们衰减与$M$重叠很高的框$b_{i}$的分类分数，而不是完全抑制它。 如果$b_{i}$包含未被$M$覆盖的对象，则不会导致在较低的检测阈值处漏掉。 然而，如果$b_{i}$不覆盖任何其他对象（而$M$覆盖了一个对象），并且即使在它的分数衰退之后，它的排名高于真实的检测结果，它仍然会产生false positive。 因此，NMS应该考虑以下条件，

* 近邻检测的得分应该降低到一定程度，这样将会在检测排名列表中高于明显的false positive 时有更小的可能提高false positive rate.
* 以较低的NMS阈值完全去除邻近检测将是次优的，这将会在以高重叠阈值进行评估时增加漏检率。
* 在使用高NMS阈值时，在一定范围的重叠阈值上测得的平均精度会下降。

我们通过6.3节的实验评估这些条件。

Soft-NMS的功能重构：与M重叠的其他检测框的分数衰减似乎是改善NMS的一种有希望的方法。 同样清楚的是，与M重叠较高的检测框的分数应该更多地衰减，因为它们具有更高的 false positive 可能性。 因此，我们建议按照以下规则更新修剪步骤，

$$
s_{i} = 
\begin{cases}
s_{i}, & iou(M,b_{i}) < N_{t}  \\
s_{i}(1-iou(M,b_{i})), & iou(M,b_{i}) \ge N_{t} 
\end{cases}
$$

上述函数会将超出阈值$N_{t}$的检测分数衰减为$IOU$的线性函数。因此，远离$M$的检测框不会受到影响，而那些非常接近的检测框将被分配更大的惩罚。

然而，$IOU$是不连续的，并且当达到NMS的阈值$N_{t}$时突然受到惩罚。 如果惩罚函数是连续的，这将是理想的，否则可能会导致排名列表中的突变。 因为$M$不应该影响与其有低$IOU$的框的分数，连续的惩罚函数应该在没有$IOU$的情况下不惩罚，在高$IOU$的情况下惩罚非常高。 而且，当$IOU$低时，应该逐渐增加惩罚。 然而，当一个检测框$b_{i}$与$M$的$IOU$接近1时，$b_{i}$应该受到显着的惩罚。 考虑到这一点，我们建议用下面的高斯罚函数更新修剪步骤，

$$
s_{i} = s_{i}e^{-\frac{iou(M,b_{i})^2}{\sigma}}, \forall b_{i} \notin D
$$

每次迭代应用此更新规则，并更新所有剩余检测框的分数。

Soft-NMS算法在图2中被正式地描述，其中$f(iou(M,b_{i}))$是基于$IOU$的加权函数。Soft-NMS中每一步的计算复杂度为$O(N)$，其中N是检测框的数量。这是因为与M重叠的所有检测框的分数被更新。因此，对于N个检测框，Soft-NMS的计算复杂度为$O(N^2)$，这与传统贪心NMS相同。 由于NMS没有应用于所有的检测框（在每次迭代中修剪最小阈值的框），这一步计算起来并不昂贵，因此不影响当前检测器的运行时间。

请注意，Soft-NMS也是一个贪婪的算法，并没有找到检测框的全局最优重新评分。 检测框的重新评分是以贪婪的方式进行的，因此具有较高局部评分的检测不被抑制。 然而，Soft-NMS是非最大抑制的广义形式，传统的NMS是一个具有不连续二进制加权函数的特例。 除了提出的两个函数之外，其他具有更多参数的函数也可以用考虑到重叠和检测分数的Soft-NMS来探索。 例如，可以使用像Gompertz函数那样的广义逻辑函数的实例，但是这样的函数会增加超参数的数量。

## 5. Datasets and Evaluation

我们在两个数据集PASCAL VOC [7]和MS-COCO [17]上进行实验。 Pascal数据集有20个对象类别，而MS-COCO数据集有80个对象类别。 我们选择VOC 2007测试样例来衡量性能。 对于MS-COCO数据集，敏感性分析是在公开可用的5000张图像的minival 上进行的。 我们还在由20,288个图像组成的MS-COCO数据集上的测试上测试结果。

为了评估我们的方法，我们使用了三种最新的检测器，即Faster-RCNN [24]，R-FCN [16] 和 Deformable-RFCN。对于PASCAL数据集，我们选择了作者提供的公开可用的预训练模型。Faster-RCNN检测器在VOC 2007训练集上训练，而R-FCN检测器在VOC 2007和2012上训练。对于MS-COCO，我们也使用Faster-RCNN的公开可用模型。然而，由于R-FCN在MS-COCO上没有公开的模型，所以我们在Caffe [14]中从ResNet-101 CNN架构[13]开始训练我们自己的模型。简单的修改，如RPN anchor 的5个尺度，最小图像尺寸800，每个 minibatch 16 张图像和每个图像256个ROI。训练是在8个GPU上并行完成的。请注意，在没有使用多尺度训练或测试时，我们的结果比的[16]（R_FCN）中报告的准确性提高了1.9％。因此，这是MS-COCO上R-FCN的强基准。这两个检测器都使用默认的NMS阈值0.3。在灵敏度分析部分，我们也改变这个参数并显示结果。我们也训练了deformable R-FCN。在10e-4的阈值下，使用4个CPU线程，对于80个类别，每个图像需要0.01s。在每次迭代之后，丢弃低于阈值的检测。这减少了计算时间。在10e-2时，单个核心的运行时间为0.005秒。我们在MS-COCO中将每个图像的最大检测设置为400，并且评估服务器选择每个类别生成度量的前100个检测结果（我们确认coco评估服务器在2017年6月之前没有选择每个图像的前100个得分检测）。将最大检测设置为100，将coco-style AP降低0.1。

## 6. Experiments

在本节中，我们将展示比较结果并进行灵敏度分析，以显示Soft-NMS与传统NMS相比的鲁棒性。 我们还进行了具体的实验，以了解Soft-NMS与传统NMS相比性能和优势。

### 6.1. Results

在表1中，我们在MS-COCO上用传统非最大抑制和Soft-NMS比较了R-FCN和Faster-RCNN。当使用线性加权函数时，我们将$N_{t}$设置为$0.3$，用高斯加权函数将$σ$设置为$0.5$。很显然，Soft-NMS（同时使用高斯和线性加权函数）在所有情况下提高了性能，特别是当在多个重叠阈值处计算AP并进行平均时。例如，R-FCN和Faster-RCNN分别提高了1.3％和1.1％，这对MS-COCO数据集有重要意义。注意，我们仅仅通过改变NMS算法获得了这种改进，因此它可以很容易地应用在多个检测器上，只需要很少的改变。我们在PASCAL VOC 2007测试集上进行了相同的实验，如表1所示。我们还报告了多重交叠阈值（如MS-COCO）的平均精确度。甚至在PASCAL VOC 2007上，Faster-RCNN和R-FCN的Soft-NMS都有1.7％的提升。对于不是基于区域提议的SSD [18]和YOLOv2 [23]等检测器，采用线性函数，Soft-NMS只能获得0.5％的提升。这是因为基于区域提议的检测器具有较高的召回率，因此Soft-NMS在更高的$O_{t}$处具有改善召回的更多潜力。

![table1]()

![table2]()

从这里开始，在所有的实验中，当我们参考Soft-NMS时，它使用高斯加权函数。 在图6中，我们也展示了MS-COCO的每个类的改进。 有趣的是，Soft-NMS在R-FCN上应用可以极大地提高动物的检测结果，在斑马，长颈鹿，绵羊，大象，马等动物群体中提高3-6%，而像烤面包机，运动球，吹风机只提升了一些，因为这些不太可能出现在同一图像。

![figure6]()

### 6.2. Sensitivity Analysis

Soft-NMS有个$σ$参数，传统的NMS具有$IOU$阈值参数$N_{t}$。 我们改变这些参数，并测量每个检测器的MS-COCO的minival 的平均精确度，参见图4. 注意$AP$稳定在$0.3$到$0.6$之间，并且在这两个范围之外显着下降。 在这个范围内，传统NMS的AP的变化在0.25％左右。 Soft-NMS从0.1到0.7的范围内获得比NMS更好的性能。 即使在我们在coco-minival集合上选择的最好的NMS门限上，对于每个探测器它的性能稳定于0.4到0.7，性能更好约有1％。 在我们的所有实验中，即使$0.6$的$σ$值似乎在coco minival set 上提供了更好的性能，我们将$σ$设置为$0.5$。 这是因为我们稍后进行了灵敏度分析实验，0.1％的差异不显着。

![figure4]()

![figure5]()

### 6.3. When does Soft-NMS work better?

**定位性能**  单独的平均精度并不能很好地解释什么时候Soft-NMS在性能上获得显着提高。因此，我们在不同的$IOU$门限值下测量NMS和Soft-NMS的平均精度。我们还改变了NMS和Soft-NMS超参数来理解这两种算法的特性。从表3可以推断，随着NMS门限的增加，平均精度下降。虽然对于一个大的$O_{t}$来说，高的$N_{t}$的性能比低的$N_{t}$获得的性能要好一些。当使用较低的$N_{t}$时，AP不会显着下降。 另一方面，使用较高的$N_{t}$导致在较低的$O_{t}$时，AP的显着下降，因此当AP在多个阈值时被平均时，我们观察到性能下降。因此，对于传统NMS，使用更高的$N_{t}$的更好的性能不会推广到的更低的$O_{t}$值( Therefore, a better performance using a higher $N_{t}$ does not generalize to lower values of $O_{t}$ for traditional NMS. )。

然而，当我们改变Soft-NMS的$σ$时，我们观察到一个不同的特征。从表3可以看出，即我们在较高的$O_{t}$获得了较好的表现，并且在较低的$O_{t}$下的表现也没有下降。此外，我们观察到 Soft-NMS 比传统的 NMS 表现得更好（〜2％），而不管在更高$O_{t}$处选定的$N_{t}$的值如何。另外，对于任何超参数（$N_{t}$或$σ$），对于选定的$O_{t}$，最好的AP总是对Soft-NMS更好。这个比较清楚地表明，在所有参数设置下，Soft-NMS的最佳$σ$参数比传统NMS中选择的硬阈值$N_{t}$要好。进一步，当所有阈值的性能平均时，由于Soft-NMS中的单个参数设置在$O_{t}$的多个值处运行良好，所以整体性能增益被放大。正如所料，低的$\sigma$在低的$O_{t}$上表现更好，高的$\sigma$在高的$O_{t}$上表现更好。与NMS不同的是，更高$N_{t}$值使得AP的改善幅度很小小，高的σ值导致AP在较高$O_{t}$时显着改善。因此，可以使用较大的$σ$来提高检测器的性能以实现更好的定位，而NMS则不是这样，因为较大的$N_{t}$获得很少的改善。

![table3]()

**精确度与召回率** 最后，我们还想知道在不同的$O_{t}$值下，哪个召回率使Soft-NMS的性能优于NMS。 请注意，我们重新评分检测分数并将其分配到较低的分数，所以我们不希望精度在较低的召回率下得到改善。 然而，随着$O_{t}$和召回率的增加，Soft-NMS在精度上获得了显着的提高。这是因为，传统的NMS对所有$IOU$大于$N_{t}$的框都赋予一个零分值。因此，很多框被遗漏，因此在召回率较高的情况下精度不会提高。 Soft-NMS对相邻的边框进行重新评分，而不是完全抑制它们，从而在召回率更高时导致精度的提高。 而且，即使在较高的$O_{t}$值，Soft-NMS低召回率也获得显着的改善，因为在这种情况下更有可能发生误判。

### 6.4. Qualitative Results

我们在图7中显示了一些定性的结果，对来自COCO验证集的图像使用$0.45$的检测阈值。使用R-FCN检测器产生检测结果。有趣的是，Soft-NMS在不良检测（false positive）与好的检测（true positive）重叠很小的情况下有帮助，以及当它们与良好的检测重叠很少时。例如，在街道图像（No.8）中，跨越多个人的大的宽边界框被抑制，因为它与具有较高分数的多个检测框重叠很小。因此，它的分数被多次减少，因此被抑制。我们在图片9中观察到类似的行为。在海滩图片（No.1）中，女士手提包附近较大边框的得分被抑制在$0.45$以下。我们也看到厨房图像（No.4）附近的碗附近的假阳性被抑制。在其他情况下，如斑马，马和长颈鹿图像（图像2,5,7和13），使用NMS抑制检测框，而Soft-NMS为邻近的框指定稍低的分数，因此我们能够检测高于$0.45$的检测阈值的true positives。

![figure7]()

## References
[1] J. Canny. A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6):679–698, 1986.
[2] R. T. Collins, A. J. Lipton, T. Kanade, H. Fujiyoshi, D. Duggins, Y. Tsin, D. Tolliver, N. Enomoto, O. Hasegawa, P. Burt, et al. A system for video surveillance and monitoring. 2000.
[3] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei. Deformable convolutional networks. arXiv preprint arXiv:1703.06211, 2017. 
[4] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on, volume 1, pages 886–893. IEEE, 2005. 
[5] C. Desai, D. Ramanan, and C. C. Fowlkes. Discriminative models for multi-class object layout. International journal
of computer vision, 95(1):1–12, 2011. 
[6] P. Dollar, C. Wojek, B. Schiele, and P. Perona. Pedestrian detection: A benchmark. In Computer Vision and Pattern
Recognition, 2009. CVPR 2009. IEEE Conference on, pages 304–311. IEEE, 2009. 
[7] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) chal-
lenge. International journal of computer vision, 88(2):303–338, 2010. 
[8] P. F. Felzenszwalb, R. B. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained partbased models. IEEE transactions on pattern analysis and machine intelligence, 32(9):1627–1645, 2010. 
[9] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11):1231–1237, 2013. 1
[10] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014. 2
[11] I. Haritaoglu, D. Harwood, and L. S. Davis. W/sup 4:real-time surveillance of people and their activities. IEEE Transactions on pattern analysis and machine intelligence, 22(8):809–830, 2000. 1
[12] C. Harris and M. Stephens. A combined corner and edge detector. In Alvey vision conference, volume 15, pages 10–5244. Citeseer, 1988. 2
[13] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages770–778, 2016. 5
[14] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell. Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia, pages 675–678. ACM, 2014. 5
[15] D. Lee, G. Cha, M.-H. Yang, and S. Oh. Individualness and determinantal point processes for pedestrian detection. In European Conference on Computer Vision, pages 330–346. Springer, 2016. 3
[16] Y. Li, K. He, J. Sun, et al. R-fcn: Object detection via regionbased fully convolutional networks. In Advances in Neural Information Processing Systems, pages 379–387, 2016. 2, 5
[17] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014. 4
[18] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. Ssd: Single shot multibox detector. In European Conference on Computer Vision, pages 21–37. Springer, 2016. 5
[19] D. G. Lowe. Distinctive image features from scale-invariant keypoints. International journal of computer vision, 60(2):91–110, 2004. 2
[20] K. Mikolajczyk and C. Schmid. Scale & affine invariant interest point detectors. International journal of computer vision, 60(1):63–86, 2004. 2
[21] D. Mrowca, M. Rohrbach, J. Hoffman, R. Hu, K. Saenko, and T. Darrell. Spatial semantic regularisation for large scale object detection. In Proceedings of the IEEE international conference on computer vision, pages 2003–2011, 2015. 2
[22] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman. Object retrieval with large vocabularies and fast spatial matching. In Computer Vision and Pattern Recognition, 2007. CVPR’07. IEEE Conference on, pages 1–8. IEEE, 2007. 1
[23] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 779–788, 2016. 5
[24] S. Ren, K. He, R. Girshick, and J. Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems, pages 91–99, 2015. 2, 3, 5
[25] A. Rosenfeld and M. Thurston. Edge and curve detection for visual scene analysis. IEEE Transactions on computers, 100(5):562–569, 1971. 2
[26] R. Rothe, M. Guillaumin, and L. Van Gool. Non-maximum suppression for object detection by passing messages between windows. In Asian Conference on Computer Vision, pages 290–306. Springer, 2014. 2
[27] S. Rujikietgumjorn and R. T. Collins. Optimized pedestrian detection for multiple and occluded people. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3690–3697, 2013. 3
[28] J. Sivic, A. Zisserman, et al. Video google: A text retrieval approach to object matching in videos. In iccv, volume 2, pages 1470–1477, 2003. 1
[29] P. Viola and M. Jones. Rapid object detection using a boosted cascade of simple features. In Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on, volume 1, pages I–I.IEEE, 2001. 2
[30] J. Zhang, S. Sclaroff, Z. Lin, X. Shen, B. Price, and R. Mech. Unconstrained salient object detection via proposal subset optimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5733–5742, 2016. 3
[31] C. L. Zitnick and P. Dollar. Edge boxes: Locating object proposals from edges. In European Conference on Computer Vision, pages 391–405. Springer, 2014. 2
