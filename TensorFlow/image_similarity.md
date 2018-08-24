# Learning Fine-grained Image Similarity with Deep Ranking

## 什么是图像相似度

图像相似度是度量两张图像的相似度的标准。将图像的相似程度进行量化。

## 图像相似度解决方法

1. 手工提取特征，如Gabor filters、SIFT、LBP、HOG ，并利用这些特征计算特征相似度；
2. 使用深度学习模型直接对输入查询图像计算相似度。

## 什么是triplet

triplet是一个三元组，可以表示为 $t_i = (p_i, p^+_i, p^-_i)$, 分别表示查询图像，正样本图像，负样本图像。triplet的几个实例如下图所示：

![triplet](images/tripletimg.png)

## 如何度量图像相似度

1. L1 norm (Manhattan distance): $D(p_i, p_j) = \|p_i-p_j\|_1$
2. L2 norm (Euclidean distance): $D(p_i, p_j) = \|p_i-p_j\|_2$
3. Squared Euclidean distance: $D(p_i, p_j) = \|p_i-p_j\|_2^2$

该论文使用的度量为

$$D(f(P),f(Q))=\|f(P)-f(Q)\|_2^2$$

其中$f(\cdot)$是图像嵌入函数(image Enbedding function)，将图像映射为欧式空间的一个点。$D(\cdot , \cdot)$是平方欧式距离。

## 损失函数

目标：

越相似的图像的平均欧氏距离越小，相关分数越高。

$$D(f(p_i),f(p_i^+)) < D(f(p_i), f(p_i^-))$$

$$r(p_i, p_i^+) > r(p_i, p_i^-)$$

hinge loss:

$$l(p_i, p_i^+, p_i^-) = max{\{0, g + D(f(pi), f(p_i^+ ))-D(f(p_i), f(p_i^-))}\}$$

其中$g$是gap参数。

## 网络结构

![](images/net_corase.png)

输入： triplets
Q、P、N: Convnets, shared architecture and parameters
Ranking Layer：计算triplets的hinge loss

