# Tutorials

## Gluon

Gluon是MXNet的高级接口。它比底层界面更直观，更易于使用。Gluon支持使用JIT编译的动态（逐个定义）图形，以实现灵活性和效率。

这是Gluon教程的一个子集，它解释了Gluon和深度学习中基本概念的基本用法。有关Gluon的综合教程，涵盖从基本统计学和概率论到强化学习和推荐系统的主题，请参阅[gluon.mxnet.io](gluon.mxnet.io)。

### Basics

* [用ndarray处理MXNet的数据](http://gluon.mxnet.io/chapter01_crashcourse/ndarray.html)
* [使用autograd自动分化](http://gluon.mxnet.io/chapter01_crashcourse/autograd.html)
* [用Gluon进行线性回归](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-gluon.html)
* [序列化 - 保存，加载和检查点](http://gluon.mxnet.io/chapter03_deep-neural-networks/serialization.html)

## neural Networks

* [Gluon中的多层感知器](http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html)
* [Gluon中的卷积神经网络](http://gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-gluon.html)
* [用Gluon实现循环神经网络](http://gluon.mxnet.io/chapter05_recurrent-neural-networks/rnns-gluon.html)

### Adcanced

* [Plumbing: A look under the hood of gluon](http://gluon.mxnet.io/chapter03_deep-neural-networks/plumbing.html)
* [用Gluon设计一个自定义图层](http://gluon.mxnet.io/chapter03_deep-neural-networks/custom-layer.html)
* [使用Gluon HybridBlocks的快速便携式神经网络](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)
* [使用Gluon在多个GPU上进行训练](http://gluon.mxnet.io/chapter07_distributed-learning/multiple-gpus-gluon.html)

## MXNet 

这些教程介绍了深度学习的一些基本概念，以及如何在*MXNet*中实现它们。在基础知识(Basics)部分包含操作数组，建立网络，载入/预处理数据等教程。训练和推理(Training and Inference)谈论实现线性回归，训练使用MLP和CNN手写数字分类，使用预训练模型运行推理，最后，高效地训练大规模图像分类器。

### Basics

* [NDArray - CPU / GPU上的强制张量运算](https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html)
* [符号 - 神经网络图和自动求导](https://mxnet.incubator.apache.org/tutorials/basic/symbol.html)
* [模块 - 神经网络训练和推理](https://mxnet.incubator.apache.org/tutorials/basic/module.html)
* [迭代器 - 加载数据](https://mxnet.incubator.apache.org/tutorials/basic/data.html)

### Training and Inference

* [线性回归](https://mxnet.incubator.apache.org/tutorials/python/linear-regression.html)
* [手写数字识别](https://mxnet.incubator.apache.org/tutorials/python/mnist.html)
* [使用预先训练的模型进行预测](https://mxnet.incubator.apache.org/tutorials/python/predict_image.html)
* [大规模图像分类](https://mxnet.incubator.apache.org/tutorials/vision/large_scale_classification.html)

### Sparse NDArray 

* [CSRNDArray - 压缩稀疏行存储格式的NDArray](https://mxnet.incubator.apache.org/tutorials/sparse/csr.html)
* [RowSparseNDArray - 用于稀疏渐变更新的NDArray](https://mxnet.incubator.apache.org/tutorials/sparse/row_sparse.html)
* [用稀疏符号训练线性回归模型](https://mxnet.incubator.apache.org/tutorials/sparse/train.html)

更多的教程和例子可以在[GitHub仓库](https://github.com/dmlc/mxnet/tree/master/example)中找到。

## Contributing Tutorials

想要贡献MXNet教程？要开始，请下载[教程模板](https://github.com/dmlc/mxnet/tree/master/example/MXNetTutorialTemplate.ipynb)。
