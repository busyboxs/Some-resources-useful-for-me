## Combined shallow layers with deeper layers for detecting small objects

1. Feature Pyramid Networks for Object Detection.[[arXiv](https://arxiv.org/abs/1612.03144)]
2. Beyond Skip Connections: Top-Down Modulation for Object Detection.[[arXiv](https://arxiv.org/abs/1612.06851)]
3. Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks.[[arXiv](https://arxiv.org/abs/1512.04143)]
4. Mask R-CNN.[[arXiv](https://arxiv.org/abs/1703.06870)]
5. SSD: Single Shot MultiBox Detector.[[arXiv](https://arxiv.org/abs/1512.02325)]

## dilated/deformable convolution is used to increase receptive fields for detecting large objects

1. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.[[arXiv](https://arxiv.org/abs/1506.01497)]
2. R-FCN: Object Detection via Region-based Fully Convolutional Networks.[[arXiv](https://arxiv.org/abs/1605.06409)]
3. Multi-Scale Context Aggregation by Dilated Convolutions.[[arXiv](https://arxiv.org/abs/1511.07122)]
4. Deformable Convolutional Networks.[[arXiv](https://arxiv.org/abs/1703.06211)]

## independent predictions at layers of different resolutions are used to capture object instances of different scales

1. Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers.
2. A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection.[[arXiv](https://arxiv.org/abs/1607.07155)]
3. Scale-aware Fast R-CNN for Pedestrian Detection.[[arXiv](https://arxiv.org/abs/1510.08160)]

## context is employed for disambiguation

1. A MultiPath Network for Object Detection.[[arXiv](https://arxiv.org/abs/1604.02135)]
2. Crafting GBD-Net for Object Detection.[[arXiv](https://arxiv.org/abs/1610.02579)]
3. Object detection via a multi-region & semantic segmentation-aware CNN model.[[arXiv](https://arxiv.org/abs/1505.01749)]


## The role of context has been well exploited in recognition and detection

1. Discriminative models for multi-class object layout.(2011)
2. Integrating context and oc- clusion for car detection by hierarchical and-or model.(2014 ECCV)
3. Detection evolution with multi-order contextual co-occurrence.(2013 CVPR)
4. Context-aware CNNs for person head detection. (2015 ICCV)
5. A multi-level contextual model for person recognition in photo albums.(2016 CVPR)

## when fuse multi layer features, difference between concate and eltsum.

(from Feature-Fused SSD: Fast Detection for Small Objects)

> In detailed fusion operation, we instantiate this feature fusion method carefully with two modules,
> concatenation module and element-sum module. Since context may introduce useless background noises, it is not always useful for small object detection. In detail, concatenation module uses a 1×1 convolution layer for learning the weights
> of the fusion of the target information and contextual information, which can reduce the interference of useless
> background noises. Element-sum module uses equivalent weights set manually and fuses the multi-level features in a
> compulsory way, which can enhance the effectiveness of useful context.

Context: Many previous studies have demonstrated that contextual information plays an important role in object
detection task, especially for small objects. The common method for introducing contextual information is exploiting the
combined feature maps within a ConvNet for prediction. For example, ION [2] extracts VGG16 [12] features from
multiple layers of each region proposal using ROI pooling [8], and concatenate them as a fixed-size descriptor for final
prediction. HyperNet [3], GBD-Net [13] and AC-CNN [14] also adopt a similar method that use the combined feature
descriptor of each region proposal for object detection. Because the combined features come from different layers, they
have different levels of abstraction of input image. So that the feature descriptors of each region proposal contain fine-
grained local features and contextual features. However, these methods are all based on region proposal method and pool
the feature descriptors from combined feature maps, which increases the memory footprint as well as decreases the speed
of detection.
