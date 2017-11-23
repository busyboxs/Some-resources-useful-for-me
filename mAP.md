# 目标检测中符号mAP，mAP@[0.5], mAP@[0.5:0.95]是什么意思？

对于检测，对目标提取是否准确的检验标准通常是使用 *Intersection Over Union(IOU)*,这里假设A是提取的目标像素框，B是真值框，则IOU的计算如下：

$$ IoU(A, B) = \frac{A \cap B}{A \cup B} $$

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />
