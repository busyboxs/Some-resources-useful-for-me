# 目标检测中符号mAP，mAP@[0.5], mAP@[0.5:0.95]是什么意思？

----
## one of the answer
 
For detection, a common way to determine if one object proposal was right is **Intersection over Union(IoU)**. This takes the set A of proposed object pixels and the set of true object pixels B and calculates:

$$IoU(A, B) = \frac{A \cap B}{A \cup B}$$

Commonly, IoU > 0.5 means that it was a hit, otherwise it was a fail. For each class, one can calculate the

* True Positive (TP(c)): a proposal was made for class **c** and there actually was an object of class **c**
* False Positive (FP(c)): a proposal was made for class **c**, but there is no object of class **c**
* Average Precision for class **c**: $\frac{\#TP(c)}{\#TP(c) + \#FP(c)}$

The mAP (mean average precision) = $\frac{1}{|classes|}\sum_{c \in classes} \frac{\#TP(c)}{\#TP(c) + \#FP(c)}$

If one wants better proposals, one does increase the IoU from 0.5 to a higher value (up to 1.0 which would be perfect). One can denote this with mAP@p, where $p∈(0,1)$ is the IoU.

----

## other answer
mAP@[.5:.95] or mAP@[.5,.95] means average mAP over different IoU thresholds, from 0.5 to 0.95, step 0.05 (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95).

>There is an associated MS COCO challenge with a new evaluation metric, that averages mAP over different IoU thresholds, from 0.5 to 0.95 (written as “0.5:0.95”). [[Ref](https://www.cs.cornell.edu/~sbell/pdf/cvpr2016-ion-bell.pdf)]

>We evaluate the mAP averaged for IoU ∈ [0.5:0.05:0.95] (COCO’s standard metric, simply denoted as mAP@[.5, .95]) and mAP@0.5 (PASCAL VOC’s metric). [[Ref](https://arxiv.org/pdf/1506.01497.pdf)]

>To evaluate our final detections, we use the official COCO API [20], which measures mAP averaged over IOU thresholds in [0.5:0.05:0.95], amongst other metrics. [[Ref](https://arxiv.org/pdf/1611.10012.pdf)]

BTW, the [source code](https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/cocoeval.py#L501) of coco shows exactly what mAP@[.5:.95] is doing:

```python
  self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
```
