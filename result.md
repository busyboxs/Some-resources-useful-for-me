

## Result on pascal_voc 2007

method|backbone|data|mAP|areo|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
Faster RCNN|VGG16|07|**69.6**|70.0|80.6|70.1|57.3|49.9|78.2|80.4|82.0|52.2|75.3|67.2|80.3|79.8|75.0|76.3|39.1|68.3|67.3|81.1|67.6|
Faster RCNN|VGG16|07+12|**73.2**|76.5|79.0|70.9|65.5|52.1|83.1|84.7|86.4|52.0|81.9|65.7|84.8|84.6|77.5|76.7|38.8|73.6|73.9|83.0|72.6|
Faster RCNN|VGG16|COCO+07+12|**78.8**|84.3|82.0|77.7|68.9|65.7|88.1|88.4|88.9|63.6|86.3|70.8|85.9|87.6|80.1|82.3|53.6|80.4|75.8|86.6|78.9|
Faster RCNN|ResNet101|07+12|**76.4**|79.8|80.7|76.2|68.3|55.9|85.1|85.3|89.8|56.7|87.8|69.4|88.3|88.9|80.9|78.4|41.7|78.6|79.8|85.3|72.0|
Faster RCNN+++|ResNet101|07+12+COCO|**85.6**|90.0|89.6|87.8|80.8|76.1|89.9|89.9|89.6|75.5|90.0|80.7|89.6|90.3|89.1|88.7|65.4|88.1|85.6|89.0|86.8|


* “Faster R-CNN +++” : uses iterative box regression, context, and multi-scale testing.

----

## Result on pascal_voc 2012

method|backbone|data|mAP|areo|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
Faster RCNN|VGG16|12|**67.0**|82.3|76.4|71.0|48.4|45.2|72.1|72.3|87.3|42.2|73.7|50.0|86.8|78.7|78.4|77.4|34.5|70.1|57.1|77.1|58.9|
Faster RCNN|VGG16|07++12|**70.4**|84.9|79.8|74.3|53.9|49.8|77.5|75.9|88.5|45.6|77.1|55.3|86.9|81.7|80.9|79.6|40.1|72.6|60.9|81.2|61.5|
Faster RCNN|ResNet101|07++12|**73.8**|86.5|81.6|77.2|58.0|51.0|78.6|76.6|93.2|48.6|80.4|59.0|92.1|85.3|84.8|80.7|48.1|77.3|66.5|84.7|65.6|
Faster RCNN|VGG16|COCO+07++12|**75.9**|87.4|83.6|76.8|62.9|59.6|81.9|82.0|91.3|54.9|82.6|59.0|89.0|85.5|84.7|84.1|52.2|78.9|65.5|85.4|70.2|
Faster RCNN+++|ResNet101|07++12+COCO|**83.8**|92.1|88.4|84.8|75.9|71.4|86.3|87.8|94.2|66.8|89.4|69.2|93.9|91.9|90.9|89.6|67.9|88.2|76.8|90.3|80.0|

* “07++12” : denotes the union set of 07 trainval+test and 12 trainval
* “Faster R-CNN +++” : uses iterative box regression, context, and multi-scale testing.

----

## RFCN result

Table 3: Comparisons between Faster R-CNN and R-FCN using ResNet-101. Timing is evaluated on a single Nvidia K40 GPU. With OHEM, N RoIs per image are computed in the forward pass, and 128 samples are selected for backpropagation. 300 RoIs are used for testing following [18].

| |depth of per-RoI subnetwork|training w/ OHEM?|train time(sec/img)|test time(sec/img)|mAP (%) on VOC07|
|:----:|:----------:|:--------:|:------------------:|:----------------:|:----------------:|
|Faster R-CNN|10| |1.2|0.42|**76.4**|
|R-FCN|0| |0.45|0.17|**76.6**|
|Faster R-CNN|10| w(300 RoIs)|1.5|0.42|**79.3**|
|R-FCN|0|w(300 RoIs)|0.45|0.17|**79.5**|
|Faster R-CNN|10|w(2000 RoIs)|2.9|0.42|**N/A**|
|R-FCN|0|w(2000 RoIs)|0.46|0.17|**79.3**|

Table 4: Comparisons on PASCAL VOC 2007 test set using ResNet-101. “Faster R-CNN +++” [9] uses iterative box regression, context, and multi-scale testing.

| |training data|mAP (%)|test time (sec/img)|
|:----:|:----:|:----:|:----:|
|Faster R-CNN|07+12|**76.4**|0.42|
|Faster R-CNN +++|07+12+COCO|**85.6**|3.36|
|R-FCN|07+12|**79.5**|0.17|
|R-FCN multi-sc train|07+12|**80.5**|0.17|
|R-FCN multi-sc train|07+12+COCO|**83.6**|0.17|


Table 5: Comparisons on PASCAL VOC 2012 test set using ResNet-101. “07++12” [6] denotes the union set of 07 trainval+test and 12 trainval.
†: http://host.robots.ox.ac.uk:8080/anonymous/44L5HI.html ‡:http://host.robots.ox.ac.uk:8080/anonymous/MVCM2L.html

| |training data|mAP (%)|test time (sec/img)|
|:----:|:----:|:----:|:----:|
|Faster R-CNN|07++12|**73.8**|0.42|
|Faster R-CNN +++|07++12+COCO|**83.8**|3.36|
|R-FCN *multi-sc train*|07++12|**77.6**†|0.17|
|R-FCN *multi-sc train*|07++12+COCO|**82.0**‡|0.17|

Table 7: Detailed detection results on the PASCAL VOC 2007 test set.(**ResNet101**)

|method|data|mAP|areo|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
|------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|Faster R-CNN|07+12|**76.4**|79.8|80.7|76.2|68.3|55.9|85.1|85.3|89.8|56.7|87.8|69.4|88.3|88.9|80.9|78.4|41.7|78.6|79.8|85.3|72.0|
|Faster R-CNN+++|07+12+CO|**85.6**|90.0|89.6|87.8|80.8|76.1|89.9|89.9|89.6|75.5|90.0|80.7|89.6|90.3|89.1|88.7|65.4|88.1|85.6|89.0|86.8|
|R-FCN|07+12|**79.5**|82.5|83.7|80.3|69.0|69.2|87.5|88.4|88.4|65.4|87.3|72.1|87.9|88.3|81.3|79.8|54.1|79.6|78.8|87.1|79.5|
|R-FCN ms train|07+12|**80.5**|79.9|87.2|81.5|72.0|69.8|86.8|88.5|89.8|67.0|88.1|74.5|89.8|90.6|79.9|81.2|53.7|81.8|81.5|85.9|79.9|
|R-FCN ms train|07+12+CO|**83.6**|88.1|88.4|81.5|76.2|73.8|88.7|89.7|89.6|71.1|89.9|76.6|90.0|90.4|88.7|86.6|59.7|87.4|84.1|88.7|82.4|

Table 8: Detailed detection results on the PASCAL VOC 2012 test set. 
†: http://host.robots.ox.ac.uk:8080/anonymous/44L5HI.html ‡: http://host.robots.ox.ac.uk:8080/anonymous/MVCM2L.html

|method|data|mAP|areo|bike|bird|boat|bottle|bus|car|cat|chair|cow|table|dog|horse|mbike|person|plant|sheep|sofa|train|tv|
|------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|Faster R-CNN|07++12|**73.8**|86.5|81.6|77.2|58.0|51.0|78.6|76.6|93.2|48.6|80.4|59.0|92.1|85.3|84.8|80.7|48.1|77.3|66.5|84.7|65.6|
|Faster R-CNN+++|07++12+CO|**83.8**|92.1|88.4|84.8|75.9|71.4|86.3|87.8|94.2|66.8|89.4|69.2|93.9|91.9|90.9|89.6|67.9|88.2|76.8|90.3|80.0|
|R-FCN *ms train†*|07++12|**77.6**|86.9|83.4|81.5|63.8|62.4|81.6|81.1|93.1|58.0|83.8|60.8|92.7|86.0|84.6|84.4|59.0|80.8|68.6|86.1|72.9|
|R-FCN *ms train‡*|07++12+CO|**82.0**|89.5|88.3|83.5|70.8|70.7|85.5|86.3|94.2|64.7|87.6|65.8|92.7|90.5|89.4|87.8|65.6|85.6|74.5|88.9|77.4|
