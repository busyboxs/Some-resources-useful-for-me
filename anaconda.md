# 环境配置

## Anaconda 安装
```
bash ~/Anaconda2-5.1.0-Linux-x86_64.sh
```
## Caffe 依赖库
```
#（**caffe**: ImportError: No module named google.protobuf.internal）
pip install protobuf 

# (**caffe**: ImportError: No module named easydict)
pip install easydict 

# copy ~/opencv/build/lib/cv2.so to ~/anaconda2/lib/python2.7/site-packages
```

## Caffe2 conda 安装

为了使得caffe2安装后，caffe或者py-faster-rcnn还能够正常编译，需要将caffe2中`~/caffe2/caffe/proto/caffe.proto`替换为py-faster-rcnn中的**caffe.proto**(主要区别在于添加了`SmoothL1LossParameter`和`ROIPoolingParameter`)，然后在进行编译，安装。

```
CONDA_INSTALL_LOCALLY=1 BUILD_ENVIRONMENT=-cuda- ./scripts/build_anaconda.sh
```
    
## COCOAPI install

```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```
 
