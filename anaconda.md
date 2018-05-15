# 环境配置

## Anaconda 安装
```
bash /home/yangshun/Anaconda2-5.1.0-Linux-x86_64.sh
```
## Caffe 依赖库
```
#（**caffe**: ImportError: No module named google.protobuf.internal）
pip install protobuf 
# (**caffe**: ImportError: No module named easydict)
pip install easydict 
```

## Caffe2 conda 安装
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
 
