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
    
      
 
