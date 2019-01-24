# jupyter中添加kernel

1. 首先激活conda环境： source activate 环境名称

2. 安装ipykernel：`conda install ipykernel`

3. 将环境写入notebook的kernel中
    `python -m ipykernel install --user --name 环境名称 --display-name "Python (环境名称)"`


## 删除kernel环境：

jupyter kernelspec remove 环境名称

