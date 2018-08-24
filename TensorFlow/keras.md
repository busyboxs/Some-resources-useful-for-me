# Keras 学习笔记

## 自定义评估标准函数

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```


## 模型结构存储与恢复

config

```python
config = model.get_config()
model = Model.from_config(config)
# 或者，对于 Sequential:
model = Sequential.from_config(config)
```

json

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

yaml

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

## keras.layers

### Dense

```python
keras.layers.Dense(units,  # 正整数，输出空间维度
                   activation=None,  # 激活函数。若不指定，则不使用激活函数
                   use_bias=True,  # 布尔值，该层是否使用偏置向量
                   kernel_initializer='glorot_uniform',  # kernel权值矩阵的初始化器
                   bias_initializer='zeros',  # 偏置向量的初始化器
                   kernel_regularizer=None,  # 运用到kernel权值矩阵的正则化函数
                   bias_regularizer=None,  # 运用到偏置向的的正则化函数
                   activity_regularizer=None,  # 运用到层的输出的正则化函数
                   kernel_constraint=None,  # 运用到kernel权值矩阵的约束函数
                   bias_constraint=None)  # 运用到偏置向量的约束函数
```

### Activation

```python
keras.layers.Activation(activation)
```

### Dropout

```python
keras.layers.Dropout(
    rate,  # 在0和1之间浮动。需要丢弃的输入比例
    noise_shape=None,   # 1D整数张量，表示将与输入相乘的二进制dropout掩层的形状
    seed=None  # 随机种子
)
```

### Flatten

```python
keras.layers.Flatten(
    data_format=None  # 一个字符串，其值为channels_last（默认值）或者channels_first
)
```

### Input

```python
keras.engine.input_layer.Input(
    shape=None,  # 尺寸元组
    batch_shape=None,  # 尺寸元组
    name=None,  # 一个可选的层的名称的字符串。
    dtype=None,  # 输入所期望的数据类型
    sparse=False,  # 指明需要创建的占位符是否是稀疏的
    tensor=None  # 可选的可封装到Input层的现有张量
)
```

### Reshape

```python
keras.layers.Reshape(target_shape)
```

### Permute

根据给定的模式置换输入的维度。

```python
keras.layers.Permute(
    dims  # dims:整数元组。置换模式,不包含样本维度。索引从1开始。例如, (2, 1) 置换输入的第一和第二个维度。
)
```

### RepeatVector

将输入重复n次
输入尺寸: 2D 张量，尺寸为 `(num_samples, features)`。
输出尺寸: 3D 张量，尺寸为 `(num_samples, n, features)`。

```python
keras.layers.RepeatVector(n)
```

### Lambda

将任意表达式封装为 `Layer` 对象

```python
keras.layers.Lambda(
    function,  # 需要封装的函数
    output_shape=None,  # 预期的函数输出尺寸。只在使用 Theano 时有意义
    mask=None, 
    arguments=None  # 可选的需要传递给函数的关键字参数
)
```

### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

### Masking, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D