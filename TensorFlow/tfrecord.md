# TFRecord 是什么？如何使用?

原文地址：[Tensorflow Records? What they are and how to use them](https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564)

自2015年11月推出以来，对 Tensorflow 的兴趣稳步增长。Tensorflow 的一个鲜为人知的组件是 TFRecord 文件格式，它是Tensorflow自己的二进制存储格式。

如果您正在处理大型数据集，则使用二进制文件格式存储数据会对导入pipeline的性能产生重大影响，从而影响模型的训练时间。二进制数据在磁盘上占用的空间更少，复制时间更短，并且可以从磁盘更有效地读取。如果您的数据存储在机械磁盘（spinning disks）上，尤其如此，因为与SSD相比，机械磁盘读/写性能要低得多。

但是，纯粹的性能并不是TFRecord文件格式的唯一优势。它针对Tensorflow以多种方式进行了优化。首先，它可以轻松组合多个数据集，并与库提供的数据导入和预处理功能无缝集成。特别是对于太大而无法完全存储在存储器中的数据集，这是一个优点，因为只有所需的数据（例如一个batch）从磁盘加载然后被处理。TFRecords的另一个主要优点是可以存储序列数据，例如，时间序列或字编码，能够非常有效（从编码角度）和方便地导入此类数据。查看[Reading Data](https://www.tensorflow.org/api_guides/python/reading_data)指南，了解有关TFRecord文件的更多信息。

因此，使用TFRecords有很多好处。但是有利也有弊，缺点是必须首先将数据转换为此格式，并且只有有限的文档说明如何使用。有一篇关于编写TFRecords的[官方教程](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py)和一些文章，但我发现它们只让我了解解决我的挑战的部分方法(大概是说教程不够详细)。

在这篇文章中，我将解释构造和编写TFRecord文件所需的组件，并详细说明如何编写不同类型的数据。这将帮助您开始应对自己的挑战。

## 构建TFRecords

TFRecord文件将数据存储为二进制字符串序列。这意味着需要在将数据写入文件之前指定数据的结构。Tensorflow为此提供了两个组件：[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example) 和 [tf.train.SequenceExample](https://www.tensorflow.org/api_docs/python/tf/train/SequenceExample)。必须将每个数据样本存储在其中一个组件中，然后对其进行序列化并使用[tf.python_io.TFRecordWriter](https://www.tensorflow.org/api_docs/python/tf/python_io/TFRecordWriter)把它写到磁盘上。

> tf.train.Example isn’t a normal Python class, but a [protocol buffer](https://en.wikipedia.org/wiki/Protocol_Buffers).

作为一名软件开发人员，我在开始时遇到的主要问题是Tensorflow API中的许多组件都没有该类的属性或方法的描述。例如，对于`tf.train.Example`，仅提供具有称为`message`的神秘结构的`.proto`文件，以及伪代码中的示例。原因是`tf.train.Example`不是普通的Python类，而是`protocol buffer`。`protocol buffer`是Google开发的一种方法，用于以有效的方式序列化结构化数据.现在将讨论构造Tensorflow TFRecords的两种主要方法，从开发人员的角度概述组件，并提供如何使用`tf.train.Example`和`tf.train.SequenceExample`的详细示例。

## 使用tf.train.Example进行电影推荐

> If your dataset consist of features, where each feature is a list of values of the same type, tf.train.Example is the right component to use.

让我们使用[Tensorflow文档](https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/example/example.proto)中的电影推荐应用程序作为示例:

![](https://cdn-images-1.medium.com/max/1000/1*At3Y8UvzwAK1bfl5EjRQiA.jpeg)

我们有许多特征，每个特征都是一个列表，其中每个条目都具有相同的数据类型。为了将这些特征存储在TFRecord中，我们需要创建构成特征的列表。

[tf.train.BytesList](https://www.tensorflow.org/api_docs/python/tf/train/BytesList), [tf.train.FloatList](https://www.tensorflow.org/api_docs/python/tf/train/FloatList), 和 [tf.train.Int64List](https://www.tensorflow.org/api_docs/python/tf/train/Int64List) 是`tf.train.Feature`的核心。这三个都有一个属性`value`，它需要一个相应bytes，float和int的列表。

```python
movie_name_list = tf.train.BytesList(value=[b'The Shawshank Redemption', b'Fight Club'])
movie_rating_list = tf.train.FloatList(value=[9.0, 9.7])
```

Python字符串需要转换为bytes（例如,*my_string.encode(‘utf-8’)*），然后才能存储在`tf.train.BytesList`中。

`tf.train.Feature`包含特定类型的数据列表，因此Tensorflow可以理解它。它有一个属性，它是`bytes_list` / `float_list` / `int64_list` 中的一种。存储的列表类型可以是`tf.train.BytesList` (属性名称为 `bytes_list`), `tf.train.FloatList` (属性名称为 `float_list`), 或者 `tf.train.Int64List` (属性名称为 `int64_list`)。

```python
movie_names = tf.train.Feature(bytes_list=movie_name_list)
movie_ratings = tf.train.Feature(float_list=movie_rating_list)
```

`tf.train.Features`是命名特征的集合。它有一个属性 `feature`，需要一个字典，其中key是特征的名称，value是`tf.train.Feature`。

```python
movie_dict = {
  'Movie Names': movie_names,
  'Movie Ratings': movie_ratings
}
movies = tf.train.Features(feature=movie_dict)
```

`tf.train.Example`是构造TFRecord的主要组件之一。`tf.train.Example`具有参数`features`,对应`tf.train.Features`。

```python
example = tf.train.Example(features=movies)
```

与之前的组件相比，`tf.python_io.TFRecordWriter`实际上是一个Python类。`path`属性接受一个文件路径，并创建一个与任何其他文件对象一样工作的`writer`对象。TFRecordWriter类提供*write*，*flush*，*close*方法。`write`方法接受一个字符串参数并写入磁盘，意味着结构化数据首先需要序列化。最后，`tf.train.Example`和`tf.train.SequenceExample`提供*SerializeToString*方法：

```python
# "example" is of type tf.train.Example.
with tf.python_io.TFRecordWriter('movie_ratings.tfrecord') as writer:
  writer.write(example.SerializeToString())
```

在我们的示例中，每个TFRecord表示单个用户（单个样本）的电影评级和相应的建议。为数据集中的所有用户编写推荐遵循相同的过程。重要的是，特征的类型（例如，电影评级是浮点型）在数据集中的所有样本中是相同的。该一致性标准和其他标准在`tf.train.Example`的protocol buffer中定义。

以下是一个完整的示例，将功能写入TFRecord文件，然后读回文件并打印解析后的特征。

----

```python
import tensorflow as tf

# Create example data
data = {
    'Age': 29,
    'Movie': ['The Shawshank Redemption', 'Fight Club'],
    'Movie Ratings': [9.0, 9.7],
    'Suggestion': 'Inception',
    'Suggestion Purchased': 1.0,
    'Purchase Price': 9.99
}
print(data)
```

```
{'Suggestion': 'Inception', 'Purchase Price': 9.99, 'Age': 29, 'Suggestion Purchased': 1.0, 'Movie': ['The Shawshank Redemption', 'Fight Club'], 'Movie Ratings': [9.0, 9.7]}
```


```python
# Create the Example
example = tf.train.Example(features=tf.train.Features(feature={
    'Age': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[data['Age']])),
    'Movie': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[m.encode('utf-8') for m in data['Movie']])),
    'Movie Ratings': tf.train.Feature(
        float_list=tf.train.FloatList(value=data['Movie Ratings'])),
    'Suggestion': tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[data['Suggestion'].encode('utf-8')])),
    'Suggestion Purchased': tf.train.Feature(
        float_list=tf.train.FloatList(
            value=[data['Suggestion Purchased']])),
    'Purchase Price': tf.train.Feature(
        float_list=tf.train.FloatList(value=[data['Purchase Price']]))
}))

print(example)
```

```
features {
  feature {
    key: "Age"
    value {
      int64_list {
        value: 29
      }
    }
  }
  feature {
    key: "Movie"
    value {
      bytes_list {
        value: "The Shawshank Redemption"
        value: "Fight Club"
      }
    }
  }
  feature {
    key: "Movie Ratings"
    value {
      float_list {
        value: 9.0
        value: 9.699999809265137
      }
    }
  }
  feature {
    key: "Purchase Price"
    value {
      float_list {
        value: 9.989999771118164
      }
    }
  }
  feature {
    key: "Suggestion"
    value {
      bytes_list {
        value: "Inception"
      }
    }
  }
  feature {
    key: "Suggestion Purchased"
    value {
      float_list {
        value: 1.0
      }
    }
  }
}

```

```python
# Write TFrecord file
with tf.python_io.TFRecordWriter('customer_1.tfrecord') as writer:
    writer.write(example.SerializeToString())

# Read and print data:
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['customer_1.tfrecord'])

_, serialized_example = reader.read(filename_queue)

# Define features
read_features = {
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Movie': tf.VarLenFeature(dtype=tf.string),
    'Movie Ratings': tf.VarLenFeature(dtype=tf.float32),
    'Suggestion': tf.FixedLenFeature([], dtype=tf.string),
    'Suggestion Purchased': tf.FixedLenFeature([], dtype=tf.float32),
    'Purchase Price': tf.FixedLenFeature([], dtype=tf.float32)}

# Extract features from serialized data
read_data = tf.parse_single_example(serialized=serialized_example,
                                    features=read_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
for name, tensor in read_data.items():
    print('{}: {}'.format(name, tensor.eval()))
```

```
Suggestion: b'Inception'
Purchase Price: 9.989999771118164
Age: 29
Suggestion Purchased: 1.0
Movie: SparseTensorValue(indices=array([[0],
       [1]]), values=array([b'The Shawshank Redemption', b'Fight Club'], dtype=object), dense_shape=array([2]))
Movie Ratings: SparseTensorValue(indices=array([[0],
       [1]]), values=array([9. , 9.7], dtype=float32), dense_shape=array([2]))
```

----

现在我们已经介绍了TFRecords的结构，读取的过程非常简单：

1. 使用`tf.TFRecordReader`读取TFRecord。
2. 使用`tf.FixedLenFeature`和`tf.VarLenFeature`定义TFRecord中所需的特征，具体取决于在`tf.train.Example`定义期间定义的内容。
3. 使用`tf.parse_single_example`一次解析一个`tf.train.Example`（一个文件）。

## 使用tf.train.SequenceExample进行电影推荐

> tf.train.SequenceExample is the right choice if you have features that consist of lists of identically typed data and maybe some contextual data.

现在，让我们从[Tensorflow文档](https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/example/example.proto)中获取一组略有不同的数据：

![](https://cdn-images-1.medium.com/max/1000/1*sDDlsCCY8hOPj6HFjRA3Tw.jpeg) 

我们有许多用户特定的上下文特征，如*Locale*, *Age*, 和 *Favorites* 以及推荐给用户的电影列表，包括*Movie Name*, *Movie Rating*, *Actors*。

数据看起来非常相似，在前面的例子中我们有一组特征，其中每个特征都由一个列表组成。列表中的每个条目表示不同电影的同类信息，例如电影评级。这没有改变，但现在我们也有Actors，这是一部在电影中扮演角色的演员。此类数据无法存储在tf.train.Example中。对于这种数据，我们需要不同类型的结构，Tensorflow提供了`tf.train.SequenceExample`的形式。和tf.train.Example相反，它不是存储一个bytes，floats或者int64的列表而是存储bytes,float,int64的列表的列表，因此非常适合我们的数据。

更正式地说，tf.train.SequenceExample有两个属性：
* 类型为tf.train.Features的`context`
* 类型为tf.train.FeatureLists的`feature_lists`

Context表中的数据存储在`context`中，Data表中的数据(Movie Name, Movie Rating, Actor)分别存储在一个单独的`tf.train.FeatureList`中。

tf.train.FeatureList具有单个参数`feature`，需要具有tf.train.Feature类型的条目的列表。起初，这可能看起来类似于tf.train.Features，它也包含多个tf.train.Feature类型的条目，但有两个很大的区别。首先，列表中的所有特征必须具有相同的内部列表类型。其次，虽然tf.train.Features是包含（无序）命名特征的字典，但tf.train.FeatureList是包含有序未命名特征的列表。

存储在tf.train.FeatureList中的数据的典型示例是时间序列，其中列表中的每个tf.train.Feature是序列的时间步长，或者是几个不同电影的演员列表。

```python
movie_1_actors = tf.train.Feature(
  bytes_list=tf.train.BytesList(
    value=[b'Tim Robbins', b'Morgan Freeman']))
movie_2_actors = tf.train.Feature(
  bytes_list=tf.train.BytesList(
    value=[b'Brad Pitt', b'Edward Norton', b'Helena Bonham Carter']))
movie_actors_list = [movie_1_actors, movie_2_actors]       
movie_actors = tf.train.FeatureList(feature=movie_actors_list)

# Short form
movie_names = tf.train.FeatureList(feature=[
    tf.train.Feature(bytes_list=tf.train.BytesList(
      value=[b'The Shawshank Redemption', b'Fight Club']))
])
movie_ratings = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(
            value=[9.7, 9.0]))
])
```

tf.train.FeatureLists是tf.train.FeatureList的命名实例的集合。该组件有一个属性`feature_list`，它需要一个dict。

```python
movies_dict = {
  'Movie Names': movie_names,
  'Movie Ratings': movie_ratings,
  'Movie Actors': movie_actors
}

movies = tf.train.FeatureLists(feature_list=movies_dict)
```

tf.train.SequenceExample，就像tf.train.Example一样，是构造TFRecord的主要组件之一。与tf.train.Example相比，它有两个属性：

1. `context`：此属性需要类型为tf.train.Features。它包含与feature_list属性中的每个特征相关的信息。context的行为与tf.train.Example的features属性相同。
2. `feature_lists`：此属性的类型为tf.train.FeatureLists。它包含特征列表，其中每个特征再次是某种顺序数据（例如时间序列或帧）

```python
# We can also add context features (short form)
customer = tf.train.Features(feature={
    'Age': tf.train.Feature(int64_list=tf.train.Int64List(value=[19])),
})

example = tf.train.SequenceExample(
    context=customer,
    feature_lists=movies)
```

您可以在[protocol buffer definition](https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/example/example.proto)中找到有关tf.train.SequenceExample的更多信息。作为旁注，虽然存在一致性标准，但它们不一定是强制执行的 - 例如FeatureList示例中的feature_list_invalid不会引发异常。

----

```python
import tensorflow as tf
# Create example data
data = {
    # Context
    'Locale': 'pt_BR',
    'Age': 19,
    'Favorites': ['Majesty Rose', 'Savannah Outen', 'One Direction'],
    # Data
    'Data': [
        {   # Movie 1
            'Movie Name': 'The Shawshank Redemption',
            'Movie Rating': 9.0,
            'Actors': ['Tim Robbins', 'Morgan Freeman']
        },
        {   # Movie 2
            'Movie Name': 'Fight Club',
            'Movie Rating': 9.7,
            'Actors': ['Brad Pitt', 'Edward Norton', 'Helena Bonham Carter']
        }
    ]
}

print(data)
```

```
{'Favorites': ['Majesty Rose', 'Savannah Outen', 'One Direction'], 'Locale': 'pt_BR', 'Age': 19, 'Data': [{'Movie Name': 'The Shawshank Redemption', 'Movie Rating': 9.0, 'Actors': ['Tim Robbins', 'Morgan Freeman']}, {'Movie Name': 'Fight Club', 'Movie Rating': 9.7, 'Actors': ['Brad Pitt', 'Edward Norton', 'Helena Bonham Carter']}]}
```

```python
# Create the context features (short form)
customer = tf.train.Features(feature={
    'Locale': tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[data['Locale'].encode('utf-8')])),
    'Age': tf.train.Feature(int64_list=tf.train.Int64List(
        value=[data['Age']])),
    'Favorites': tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[m.encode('utf-8') for m in data['Favorites']]))
})

# Create sequence data
names_features = []
ratings_features = []
actors_features = []
for movie in data['Data']:
    # Create each of the features, then add it to the
    # corresponding feature list
    movie_name_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[movie['Movie Name'].encode('utf-8')]))
    names_features.append(movie_name_feature)
    
    movie_rating_feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=[movie['Movie Rating']]))
    ratings_features.append(movie_rating_feature)
                                             
    movie_actors_feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[m.encode('utf-8') for m in movie['Actors']]))
    actors_features.append(movie_actors_feature)

movie_names = tf.train.FeatureList(feature=names_features)
movie_ratings = tf.train.FeatureList(feature=ratings_features)
movie_actors = tf.train.FeatureList(feature=actors_features)

movies = tf.train.FeatureLists(feature_list={
    'Movie Names': movie_names,
    'Movie Ratings': movie_ratings,
    'Movie Actors': movie_actors
})

# Create the SequenceExample
example = tf.train.SequenceExample(context=customer,
                                   feature_lists=movies)

print(example)
```

```
context {
  feature {
    key: "Age"
    value {
      int64_list {
        value: 19
      }
    }
  }
  feature {
    key: "Favorites"
    value {
      bytes_list {
        value: "Majesty Rose"
        value: "Savannah Outen"
        value: "One Direction"
      }
    }
  }
  feature {
    key: "Locale"
    value {
      bytes_list {
        value: "pt_BR"
      }
    }
  }
}
feature_lists {
  feature_list {
    key: "Movie Actors"
    value {
      feature {
        bytes_list {
          value: "Tim Robbins"
          value: "Morgan Freeman"
        }
      }
      feature {
        bytes_list {
          value: "Brad Pitt"
          value: "Edward Norton"
          value: "Helena Bonham Carter"
        }
      }
    }
  }
  feature_list {
    key: "Movie Names"
    value {
      feature {
        bytes_list {
          value: "The Shawshank Redemption"
        }
      }
      feature {
        bytes_list {
          value: "Fight Club"
        }
      }
    }
  }
  feature_list {
    key: "Movie Ratings"
    value {
      feature {
        float_list {
          value: 9.0
        }
      }
      feature {
        float_list {
          value: 9.699999809265137
        }
      }
    }
  }
}
```

```python
# Write TFrecord file
with tf.python_io.TFRecordWriter('customer_1.tfrecord') as writer:
    writer.write(example.SerializeToString())

# Read and print data:
sess = tf.InteractiveSession()

# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['customer_1.tfrecord'])

_, serialized_example = reader.read(filename_queue)

# Define features
context_features = {
    'Locale': tf.FixedLenFeature([], dtype=tf.string),
    'Age': tf.FixedLenFeature([], dtype=tf.int64),
    'Favorites': tf.VarLenFeature(dtype=tf.string)
}
sequence_features = {
    'Movie Names': tf.FixedLenSequenceFeature([], dtype=tf.string),
    'Movie Ratings': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    'Movie Actors': tf.VarLenFeature(dtype=tf.string)
}

# Extract features from serialized data
context_data, sequence_data = tf.parse_single_sequence_example(
    serialized=serialized_example,
    context_features=context_features,
    sequence_features=sequence_features)

# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)

# Print features
print('Context:')
for name, tensor in context_data.items():
    print('{}: {}'.format(name, tensor.eval()))

print('\nData')
for name, tensor in sequence_data.items():
    print('{}: {}'.format(name, tensor.eval()))
```

```
Context:
Favorites: SparseTensorValue(indices=array([[0],
       [1],
       [2]]), values=array([b'Majesty Rose', b'Savannah Outen', b'One Direction'], dtype=object), dense_shape=array([3]))
Locale: b'pt_BR'
Age: 19

Data
Movie Names: [b'The Shawshank Redemption' b'Fight Club']
Movie Actors: SparseTensorValue(indices=array([[0, 0],
       [0, 1],
       [1, 0],
       [1, 1],
       [1, 2]]), values=array([b'Tim Robbins', b'Morgan Freeman', b'Brad Pitt', b'Edward Norton',
       b'Helena Bonham Carter'], dtype=object), dense_shape=array([2, 3]))
Movie Ratings: [9.  9.7]
```

----

基于 `tf.train.SequenceExample` 读取TFRecords的工作方式与tf.train.Examples相同。唯一的区别是，我们需要定义两个：context and sequence features，而不仅仅是一组特征。上下文功能与之前显示的完全相同。序列特征必须是 [tf.VarLenFeature](https://www.tensorflow.org/api_docs/python/tf/VarLenFeature) 或 [tf.FixedLenSequenceFeature](https://www.tensorflow.org/api_docs/python/tf/FixedLenSequenceFeature) 类型，并使用 [tf.parse_single_sequence_example](https://www.tensorflow.org/api_docs/python/tf/parse_single_sequence_example) 进行解析。

## 总结

使用Tensorflow TFRecords是将数据导入机器学习管道的便捷方式，但是在开始时理解所有的点点滴滴可能是令人生畏的。这篇文章中的例子应该澄清整个过程并让你开始。

在[Github](https://gist.github.com/eega)上使用的所有代码片段以及更多内容 - 随意复制并以您喜欢的任何方式使用它们。