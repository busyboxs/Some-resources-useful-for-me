# Python API Guide

----

## Tensor Transformations

### Casting

|API|Description|
|:----|:----|
|[tf.string_to_number](https://www.tensorflow.org/api_docs/python/tf/string_to_number?hl=zh-cn)|Converts each `string` in the input Tensor to the specified `numeric` type.|
|[tf.to_double](https://www.tensorflow.org/api_docs/python/tf/to_double?hl=zh-cn)|Casts a tensor to type `float64`.|
|[tf.to_float](https://www.tensorflow.org/api_docs/python/tf/to_float?hl=zh-cn)|Casts a tensor to type `float32`.|
|[tf.to_bfloat16](https://www.tensorflow.org/api_docs/python/tf/to_bfloat16?hl=zh-cn)|Casts a tensor to type `bfloat16`.|
|[tf.to_int32](https://www.tensorflow.org/api_docs/python/tf/to_int32?hl=zh-cn)|Casts a tensor to type `int32`.|
|[tf.to_int64](https://www.tensorflow.org/api_docs/python/tf/to_int64?hl=zh-cn)|Casts a tensor to type `int64`.|
|[tf.cast](https://www.tensorflow.org/api_docs/python/tf/cast?hl=zh-cn)|Casts a tensor to a new type.|
|[tf.bitcast](https://www.tensorflow.org/api_docs/python/tf/bitcast?hl=zh-cn) [详细见表格后内容]|Bitcasts a tensor from one type to another without copying data.|
|[tf.saturate_cast](https://www.tensorflow.org/api_docs/python/tf/saturate_cast?hl=zh-cn) [详细见表格后内容]|Performs a safe saturating cast of `value` to `dtype`.|

#### tf.bitcast

> Given a tensor `input`, this operation returns a tensor that has the same buffer data as `input` with datatype `type`.
> 
> If the input datatype `T` is larger than the output datatype type then the shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].
>
> If T is smaller than type, the operator requires that the rightmost dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from [..., sizeof(`type`)/sizeof(`T`)] to [...].
>
> NOTE: Bitcast is implemented as a low-level cast, so machines with different endian orderings will give different results.

#### tf.saturate_cast

>This function casts the input to `dtype` without applying any scaling. If there is a danger that values would over or underflow in the cast, this op applies the appropriate clamping before the cast.

----

### Shapes and Shaping

|API|Parameters|Description|
|:----|:----|:----|
|[tf.broadcast_dynamic_shape](https://www.tensorflow.org/api_docs/python/tf/broadcast_dynamic_shape?hl=zh-cn)|`shape_x`,`shape_y`|Returns the broadcasted dynamic shape between `shape_x` and `shape_y`|
|[tf.broadcast_static_shape](https://www.tensorflow.org/api_docs/python/tf/broadcast_static_shape?hl=zh-cn)|`shape_x`,`shape_y`|Returns the broadcasted static shape between `shape_x` and `shape_y`.|
|[tf.shape](https://www.tensorflow.org/api_docs/python/tf/shape?hl=zh-cn)|`input`, `name=None`, `out_type=tf.int32`|Returns the shape of a tensor.|
|[tf.shape_n](https://www.tensorflow.org/api_docs/python/tf/shape_n?hl=zh-cn)|`input`, `name=None`, `out_type=tf.int32`|Returns shape of tensors.|
|[tf.size](https://www.tensorflow.org/api_docs/python/tf/size?hl=zh-cn)|`input`, `name=None`, `out_type=tf.int32`|Returns the size of a tensor.|
|[tf.rank](https://www.tensorflow.org/api_docs/python/tf/rank?hl=zh-cn)|`input`, `name=None`|Returns the rank of a tensor.|
|[tf.reshape](https://www.tensorflow.org/api_docs/python/tf/reshape?hl=zh-cn)|`tensor`, `shape`,  `name=None`|Reshapes a tensor.|
|[tf.squeeze](https://www.tensorflow.org/api_docs/python/tf/squeeze?hl=zh-cn) [详见表格下]|`input`, `axis=None`, `name=None`, `squeeze_dims=None`|Removes dimensions of size 1 from the shape of a tensor.|
|[tf.expand_dims](https://www.tensorflow.org/api_docs/python/tf/expand_dims?hl=zh-cn) [详见表格下]|`input`, `axis=None`, `name=None`, `dim=None`|Inserts a dimension of 1 into a tensor's shape.|
|[tf.meshgrid](https://www.tensorflow.org/api_docs/python/tf/meshgrid?hl=zh-cn) [详见表格下]|`*args`, ` **kwargs`|Broadcasts parameters for evaluation on an N-D grid.|

#### tf.squeeze

> Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed. If you don't want to remove all size 1 dimensions, you can remove specific size 1 dimensions by specifying axis.
>
> For example:
> 
> 
```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t))  # [2, 3]
```

> Or, to remove specific size 1 dimensions:

```python
# 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
tf.shape(tf.squeeze(t, [2, 4]))  # [1, 2, 3, 1]
```

#### tf.expand_dims

> Given a tensor `input`, this operation inserts a dimension of 1 at the dimension index `axis` of `input`'s shape. The dimension index `axis` starts at zero; if you specify a negative number for axis it is counted backward from the end.

> This operation is useful if you want to add a batch dimension to a single element. For example, if you have a single image of shape `[height, width, channels]`, you can make it a batch of 1 image with expand_dims(image, 0), which will make the shape `[1, height, width, channels]`.

> Other examples:
```python
# 't' is a tensor of shape [2]
tf.shape(tf.expand_dims(t, 0))  # [1, 2]
tf.shape(tf.expand_dims(t, 1))  # [2, 1]
tf.shape(tf.expand_dims(t, -1))  # [2, 1]

# 't2' is a tensor of shape [2, 3, 5]
tf.shape(tf.expand_dims(t2, 0))  # [1, 2, 3, 5]
tf.shape(tf.expand_dims(t2, 2))  # [2, 3, 1, 5]
tf.shape(tf.expand_dims(t2, 3))  # [2, 3, 5, 1]
```
> This operation requires that:

>* `-1-input.dims() <= dim <= input.dims()`

> This operation is related to squeeze(), which removes dimensions of size 1.

#### tf.meshgrid

> Given N one-dimensional coordinate arrays `*args`, returns a list `outputs` of N-D coordinate arrays for evaluating expressions on an N-D grid.

> Notes:

> `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions. When the `indexing` argument is set to 'xy' (the default), the broadcasting instructions for the first two dimensions are swapped.

> Examples:

> Calling `X, Y = meshgrid(x, y)` with the tensors
```python
x = [1, 2, 3]
y = [4, 5, 6]
X, Y = tf.meshgrid(x, y)
# X = [[1, 2, 3],
#      [1, 2, 3],
#      [1, 2, 3]]
# Y = [[4, 4, 4],
#      [5, 5, 5],
#      [6, 6, 6]]
```

----

### Slicing and Joining

|API|Description|
|:----|:----|
|[tf.slice](https://www.tensorflow.org/api_docs/python/tf/slice?hl=zh-cn) 详见表下|Extracts a slice from a tensor.|
|[tf.strided_slice](https://www.tensorflow.org/api_docs/python/tf/strided_slice?hl=zh-cn) 详见表下|Extracts a strided slice of a tensor (generalized python array indexing).|
|[tf.split](https://www.tensorflow.org/api_docs/python/tf/split?hl=zh-cn) 详见表下|Splits a tensor into sub tensors.|
|[tf.tile](tf.tile) 详见表下|Constructs a tensor by tiling a given tensor.|
|[tf.pad](https://www.tensorflow.org/api_docs/python/tf/pad?hl=zh-cn) 详见表下|Pads a tensor.|
|[tf.concat](https://www.tensorflow.org/api_docs/python/tf/concat?hl=zh-cn) 详见表下|Concatenates tensors along one dimension.|
|[tf.stack](https://www.tensorflow.org/api_docs/python/tf/stack?hl=zh-cn)|Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor.|
|[tf.parallel_stack](https://www.tensorflow.org/api_docs/python/tf/parallel_stack?hl=zh-cn)|Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.|
|[tf.unstack](https://www.tensorflow.org/api_docs/python/tf/unstack?hl=zh-cn)|Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.|
|[tf.reverse_sequence](https://www.tensorflow.org/api_docs/python/tf/reverse_sequence?hl=zh-cn)|Reverses variable length slices.|
|[tf.reverse](https://www.tensorflow.org/api_docs/python/tf/reverse?hl=zh-cn)|Reverses specific dimensions of a tensor.|
|[tf.reverse_v2](https://www.tensorflow.org/api_docs/python/tf/reverse_v2?hl=zh-cn)|Reverses specific dimensions of a tensor.|
|[tf.transpose](https://www.tensorflow.org/api_docs/python/tf/transpose?hl=zh-cn)|Transposes `a`. Permutes the dimensions according to `perm`.|
|[tf.extract_image_patches](https://www.tensorflow.org/api_docs/python/tf/extract_image_patches?hl=zh-cn)|Extract patches from images and put them in the "depth" output dimension.|
|[tf.space_to_batch_nd](https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd?hl=zh-cn)|SpaceToBatch for N-D tensors of type T.|
|[tf.space_to_batch](https://www.tensorflow.org/api_docs/python/tf/space_to_batch?hl=zh-cn)|SpaceToBatch for 4-D tensors of type T.|
|[tf.required_space_to_batch_paddings](https://www.tensorflow.org/api_docs/python/tf/required_space_to_batch_paddings?hl=zh-cn)|Calculate padding required to make block_shape divide input_shape.|
|[tf.batch_to_space_nd](https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd?hl=zh-cn)|BatchToSpace for N-D tensors of type T.|
|[tf.batch_to_space](https://www.tensorflow.org/api_docs/python/tf/batch_to_space?hl=zh-cn)|BatchToSpace for 4-D tensors of type T.|
|[tf.space_to_depth](https://www.tensorflow.org/api_docs/python/tf/space_to_depth?hl=zh-cn)|SpaceToDepth for tensors of type T.|
|[tf.depth_to_space](https://www.tensorflow.org/api_docs/python/tf/depth_to_space?hl=zh-cn)|DepthToSpace for tensors of type T.|
|[tf.gather](https://www.tensorflow.org/api_docs/python/tf/gather?hl=zh-cn)|Gather slices from params axis axis according to indices.|
|[tf.gather_nd](https://www.tensorflow.org/api_docs/python/tf/gather_nd?hl=zh-cn)|Gather slices from params into a Tensor with shape specified by indices.|
|[tf.unique_with_counts](https://www.tensorflow.org/api_docs/python/tf/unique_with_counts?hl=zh-cn)|Finds unique elements in a 1-D tensor.|
|[tf.scatter_nd](https://www.tensorflow.org/api_docs/python/tf/scatter_nd?hl=zh-cn)|Scatter updates into a new (initially zero) tensor according to indices.|
|[tf.dynamic_partition](https://www.tensorflow.org/api_docs/python/tf/dynamic_partition?hl=zh-cn)|Partitions data into num_partitions tensors using indices from partitions.|
|[tf.dynamic_stitch](https://www.tensorflow.org/api_docs/python/tf/dynamic_stitch?hl=zh-cn)|Interleave the values from the data tensors into a single tensor.|
|[tf.boolean_mask](https://www.tensorflow.org/api_docs/python/tf/boolean_mask?hl=zh-cn)|Apply boolean mask to tensor. Numpy equivalent is tensor[mask].|
|[tf.one_hot](https://www.tensorflow.org/api_docs/python/tf/one_hot?hl=zh-cn)|Returns a one-hot tensor.|
|[tf.sequence_mask](https://www.tensorflow.org/api_docs/python/tf/sequence_mask?hl=zh-cn)|Returns a mask tensor representing the first N positions of each cell.|
|[tf.dequantize](https://www.tensorflow.org/api_docs/python/tf/dequantize?hl=zh-cn)|Dequantize the 'input' tensor into a float Tensor.|
|[tf.quantize_v2](https://www.tensorflow.org/api_docs/python/tf/quantize_v2?hl=zh-cn)|Please use `tf.quantize` instead.|
|[tf.quantized_concat](https://www.tensorflow.org/api_docs/python/tf/quantized_concat?hl=zh-cn)|Concatenates quantized tensors along one dimension.|
|[tf.setdiff1d](https://www.tensorflow.org/api_docs/python/tf/setdiff1d?hl=zh-cn)|Computes the difference between two lists of numbers or strings.|

#### tf.slice

```python
tf.slice(input_, begin, size, name=None)
```
> This operation extracts a slice of size `size` from a tensor `input` starting at the location specified by `begin`. The slice `size` is represented as a tensor shape, where `size[i]` is the number of elements of the 'i'th dimension of `input` that you want to slice. The starting location (`begin`) for the slice is represented as an offset in each dimension of input. In other words, `begin[i]` is the offset into the 'i'th dimension of `input` that you want to slice from.
>
> Note that `tf.Tensor.getitem` is typically a more pythonic way to perform slices, as it allows you to write `foo[3:7, :-2] `instead of `tf.slice(foo, [3, 0], [4, foo.get_shape()[1]-2])`.
>
> `begin` is zero-based; `size` is one-based. If `size[i]` is -1, all remaining elements in dimension i are included in the slice. In other words, this is equivalent to setting:
> 
>* `size[i] = input.dim_size(i) - begin[i]`
>
> This operation requires that:
>
>* `0 <= begin[i] <= begin[i] + size[i] <= Di for i in [0, n]`
>
> For example:
```
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
tf.slice(t, [1, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
                                   #   [4, 4, 4]]]
tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
                                   #  [[5, 5, 5]]]
```

#### tf.strided_slice

```python
tf.strided_slice(input_, begin, end, strides=None, begin_mask=0, end_mask=0,
    ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0, var=None,name=None)
```

> **Instead of calling this op directly most users will want to use the NumPy-style slicing syntax (e.g. `tensor[..., 3:4:-1, tf.newaxis, 3]`), which is supported via `tf.Tensor.getitem` and `tf.Variable.getitem`**. The interface of this op is a low-level encoding of the slicing syntax.
> 
> Roughly speaking, this op extracts a slice of size `(end-begin)/stride` from the given `input_` tensor. Starting at the location specified by `begin` the slice continues by adding `stride` to the index until all dimensions are not less than `end`. Note that a stride can be negative, which causes a reverse slice.
> 
> Given a Python slice `input[spec0, spec1, ..., specn]`, this function will be called as follows.
> 
> `begin`, `end`, and `strides` will be vectors of length n. n in general is not equal to the rank of the `input_` tensor.
> 
> In each mask field (`begin_mask`, `end_mask`, `ellipsis_mask`, `new_axis_mask`, `shrink_axis_mask`) the ith bit will correspond to the ith spec.
> 
> If the ith bit of `begin_mask` is set, `begin[i]` is ignored and the fullest possible range in that dimension is used instead. `end_mask` works analogously, except with the end range.
> 
> `foo[5:,:,:3]` on a 7x8x9 tensor is equivalent to `foo[5:7,0:8,0:3]`. `foo[::-1]` reverses a tensor with shape 8.
> 
> If the ith bit of `ellipsis_mask` is set, as many unspecified dimensions as needed will be inserted between other dimensions. Only one non-zero bit is allowed in `ellipsis_mask`.
> 
> For example `foo[3:5,...,4:5]` on a shape 10x3x3x10 tensor is equivalent to `foo[3:5,:,:,4:5]` and `foo[3:5,...]` is equivalent to `foo[3:5,:,:,:]`.
> 
> If the ith bit of `new_axis_mask` is set, then `begin`, `end`, and `stride` are ignored and a new length 1 dimension is added at this point in the output tensor.
> 
> For example, `foo[:4, tf.newaxis, :2]` would produce a shape (`4, 1, 2`) tensor.
> 
> If the ith bit of `shrink_axis_mask` is set, it implies that the ith specification shrinks the dimensionality by 1. `begin[i]`, `end[i]` and `strides[i]` must imply a slice of size 1 in the dimension. For example in Python one might do `foo[:, 3, :]` which would result in `shrink_axis_mask equal` to 2.
> 
> NOTE: `begin` and `end` are zero-indexed. `strides` entries must be non-zero.
```python
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1])  # [[[3, 3, 3],
                                                      #   [4, 4, 4]]]
tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])  # [[[4, 4, 4],
                                                         #   [3, 3, 3]]]
```

#### tf.split

```python
tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
```

> If `num_or_size_splits` is an integer type, `num_split`, then splits `value` along dimension axis into `num_split `smaller tensors. Requires that `num_split` evenly divides `value.shape[axis]`.
> 
> If `num_or_size_splits` is not an integer type, it is presumed to be a Tensor `size_splits`, then splits `value` into `len(size_splits)` pieces. The shape of the i-th piece has the same size as the `value` except along dimension axis where the size is `size_splits[i]`.
> 
> For example:
```python
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0)  # [5, 4]
tf.shape(split1)  # [5, 15]
tf.shape(split2)  # [5, 11]
# Split 'value' into 3 tensors along dimension 1
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0)  # [5, 10]
```

#### tf.tile

```python
tf.tile(input, multiples, name=None)
```

> This operation creates a new tensor by replicating `input` `multiples` times. The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements, and the values of `input` are replicated `multiples[i]` times along the 'i'th dimension. For example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.

#### tf.pad

```python
tf.pad(tensor, paddings, mode='CONSTANT', name=None, constant_values=0)
```

> This operation pads a `tensor` according to the `paddings` you specify. `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of `tensor`. For each dimension D of input, `paddings[D, 0]` indicates how many values to add before the contents of `tensor` in that dimension, and `paddings[D, 1]` indicates how many values to add after the contents of `tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If `mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be no greater than `tensor.dim_size(D)`.
> 
> The padded size of each dimension D of the output is:
> 
>* `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`
> 
> For example:
```python
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
# 'constant_values' is 0.
# rank of 't' is 2.
tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]

tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1],
                                #  [6, 5, 4, 5, 6, 5, 4],
                                #  [3, 2, 1, 2, 3, 2, 1]]

tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                  #  [2, 1, 1, 2, 3, 3, 2],
                                  #  [5, 4, 4, 5, 6, 6, 5],
                                  #  [5, 4, 4, 5, 6, 6, 5]]
```

#### tf.concat

```python
tf.concat(values, axis, name='concat')
```

> Concatenates the list of tensors `values` along dimension `axis`. If `values[i].shape = [D0, D1, ... Daxis(i), ...Dn]`, the concatenated result has shape `[D0, D1, ... Raxis, ...Dn]` where `Raxis = sum(Daxis(i))` That is, the data from the input tensors is joined along the `axis` dimension.
> 
> The number of dimensions of the input tensors must match, and all dimensions except `axis` must be equal.
> 
> For example:
> 
```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0)  # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1)  # [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
```

> As in Python, the `axis` could also be negative numbers. Negative `axis` are interpreted as counting from the end of the rank, i.e., `axis + rank(values)`-th dimension.
> 
> For example:

```python
t1 = [[[1, 2], [2, 3]], [[4, 4], [5, 3]]]
t2 = [[[7, 4], [8, 4]], [[2, 10], [15, 11]]]
tf.concat([t1, t2], -1)
```

> would produce:

```python
[[[ 1,  2,  7,  4],
  [ 2,  3,  8,  4]],

 [[ 4,  4,  2, 10],
  [ 5,  3, 15, 11]]]
```

> Note: If you are concatenating along a new axis consider using stack. E.g.
```python
tf.concat([tf.expand_dims(t, axis) for t in tensors], axis)
```

> can be rewritten as
> 
```python
tf.stack(tensors, axis=axis)
```

#### tf.stack

```python
tf.stack(values, axis=0, name='stack')
```

> Packs the list of tensors in `values` into a tensor with rank one higher than each tensor in `values`, by packing them along the `axis` dimension. Given a list of length `N` of tensors of shape `(A, B, C)`;
> 
> if `axis == 0` then the output tensor will have the shape (`N, A, B, C`). if `axis == 1` then the output tensor will have the shape (`A, N, B, C`). Etc.
> 
> For example:

```python
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]] (Pack along first dim.)
tf.stack([x, y, z], axis=1)  # [[1, 2, 3], [4, 5, 6]]
This is the opposite of unstack. The numpy equivalent is

tf.stack([x, y, z]) = np.stack([x, y, z])
```

#### tf.parallel_stack

```python
tf.parallel_stack(values, name='parallel_stack')
```

> Stacks a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.
> 
> Requires that the shape of inputs be known at graph construction time.
> 
> Packs the list of tensors in values into a tensor with rank one higher than each tensor in values, by packing them along the first dimension. Given a list of length `N` of tensors of shape `(A, B, C)`; the output tensor will have the shape `(N, A, B, C)`.
> 
> For example:

```python
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.parallel_stack([x, y, z])  # [[1, 4], [2, 5], [3, 6]]
```

> The difference between `stack` and `parallel_stack` is that stack requires all the inputs be computed before the operation will begin but doesn't require that the input shapes be known during graph construction.

> `parallel_stack` will copy pieces of the input into the output as they become available, in some situations this can provide a performance benefit.
> 
> Unlike `stack`, `parallel_stack` does NOT support backpropagation.
> 
> This is the opposite of unstack. The numpy equivalent is
> 
```python
tf.parallel_stack([x, y, z]) = np.asarray([x, y, z])
```

#### tf.unstack

```python
tf.unstack(value, num=None, axis=0, name='unstack')
```

> Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
> 
> Unpacks num tensors from value by chipping it along the `axis` dimension. If num is not specified (the default), it is inferred from `value`'s shape. If `value.shape[axis]` is not known, ValueError is raised.
> 
> For example, given a tensor of shape `(A, B, C, D)`;
> 
> If `axis == 0` then the i'th tensor in output is the slice value[i, :, :, :] and each tensor in output will have shape (B, C, D). (Note that the dimension unpacked along is gone, unlike split).
> 
> If `axis == 1` then the i'th tensor in output is the slice value[:, i, :, :] and each tensor in output will have shape (A, C, D). Etc.
> 
> This is the opposite of stack. The numpy equivalent is
> 
```python
tf.unstack(x, n) = np.unstack(x)
```

#### tf.reverse_sequence

```python
tf.reverse_sequence(input, seq_lengths, seq_axis=None, batch_axis=None,
    name=None, seq_dim=None, batch_dim=None)
```

> Reverses variable length slices.
> 
> This op first slices input along the dimension `batch_axis`, and for each slice `i`, reverses the first `seq_lengths[i] `elements along the dimension `seq_axis`.
> 
> The elements of `seq_lengths` must obey `seq_lengths[i] <= input.dims[seq_dim]`, and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.
> 
> The output slice `i` along dimension `batch_axis` is then given by input slice i, with the first `seq_lengths[i]` slices along dimension `seq_axis` reversed.
> 
> For example:

```python
# Given this:
batch_dim = 0
seq_dim = 1
input.dims = (4, 8, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

# while entries past seq_lens are copied through:
output[0, 7:, :, ...] = input[0, 7:, :, ...]
output[1, 2:, :, ...] = input[1, 2:, :, ...]
output[2, 3:, :, ...] = input[2, 3:, :, ...]
output[3, 2:, :, ...] = input[3, 2:, :, ...]
```

> In contrast, if:

```python
# Given this:
batch_dim = 2
seq_dim = 0
input.dims = (8, ?, 4, ...)
seq_lengths = [7, 2, 3, 5]

# then slices of input are reversed on seq_dim, but only up to seq_lengths:
output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

# while entries past seq_lens are copied through:
output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
```

#### tf.reverse

```python
tf.reverse(tensor, axis, name=None)
```

> Reverses specific dimensions of a tensor.
> 
> NOTE `tf.reverse` has now changed behavior in preparation for 1.0. `tf.reverse_v2` is currently an alias that will be deprecated before TF 1.0.
> 
> Given a `tensor`, and a `int32` tensor `axis` representing the set of dimensions of tensor to reverse. This operation reverses each dimension `i` for which there exists `j` s.t. `axis[j] == i`.
> 
> `tensor` can have up to 8 dimensions. The number of dimensions specified in axis may be 0 or more entries. If an index is specified more than once, a InvalidArgument error is raised.
>
> For example:

```python
# tensor 't' is [[[[ 0,  1,  2,  3],
#                  [ 4,  5,  6,  7],
#                  [ 8,  9, 10, 11]],
#                 [[12, 13, 14, 15],
#                  [16, 17, 18, 19],
#                  [20, 21, 22, 23]]]]
# tensor 't' shape is [1, 2, 3, 4]

# 'dims' is [3] or 'dims' is [-1]
reverse(t, dims) ==> [[[[ 3,  2,  1,  0],
                        [ 7,  6,  5,  4],
                        [ 11, 10, 9, 8]],
                       [[15, 14, 13, 12],
                        [19, 18, 17, 16],
                        [23, 22, 21, 20]]]]

# 'dims' is '[1]' (or 'dims' is '[-3]')
reverse(t, dims) ==> [[[[12, 13, 14, 15],
                        [16, 17, 18, 19],
                        [20, 21, 22, 23]
                       [[ 0,  1,  2,  3],
                        [ 4,  5,  6,  7],
                        [ 8,  9, 10, 11]]]]

# 'dims' is '[2]' (or 'dims' is '[-2]')
reverse(t, dims) ==> [[[[8, 9, 10, 11],
                        [4, 5, 6, 7],
                        [0, 1, 2, 3]]
                       [[20, 21, 22, 23],
                        [16, 17, 18, 19],
                        [12, 13, 14, 15]]]]
```

#### tf.transpose

```python
tf.transpose(a, perm=None, name='transpose', conjugate=False)
```

> Transposes `a`. Permutes the dimensions according to `perm`.
> 
> The returned tensor's dimension i will correspond to the input dimension `perm[i]`. If perm is not given, it is set to (n-1...0), where n is the rank of the input tensor. Hence by default, this operation performs a regular matrix transpose on 2-D input Tensors. If conjugate is True and `a.dtype` is either `complex64` or `complex128` then the values of `a` are conjugated and transposed.
> 
> For example:

```python
x = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.transpose(x)  # [[1, 4]
                 #  [2, 5]
                 #  [3, 6]]

# Equivalently
tf.transpose(x, perm=[1, 0])  # [[1, 4]
                              #  [2, 5]
                              #  [3, 6]]

# If x is complex, setting conjugate=True gives the conjugate transpose
x = tf.constant([[1 + 1j, 2 + 2j, 3 + 3j],
                 [4 + 4j, 5 + 5j, 6 + 6j]])
tf.transpose(x, conjugate=True)  # [[1 - 1j, 4 - 4j],
                                 #  [2 - 2j, 5 - 5j],
                                 #  [3 - 3j, 6 - 6j]]

# 'perm' is more useful for n-dimensional tensors, for n > 2
x = tf.constant([[[ 1,  2,  3],
                  [ 4,  5,  6]],
                 [[ 7,  8,  9],
                  [10, 11, 12]]])

# Take the transpose of the matrices in dimension-0
# (this common operation has a shorthand `matrix_transpose`)
tf.transpose(x, perm=[0, 2, 1])  # [[[1,  4],
                                 #   [2,  5],
                                 #   [3,  6]],
                                 #  [[7, 10],
                                 #   [8, 11],
                                 #   [9, 12]]]
```

#### tf.extract_image_patches

```python
tf.extract_image_patches(images, ksizes, strides, rates, padding, name=None)
```

> Extract `patches` from `images` and put them in the "depth" output dimension.
> 
> Args:
>* `images`: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`,` uint64`. 4-D Tensor with shape [`batch`, `in_rows`, `in_cols`, `depth`].
>* `ksizes`: A list of `ints` that has `length >= 4`. The size of the sliding window for each dimension of `images`.
>* `strides`: A list of `ints` that has `length >= 4`. 1-D of length 4. How far the centers of two consecutive patches are in the images. Must be: `[1, stride_rows, stride_cols, 1]`.
>* `rates`: A list of `ints` that has `length >= 4`. 1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the input stride, specifying how far two consecutive patch samples are in the input. Equivalent to extracting patches with `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1)`, followed by subsampling them spatially by a factor of `rates`. This is equivalent to `rate` in dilated (a.k.a. `Atrous`) convolutions.
>* `padding`: A string from: `"SAME"`, `"VALID"`. The type of padding algorithm to use.
>
> We specify the size-related attributes as:
> python `ksizes = [1, ksize_rows, ksize_cols, 1]` `strides = [1, strides_rows, strides_cols, 1]` `rates = [1, rates_rows, rates_cols, 1]` 
> * `name`: A name for the operation (optional).

#### tf.space_to_batch_nd

```python
tf.space_to_batch_nd(input, block_shape, paddings, name=None)
```

