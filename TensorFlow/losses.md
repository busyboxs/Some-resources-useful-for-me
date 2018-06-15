# tf.losses API

----

## Functions

* `absolute_difference(...)`: Adds an Absolute Difference loss to the training procedure.
* `add_loss(...)`: Adds a externally defined loss to the collection of losses.
* `compute_weighted_loss(...)`: Computes the weighted loss.
*  `cosine_distance(...)`: Adds a cosine-distance loss to the training procedure. (deprecated arguments)
* `get_losses(...)`: Gets the list of losses from the loss_collection.
* `get_regularization_loss(...)`: Gets the total regularization loss.
* `get_regularization_losses(...)`: Gets the list of regularization losses.
* `get_total_loss(...)`: Returns a tensor whose value represents the total loss.
* `hinge_loss(...)`: Adds a hinge loss to the training procedure.
* `huber_loss(...)`: Adds a Huber Loss term to the training procedure.
* `log_loss(...)`: Adds a Log Loss term to the training procedure.
* `mean_pairwise_squared_error(...)`: Adds a pairwise-errors-squared loss to the training procedure.
* `mean_squared_error(...)`: Adds a Sum-of-Squares loss to the training procedure.
* `sigmoid_cross_entropy(...)`: Creates a cross-entropy loss using `tf.nn.sigmoid_cross_entropy_with_logits`.
* `softmax_cross_entropy(...)`: Creates a cross-entropy loss using `tf.nn.softmax_cross_entropy_with_logits`.
* `sparse_softmax_cross_entropy(...)`: Cross-entropy loss using `tf.nn.sparse_softmax_cross_entropy_with_logits`.

## tf.losses.absolute_difference

差的绝对值损失

$$loss = \lvert y - \hat{y} \rvert$$

```python
losses = math_ops.abs(math_ops.subtract(predictions, labels))`
```

## tf.losses.add_loss

添加额外定义的损失到损失集合中

`ops.add_to_collection(loss_collection, loss)`

## tf.losses.compute_weighted_loss

计算损失的函数接口，其他损失函数调用以便计算损失

##  tf.losses.cosine_distance

余弦距离损失

$$loss = \frac{\hat{y} \cdot y}{\sqrt{\hat{y}^2} \cdot \sqrt{y^2}}$$

```python
radial_diffs = math_ops.multiply(predictions, labels)
losses = 1 - math_ops.reduce_sum(radial_diffs, axis=(axis,), keepdims=True)
```

## tf.losses.hinge_loss

hinge 损失

$$loss = max(0, 1-z)$$ 

```python
all_ones = array_ops.ones_like(labels)
labels = math_ops.subtract(2 * labels, all_ones)
losses = nn_ops.relu(math_ops.subtract(all_ones, math_ops.multiply(labels, logits)))
```

## tf.losses.huber_loss

huber 损失

```tex
0.5 * x^2                  if |x| <= d
0.5 * d^2 + d * (|x| - d)  if |x| > d
```

$$loss =
\begin{cases}
0.5x^2 & \text{if $\lvert x \rvert \le d$ }  \\
0.5d^2+d(\lvert x \rvert - d) & \text{if $\lvert x \rvert > d$}
\end{cases}
$$

```python
error = math_ops.subtract(predictions, labels)
abs_error = math_ops.abs(error)
quadratic = math_ops.minimum(abs_error, delta)
# The following expression is the same in value as
# tf.maximum(abs_error - delta, 0), but importantly the gradient for the
# expression when abs_error == delta is 0 (for tf.maximum it would be 1).
# This is necessary to avoid doubling the gradient, since there is already a
# nonzero contribution to the gradient from the quadratic term.
linear = math_ops.subtract(abs_error, quadratic)
losses = math_ops.add(
    math_ops.multiply(
        ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
        math_ops.multiply(quadratic, quadratic)),
    math_ops.multiply(delta, linear))
```

## tf.losses.log_loss

log 损失

$$loss = - y log(\hat{y})-(1-y)log(1-\hat{y})$$

```python
losses = -math_ops.multiply(
    labels,
    math_ops.log(predictions + epsilon)) - math_ops.multiply(
        (1 - labels), math_ops.log(1 - predictions + epsilon))
```

## tf.losses.mean_pairwise_squared_error

pairwise-errors-squared 损失

