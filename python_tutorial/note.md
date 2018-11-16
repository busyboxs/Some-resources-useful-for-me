## Python notes

### String

1. 当字符串里面的字符不需要进行转义时，可以使用`r" "`;
2. 对于多行字符串可以使用`""" """`或者`''' '''`;
3. 可以使用`+`进行级联，`*`进行重复;
4. 多个相邻的字符串会自动拼接; eg:`"py" "thon" = "python"`
5. 可以通过索引取字符，通过切片取子字符串;
6. python字符串是不可改变的，不能改变其中一个字符的值，如果需要不同的字符串，则必须创建新字符串;


### List

1. list 可以有不同类型的项目（items）;
2. list 可以使用索引和切片，切片操作会返回一个新的 list;
3. list 可以使用`+`进行级联;
4. list 值是可改变的，可以通过索引或者切片来改变对应的值;
5. list 可以使用 `append()` 在 list 末尾添加元素;
6. `print(a, end=',)` 可以指定结束字符;

### Function

1. 函数传参时，`*name` 代表tuple， `**name`代表字典类型;
   
    ```python
    # eg
    def cheeseshop(kind, *arguments, **keywords):
        print("-- Do you have any", kind, "?")
        print("-- I'm sorry, we're all out of", kind)
        for arg in arguments:
            print(arg)
        print("-" * 40)
        for kw in keywords:
            print(kw, ":", keywords[kw])


    cheeseshop("Limburger", "It's very runny, sir.",
            "It's really very, VERY runny, sir.",
            shopkeeper="Michael Palin",
            client="John Cleese",
            sketch="Cheese Shop Sketch")
    ```

2. 如果形参`*name`、`**name`都存在，则`*name`必须在`**name`之前;
3. 实参中使用`*name`能够解包list，使用`**name`能够解包字典;

    ```bash
    # eg
    >>> list(range(3, 6)) # normal call with separate arguments
    [3, 4, 5]
    >>> args = [3, 6]
    >>> list(range(*args)) # call with arguments unpacked from a list
    [3, 4, 5]
    ```

    ```bash
    >>> def parrot(voltage, state='a stiff', action='voom'):
    ...     print("-- This parrot wouldn't", action, end=' ')
    ...     print("if you put", voltage, "volts through it.", end=' ')
    ...     print("E's", state, "!")
    ...
    >>> d = {"voltage": "four million", "state": "bleedin' demised", "action": "VOOM"}
    >>> parrot(**d)
    -- This parrot wouldn't VOOM if you put four million volts through it. E's bleedin' demised !
    ```
4. 文档字符串要求

    * 第一行应始终是目的的简短概述。该行应以大写字母开头，以句点结尾;
    * 如果文档字符串中有更多行，则第二行应为空白，从而在视觉上将摘要与其余描述分开;
    * 余下行应该是一个或多个段落，描述对象的调用约定，其副作用等。

5. 代码风格（PEP 8）

    * 使用4个空格缩进，不用tab;
    * 换行，使其不超过79个字符;
    * 使用空行分隔函数和类，以及函数内的较大代码块。
    * 如果可能，将评论放在代码同一行上。
    * 使用docstrings;
    * 在操作符周围和逗号后面使用空格：`a = f(1, 2) + g(3, 4)`;
    * 命名类和函数时保持一致，惯例是将驼峰式(CamelCase)用于类，使用下划线式(lower_case_with_underscores)用于函数和方法;
    * 如果您的代码旨在用于国际环境，请不要使用花哨的编码;Python的默认值，UTF-8甚至纯ASCII在任何情况下都是最好的选择;

### Data Structures

1. **List**

    * `list.append(x)`: 在 list 结尾添加 item;
    * `list.extend(iterable)`: 在 list 中添加 iterable 中的所有 items;
    * `list.insert(i, x)`：在给定位置插入 item;
    * `list.remove(x)`：删除第一个值为 x 的item，如果没有值为 x 的item，会出现 `ValueError`;
    * `list.pop([i])`：删除给定位置的元素，i为可选，默认为最后一个元素的位置;
    * `list.clear()`：删除 list 中所有 items;
    * `list.index(x[, start[, end]])`：在[start, end]中返回第一个值为 x 的索引，如果没有值为 x 的item，会出现 `ValueError`;
    * `list.count(x)`：返回 list 中 x 出现的次数;
    * `list.sort(key=None, reverse=False)`：对list进行排序，in place(list内操作，不会创建新的list);
    * `list.reverse()`：反转list，in place;
    * `list.copy()`：返回 list 的浅拷贝;
    * 将list作为栈使用，`append()`入栈，`pop()`出栈;
    * 将list作为队列使用，但是入队和出队很慢(会有元素移动)，可以使用`collections.deque`，`append()`入队，`popleft()`出队;
    * list 可以用生成器来生成，eg：`squares = [x**2 for x in range(10)]`

2. del 语句可以通过索引或者片段（slice）删除部分元素，或者直接删除整个数据结构;
3. **tuple**

    * 元组由逗号分隔的许多值组成，eg:`t = 12345, 54321, 'hello!'`
    * 元组可以嵌套;
    * 元组中item不可改变;但是可以包含可变的对象;

4. **Sets**

    * 集合是无序集合，没有重复元素;
    * 集合可用于测试成员是否存在;
    * 集合可以用于去除重复元素(set(a))，或者求差集（a有的，b没有的元素）;
    * 集合对象还支持数学运算，如并集，交集，差即和对称差集。
    * `{}`或者`set()`用于创建集合，要创建一个空集，必须使用`set()`，不能使用`{}`;
    * 集合可以用生成器来生成，eg:`a = {x for x in 'abracadabra' if x not in 'abc'}`

    ```bash
    >>> basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
    >>> print(basket) # show that duplicates have been removed
    {'orange', 'banana', 'pear', 'apple'}
    >>> 'orange' in basket # fast membership testing
    True
    >>> 'crabgrass' in basket
    False

    >>> # Demonstrate set operations on unique letters from two words
    ...
    >>> a = set('abracadabra')
    >>> b = set('alacazam')
    >>> a  # unique letters in a
    {'a', 'r', 'b', 'c', 'd'}
    >>> a - b  # letters in a but not in b
    {'r', 'd', 'b'}
    >>> a | b  # letters in a or b or both
    {'a', 'c', 'r', 'd', 'b', 'm', 'z', 'l'}
    >>> a & b  # letters in both a and b
    {'a', 'c'}
    >>> a ^ b  # letters in a or b but not both
    {'r', 'd', 'b', 'm', 'z', 'l'}
    ```

5. Dictionaries

    * 字典上的主要操作是使用某个键存储值并提取给定键的值;
    * 可以用 `del` 删除一个键值对;
    * 字典键值是唯一的，如果存储已经存在的键，则对应的值会被替换;
    * 使用不存在的键取值会出现错误;
    * 在字典上执行`list(d)`会返回所有键构成的list;
    * 可以使用`in`关键字检测字典中是否存在某个键;
    * `dict()`构造函数直接从键值对序列构建字典, eg:`dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])`;
    * 字典可以使用生成器来生成，eg：`{x: x**2 for x in (2, 4, 6)}`
    * 当键是简单字符串时，有时使用关键字参数指定对更容易，eg：`dict(sape=4139, guido=4127, jack=4098)`

6. 循环技巧

    * `dict.item()`返回键值对，eg：`for k, v in knights.items():`;
    * `enumerate(list)`返回list索引和值，eg：`for i, v in enumerate(['tic', 'tac', 'toe'])`;
    * `zip()`可以组合多个序列，eg: `for q, a in zip(questions, answers)`;
    * `reversed()`可以用来做反向循环， eg:`for i in reversed(range(1, 10, 2))`;

7. 条件语句

    * while和if语句中使用的条件可以包含任何运算符，而不仅仅是比较;
    * 比较运算符`in`和`not in`检查序列中是否有（没有）值;
    * `is`和`is not`比较两个对象是否真正相同;
    * 比较是可以连接的，eg：`a < b < c`测试b是否大于a并且小于c;
    * `and`和`or`可以对多个条件进行逻辑运算，并且会有“短路”问题;

### Module

1. 模块是包含Python定义和语句的文件;
2. 导入模块方法，eg：`from modname import itemname1, itemname2`或者`from modname import *`或者`import modname as shortname`
   
```python
from fibo import fib, fib2
from fibo import *
import fibo as fib
```
3. `from modname import *` 会导入除了下划线开头以外的其他函数;
4. 导入名为*spam*的模块时，解释器首先搜索具有该名称的内置模块。如果未找到，则会在变量`sys.path`给出的目录列表中搜索名为 *spam.py* 的文件;`sys.path`包括当前脚本所在的目录，`PYTHONPATH`目录和安装依赖目录;
5. `sys.path`:返回表示模块搜索路径的列表;`sys.path.append()`用于添加模块搜索路径;
6. 内置函数`dir()`用于返回模块中定义的名字（变量，模块，方法...），不包含内置方法或者函数的名字;
7. 
