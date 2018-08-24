# Python 学习笔记

本文档主要记录一些python的知识点，文档会比较乱，待有时间后会再整理，主要资料是python3.7.0版本的[官方文档](https://docs.python.org/3/reference/index.html#reference-index)，因此主要记录目录会根据官方文档的目录来。

## 1. Introduction

python的几个其他语言实现：

1. CPython : C语言编写的python实现
2. [Jython](http://www.jython.org/) : Java语言编写的python实现
3. Python for .NET
4. IronPython
5. [PyPy](http://pypy.org/) : python语言的python实现

## 2. Lexical analysis

#### 编码声明

编码声明正则表达式: `coding[=:]\s*([-\w.]+)`, 通长写成以下形式：

```python
# -*- coding: <encoding-name> -*-
```

#### 特殊语法

python3.6之后添加`f-string`和数字下划线分割(方便分组)，eg.

```python
name = 'Peter'
s = f'Hello {name!r}'
print(s)

a = 1_234_567
print(a)
b = 3.14_15_926
print(b)
```

f-string 语法：

```
f_string          ::=  (literal_char | "{{" | "}}" | replacement_field)*
replacement_field ::=  "{" f_expression ["!" conversion] [":" format_spec] "}"
f_expression      ::=  (conditional_expression | "*" or_expr)
                         ("," conditional_expression | "," "*" or_expr)* [","]
                       | yield_expression
conversion        ::=  "s" | "r" | "a"
format_spec       ::=  (literal_char | NULL | replacement_field)*
literal_char      ::=  <any code point except "{", "}" or NULL>
```

官方例子：

```
>>> name = "Fred"
>>> f"He said his name is {name!r}."
"He said his name is 'Fred'."
>>> f"He said his name is {repr(name)}."  # repr() is equivalent to !r
"He said his name is 'Fred'."
>>> width = 10
>>> precision = 4
>>> value = decimal.Decimal("12.34567")
>>> f"result: {value:{width}.{precision}}"  # nested fields
'result:      12.35'
>>> today = datetime(year=2017, month=1, day=27)
>>> f"{today:%B %d, %Y}"  # using date format specifier
'January 27, 2017'
>>> number = 1024
>>> f"{number:#0x}"  # using integer format specifier
'0x400'
```

## 3. Data model

#### 类型层级

![](images/type.png)

`ord()`:将字符转换为数字
`chr()`:将数字转换为字符
`str.encode()`:将str转换为bytes
`bytes.decode()`:将bytes转换为str

set是无顺序的，无重复元素的。通常用于序列去重复，或序列交，并，差运算。

module具有的属性： 
* `__name__`:返回模块的名字；
* `__doc__`:返回模块的文档，如果没有，则为None；
* `__file__`:返回模块的文件路径。







