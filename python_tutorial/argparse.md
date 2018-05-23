# [argparse](https://docs.python.org/2.7/library/argparse.html) - 用于命令行选项，参数和子命令的解析器

argparse模块可以轻松编写用户友好的命令行界面。该程序定义了它需要的参数，argparse将找出如何解析sys.argv中的那些参数。argparse模块还会自动生成 help (帮助)和 usage(用法) 消息，并在用户给出程序无效参数时发出错误。

## Example

以下代码是一个Python程序，它接受一个整数列表并求出总和或最大值

```python
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print args.accumulate(args.integers)
```

假设上面的Python代码被保存到名为 prog.py 的文件中，它可以在命令行运行并提供有用的帮助信息：

```bash
$ python prog.py -h
usage: prog.py [-h] [--sum] N [N ...]

Process some integers.

positional arguments:
 N           an integer for the accumulator

optional arguments:
 -h, --help  show this help message and exit
 --sum       sum the integers (default: find the max)
```

当使用适当的参数运行时，它会输出命令行整数的总和或最大值：

```
$ python prog.py 1 2 3 4
4

$ python prog.py 1 2 3 4 --sum
10
```

如果传入无效参数，它将发出错误：

```
$ python prog.py a b c
usage: prog.py [-h] [--sum] N [N ...]
prog.py: error: argument N: invalid int value: 'a'
```

以下部分将引导您完成此示例。

### Creating a parser

使用argparse的第一步是创建一个ArgumentParser对象：

```python
parser = argparse.ArgumentParser(description='Process some integers.')
```

ArgumentParser对象将包含将命令行解析为Python数据类型所需的所有信息。

### Adding arguments

通过调用`add_argument()`法完成关于程序参数信息的 ArgumentParser 填充。通常，这些调用告诉ArgumentParser如何在命令行上获取字符串并将它们转换为对象。这个信息在`parse_args()`调用时被存储和使用。例如：

```python
parser.add_argument('integers', metavar='N', type=int, nargs='+',
					help='an integer for the accumulator')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
```

稍后，调用`parse_args()`返回一个具有两个属性(integers, accumulate)的对象。integers 属性将是一个或多个整数的列表，如果在命令行指定了`--sum`，accumulate属性将是`sum()`函数，如果不指定则为`max()`函数。

### Parsing arguments

ArgumentParser通过`parse_args()`法解析参数。这将检查命令行，将每个参数转换为适当的类型，然后调用相应的操作。在大多数情况下，这意味着一个简单的 Namespace 对象将由从命令行解析出来的属性构建而成：

```python
parser.parse_args(['--sum', '7', '-1', '42'])
Namespace(accumulate=<built-in function sum>, integers=[7, -1, 42])
```

在脚本中，`parse_args()`不带参数被调用，而 ArgumentParser 将自动从 sys.argv 中确定命令行参数。

## ArgumentParser objects
```
class argparse.ArgumentParser（prog = None，usage = None，description = None，epilog = None，parents = []，formatter_class = argparse.HelpFormatter，prefix_chars =' - '，fromfile_prefix_chars = None，argument_default = None，conflict_handler ='error'，add_help = True ）
```

创建一个新的ArgumentParser对象。所有参数都应该作为关键字参数传递。下面的每个参数都有自己的更详细的描述，

* `prog` - The name of the program (default: sys.argv[0])
* `usage` - The string describing the program usage (default: generated from arguments added to parser)
* `description` - Text to display before the argument help (default: none)
* `epilog` - Text to display after the argument help (default: none)
* `parents` - A list of ArgumentParser objects whose arguments should also be included
* `formatter_class` - A class for customizing the help output
* `prefix_chars` - The set of characters that prefix optional arguments (default: ‘-‘)
* `fromfile_prefix_chars` - The set of characters that prefix files from which additional arguments should be read (default: None)
* `argument_default `- The global default value for arguments (default: None)
* `conflict_handler` - The strategy for resolving conflicting optionals (usually unnecessary)
* `add_help` - Add a -h/--help option to the parser (default: True)

以下部分描述如何使用每个这些。

### prog

默认情况下，ArgumentParser对象用于`sys.argv[0]`确定如何在帮助消息中显示程序的名称。这个默认值几乎总是可取的，因为它会使帮助消息与命令行上的程序调用方式相匹配。例如，考虑 myprogram.py 使用以下代码命名的文件 ：

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()
```

该程序的帮助将显示myprogram.py为程序名称（不管程序从何处被调用）：

```tex
$ python myprogram.py --help
usage: myprogram.py [-h] [--foo FOO]

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO   foo help
$ cd ..
$ python subdir/myprogram.py --help
usage: myprogram.py [-h] [--foo FOO]

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO   foo help
```

要更改此默认行为，可以使用以下`prog=`参数提供另一个值 给 ArgumentParser：

```python
parser = argparse.ArgumentParser(prog='myprogram')
parser.print_help()
```

```
usage: myprogram [-h]

optional arguments:
 -h, --help  show this help message and exit
```
 
请注意，程序名称（无论从参数确定`sys.argv[0]`还是从 `prog=`参数确定）都可用于使用`%(prog)s`格式说明符来帮助消息。

```
>>> parser = argparse.ArgumentParser(prog='myprogram')
>>> parser.add_argument('--foo', help='foo of the %(prog)s program')
>>> parser.print_help()
usage: myprogram [-h] [--foo FOO]

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO   foo of the myprogram program
```

### usage

默认情况下，ArgumentParser根据它包含的参数计算使用情况消息：

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('--foo', nargs='?', help='foo help')
>>> parser.add_argument('bar', nargs='+', help='bar help')
>>> parser.print_help()
usage: PROG [-h] [--foo [FOO]] bar [bar ...]

positional arguments:
 bar          bar help

optional arguments:
 -h, --help   show this help message and exit
 --foo [FOO]  foo help
```

默认消息可以用`usage=`关键字参数覆盖：

```
>>> parser = argparse.ArgumentParser(prog='PROG', usage='%(prog)s [options]')
>>> parser.add_argument('--foo', nargs='?', help='foo help')
>>> parser.add_argument('bar', nargs='+', help='bar help')
>>> parser.print_help()
usage: PROG [options]

positional arguments:
 bar          bar help

optional arguments:
 -h, --help   show this help message and exit
 --foo [FOO]  foo help
```

该`%(prog)s`格式说明可填写程序名称到您的使用情况的信息。

### description

大多数对ArgumentParser构造函数的调用将使用 `description=`关键字参数。这个观点简要介绍了程序的功能和工作原理。在帮助消息中，说明显示在命令行用法字符串和各种参数的帮助消息之间：

```
>>> parser = argparse.ArgumentParser(description='A foo that bars')
>>> parser.print_help()
usage: argparse.py [-h]

A foo that bars

optional arguments:
 -h, --help  show this help message and exit
```

默认情况下，描述将被行包裹，以便它符合给定的空间。要改变这种行为，请参阅`formatter_class`参数。

### epilog

一些程序喜欢在参数描述之后显示程序的附加描述。这样的文本可以使用`epilog=`参数来指定 ArgumentParser：

```
>>> parser = argparse.ArgumentParser(
...     description='A foo that bars',
...     epilog="And that's how you'd foo a bar")
>>> parser.print_help()
usage: argparse.py [-h]

A foo that bars

optional arguments:
 -h, --help  show this help message and exit

And that's how you'd foo a bar
```

与`description`参数一样，`epilog=`文本在默认情况下是行包装的，但此行为可以使用`formatter_class`参数进行调整ArgumentParser。

### parents

有时，几个解析器共享一组通用参数。 可以使用具有所有共享参数并传递给`parents=`参数的单个解析器，而不是重复这些参数的定义ArgumentParser。该`parents=`参数获取ArgumentParser 对象列表，收集所有位置和可选操作，并将这些操作添加到ArgumentParser正在构建的对象中：

```
>>> parent_parser = argparse.ArgumentParser(add_help=False)
>>> parent_parser.add_argument('--parent', type=int)

>>> foo_parser = argparse.ArgumentParser(parents=[parent_parser])
>>> foo_parser.add_argument('foo')
>>> foo_parser.parse_args(['--parent', '2', 'XXX'])
Namespace(foo='XXX', parent=2)

>>> bar_parser = argparse.ArgumentParser(parents=[parent_parser])
>>> bar_parser.add_argument('--bar')
>>> bar_parser.parse_args(['--bar', 'YYY'])
Namespace(bar='YYY', parent=None)
```

请注意，大多数父解析器将指定`add_help=False`。否则， ArgumentParser将会看到两个`-h/--help`选项（一个在父项中，一个在子项中）并引发错误。

> 注意 在传递它们之前，您必须完全初始化解析器`parents=`。如果在子解析器之后更改父解析器，那些更改将不会反映到子解析器中。

### formatter_class

ArgumentParser对象允许通过指定备用格式类来自定义帮助格式。目前，有三个这样的类：

```
class argparse.RawDescriptionHelpFormatter
class argparse.RawTextHelpFormatter
class argparse.ArgumentDefaultsHelpFormatter
```

前两个允许更多的控制如何显示文本描述，而最后一个自动添加有关参数默认值的信息。

默认情况下，ArgumentParser对象将命令行帮助消息中的 `description`和 `epilog` 文本换行：

```
>>> parser = argparse.ArgumentParser(
...     prog='PROG',
...     description='''this description
...         was indented weird
...             but that is okay''',
...     epilog='''
...             likewise for this epilog whose whitespace will
...         be cleaned up and whose words will be wrapped
...         across a couple lines''')
>>> parser.print_help()
usage: PROG [-h]

this description was indented weird but that is okay

optional arguments:
 -h, --help  show this help message and exit

likewise for this epilog whose whitespace will be cleaned up and whose words
will be wrapped across a couple lines
```

`RawDescriptionHelpFormatter`作为`formatter_class=` 指示`description`和 `epilog`已经被正确格式化并且不应该被行包裹传递：

```
>>> parser = argparse.ArgumentParser(
...     prog='PROG',
...     formatter_class=argparse.RawDescriptionHelpFormatter,
...     description=textwrap.dedent('''\
...         Please do not mess up this text!
...         --------------------------------
...             I have indented it
...             exactly the way
...             I want it
...         '''))
>>> parser.print_help()
usage: PROG [-h]

Please do not mess up this text!
--------------------------------
   I have indented it
   exactly the way
   I want it

optional arguments:
 -h, --help  show this help message and exit
```
 
`RawTextHelpFormatter`为各种帮助文本保留空格，包括参数说明。但是，多个新行被替换为一个。如果您希望保留多个空行，请在换行符之间添加空格。

其他格式化程序类可用，`ArgumentDefaultsHelpFormatter`将添加有关每个参数的默认值的信息：

```
>>> parser = argparse.ArgumentParser(
...     prog='PROG',
...     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
>>> parser.add_argument('--foo', type=int, default=42, help='FOO!')
>>> parser.add_argument('bar', nargs='*', default=[1, 2, 3], help='BAR!')
>>> parser.print_help()
usage: PROG [-h] [--foo FOO] [bar [bar ...]]

positional arguments:
 bar         BAR! (default: [1, 2, 3])

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO   FOO! (default: 42)
```

### prefix_chars

大多数命令行选项将`-`用作前缀，例如`-f/--foo`。需要支持不同或额外前缀字符的解析器，例如像`+for`或`/foo`这样的选项，可以使用`prefix_chars=`参数来指定它们：

```
>>> parser = argparse.ArgumentParser(prog='PROG', prefix_chars='-+')
>>> parser.add_argument('+f')
>>> parser.add_argument('++bar')
>>> parser.parse_args('+f X ++bar Y'.split())
Namespace(bar='Y', f='X')
```

该`prefix_chars=`参数默认为`'-'`。提供一组不包含的字符`-`将导致`-f/--foo`选项不被允许。

### fromfile_prefix_chars

有时候，例如当处理特别长的参数列表时，将参数列表保存在文件中而不是在命令行输入它可能是有意义的。如果将`fromfile_prefix_chars=`参数赋予 ArgumentParser构造函数，那么以任何指定字符开头的参数将被视为文件，并将被它们包含的参数替换。例如：

```
>>> with open('args.txt', 'w') as fp:
...     fp.write('-f\nbar')
>>> parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
>>> parser.add_argument('-f')
>>> parser.parse_args(['-f', 'foo', '@args.txt'])
Namespace(f='bar')
```

从文件中读取的参数必须默认为每行一个（但也可参见` convert_arg_line_to_args()`），并且将它们视为与命令行中原始文件引用参数位于同一位置。所以在上面的例子中，表达式`['-f', 'foo', '@args.txt']`被认为等同于表达式`['-f', 'foo', '-f', 'bar']`.

该`fromfile_prefix_chars=`参数默认为None，这意味着参数就永远不会文件引用处理。

### argument_default 

一般来说，参数默认值是通过传递一个默认值给`add_argument()`来指定，或者通过调用`set_defaults()`具有一组特定名称/值对的 方法来指定参数默认值。然而，有时候，为参数指定单个解析器范围的默认值可能很有用。这可以通过传递 `argument_default=`关键字参数来完成ArgumentParser。例如，要全局大写`parse_args() `调用中的属性创建，我们提供`argument_default=SUPPRESS`：

```
>>> parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
>>> parser.add_argument('--foo')
>>> parser.add_argument('bar', nargs='?')
>>> parser.parse_args(['--foo', '1', 'BAR'])
Namespace(bar='BAR', foo='1')
>>> parser.parse_args([])
Namespace()
```
### conflict_handler 

ArgumentParser对象不允许具有相同选项字符串的两个操作。默认情况下，ArgumentParser如果尝试使用已在使用的选项字符串创建参数，则会引发异常：

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-f', '--foo', help='old foo help')
>>> parser.add_argument('--foo', help='new foo help')
Traceback (most recent call last):
 ..
ArgumentError: argument --foo: conflicting option string(s): --foo
```

有时（例如，在使用`parents`时），使用相同的选项字符串覆盖任何较旧的参数可能会很有用。为了获得这种行为，值 `'resolve'`可以提供给ArgumentParser的`conflict_handler=`参数 ：

```
>>> parser = argparse.ArgumentParser(prog='PROG', conflict_handler='resolve')
>>> parser.add_argument('-f', '--foo', help='old foo help')
>>> parser.add_argument('--foo', help='new foo help')
>>> parser.print_help()
usage: PROG [-h] [-f FOO] [--foo FOO]

optional arguments:
 -h, --help  show this help message and exit
 -f FOO      old foo help
 --foo FOO   new foo help
```
 
请注意，ArgumentParser如果对象的所有选项字符串都被覆盖，对象只会删除一个操作。因此，在上面的示例中，`old -f/--foo` 操作保留为`-f`操作，因为只有`--foo`选项字符串被覆盖。

### add_help 

默认情况下，ArgumentParser对象添加一个选项，该选项只显示解析器的帮助消息。例如，考虑一个名为myprogram.py包含以下代码的文件 ：

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()
```

如果`-h`或`--help`在命令行中提供，则会打印参数帮助器的帮助：

```
$ python myprogram.py --help
usage: myprogram.py [-h] [--foo FOO]

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO   foo help
```

偶尔，禁用此帮助选项可能会有用。这可以通过`add_help=False`参数传递给 ArgumentParser：

```
>>> parser = argparse.ArgumentParser(prog='PROG', add_help=False)
>>> parser.add_argument('--foo', help='foo help')
>>> parser.print_help()
usage: PROG [--foo FOO]

optional arguments:
 --foo FOO  foo help
```

帮助选项通常是`-h/--help`。这是一个例外，如果`prefix_chars=`指定并且不包括`-`，在这种情况下`-h`和`--help`不是有效的选项。在这种情况下，`prefix_chars`第一个字符用于作为帮助选项的前缀：

```
>>> parser = argparse.ArgumentParser(prog='PROG', prefix_chars='+/')
>>> parser.print_help()
usage: PROG [+h]

optional arguments:
  +h, ++help  show this help message and exit
```

## The add_argument() method

```
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
```

定义如何解析单个命令行参数。下面的每个参数都有自己的更详细的描述：

* `name or flags` - Either a name or a list of option strings, e.g. `foo` or `-f, --foo`.
* `action` - The basic type of action to be taken when this argument is encountered at the command line.
* `nargs` - The number of command-line arguments that should be consumed.
* `const` - A constant value required by some action and nargs selections.
* `default` - The value produced if the argument is absent from the command line.
* `type` - The type to which the command-line argument should be converted.
* `choices` - A container of the allowable values for the argument.
* `required` - Whether or not the command-line option may be omitted (optionals only).
* `help` - A brief description of what the argument does.
* `metavar` - A name for the argument in usage messages.
* `dest` - The name of the attribute to be added to the object returned by `parse_args()`.

以下部分描述如何使用每个这些。

### name or flags

`add_argument()`方法必须知道是否需要一个可选参数，如`-f`或`--foo`，或`positional argument`（如文件名列表）。因此传递给`add_argument()`的第一个参数必须是一系列flags或一个简单的参数name。例如，可以创建一个可选参数，如下所示：

```
>>> parser.add_argument('-f', '--foo')
```

而一个`positional argument`可以创建如下：

```
>>> parser.add_argument('bar')
```

当`parse_args()`被调用时，可选参数将由`-`前缀标识，其余参数将被假定为position：

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-f', '--foo')
>>> parser.add_argument('bar')
>>> parser.parse_args(['BAR'])
Namespace(bar='BAR', foo=None)
>>> parser.parse_args(['BAR', '--foo', 'FOO'])
Namespace(bar='BAR', foo='FOO')
>>> parser.parse_args(['--foo', 'FOO'])
usage: PROG [-h] [-f FOO] bar
PROG: error: too few arguments
```

### action

ArgumentParser对象将命令行参数与操作相关联。这些操作可以完成任何与它们相关的命令行参数的任何操作，尽管大多数操作只是为`parse_args()`返回的对象添加一个属性 。`action`关键字参数指定命令行参数应该如何处理。提供的操作是：

* `'store'` - 这只是存储参数的值。这是默认操作。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo')
>>> parser.parse_args('--foo 1'.split())
Namespace(foo='1')
```

* `'store_const'`- 这存储由const关键字参数指定的值。`'store_const'`操作最常用于指定某种flag的可选参数。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action='store_const', const=42)
>>> parser.parse_args(['--foo'])
Namespace(foo=42)
```

* `'store_true'`和`'store_false'`-这些都是特殊情况下 `'store_const'`使用，分别用于存储值True和False 。此外，他们分别创造的默认值False和True 。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action='store_true')
>>> parser.add_argument('--bar', action='store_false')
>>> parser.add_argument('--baz', action='store_false')
>>> parser.parse_args('--foo --bar'.split())
Namespace(bar=False, baz=True, foo=True)
```

* `'append'` - 这存储一个列表，并将每个参数值附加到列表中。这对于允许多次指定选项很有用。用法示例：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action='append')
>>> parser.parse_args('--foo 1 --foo 2'.split())
Namespace(foo=['1', '2'])
```

* `'append_const'`- 存储一个列表，并将const关键字参数指定的值附加到列表中。（请注意，`const`关键字参数默认为`None`。）`'append_const'`当多个参数需要将常量存储到同一列表时，该操作通常很有用。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--str', dest='types', action='append_const', const=str)
>>> parser.add_argument('--int', dest='types', action='append_const', const=int)
>>> parser.parse_args('--str --int'.split())
Namespace(types=[<type 'str'>, <type 'int'>])
```

* `'count'` - 这会计算关键字参数发生的次数。例如，这对增加冗长级别很有用：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--verbose', '-v', action='count')
>>> parser.parse_args(['-vvv'])
Namespace(verbose=3)
```

* `'help'` - 这将显示当前解析器中所有选项的完整帮助消息，然后退出。默认情况下，帮助操作会自动添加到解析器中。请参阅ArgumentParser有关如何创建输出的详细信息。

* `'version'`- 这需要调用`add_argument()`中的`version=`关键字参数 ，并打印版本信息并在调用时退出：

```
>>> import argparse
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('--version', action='version', version='%(prog)s 2.0')
>>> parser.parse_args(['--version'])
PROG 2.0
```

您也可以通过传递一个Action子类或其他实现相同接口的对象来指定一个任意的动作。推荐的方法是扩展`Action`，覆盖`__call__`方法和可选的`__init__`方法。

自定义操作的示例：

```
>>> class FooAction(argparse.Action):
...     def __init__(self, option_strings, dest, nargs=None, **kwargs):
...         if nargs is not None:
...             raise ValueError("nargs not allowed")
...         super(FooAction, self).__init__(option_strings, dest, **kwargs)
...     def __call__(self, parser, namespace, values, option_string=None):
...         print '%r %r %r' % (namespace, values, option_string)
...         setattr(namespace, self.dest, values)
...
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action=FooAction)
>>> parser.add_argument('bar', action=FooAction)
>>> args = parser.parse_args('1 --foo 2'.split())
Namespace(bar=None, foo=None) '1' None
Namespace(bar='1', foo=None) '2' '--foo'
>>> args
Namespace(bar='1', foo='2')
```

有关更多详情，请参阅Action。

### nargs

ArgumentParser对象通常会将单个命令行参数与要执行的单个操作相关联。`nargs`关键字参数将不同数量的命令行参数和与单个动作关联起来。支持的值是：

* `N`（一个整数）。 命令行中的`N`参数将汇集到一个列表中。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', nargs=2)
>>> parser.add_argument('bar', nargs=1)
>>> parser.parse_args('c --foo a b'.split())
Namespace(bar=['c'], foo=['a', 'b'])
```

请注意，`nargs=1`生成一个项目的列表。这与项目自行生成的默认值不同。

* `'?'`。如有可能，将从命令行中消耗一个参数，并将其作为单个项目生成。如果没有命令行参数，则会生成默认值 。请注意，对于可选参数，还有一个额外的情况 - 选项字符串存在，但后面没有命令行参数。在这种情况下，`const`的值将被生成。一些例子来说明这一点：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', nargs='?', const='c', default='d')
>>> parser.add_argument('bar', nargs='?', default='d')
>>> parser.parse_args(['XX', '--foo', 'YY'])
Namespace(bar='XX', foo='YY')
>>> parser.parse_args(['XX', '--foo'])
Namespace(bar='XX', foo='c')
>>> parser.parse_args([])
Namespace(bar='d', foo='d')
```

更常见的`nargs='?'`用途之一是允许可选的输入和输出文件：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
...                     default=sys.stdin)
>>> parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
...                     default=sys.stdout)
>>> parser.parse_args(['input.txt', 'output.txt'])
Namespace(infile=<open file 'input.txt', mode 'r' at 0x...>,
          outfile=<open file 'output.txt', mode 'w' at 0x...>)
>>> parser.parse_args([])
Namespace(infile=<open file '<stdin>', mode 'r' at 0x...>,
          outfile=<open file '<stdout>', mode 'w' at 0x...>)
```

* `'*'`。所有存在的命令行参数都被收集到一个列表中。请注意，`nargs='*'`具有多个位置参数通常没有多大意义，但`nargs='*'`可以使用多个可选参数。例如：

```
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', nargs='*')
>>> parser.add_argument('--bar', nargs='*')
>>> parser.add_argument('baz', nargs='*')
>>> parser.parse_args('a b --foo x y --bar 1 2'.split())
Namespace(bar=['1', '2'], baz=['a', 'b'], foo=['x', 'y'])
```

* `'+'`。就像`'*'`，现在所有的命令行参数都被收集到一个列表中。此外，如果没有至少一个命令行参数，则会生成错误消息。例如：

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('foo', nargs='+')
>>> parser.parse_args(['a', 'b'])
Namespace(foo=['a', 'b'])
>>> parser.parse_args([])
usage: PROG [-h] foo [foo ...]
PROG: error: too few arguments
```

* `argparse.REMAINDER`。所有其余的命令行参数都被收集到一个列表中。这对于派发到其他命令行实用程序的命令行实用程序通常很有用：

```
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('--foo')
>>> parser.add_argument('command')
>>> parser.add_argument('args', nargs=argparse.REMAINDER)
>>> print parser.parse_args('--foo B cmd --arg1 XX ZZ'.split())
Namespace(args=['--arg1', 'XX', 'ZZ'], command='cmd', foo='B')
```

如果`nargs`没有提供关键字参数，则消耗的参数数量由`action`决定。通常这意味着一个命令行参数将被消耗，并且将生成单个项目（不是列表）。

### const

`add_argument()`的`const`参数是用来装未在命令行中读取但ArgumentParser的`action`所需要的各种常数值。它最常见的两种用途是：

* 当`add_argument()`调用`action='store_const'` 或者`action='append_const'`。这些`action`将`const`值添加到返回的对象的某个属性中 。有关示例，请参阅`action`说明。
* 当`add_argument()`用选项字符串（如`-f` or `--foo`）和`nargs='?'`时。这会创建一个可选参数，后面跟零个或一个命令行参数。在解析命令行时，如果选项字符串遇到后面没有命令行参数，`const`的值则会取而代之。有关示例，请参阅`nargs`说明。

使用`'store_const'`和`'append_const'`操作时，`const `必须给出关键字参数。对于其他操作，它默认为`None`。

15.4.3.5。默认
所有可选参数和一些位置参数可以在命令行中省略。如果命令行参数不存在，那么其值缺省为的default关键字参数 add_argument()将None指定应使用的值。对于可选参数，default当选项字符串不存在于命令行时，将使用该值：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', default=42)
>>> parser.parse_args(['--foo', '2'])
Namespace(foo='2')
>>> parser.parse_args([])
Namespace(foo=42)
如果该default值是一个字符串，则解析器会将该值解析为一个命令行参数。特别是， 在设置返回值的属性之前，解析器应用任何类型转换参数（如果提供的话） Namespace。否则，解析器按原样使用该值：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--length', default='10', type=int)
>>> parser.add_argument('--width', default=10.5, type=int)
>>> parser.parse_args()
Namespace(length=10, width=10.5)
对于nargs等于?or的位置参数，当没有命令行参数存在时使用*该default值：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('foo', nargs='?', default=42)
>>> parser.parse_args(['a'])
Namespace(foo='a')
>>> parser.parse_args([])
Namespace(foo=42)
default=argparse.SUPPRESS如果命令行参数不存在，则提供不导致添加属性：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', default=argparse.SUPPRESS)
>>> parser.parse_args([])
Namespace()
>>> parser.parse_args(['--foo', '1'])
Namespace(foo='1')
15.4.3.6。键入
默认情况下，ArgumentParser对象以简单的字符串读取命令行参数。但是，通常命令行字符串应该被解释为另一种类型，如a float或int。所述 type的关键字参数add_argument()允许执行任何必要的类型检查和类型转换。常见的内置类型和函数可以直接用作type参数的值：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('foo', type=int)
>>> parser.add_argument('bar', type=file)
>>> parser.parse_args('2 temp.txt'.split())
Namespace(bar=<open file 'temp.txt', mode 'r' at 0x...>, foo=2)
有关何时将参数应用于默认参数的信息，请参阅default关键字参数 部分type。

为了简化各种类型的文件的使用，argparse模块提供了工厂文件类型，该文件类型包含对象的参数mode=和bufsize=参数 file。例如，FileType('w')可以用来创建一个可写文件：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('bar', type=argparse.FileType('w'))
>>> parser.parse_args(['out.txt'])
Namespace(bar=<open file 'out.txt', mode 'w' at 0x...>)
type= 可以采用任何可以调用单个字符串参数的可调用方法，并返回转换后的值：

>>> def perfect_square(string):
...     value = int(string)
...     sqrt = math.sqrt(value)
...     if sqrt != int(sqrt):
...         msg = "%r is not a perfect square" % string
...         raise argparse.ArgumentTypeError(msg)
...     return value
...
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('foo', type=perfect_square)
>>> parser.parse_args(['9'])
Namespace(foo=9)
>>> parser.parse_args(['7'])
usage: PROG [-h] foo
PROG: error: argument foo: '7' is not a perfect square
该选择关键字参数可以是用于类型检查，简单地核对值的范围更方便：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('foo', type=int, choices=xrange(5, 10))
>>> parser.parse_args(['7'])
Namespace(foo=7)
>>> parser.parse_args(['11'])
usage: PROG [-h] {5,6,7,8,9}
PROG: error: argument foo: invalid choice: 11 (choose from 5, 6, 7, 8, 9)
请参阅选项部分了解更多详情。

15.4.3.7。选择
一些命令行参数应该从一组受限制的值中选择。这些可以通过将容器对象作为选择关键字参数传递给add_argument()。在解析命令行时，将检查参数值，如果参数不是可接受值之一，则会显示错误消息：

>>> parser = argparse.ArgumentParser(prog='game.py')
>>> parser.add_argument('move', choices=['rock', 'paper', 'scissors'])
>>> parser.parse_args(['rock'])
Namespace(move='rock')
>>> parser.parse_args(['fire'])
usage: game.py [-h] {rock,paper,scissors}
game.py: error: argument move: invalid choice: 'fire' (choose from 'rock',
'paper', 'scissors')
请注意，在 执行任何类型转换后，将检查包含在选择容器中的内容，以便选择 容器中的对象类型应与指定的类型匹配：

>>> parser = argparse.ArgumentParser(prog='doors.py')
>>> parser.add_argument('door', type=int, choices=range(1, 4))
>>> print(parser.parse_args(['3']))
Namespace(door=3)
>>> parser.parse_args(['4'])
usage: doors.py [-h] {1,2,3}
doors.py: error: argument door: invalid choice: 4 (choose from 1, 2, 3)
任何支持in操作符的对象都可以作为选择 值传递，因此所有dict对象，set对象，自定义容器等都是受支持的。

15.4.3.8。需要
通常情况下，argparse模块假定标志像-f和--bar 表示可选参数，这些参数在命令行中总是可以省略。要创建所需的选项，True可以为required= 关键字参数指定add_argument()：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', required=True)
>>> parser.parse_args(['--foo', 'BAR'])
Namespace(foo='BAR')
>>> parser.parse_args([])
usage: argparse.py [-h] [--foo FOO]
argparse.py: error: option --foo is required
如示例所示，如果某个选项被标记为required， parse_args()则会在命令行中不存在该选项时报告错误。

注意 所需的选项通常被认为是不好的形式，因为用户期望 选项是可选的，因此应尽可能避免。
15.4.3.9。帮助
该help值是一个包含参数简要描述的字符串。当用户请求帮助时（通常通过使用-h或--help通过命令行），这些help描述将与每个参数一起显示：

>>> parser = argparse.ArgumentParser(prog='frobble')
>>> parser.add_argument('--foo', action='store_true',
...                     help='foo the bars before frobbling')
>>> parser.add_argument('bar', nargs='+',
...                     help='one of the bars to be frobbled')
>>> parser.parse_args(['-h'])
usage: frobble [-h] [--foo] bar [bar ...]

positional arguments:
 bar     one of the bars to be frobbled

optional arguments:
 -h, --help  show this help message and exit
 --foo   foo the bars before frobbling
这些help字符串可以包含各种格式说明符，以避免重复诸如程序名称或参数默认值之类的内容。可用符包括节目名称，%(prog)s和大多数关键字参数 add_argument()，如%(default)s，%(type)s等：

>>> parser = argparse.ArgumentParser(prog='frobble')
>>> parser.add_argument('bar', nargs='?', type=int, default=42,
...                     help='the bar to %(prog)s (default: %(default)s)')
>>> parser.print_help()
usage: frobble [-h] [bar]

positional arguments:
 bar     the bar to frobble (default: 42)

optional arguments:
 -h, --help  show this help message and exit
argparse支持通过将某些选项的帮助条目设置help为argparse.SUPPRESS：

>>> parser = argparse.ArgumentParser(prog='frobble')
>>> parser.add_argument('--foo', help=argparse.SUPPRESS)
>>> parser.print_help()
usage: frobble [-h]

optional arguments:
  -h, --help  show this help message and exit
15.4.3.10。metavar 
当ArgumentParser生成帮助消息时，需要一些方法来引用每个预期的参数。默认情况下，ArgumentParser对象使用dest 值作为每个对象的“名称”。默认情况下，对于位置参数操作，dest值直接使用，对于可选参数操作，dest值是大写的。所以，一个单独的位置参数 dest='bar'将被称为bar。--foo单个命令行参数后面的单个可选参数将被称为FOO。一个例子：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo')
>>> parser.add_argument('bar')
>>> parser.parse_args('X --foo Y'.split())
Namespace(bar='X', foo='Y')
>>> parser.print_help()
usage:  [-h] [--foo FOO] bar

positional arguments:
 bar

optional arguments:
 -h, --help  show this help message and exit
 --foo FOO
另一个名称可以用以下方式指定metavar：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', metavar='YYY')
>>> parser.add_argument('bar', metavar='XXX')
>>> parser.parse_args('X --foo Y'.split())
Namespace(bar='X', foo='Y')
>>> parser.print_help()
usage:  [-h] [--foo YYY] XXX

positional arguments:
 XXX

optional arguments:
 -h, --help  show this help message and exit
 --foo YYY
请注意，metavar只会更改显示的名称 - parse_args()对象上属性的名称仍由dest值确定。

不同的值nargs可能会导致metavar被多次使用。提供一个元组来metavar为每个参数指定一个不同的显示：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-x', nargs=2)
>>> parser.add_argument('--foo', nargs=2, metavar=('bar', 'baz'))
>>> parser.print_help()
usage: PROG [-h] [-x X X] [--foo bar baz]

optional arguments:
 -h, --help     show this help message and exit
 -x X X
 --foo bar baz
15.4.3.11。DEST 
大多数ArgumentParser操作都会添加一些值作为返回对象的属性parse_args()。该属性的名称由dest关键字参数确定 add_argument()。对于位置参数操作， dest通常作为第一个参数提供给 add_argument()：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('bar')
>>> parser.parse_args(['XXX'])
Namespace(bar='XXX')
对于可选的参数操作，dest通常从选项字符串推断值。 通过取第一个长选项字符串并剥离初始 字符串来ArgumentParser生成值。如果没有提供长选项字符串，则将通过剥离初始字符从第一个短选项字符串派生。任何内部字符都将转换为字符以确保该字符串是有效的属性名称。下面的例子说明了这种行为：dest--dest--_

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('-f', '--foo-bar', '--foo')
>>> parser.add_argument('-x', '-y')
>>> parser.parse_args('-f 1 -x 2'.split())
Namespace(foo_bar='1', x='2')
>>> parser.parse_args('--foo 1 -y 2'.split())
Namespace(foo_bar='1', x='2')
dest 允许提供自定义属性名称：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', dest='bar')
>>> parser.parse_args('--foo XXX'.split())
Namespace(bar='XXX')
15.4.3.12。动作类
Action类实现Action API，这是一个可调用的函数，它返回一个可从命令行处理参数的可调用对象。任何遵循此API的对象都可以作为action参数传递给add_argument()。

class argparse.Action（option_strings，dest，nargs = None，const = None，default = None，type = None，choices = None，required = False，help = None，metavar = None ）
ArgumentParser使用Action对象来表示解析来自命令行的一个或多个字符串的单个参数所需的信息。Action类必须接受两个位置参数以及传递给ArgumentParser.add_argument()除action它自身以外的任何关键字参数。

Action的实例（或任何可调用的action 参数的返回值）应该具有定义的属性“dest”，“option_strings”，“default”，“type”，“required”，“help”等。确定这些属性的最简单方法是调用Action.__init__。

动作实例应该是可调用的，所以子类必须重载该 __call__方法，该方法应该接受四个参数：

parser - 包含此操作的ArgumentParser对象。
namespace- Namespace将被返回的对象 parse_args()。大多数操作都使用此对象添加属性setattr()。
values - 关联的命令行参数，应用了任何类型的转换。类型转换用type关键字参数 指定add_argument()。
option_string - 用于调用此操作的选项字符串。该option_string参数是可选的，如果该动作与位置参数相关联，则该参数将不存在。
该__call__方法可能会执行任意操作，但通常会在namespace基于dest和上设置属性values。

15.4.4。parse_args（）方法
ArgumentParser.parse_args（args = None，namespace = None ）
将参数字符串转换为对象并将它们分配为命名空间的属性。返回填充的命名空间。

先前的调用add_argument()确切地确定创建了哪些对象以及如何分配对象。有关add_argument()详细信息，请参阅文档 。

args - 要解析的字符串列表。默认取自 sys.argv。
命名空间 - 一个获取属性的对象。默认值是一个新的空 Namespace对象。
15.4.4.1。选项值语法
该parse_args()方法支持多种指定选项值的方式（如果需要的话）。在最简单的情况下，该选项及其值作为两个单独的参数传递：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-x')
>>> parser.add_argument('--foo')
>>> parser.parse_args(['-x', 'X'])
Namespace(foo=None, x='X')
>>> parser.parse_args(['--foo', 'FOO'])
Namespace(foo='FOO', x=None)
对于长选项（名称长于单个字符的选项），该选项和值也可以作为单个命令行参数传递，=用于将它们分开：

>>> parser.parse_args(['--foo=FOO'])
Namespace(foo='FOO', x=None)
对于短期选项（选项只有一个字符长），选项和它的值可以连接起来：

>>> parser.parse_args(['-xX'])
Namespace(foo=None, x='X')
-只要使用一个前缀，几个简短的选项就可以结合在一起，只要最后一个选项（或者它们中没有一个）需要一个值：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-x', action='store_true')
>>> parser.add_argument('-y', action='store_true')
>>> parser.add_argument('-z')
>>> parser.parse_args(['-xyzZ'])
Namespace(x=True, y=True, z='Z')
15.4.4.2。无效的参数
在解析命令行时，parse_args()检查各种错误，包括不明确的选项，无效的类型，无效的选项，错误的位置参数数量等。遇到此类错误时，它会退出并将错误与使用消息一起打印出来：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('--foo', type=int)
>>> parser.add_argument('bar', nargs='?')

>>> # invalid type
>>> parser.parse_args(['--foo', 'spam'])
usage: PROG [-h] [--foo FOO] [bar]
PROG: error: argument --foo: invalid int value: 'spam'

>>> # invalid option
>>> parser.parse_args(['--bar'])
usage: PROG [-h] [--foo FOO] [bar]
PROG: error: no such option: --bar

>>> # wrong number of arguments
>>> parser.parse_args(['spam', 'badger'])
usage: PROG [-h] [--foo FOO] [bar]
PROG: error: extra arguments found: badger
15.4.4.3。包含参数-
parse_args()每当用户明确犯了错误，该方法都会尝试给出错误，但有些情况本质上是不明确的。例如，命令行参数-1可能是尝试指定选项或试图提供位置参数。这里的parse_args()方法很谨慎：位置参数可能只有-在它们看起来像负数时才会开始，解析器中没有任何选项看起来像负数：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-x')
>>> parser.add_argument('foo', nargs='?')

>>> # no negative number options, so -1 is a positional argument
>>> parser.parse_args(['-x', '-1'])
Namespace(foo=None, x='-1')

>>> # no negative number options, so -1 and -5 are positional arguments
>>> parser.parse_args(['-x', '-1', '-5'])
Namespace(foo='-5', x='-1')

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-1', dest='one')
>>> parser.add_argument('foo', nargs='?')

>>> # negative number options present, so -1 is an option
>>> parser.parse_args(['-1', 'X'])
Namespace(foo=None, one='X')

>>> # negative number options present, so -2 is an option
>>> parser.parse_args(['-2'])
usage: PROG [-h] [-1 ONE] [foo]
PROG: error: no such option: -2

>>> # negative number options present, so both -1s are options
>>> parser.parse_args(['-1', '-1'])
usage: PROG [-h] [-1 ONE] [foo]
PROG: error: argument -1: expected one argument
如果您的位置参数必须以开头-且看起来不像负数，那么您可以插入伪参数'--'， 该伪参数parse_args()指示之后的所有内容都是位置参数：

>>> parser.parse_args(['--', '-f'])
Namespace(foo='-f', one=None)
15.4.4.4。参数缩写（前缀匹配）
parse_args()如果缩写是明确的（前缀匹配唯一选项），该方法允许将长选项缩写为前缀：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('-bacon')
>>> parser.add_argument('-badger')
>>> parser.parse_args('-bac MMM'.split())
Namespace(bacon='MMM', badger=None)
>>> parser.parse_args('-bad WOOD'.split())
Namespace(bacon=None, badger='WOOD')
>>> parser.parse_args('-ba BA'.split())
usage: PROG [-h] [-bacon BACON] [-badger BADGER]
PROG: error: ambiguous option: -ba could match -badger, -bacon
对于可能产生多个选项的参数会产生错误。

15.4.4.5。超越sys.argv
有时候可能有一个ArgumentParser解析其他的参数sys.argv。这可以通过传递一个字符串列表来完成 parse_args()。这对于在交互式提示下进行测试很有用：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument(
...     'integers', metavar='int', type=int, choices=xrange(10),
...     nargs='+', help='an integer in the range 0..9')
>>> parser.add_argument(
...     '--sum', dest='accumulate', action='store_const', const=sum,
...     default=max, help='sum the integers (default: find the max)')
>>> parser.parse_args(['1', '2', '3', '4'])
Namespace(accumulate=<built-in function max>, integers=[1, 2, 3, 4])
>>> parser.parse_args(['1', '2', '3', '4', '--sum'])
Namespace(accumulate=<built-in function sum>, integers=[1, 2, 3, 4])
15.4.4.6。名字空间对象
类argparse.Namespace
简单的类默认使用parse_args()创建一个对象来保存属性并返回它。

这个类是故意简单的，只是一个object具有可读字符串表示的子类。如果您更喜欢使用类似字典的属性视图，则可以使用标准的Python语言vars()：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo')
>>> args = parser.parse_args(['--foo', 'BAR'])
>>> vars(args)
{'foo': 'BAR'}
ArgumentParser将属性赋值给已经存在的对象而不是新Namespace对象也可能是有用的。这可以通过指定namespace=关键字参数来实现：

>>> class C(object):
...     pass
...
>>> c = C()
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo')
>>> parser.parse_args(args=['--foo', 'BAR'], namespace=c)
>>> c.foo
'BAR'
15.4.5。其他工具
15.4.5.1。子命令
ArgumentParser.add_subparsers([title][, description][, prog][, parser_class][, action][, option_string][, dest][, help][, metavar])
许多程序分割它们的功能分成若干子命令，例如，该svn程序可以调用像子命令，和。当程序执行需要不同种类的命令行参数的几个不同功能时，以这种方式拆分功能可能是一个特别好的想法。 支持用该方法创建这样的子命令 。该方法通常不带参数调用，并返回一个特殊的操作对象。该对象有一个方法，它接受一个命令名和任何构造函数参数，并返回一个可以像平常一样修改的对象。svn checkoutsvn updatesvn commitArgumentParseradd_subparsers()add_subparsers()add_parser()ArgumentParserArgumentParser

参数描述：

title - 帮助输出中的子分析器组的标题; 如果提供了描述，则默认为“子命令”，否则使用标题作为位置参数
描述 - 默认情况下帮助输出中的子分析器组的说明 None
将使用子命令帮助显示的prog - 使用信息，默认情况下程序的名称和子分析器参数前的任何位置参数
parser_class - 将用于创建子分析器实例的类，默认情况下是当前分析器的类（例如，ArgumentParser）
操作 - 在命令行中遇到此参数时要采取的基本操作类型
dest - 存储子命令名称的属性名称; 默认情况下None不存储任何值
帮助 - 默认帮助帮助输出中的子分析器组None
metavar - 字符串在帮助中显示可用的子命令; 默认情况下它是None和呈现形式为{cmd1，cmd2，..}的子命令
一些示例用法：

>>> # create the top-level parser
>>> parser = argparse.ArgumentParser(prog='PROG')
>>> parser.add_argument('--foo', action='store_true', help='foo help')
>>> subparsers = parser.add_subparsers(help='sub-command help')
>>>
>>> # create the parser for the "a" command
>>> parser_a = subparsers.add_parser('a', help='a help')
>>> parser_a.add_argument('bar', type=int, help='bar help')
>>>
>>> # create the parser for the "b" command
>>> parser_b = subparsers.add_parser('b', help='b help')
>>> parser_b.add_argument('--baz', choices='XYZ', help='baz help')
>>>
>>> # parse some argument lists
>>> parser.parse_args(['a', '12'])
Namespace(bar=12, foo=False)
>>> parser.parse_args(['--foo', 'b', '--baz', 'Z'])
Namespace(baz='Z', foo=True)
请注意，返回的对象parse_args()将仅包含由命令行选择的主分析器和子分析器的属性（而不是任何其他子分析器）。因此，在上面的示例中，当a指定命令时，只有foo和bar属性存在，并且在b指定命令时，只有foo和 baz属性存在。

同样，当从分析器请求帮助消息时，只会打印该特定分析器的帮助。帮助消息将不包括父解析器或同级解析器消息。（但是，每个子分析器命令的帮助消息可以通过提供help=参数给出，add_parser()如上所示。）

>>> parser.parse_args(['--help'])
usage: PROG [-h] [--foo] {a,b} ...

positional arguments:
  {a,b}   sub-command help
    a     a help
    b     b help

optional arguments:
  -h, --help  show this help message and exit
  --foo   foo help

>>> parser.parse_args(['a', '--help'])
usage: PROG a [-h] bar

positional arguments:
  bar     bar help

optional arguments:
  -h, --help  show this help message and exit

>>> parser.parse_args(['b', '--help'])
usage: PROG b [-h] [--baz {X,Y,Z}]

optional arguments:
  -h, --help     show this help message and exit
  --baz {X,Y,Z}  baz help
该add_subparsers()方法还支持title和description 关键字参数。当出现任何一个时，子分析器的命令将出现在帮助输出中的他们自己的组中。例如：

>>> parser = argparse.ArgumentParser()
>>> subparsers = parser.add_subparsers(title='subcommands',
...                                    description='valid subcommands',
...                                    help='additional help')
>>> subparsers.add_parser('foo')
>>> subparsers.add_parser('bar')
>>> parser.parse_args(['-h'])
usage:  [-h] {foo,bar} ...

optional arguments:
  -h, --help  show this help message and exit

subcommands:
  valid subcommands

  {foo,bar}   additional help
处理子命令的一种特别有效的方式是将该add_subparsers()方法的使用与调用结合起来，set_defaults()以便每个子分析器知道应该执行哪个Python函数。例如：

>>> # sub-command functions
>>> def foo(args):
...     print args.x * args.y
...
>>> def bar(args):
...     print '((%s))' % args.z
...
>>> # create the top-level parser
>>> parser = argparse.ArgumentParser()
>>> subparsers = parser.add_subparsers()
>>>
>>> # create the parser for the "foo" command
>>> parser_foo = subparsers.add_parser('foo')
>>> parser_foo.add_argument('-x', type=int, default=1)
>>> parser_foo.add_argument('y', type=float)
>>> parser_foo.set_defaults(func=foo)
>>>
>>> # create the parser for the "bar" command
>>> parser_bar = subparsers.add_parser('bar')
>>> parser_bar.add_argument('z')
>>> parser_bar.set_defaults(func=bar)
>>>
>>> # parse the args and call whatever function was selected
>>> args = parser.parse_args('foo 1 -x 2'.split())
>>> args.func(args)
2.0
>>>
>>> # parse the args and call whatever function was selected
>>> args = parser.parse_args('bar XYZYX'.split())
>>> args.func(args)
((XYZYX))
这样，您可以parse_args()在参数分析完成后调用适当的函数。将功能与这样的动作相关联通常是处理每个子分析器的不同动作的最简单方法。但是，如果需要检查被调用的子分析器的名称，则该调用的dest关键字参数add_subparsers()将起作用：

>>> parser = argparse.ArgumentParser()
>>> subparsers = parser.add_subparsers(dest='subparser_name')
>>> subparser1 = subparsers.add_parser('1')
>>> subparser1.add_argument('-x')
>>> subparser2 = subparsers.add_parser('2')
>>> subparser2.add_argument('y')
>>> parser.parse_args(['2', 'frobble'])
Namespace(subparser_name='2', y='frobble')
15.4.5.2。FileType对象
class argparse.FileType（mode ='r'，bufsize = None ）
该FileType工厂创建一个可以传递给类型参数的对象ArgumentParser.add_argument()。将 FileType对象作为类型的参数将以具有所请求模式和缓冲区大小的文件的形式打开命令行参数：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--output', type=argparse.FileType('wb', 0))
>>> parser.parse_args(['--output', 'out'])
Namespace(output=<open file 'out', mode 'wb' at 0x...>)
FileType对象理解伪参数'-'并自动将其转换sys.stdin为可读FileType对象和 sys.stdout可写FileType对象：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('infile', type=argparse.FileType('r'))
>>> parser.parse_args(['-'])
Namespace(infile=<open file '<stdin>', mode 'r' at 0x...>)
15.4.5.3。参数组
ArgumentParser.add_argument_group（title = None，description = None ）
默认情况下，ArgumentParser在显示帮助消息时将命令行参数分组为“位置参数”和“可选参数”。当比这个默认参数有一个更好的概念组参数时，可以使用该add_argument_group()方法创建适当的组 ：

>>> parser = argparse.ArgumentParser(prog='PROG', add_help=False)
>>> group = parser.add_argument_group('group')
>>> group.add_argument('--foo', help='foo help')
>>> group.add_argument('bar', help='bar help')
>>> parser.print_help()
usage: PROG [--foo FOO] bar

group:
  bar    bar help
  --foo FOO  foo help
该add_argument_group()方法返回一个参数组对象，它有一个add_argument()像普通的方法 ArgumentParser。当将参数添加到组中时，解析器将其视为与普通参数类似，但将参数显示在单独的组中以获取帮助消息。该add_argument_group()方法接受可用于自定义此显示的标题和描述参数：

>>> parser = argparse.ArgumentParser(prog='PROG', add_help=False)
>>> group1 = parser.add_argument_group('group1', 'group1 description')
>>> group1.add_argument('foo', help='foo help')
>>> group2 = parser.add_argument_group('group2', 'group2 description')
>>> group2.add_argument('--bar', help='bar help')
>>> parser.print_help()
usage: PROG [--bar BAR] foo

group1:
  group1 description

  foo    foo help

group2:
  group2 description

  --bar BAR  bar help
请注意，任何不在用户定义组中的参数将返回到通常的“位置参数”和“可选参数”部分。

15.4.5.4。互斥
ArgumentParser.add_mutually_exclusive_group（required = False ）
创建一个互斥组。argparse将确保在命令行中仅存在互斥组中的一个参数：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> group = parser.add_mutually_exclusive_group()
>>> group.add_argument('--foo', action='store_true')
>>> group.add_argument('--bar', action='store_false')
>>> parser.parse_args(['--foo'])
Namespace(bar=True, foo=True)
>>> parser.parse_args(['--bar'])
Namespace(bar=False, foo=False)
>>> parser.parse_args(['--foo', '--bar'])
usage: PROG [-h] [--foo | --bar]
PROG: error: argument --bar: not allowed with argument --foo
该add_mutually_exclusive_group()方法还接受必需的 参数，以指示至少需要一个互斥参数：

>>> parser = argparse.ArgumentParser(prog='PROG')
>>> group = parser.add_mutually_exclusive_group(required=True)
>>> group.add_argument('--foo', action='store_true')
>>> group.add_argument('--bar', action='store_false')
>>> parser.parse_args([])
usage: PROG [-h] (--foo | --bar)
PROG: error: one of the arguments --foo --bar is required
请注意，目前互斥的参数组不支持 标题和描述参数 add_argument_group()。

15.4.5.5。解析器默认值
ArgumentParser.set_defaults（** kwargs ）
大多数情况下，parse_args() 通过检查命令行参数和参数操作来完全确定返回的对象的属性。 set_defaults()允许一些额外的属性被确定，而不需要检查命令行的添加：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('foo', type=int)
>>> parser.set_defaults(bar=42, baz='badger')
>>> parser.parse_args(['736'])
Namespace(bar=42, baz='badger', foo=736)
请注意，解析器级别的默认值总是覆盖参数级别的默认值：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', default='bar')
>>> parser.set_defaults(foo='spam')
>>> parser.parse_args([])
Namespace(foo='spam')
解析器级别的默认值在使用多个解析器时可能特别有用。请参阅该add_subparsers()方法以获取此类型的示例。

ArgumentParser.get_default（dest ）
获取默认值的命名空间属性，为通过设置 add_argument()或通过 set_defaults()：

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', default='badger')
>>> parser.get_default('foo')
'badger'
15.4.5.6。打印帮助
在大多数典型的应用程序中，parse_args()将负责格式化和打印任何使用或错误消息。但是，有几种格式化方法可用：

ArgumentParser.print_usage（file = None ）
打印关于ArgumentParser应该如何在命令行上调用的简要说明。如果文件是None，sys.stdout假定。

ArgumentParser.print_help（file = None ）
打印帮助信息，包括程序使用情况和关于使用注册的参数的信息ArgumentParser。如果文件是 None，sys.stdout假定。

也有这些方法的变体，只是返回一个字符串，而不是打印它：

ArgumentParser.format_usage（）
返回一个字符串，其中包含ArgumentParser应该如何在命令行上调用的简要说明 。

ArgumentParser.format_help（）
返回一个包含帮助信息的字符串，包括程序使用情况和有关使用注册的参数的信息ArgumentParser。

15.4.5.7。部分解析
ArgumentParser.parse_known_args（args = None，namespace = None ）
有时脚本可能只解析一些命令行参数，将剩下的参数传递给另一个脚本或程序。在这些情况下，该 parse_known_args()方法可能很有用。它的工作方式很像， parse_args()只是在出现额外参数时不会产生错误。相反，它会返回一个包含已填充名称空间和剩余参数字符串列表的两个项目元组。

>>> parser = argparse.ArgumentParser()
>>> parser.add_argument('--foo', action='store_true')
>>> parser.add_argument('bar')
>>> parser.parse_known_args(['--foo', '--badger', 'BAR', 'spam'])
(Namespace(bar='BAR', foo=True), ['--badger', 'spam'])
警告 前缀匹配规则适用于 parse_known_args()。解析器可能会使用一个选项，即使它只是其中一个已知选项的前缀，而不是将它留在剩余的参数列表中。
15.4.5.8。自定义文件解析
ArgumentParser.convert_arg_line_to_args（arg_line ）
从文件读取的参数（请参阅构造函数的fromfile_prefix_chars 关键字参数ArgumentParser）每行读取一个参数。convert_arg_line_to_args()可以替代更好的阅读。

该方法采用一个参数arg_line，它是从参数文件中读取的字符串。它返回从此字符串解析的参数列表。按顺序从参数文件中读取每行一次的方法。

这种方法的有用重写是将每个空格分隔的单词作为参数对待：

def convert_arg_line_to_args(self, arg_line):
    return arg_line.split()
15.4.5.9。退出方法
ArgumentParser.exit（status = 0，message = None ）
该方法终止程序，以指定的状态退出， 并且如果给定，则在此之前打印消息。

ArgumentParser.error（消息）
此方法将包含消息的使用消息打印到标准错误，并终止状态代码为2的程序。

15.4.6。升级optparse代码
最初，argparse模块试图保持兼容性optparse。但是，optparse很难透明地进行扩展，尤其是需要支持新的nargs=说明符和更好的使用消息所需的更改 。当大多数内容 optparse都被复制粘贴或猴子修补时，尝试维持向后兼容性似乎不再实际可行。

该argparse模块optparse 以多种方式改进了标准库模块，包括：

处理位置参数。
支持子命令。
允许像+和的替代选项前缀/。
处理零个或多个以及一个或多个样式的参数。
生成更多信息的使用信息。
为定制type和提供更简单的界面action。
从A部分升级路径optparse到argparse：

用optparse.OptionParser.add_option()呼叫 替换所有ArgumentParser.add_argument()呼叫。
更换用并添加额外 的定位参数调用。请记住，现在在上下文中调用的是之前称为的内容。(options, args) = parser.parse_args()args = parser.parse_args()ArgumentParser.add_argument()optionsargparseargs
更换optparse.OptionParser.disable_interspersed_args() 设定nargs一个位置参数来argparse.REMAINDER，或使用parse_known_args()收集未解析字符串参数在一个单独的列表。
callback_*使用type或action参数替换回调操作和关键字 参数。
将type关键字参数的字符串名称替换为相应的类型对​​象（例如int，float，complex等）。
更换optparse.Values用Namespace和 optparse.OptionError与optparse.OptionValueError用 ArgumentError。
用隐式参数替换字符串，例如%default或%prog使用标准Python语法来使用字典格式化字符串，也就是 %(default)s和%(prog)s。
用version调用来替换OptionParser构造函数参数 。parser.add_argument('--version', action='version', version='<the version>')
