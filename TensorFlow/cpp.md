## string初始化

```cpp
string s1;
string s2(s1);
string s2 = s1;
string s3("hello");
string s3 = "hello";
string s4 = {5, 'c'}; //ccccc
```

## string 操作

|operate|describe|
|:----|:----|
|os<<s|将s写到输出流os当中，返回os|
|is>>s|从is中读取字符串给s，字符串以空白分隔，返回is|
|getline(is, s)|从is中读取一行赋给s，返回is|
|s.empty()|s为空返回true，否则返回false|
|s.size()|返回s中字符的个数|
|s[n]|返回s中第n个字符的引用，位置n从0计起|
|s1+s2|返回s1和s2连接后的结果|
|s1=s2|用s2的副本代替s1中原来的字符|
|s1==s2|s1和s2字符完全一样为true|
|s1!=s2|和上相反|
|<, <=, >, >=|字典顺序比较|

## cctype头文件对字符操作

|function|description|
|:----|:----|
|isalnum(c)|c为字母或者数字时为true|
|isalpha(c)|c为字母时为true|
|iscntrl(c)|c为控制字符时为ture|
|isdigit(c)|c为数字时为true|
|isgraph(c)|c不是空格但可打印时为true|
|islower(c)|c为小写字母时为true|
|isprint(c)|c是可打印字符时为true|
|ispunct(c)|c为标点符号是为true|
|isspace(c)|c为空白时为true|
|isupper(c)|c为大写字母时为true|
|isxdigit(c)|c为16进制数字时为ture|
|tolower(c)|大写转小写|
|toupper(c)|小写转大写|


## vector初始化方法

|initial|description|
|:----|:----|
|vector<T> v1|空vector|
|vector<T> v2(v1)|v2中包含v1所有元素的副本|
|vector<T> v2 = v1|同上|
|vector<T> v3(n, val)|包含n个val元素|
|vector<T> v4(n)|v4包含了n个重复地执行了值初始化的对象|
|vector<T> v5{a,b,c...}|v5包含了初始值个数的元素，每个元素被赋予相应的初始值|
|vector<T> v5 = {a,b,c...}|同上|

```cpp
//数组初始化vector
int arr = {0, 1, 2, 3, 4, 5};
vector<int> vec(begin(arr), end(arr));
vector<int> subVec(arr+1, arr+4); //arr[1], arr[2], arr[3]
```

## vector操作

|operator|description|
|:----|:----|
|v.empty()|判断是否为空|
|v.size()|返回v中元素个数|
|v.push_back(t)|向v尾端添加t元素|
|v[n]|取v中第n个元素|
|v1 = v2|v2中元素的拷贝替换v1中的元素|
|v1 = {a,b,c...}|列表元素的拷贝替换v1中的元素|
|v1 == v2|v1和v2数量相等且对应位置元素相同时为true|
|v1 != v2|与上相反|
|<, <=, >, >=|字典顺序比较|

## 迭代器运算符

|operator|description|
|:----|:----|
|*iter|返回迭代器iter所指元素的引用|
|iter->mem|解引用iter并获取该元素的名为mem的成员，等价于(*iter).mem|
|++iter|令iter指示容器中的下一个元素|
|--iter|令iter指示容器中的上一个元素|
|iter1 == iter2|判断两个迭代器是否相等，如果指示同一个元素或者它们是同一容器的尾后迭代器，则相等|
|iter1 != iter2|与上相反|

## vector和string迭代器支持的运算

```cpp
iter + n
iter - n
iter1 += n
iter1 -= n
iter1 - iter2
>, >=, <, <=
```

## 多维数组范围for语句

```cpp
size_t cnt = 0;
for (auto &row : ia){
    for (auto &col : row){
        col = cnt;
        ++cnt;
    }
}
```

## 可变数量形参表示

initializer_list

* `initializer_list<T> lst`
* `initializer_list<T> lst{a,b,c...}`
* `lst2(lst)`
* `lst2 = lst`
* `lst.size()`
* `lst.begin()`
* `lst.end()`

```cpp
void error_msg(initializer_list<string> il){
    for(auto beg = il.begin(); beg != il.end(); ++beg)
        cout << *beg << " ";
    cout << endl;
}

if (expected != actual)
    error_msg({"functionX", expected, actual});
else
    error_msg({"functionX", "okay"});
```