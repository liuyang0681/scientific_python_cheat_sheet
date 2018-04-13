Python科学计算备忘单
============================
<!-- markdown-toc start - Don't forget to edit this section according to your modifications -->
**目录**

* [Python科学计算备忘单](#scientific-python-cheatsheet)
   
   * [纯python](#纯python)
       
       * [类型](#类型)
        * [列表](#列表)
        * [字典](#字典)
        * [集合](#集合)
        * [字符串](#字符串)
        * [操作符](#操作符)
        * [控制流](#控制流)
        * [函数、类、生成器和修饰器](#函数-类-生成器和修饰器)
    * [IPython](#ipython)
        * [控制台](#控制台)
        * [调试器](#调试器)
        * [命令行](#命令行)
    * [NumPy](#numpy-import-numpy-as-np)
        * [数组初始化](#数组初始化)
        * [索引](#索引)
        * [数组属性和操作](#数组属性和操作)
        * [布尔数组](#布尔数组)
        * [元素对操作与数学函数](#元素对操作与数学函数)
        * [内/外积](#内-外积)
        * [线性代数/矩阵数学](#线性代数-矩阵数学)
        * [读/写文件](#读-写文件)
        * [插值、积分与优化](#插值-积分与优化)
        * [FFT](#fft)
        * [舍入](#舍入)
        * [随机变量](#随机变量)
    * [Matplotlib](#matplotlib-import-matplotlib.pyplot-as-plt)
        * [图形与轴](#图形与轴)
        * [图像与轴属性](#图像与轴属性)
        * [绘制常规图](#绘制常规图)
    * [Scipy](#scipy-import-scipy-as-sci)
        * [插值](#插值)
        * [线性代数](#线性代数)
        * [积分](#积分)
    * [Pandas](#pandas-import-pandas-as-pd)
        * [数据结构](#数据结构)
        * [DataFrame](#dataframe)
        
<!-- markdown-toc end -->

## 纯python

### 类型
```python
a = 2           # 整数
b = 5.0         # 浮点数
c = 8.3e5       # 指数
d = 1.5 + 0.5j  # 浮点数
e = 4 > 5       # 布尔值
f = 'word'      # 字符串
```

### 列表

```python
a = ['red', 'blue', 'green']       # 手动初始化
b = list(range(5))                 # 通过迭代初始化
c = [nu**2 for nu in b]            # 列表推导式
d = [nu**2 for nu in b if nu < 3]  # 条件列表推导式
e = c[0]                           # 访问元素
f = c[1:2]                         # 切片
g = c[-1]                          # 访问最后一个元素
h = ['re', 'bl'] + ['gr']          # 列表粘连
i = ['re'] * 5                     # 重复列表
['re', 'bl'].index('re')           # 返回're'索引
a.append('yellow')                 # 添加新元素到列表尾部
a.extend(b)                        # 添加列表b的元素到a的尾部
a.insert(1, 'yellow')              # 插入元素到特定位置
're' in ['re', 'bl']               # 如果're'在列表中，返回true
'fi' not in ['re', 'bl']           # 如果'fi'不在列表中，返回true
sorted([3, 2, 1])                  # 返回排好序的列表
a.pop(2)                           # 移除和返回索引处的元素（默认最后一个）
```

### 字典

```python
a = {'red': 'rouge', 'blue': 'bleu'}         # 字典
b = a['red']                                 # 翻译（索引）条目
'red' in a                                   # 如果字典包含'red'键，返回true
c = [value for key, value in a.items()]      # 字典条目循环
d = a.get('yellow', 'no translation found')  # 返回默认值
a.setdefault('extra', []).append('cyan')     # 使用默认值初始化键
a.update({'green': 'vert', 'brown': 'brun'}) # 通过其他数据更新字典
a.keys()                                     # 获取键的列表
a.values()                                   # 获取值的列表
a.items()                                    # 获取键值对列表
del a['red']                                 # 删除键和相关联的值
a.pop('blue')                                # 移除指定的键并返回对应的值
```


### 集合

```python
a = {1, 2, 3}                                # 手动初始化
b = set(range(5))                            # 通过迭代初始化
a.add(13)                                    # 添加新元素到集合
a.discard(13)                                # 从集合中丢弃元素
a.update([21, 22, 23])                       # 从迭代变量中更新集合
a.pop()                                      # 移除和返回任意的集合元素
2 in {1, 2, 3}                               # 如果2在集合中，返回true
5 not in {1, 2, 3}                           # 如果5不在集合中，返回true
a.issubset(b)                                # 检测是否a中每一个元素都在b里
a <= b                                       # issubset的操作符形式
a.issuperset(b)                              # 检测是否b中的每个元素都在a里
a >= b                                       # issuperset的操作符形式
a.intersection(b)                            # 返回两个集合的交集
a.difference(b)                              # 返回两个或多个集合的差集
a - b                                        # difference的操作符形式
a.symmetric_difference(b)                    # 返回对称差集
a.union(b)                                   # 返回并集
c = frozenset()                              # 不可变集合
```

### 字符串

```python
a = 'red'                      # 赋值
char = a[2]                    # 获取单个字符
'red ' + 'blue'                # 字符联接
'1, 2, three'.split(',')       # 将字符分割为列表
'.'.join(['1', '2', 'three'])  # 将列表联接为字符
```

### 操作符

```python
a = 2             # 赋值
a += 1 (*=, /=)   # 改变和赋值
3 + 2             # 加法
3 / 2             # 整数(python2) 或者 浮点数(python3)除法
3 // 2            # 整除
3 * 2             # 乘法
3 ** 2            # 指数
3 % 2             # 求余
abs(a)            # 绝对值
1 == 1            # 相等
2 > 1             # 大于
2 < 1             # 小于
1 != 2            # 不等
1 != 2 and 2 < 3  # 逻辑与
1 != 2 or 2 < 3   # 逻辑或
not 1 == 2        # 逻辑非
'a' in b          # 检测是否'a'在b中
a is b            # 检测是否对象映射到相同的内存(id)
```

### 控制流

```python
# if/elif/else
a, b = 1, 2
if a + b == 3:
    print('True')
elif a + b == 1:
    print('False')
else:
    print('?')

# for
a = ['red', 'blue', 'green']
for color in a:
    print(color)

# while
number = 1
while number < 10:
    print(number)
    number += 1

# break
number = 1
while True:
    print(number)
    number += 1
    if number > 10:
        break

# continue
for i in range(20):
    if i % 2 == 0:
        continue
    print(i)
```

### 函数、类、生成器和修饰器

```python
# 函数将代码语句分类并返回一个派生值
def myfunc(a1, a2):
    return a1 + a2

x = myfunc(a1, a2)

# 类将属性（数据）和关联的方法（函数）进行分类
class Point(object):
    def __init__(self, x):
        self.x = x
    def __call__(self):
        print(self.x)

x = Point(3)

# 生成器不用一次性创建所有值来进行迭代
def firstn(n):
    num = 0
    while num < n:
        yield num
        num += 1

x = [i for i in firstn(10)]

# 修饰器可以用来修饰函数的行为
class myDecorator(object):
    def __init__(self, f):
        self.f = f
    def __call__(self):
        print("call")
        self.f()

@myDecorator
def my_funct():
    print('func')

my_funct()
```

## IPython

### 控制台
```python
<object>?                   # 关于变量的信息
<object>.<TAB>              # tab补全

# 运行脚本 / profile / debug
%run myscript.py

%timeit range(1000)         # 测量语句运行时间
%run -t  myscript.py        # 测量脚本执行时间

%prun <statement>           # 使用profiler执行语句
%prun -s <key> <statement>  # 通过键排序，例如 "cumulative" or "calls"
%run -p  myfile.py          # profile脚本

%run -d myscript.py         # 以调试模式运行脚本
%debug                      # 遇到意外后跳转到调试器
%pdb                        # 遇到意外自动运行调试器

# 检查历史
%history
%history ~1/1-5  # 最后线程1-5行

# 运行shell命令
!make  # 使用"!"前缀

# 清除命名空间
%reset

# 运行剪贴板代码
%paste
```

### 调试器

```python
n               # 执行下一行
b 42            # 在主文件42行设定断点
b myfile.py:42  # 在'myfile.py'第42行设定断点
c               # 继续执行
l               # 显示代码中当前位置
p data          # 打印'data'变量
pp data         # 以美观的方式打印'data'变量
s               # 步入子程序
a               # 打印函数收到的参数
pp locals()     # 显示所有的局部变量
pp globals()    # 显示所有的全局变量
```

### 命令行

```bash
ipython --pdb -- myscript.py argument1 --option1  # 遇到意外后进入调试
ipython -i -- myscript.py argument1 --option1     # 完成后进入控制台
```

## NumPy (`import numpy as np`)

### 数组初始化

```python
np.array([2, 3, 4])             # 直接初始化
np.empty(20, dtype=np.float32)  # 大小为20的单精度数组
np.zeros(200)                   # 初始化200个0
np.ones((3,3), dtype=np.int32)  # 3 x 3的全1整数矩阵
np.eye(200)                     # 对角矩阵
np.zeros_like(a)                # 与a大小一样的全0矩阵
np.linspace(0., 10., 100)       # 0到10,100个等分点
np.arange(0, 100, 2)            # 步长为2，从0到<100
np.logspace(-5, 2, 100)         # 从1e-5 -> 1e2的100个对数间隔值
np.copy(a)                      # 拷贝数组到新的内存
```

### 索引

```python
a = np.arange(100)          # 用0 - 99初始化
a[:3] = 0                   # 设置第一个到第3个为0
a[2:5] = 1                  # 设置索引2-4为0
a[:-3] = 2                  # 设置除了最后三个的其他所有值为2
a[start:stop:step]          # 索引/切片的通用形式
a[None, :]                  # 转换为列向量
a[[1, 1, 3, 8]]             # 使用索引值返回数组
a = a.reshape(10, 10)       # 转换为10 x 10 矩阵
a.T                         # 返回矩阵的倒置
b = np.transpose(a, (1, 0)) # 调换矩阵到新的轴序
a[a < 2]                    # 返回元素对（向量）满足条件的值
```

### 数组属性和操作

```python
a.shape                # 一个包含每个轴长度的元组
len(a)                 # 0轴的长度
a.ndim                 # 维度(axes)数目
a.sort(axis=1)         # 按轴对数组排序
a.flatten()            # 塌缩数组到1维
a.conj()               # 返回共轭复数
a.astype(np.int16)     # 投射为整数
a.tolist()             # 转换（可能多维矩阵）为列表
np.argmax(a, axis=1)   # 返回给定轴次最大值的索引
np.cumsum(a)           # 返回累积和
np.any(a)              # 如果任意值为True，返回True
np.all(a)              # 如果所有值为True，返回True
np.argsort(a, axis=1)  # 返回按轴排序的索引数组
np.where(cond)         # 返回cond（条件）为True处的索引
np.where(cond, x, y)   # 返回满足条件的元素
```

### 布尔数组

```python
a < 2                         # 返回布尔值数组
(a < 2) & (b > 10)            # 元素对逻辑与
(a < 2) | (b > 10)            # 元素对逻辑或
~a                            # 逻辑矩阵的反（非）
```

### 元素对操作与数学函数

```python
a * 5              # 用标度乘
a + 5              # 用标度加
a + b              # 与数组b相加
a / b              # 与数组b相除 (如果除以0，返回np.NaN)
np.exp(a)          # 指数 (复数与实数)
np.power(a, b)     # a的b次幂
np.sin(a)          # sine函数
np.cos(a)          # cosine函数
np.arctan2(a, b)   # arctan(a/b)函数
np.arcsin(a)       # arcsin函数
np.radians(a)      # 度到弧度
np.degrees(a)      # 弧度到度
np.var(a)          # 数组的方差
np.std(a, axis=1)  # 数组的标准差
```

### 内 外积

```python
np.dot(a, b)                  # 内积: a_mi b_in
np.einsum('ij,kj->ik', a, b)  # 爱因斯坦求和约定
np.sum(a, axis=1)             # 轴1求和
np.abs(a)                     # 返回绝对值
a[None, :] + b[:, None]       # 外部和
a[None, :] * b[:, None]       # 外积
np.outer(a, b)                # 外积
np.sum(a * a.T)               # 矩阵标准化
```


### 线性代数 矩阵数学

```python
evals, evecs = np.linalg.eig(a)      # 寻找特征值与特征向量
evals, evecs = np.linalg.eigh(a)     # 厄尔米特矩阵np.linalg.eig
```


### 读 写文件

```python

np.loadtxt(fname/fobject, skiprows=2, delimiter=',')   # 读ascii数据文件
np.savetxt(fname/fobject, array, fmt='%.5f')           # 写ascii数据文件
np.fromfile(fname/fobject, dtype=np.float32, count=5)  # 读二进制数据文件
np.tofile(fname/fobject)                               # 写二进制数据文件
np.save(fname/fobject, array)                          # 保存为numpy 二进制文件(.npy)
np.load(fname/fobject, mmap_mode='c')                  # 导入.npy文件
```

### 插值、积分与优化

```python
np.trapz(a, x=x, axis=1)  # 沿轴1积分
np.interp(x, xp, yp)      # 在x点的插值函数xp, yp
np.linalg.lstsq(a, b)     # 用最小二乘法求解a x = b
```

### fft

```python
np.fft.fft(a)                # a的复数傅里叶变换
f = np.fft.fftfreq(len(a))   # fft频率
np.fft.fftshift(f)           # 将频率0移到中间
np.fft.rfft(a)               # a的实数傅里叶变换
np.fft.rfftfreq(len(a))      # 实数傅里叶变换频率
```

### 舍入

```python
np.ceil(a)   # 向上取整
np.floor(a)  # 向下取整
np.round(a)  # 临近取整
```

### 随机变量

```python
from np.random import normal, seed, rand, uniform, randint
normal(loc=0, scale=2, size=100)  # 100个正态分布数据点
seed(23032)                       # 设定种子数
rand(200)                         # [0, 1)区间200个随机数
uniform(1, 30, 200)               # [1, 30) 200个随机数
randint(1, 16, 300)               # [1, 16) 200个随机整数
```

## Matplotlib (`import matplotlib.pyplot as plt`)

### 图形与轴

```python
fig = plt.figure(figsize=(5, 2))  # 初始化图
fig.savefig('out.png')            # 保存png图像
fig, axes = plt.subplots(5, 2, figsize=(5, 5)) # 绘制子图
ax = fig.add_subplot(3, 2, 2)     # 添加子图到
ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)  # 多个轴
ax = fig.add_axes([left, bottom, width, height])  # 添加自定义轴
```

### 图像与轴属性

```python
fig.suptitle('title')            # 大的图标题
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace=0.2,
                    hspace=0.5)  # 调整子图位置
fig.tight_layout(pad=0.1, h_pad=0.5, w_pad=0.5,
                 rect=None)      # 调整子图
ax.set_xlabel('xbla')            # 设置 xlabel
ax.set_ylabel('ybla')            # 设置 ylabel
ax.set_xlim(1, 2)                # 设置 x limits
ax.set_ylim(3, 4)                # 设置 y limits
ax.set_title('blabla')           # 设置轴标题
ax.set(xlabel='bla')             # 一次性设置多个参数
ax.legend(loc='upper center')    # 激活图例
ax.grid(True, which='both')      # 激活网格
bbox = ax.get_position()         # 返回轴边界框
bbox.x0 + bbox.width             # 边界框参数
```

### 绘制常规图

```python
ax.plot(x,y, '-o', c='red', lw=2, label='bla')  # 线图
ax.scatter(x,y, s=20, c=color)                  # 点图
ax.pcolormesh(xx, yy, zz, shading='gouraud')    # fast colormesh
ax.colormesh(xx, yy, zz, norm=norm)             # slower colormesh
ax.contour(xx, yy, zz, cmap='jet')              # 彩线
ax.contourf(xx, yy, zz, vmin=2, vmax=4)         # 颜色填充
n, bins, patch = ax.hist(x, 50)                 # 直方图
ax.imshow(matrix, origin='lower',
          extent=(x1, x2, y1, y2))              # 展示图形
ax.specgram(y, FS=0.1, noverlap=128,
            scale='linear')                     # 绘制频谱图
ax.text(x, y, string, fontsize=12, color='m')   # 添加文字
```

## Scipy (`import scipy as sci`)

### 插值

```python
# 在索引位置插入数据：
from scipy.ndimage import map_coordinates
pts_new = map_coordinates(data, float_indices, order=3)

# 简单1维插值
from scipy.interpolate import interp1d
interpolator = interp1d(x, y, axis=2, fill_value=0., bounds_error=False)
y_new = interpolator(x_new)
```

### 积分

```python
from scipy.integrate import quad     # python中的定积分
value = quad(func, low_lim, up_lim)  # 函数/方法
```

### 线性代数

```python
from scipy import linalg
evals, evecs = linalg.eig(a)      # 寻找特征值和特征向量
evals, evecs = linalg.eigh(a)     # hermitian的求解函数
b = linalg.expm(a)                # 矩阵指数
c = linalg.logm(a)                # 矩阵对数
```


## Pandas (`import pandas as pd`)

### 数据结构

```python
s = pd.Series(np.random.rand(1000), index=range(1000))  # 序列
index = pd.date_range("13/06/2016", periods=1000)       # 时间索引
df = pd.DataFrame(np.zeros((1000, 3)), index=index,
                    columns=["A", "B", "C"])            # DataFrame（数据框）
```

### DataFrame

```python
df = pd.read_csv("filename.csv")   # 读和导入CSV文件到一个DataFrame
raw = df.values                    # 获取DataFrame对象原始数据
cols = df.columns                  # 获取列名的列表
df.dtype                           # 获取所有列的数据类型
df.head(5)                         # 获取头5行
df.describe()                      # 获取基本统计信息
df.index                           # 获取列范围索引

# 列切片
# (.loc[] 和 .ix[]都内含所选择的值)
df.col_name                         # 通过列名选择列值作为序列
df[['col_name']]                    # 通过列名选择列值作为DataFrame
df.loc[:, 'col_name']               # 通过列名选择列值作为序列
df.loc[:, ['col_name']]             # 通过列名选择列值作为DataFrame
df.iloc[:, 0]                       # 通过列索引选择
df.iloc[:, [0]]                     # 通过列索引选择，但不作为DataFrame 
df.ix[:, 'col_name']                # 用列名的混合方法
df.ix[:, 0]                         # 用列索引的混合方法

# 行切片
print(df[:2])                      # 打印dataframe前两行
df.iloc[0:2, :]                    # 选择dataframe的头2行
df.loc[0:2,'col_name']             # 选择dataframe的头3行
df.loc[0:2, ['col_name1', 'col_name3', 'col_name6']]    # 选择dataframe的头3行与根据列名选择3列
df.loc[0:2,0:2]                   # select fisrt 3 rows and first 3 columns
# 同样，.loc[] 和 .ix[]都内含所选择的值

# Dicin
df[ df.col_name < 7 ]                            # 选择符合col_name < 7的所有行
df[ (df.col_name1 < 7) & (df.col_name2 == 0) ]       # 用按位操作符结合多个逻辑条件
                                                     # 标准的Python布尔操作符不能在这里使用(and, or) cannot be used here. 
                                                     # 确保将每个条件封装在括号中以使其工作。
df[df.recency < 7] = -100                        # 切片赋值
```
