# Pytorch

## 概论

### 学习系统：

![image-20240520221757161](assets/image-20240520221757161.png)

>  Deep Learning 端到端



### 神经网络：





## Lecture1:线性模型

- DataSet;//数据集合
- Model;//模型
- Training;//训练
- inferring;//推理

### 问题：

![image-20240521165322667](assets/image-20240521165322667.png)

### 流程：

![image-20240521165615731](assets/image-20240521165615731.png)

### DataSet;//数据集合：

- 训练数据集合（包含输入和输出）；使模型有更强泛化能力；
  - 开发集合：类似测试集合，用于评估模型；
- 测试数据集合（只有输入）；



### Model;//模型：

> y = f(x);

- 线性模型：Linear Model：

![image-20240521170939071](assets/image-20240521170939071.png)

- 随机猜测（w = a random guess）；

- 求偏移程度；

​	![image-20240521171412308](assets/image-20240521171412308.png)

- 对样本评估`Evaluate`模型，求平均损失；

  ![image-20240521171552324](assets/image-20240521171552324.png)

- 平均平方误差MSE;

  ![image-20240521172110059](assets/image-20240521172110059.png)



Code

![image-20240521173550703](assets/image-20240521173550703.png)

Draw Graph

![image-20240521174836083](assets/image-20240521174836083.png)



## 梯度下降

### ![image-20240521231555609](assets/image-20240521231555609.png)

### 梯度：

![image-20240521231640632](assets/image-20240521231640632.png)

- 如果>0，函数上升；
- 如果<0，函数下降；

> 梯度负方向；



### 梯度更新：（贪心）

![image-20240521231839084](assets/image-20240521231839084.png)

> a学习率要尽量取小点0.01；



**非凸函数 全局最优**

![image-20240521232242603](assets/image-20240521232242603.png)



**鞍点** 

> 梯度为0;

![image-20240521232605275](assets/image-20240521232605275.png)



### 梯度计算：

![image-20240521232901782](assets/image-20240521232901782.png)



### 发散：

![image-20240522090513300](assets/image-20240522090513300.png)



### 随机梯度下降:

![image-20240522090755658](assets/image-20240522090755658.png)

- 单个数据的损失`(pre_y - y)**2`
- 对每个样本求梯度`2*x*(w*x - y)`

### Code；

![image-20240522090231480](assets/image-20240522090231480.png)





![image-20240522090949073](assets/image-20240522090949073.png)



## 反向传播 

![image-20240522212420013](assets/image-20240522212420013.png)

- 更新权重：![image-20240522212509925](assets/image-20240522212509925.png)



![image-20240522212528892](assets/image-20240522212528892.png)



### 反向传播：

#### 层：

![image-20240522213235840](assets/image-20240522213235840.png)

> X输入后x乘转置矩阵；
>
> 加上偏移量；



![image-20240522213526383](assets/image-20240522213526383.png)

> 对每一层加非线性函数；
>
> 必须分层计算；



### 链式求导：

![image-20240522213703519](assets/image-20240522213703519.png)

> 链式法则;

#### 过程：



![image-20240522214115853](assets/image-20240522214115853.png)

> 先前馈后反馈；

![image-20240522215506716](assets/image-20240522215506716.png)

> 为了求梯度：
> $$
> \dfrac{\partial loss}{\partial \omega}
> $$
> 转化为：
> $$
> \dfrac{\partial loss}{\partial r}
> \times
> \dfrac{\partial r}{\partial \widehat{y}}
> \times
> \dfrac{\partial \widehat{y}}{\partial \omega}
> $$
> 反向求梯度；



### Tensor:建立计算图

- Data;

- $$
  \dfrac{\partial loss}{\partial \omega}
  $$

  

#### 设定需要梯度计算：

<img src="assets/image-20240522221627839.png" alt="image-20240522221627839" style="zoom:67%;" />

> 默认Tensor不计算梯度；



![image-20240522222116814](assets/image-20240522222116814.png)

> `x`需要转为`Tensor`；
>
> 结果也需要梯度



![image-20240522222308586](assets/image-20240522222308586.png)

> .backward();求链上梯度；
>
> 取梯度的Data进行数据计算（标量）；//grad计算会生成计算图；
>
> .item()取值计算；
>
> .grad.data.zero()清零运算；//每次都要



### 总结：

![image-20240522223240259](assets/image-20240522223240259.png)





## PyTorch线性回归

> Pytorch工具使用；

<img src="assets/image-20240523212811828.png" alt="image-20240523212811828" style="zoom:50%;" />

### PyTorch:

#### 广播：

> 非矩阵数据和矩阵数据计算，自动广播为对应大小矩阵；



> ![image-20240523152552833](assets/image-20240523152552833.png)

<img src="assets/image-20240523152334242.png" alt="image-20240523152334242" style="zoom:50%;" />





### 数据集合：

> mini-batch;

```python
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
```

> 符合构造计算图的输入；Linear期待二维张量的输入

### 设计模型：

> 构造计算图；

![image-20240523152847339](assets/image-20240523152847339.png)

> 由输出y 输入x 确定w b矩阵类型；
>
> loss 为标量；



#### model类：

![image-20240523153339943](assets/image-20240523153339943.png)

> 继承Module(构建计算图) `super`继承父类；
>
> Linear 包含 权重w  b ；
>
> `forward()`为前馈函数 `Moduls`本身包含需要重构；



#### Linear:

 PyTorch的nn.Linear（）是用于设置网络中的全连接层的，需要注意在二维图像处理的任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]，不同于卷积层要求输入输出是四维张量。其用法与形参说明如下：

-   in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size，**每个样本的大小**。

-   out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。

  从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。


![image-20240523154351622](assets/image-20240523154351622.png)

> 默认`bias`为`True`

#### Override : forward()

![image-20240523155110905](assets/image-20240523155110905.png)

### 损失函数 

![image-20240523155356586](assets/image-20240523155356586.png)

> nn.MSELoss(继承自Moduls)；

#### MSELoss:

- `size_avarege` 是否求均值；
- `reduce` 是否降维度；



### 优化器：

<img src="assets/image-20240523161522083.png" alt="image-20240523161522083" style="zoom: 50%;" />

![image-20240523155956413](assets/image-20240523155956413.png)

#### `optim.SGD`

- `params` 确定`Moduls`成员中需要梯度更新的成员；
- `lr` 学习率



`reduction='sum'` 是 PyTorch 损失函数中的一个参数，用于指定如何对批次中的损失进行聚合。PyTorch 的损失函数（例如 `nn.MSELoss`）支持三种类型的聚合方式：`'none'`、`'mean'` 和 `'sum'`。

- `'none'`：不进行任何聚合，直接返回每个样本的损失。
- `'mean'`：对所有样本的损失求平均值。
- `'sum'`：对所有样本的损失求和。

### 训练周期：

![image-20240523160801408](assets/image-20240523160801408.png)

> `print`中`loss` 为标量；
>
> 梯度清0；
>
> `Backward`反向传播；



#### 打印:

<img src="assets/image-20240523161056975.png" alt="image-20240523161056975" style="zoom:50%;" />

#### 测试：

<img src="assets/image-20240523161103663.png" alt="image-20240523161103663" style="zoom:50%;" />





![image-20240523161554454](assets/image-20240523161554454.png)



## Logistic Regression分类

> 输出概率值；



### 官方数据集合：

#### MNIST:

![image-20240523214014250](assets/image-20240523214014250.png)

> `train`是否为训练集；
>
> `download`第一次是否从网络下载；



#### CIFAR-10：

![image-20240523214253492](assets/image-20240523214253492.png)

### 二分类：

![image-20240523214401094](assets/image-20240523214401094.png)

> 计算通过概率；
>
> `pass` or`fail`
> $$
> p(\widehat{y}=1)+p(\widehat{y}=0)=1
> $$

> 近似0.5处于不确定；

#### 实数映射概率：

<img src="assets/image-20240523214918240.png" alt="image-20240523214918240" style="zoom:80%;" />

> 将预测的实数值映射[0,1]的概率；

> 将Logistic函数中变量x换为为 ：
> $$
> \widehat{y} = w*x+b
> $$



#### sigmoid函数

![image-20240523215540991](assets/image-20240523215540991.png)

### 回归模型：

> Logistic函数；

![image-20240523221609564](assets/image-20240523221609564.png)

> 函数包：`torch.nn.functional`



### BCE Loss损失函数：

#### 函数：

![image-20240523220010178](assets/image-20240523220010178.png)

<img src="assets/image-20240523221534272.png" alt="image-20240523221534272" style="zoom:67%;" />

> 分布的差异;越接近y，loss越小；

> $$
> y=1时\quad loss=-ylog\widehat{y}
> $$

> $$
> y=0时\quad loss=-ylog(1-\widehat{y})
> $$



#### 代码：

![image-20240523221858421](assets/image-20240523221858421.png)

> `sive_average` 影响后期学习率；



### Code：

![image-20240523222243476](C:/Users/27836/Desktop/image-20240523222243476.png)



### 绘图：

![image-20240523222523859](assets/image-20240523222523859.png)

> `x = np.linspace(0, 10, 200)`创建一个从0到10的200个等间隔的点；

> `x_t = torch.Tensor(x).view((200, 1))`将 numpy 数组转换为 PyTorch 张量，并调整其形状为 (200, 1)；

> `y = y_t.data.numpy()`将 PyTorch 张量 y_t 转换为 numpy 数组 y；

> `plt.plot([0, 10], [0.5, 0.5], c='r')`绘制一条红色的线，表示在 x 从 0 到 10 范围内，y 的值为 0.5 的水平线;

> `plt.grid()`显示网格；





## 多维数据：



![image-20240524164357721](assets/image-20240524164357721.png)

> x为一个数据集合；
>
> y为一个数据集合；



### 模型：

#### 公式：

<img src="assets/image-20240524164957506.png" alt="image-20240524164957506" style="zoom:50%;" />

#### Logistic回归：

<img src="assets/image-20240524165248533.png" alt="image-20240524165248533" style="zoom:50%;" />



#### Mini_Batch:



![image-20240524165726070](assets/image-20240524165726070.png)



### Model：

![image-20240524165845548](assets/image-20240524165845548.png)



### 降维：

#### 一维：

![image-20240524170034185](assets/image-20240524170034185.png)

> 输入数据的行数为`Linear()`输入维度；



#### 高维：

![image-20240524170607615](assets/image-20240524170607615.png)

> 输入矩阵从8维将为6维；

> Logistic函数进行非线性变换；



增强泛化能力：

![image-20240524171224510](assets/image-20240524171224510.png)



![image-20240524171441252](assets/image-20240524171441252.png)



### 流程：

#### 数据读入：

![image-20240524172031960](assets/image-20240524172031960.png)

> `loadtxt`读取数据；
>
> delimiter = 分割符号；
>
> dtape 数据 类型：float32 浮点数；

> x切片：`[:,:-1]`第一列到最后一列，不包括最后一列；
>
> y切片: `[:,[-1]]` `[-1]`表示拿出来是矩阵；



#### 模型：

![image-20240524172830383](assets/image-20240524172830383.png)

> `nn.Sigmoid()`继承`Module`无参数；

> `forward`中使用唯一变量`x`防止出错；



#### 损失函数：

<img src="assets/image-20240524173312024.png" alt="image-20240524173312024" style="zoom:67%;" />

![image-20240524173321293](assets/image-20240524173321293.png)

#### 优化器：

<img src="assets/image-20240524173328522.png" alt="image-20240524173328522" style="zoom:67%;" />

![image-20240524173333857](assets/image-20240524173333857.png)



#### 训练：

![image-20240524173424510](assets/image-20240524173424510.png)

### 激活函数：

![image-20240524173512147](assets/image-20240524173512147.png)

![image-20240524173818322](assets/image-20240524173818322.png)

> `activate`修改为不同的激活函数；
>
> `forward`中最后`activate`改为`sigmoid`使数据保持在（0，1）之间不为0防止后续计算出错；



## Dataset & DataLoader

### Mine-Batch:

#### 训练循环：

<img src="assets/image-20240524222400408.png" alt="image-20240524222400408" style="zoom:67%;" />

> 外层，整个样本运行次数；
>
> 内层，单个Batch；；



![image-20240524222641632](assets/image-20240524222641632.png)

- `Epoch` 所有样本训练次数；

- `Batch-Size`单个样本大小；
- `Iteration` 表示`Batch`个数；



### `DataLoader`:

- `dataset`

- `batch-size`;
- `shuffle`是否打乱样本；

#### `Dataset`:

- 提供索引；
- `len`长度；

#### 流程图：

![image-20240524223148237](assets/image-20240524223148237.png)



### Code:

![image-20240524223328195](assets/image-20240524223328195.png)

> `Dataset` 是一个抽象类 只能被继承；

> `__getitem__`返回第N个样本；
>
> `__len___`返回长度；



> `DataLoader`：

- `dataset`数据集合；
- `batch-size` 单个集合大小；
- `shuffle`是否打乱样本顺序；
- `num workers =`样本加载线程数目；

#### 数据读取方式：

- 数据较小，全部读入；
- 数据较大，读入文件标签；



### Windons 报错;

![image-20240524224700653](assets/image-20240524224700653.png)

#### 解决:

![image-20240524224742658](assets/image-20240524224742658.png)

- 训练代码封装到 函数 或 条件语句中；



### Dataset:

![image-20240524224931508](assets/image-20240524224931508.png)

> `xy.shape[0]` 数据行数 即`len`

> `__getitem__`返回对应序元组；



#### `x[m,n]`

**x[m,n]是通过numpy库引用数组或矩阵中的某一段数据集的一种写法，**

- m代表第m维
- n代表m维中取第几段特征数据。

通常用法：

**x[:,n]或者x[n,:]**

**x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,** 

举例说明：

**x[:,0]**

[python] [view plain](https://blog.csdn.net/csj664103736/article/details/72828584#) [copy](https://blog.csdn.net/csj664103736/article/details/72828584#)

1. import numpy as np 
2.  
3. X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]) 
4. print X[:,0] 
5.   

输出结果是：

![img](https://img-blog.csdn.net/20170601100806232)

**x[:,1]**

[python] [view plain](https://blog.csdn.net/csj664103736/article/details/72828584#) [copy](https://blog.csdn.net/csj664103736/article/details/72828584#)

1. import numpy as np 
2.  
3. X = np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]) 
4. print X[:,1]  

输出结果是：

![img](https://img-blog.csdn.net/20170601100855592)

**x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据,** 

x[1,:]即取第一维中下标为1的元素的所有值，输出结果：

![img](https://img-blog.csdn.net/20170711120240934)

扩展用法

**x[:,m:n]，即取所有数据集的第m到n-1列数据**

例：输出X数组中所有行第1到2列数据

输出结果：

![img](https://img-blog.csdn.net/20170712103836599)

#### `np.loadtxt()`

> 读取文件；

```python
# 这里的skiprows是指跳过前1行, 如果设置skiprows=2, 就会跳过前两行,数据类型设置为整形.
a = np.loadtxt('./data/test.txt', skiprows=1, dtype=int)
print(a)
```



#### `torch.from_numpy()`

简单说一下，就是[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。

```python
Example:
>>> a = numpy.array([1, 2, 3])
>>> t = torch.from_numpy(a)
>>> t
tensor([ 1, 2, 3])
>>> t[0] = -1
>>> a
array([-1, 2, 3])
```

### Train_loader:

```python
train_loader = DataLoader(dataset=train_data, batch_size=5,
                        	shuffle=True,num_workers=3)
```

- Dataset;
- batch_size 大小；
- 是否打乱顺序；
- 读入线程；

### Train:

![image-20240524225349351](assets/image-20240524225349351.png)

> `data=(x,y)`;
>
> `data`中`x,y`已经自动转化为`Tanser`;



### 数据集合：

![image-20240524230232318](assets/image-20240524230232318.png)



#### MINST

![image-20240524230420065](assets/image-20240524230420065.png)

- `root` 文件源；
- `train` 是否为训练集；
- `transform`做变换转换张量；



### 竞赛：

![image-20240524231132207](assets/image-20240524231132207.png)





## 多分类：

> `net`最后一层不激活  输出`ouoputs`直接接`criterion(outputs,targets)`

![image-20240525142201098](assets/image-20240525142201098.png)

> 输出要求；
>
> - 输出概率大于0；
>
> - 概率和为1；



### Softmax:

![image-20240525142520520](assets/image-20240525142520520.png)

<img src="assets/image-20240525143056539.png" alt="image-20240525143056539" style="zoom: 50%;" />

> 输出[0,9]的可能性；

#### 计算公式：

![image-20240525142614064](assets/image-20240525142614064.png)

> 对线性层做指数运算，使大于0；
>
> 再求取权重；

> 输出[0,9]的可能性；总和为1；

### 损失函数：

![image-20240525143506012](assets/image-20240525143506012.png)

### Code:

<img src="assets/image-20240525143616573.png" alt="image-20240525143616573" style="zoom:67%;" />



#### CrossEntropyLoss:

![image-20240525143904382](assets/image-20240525143904382.png)

> 实际标签数据Y`.LongTenser()`要为长整型张量；
>
> 线性层后 **直接接交叉熵损失**；

##### CrossEntropyLoss输入

`nn.CrossEntropyLoss` 接受两个输入参数：`input` 和 `target`。

1. **input (logits)**：
   - 形状：`(N, C)` 或 `(N, C, d1, d2, ..., dk)`，其中 `N` 是批次大小，`C` 是类别数，`d1, d2, ..., dk` 是额外的维度（例如图像的高和宽）。
   - 描述：未经过 softmax 激活的原始分数（logits）。这些分数表示模型对每个类别的置信度。
2. **target (labels)**：
   - 形状：`(N)` 或 `(N, d1, d2, ..., dk)`，其中 `N` 是批次大小，`d1, d2, ..., dk` 是额外的维度。
   - 描述：真实的类别标签，标签值范围为 `[0, C-1]`。这些标签是整型的，表示每个样本的真实类别。

##### CrossEntropyLoss输出

`nn.CrossEntropyLoss` 的输出是一个标量，表示整个批次的平均损失值。



![image-20240525144251656](assets/image-20240525144251656.png)



![image-20240525144316885](assets/image-20240525144316885.png)

### MNIST:

#### 原理：



![image-20240526174351845](assets/image-20240526174351845.png)



> **将图片分为[1,28,28]的矩阵[0,255]的色块转为[0,1]的值映射每一个像素，作为元素填充矩阵**；





<img src="assets/image-20240525144621566.png" alt="image-20240525144621566" style="zoom:67%;" />

#### Data:

![image-20240525144720442](assets/image-20240525144720442.png)

> `batch_size = 64` 通常在大多数 GPU 上都能高效运行；
>
> `transform`中转化为图像张量；
>
> MNIST 图像的张量格式是 `[通道数, 高度, 宽度]`，具体为 `[1, 28, 28]`
>
> - 转为 `颜色通道` x`宽`x`高`;
> - 颜色区域转为[0,1];

#### `transform`:

```
# 定义数据预处理操作
data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),# 调整图像大小
        transforms.ToTensor(),#转化为张量
        # 归一化至 [0, 1] 范围内（假设图像为 RGB）
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

```

#### transforms.Compose：

```text
transforms.Compose([transforms.RandomResizedCrop(224),
 		    		transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

具体是对图像进行各种转换操作，并用函数compose将这些转换操作组合起来；

先读取一张图片：

```text
from PIL import Image
img = Image.open("./tulip.jpg")
```

**transforms.RandomResizedCrop(224)**将给定图像随机裁剪为不同的大小和宽高比，然后缩放所裁剪得到的图像为制定的大小；（即先随机采集，然后对裁剪得到的图像缩放为同一大小），默认scale=(0.08, 1.0)

```text
img = Image.open("./demo.jpg")
print("原图大小：",img.size)
data1 = transforms.RandomResizedCrop(224)(img)
print("随机裁剪后的大小:",data1.size)
data2 = transforms.RandomResizedCrop(224)(img)
data3 = transforms.RandomResizedCrop(224)(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2),plt.imshow(data1),plt.title("转换后的图1")
plt.subplot(2,2,3),plt.imshow(data2),plt.title("转换后的图2")
plt.subplot(2,2,4),plt.imshow(data3),plt.title("转换后的图3")
plt.show()
```

结果为：

```text
原图大小： (500, 721)
随机裁剪后的大小: (224, 224)
```

该操作的含义在于：即使只是该物体的一部分，我们也认为这是该类物体；

![img](https://pic4.zhimg.com/80/v2-597eca3bdabba6527253e201b5c39e6b_1440w.webp)

**transforms.RandomHorizontalFlip()** **以给定的概率随机水平旋转给定的PIL的图像，默认为0.5；**

```text
img = Image.open("./demo.jpg")
img1 = transforms.RandomHorizontalFlip()(img)
img2 = transforms.RandomHorizontalFlip()(img)
img3 = transforms.RandomHorizontalFlip()(img)

plt.subplot(2,2,1),plt.imshow(img),plt.title("原图")
plt.subplot(2,2,2), plt.imshow(img1), plt.title("变换后的图1")
plt.subplot(2,2,3), plt.imshow(img2), plt.title("变换后的图2")
plt.subplot(2,2,4), plt.imshow(img3), plt.title("变换后的图3")
plt.show()
```

![img](https://pic3.zhimg.com/80/v2-bda5b3c8830b6b667f87754e956b2d3e_1440w.webp)

**transforms.ToTensor() 将给定图像转为Tensor**

```text
img = Image.open("./demo.jpg")
img = transforms.ToTensor()(img)
print(img)
```

输出为：

```text
tensor([[[0.4549, 0.4549, 0.4471,  ..., 0.5216, 0.5294, 0.5294],
         [0.4510, 0.4510, 0.4431,  ..., 0.5216, 0.5255, 0.5255],
         [0.4471, 0.4431, 0.4392,  ..., 0.5176, 0.5255, 0.5216],
         ...,
         [0.5529, 0.5333, 0.5059,  ..., 0.7922, 0.7922, 0.7922],
         [0.5647, 0.5451, 0.5176,  ..., 0.7922, 0.7922, 0.7922],
         [0.5882, 0.5725, 0.5451,  ..., 0.7843, 0.7843, 0.7843]],

        [[0.4980, 0.4980, 0.4902,  ..., 0.5059, 0.5137, 0.5137],
         [0.4941, 0.4941, 0.4863,  ..., 0.5059, 0.5098, 0.5098],
         [0.4902, 0.4863, 0.4824,  ..., 0.5020, 0.5098, 0.5059],
         ...,
         [0.5059, 0.4863, 0.4588,  ..., 0.7373, 0.7373, 0.7373],
         [0.5176, 0.4980, 0.4706,  ..., 0.7373, 0.7373, 0.7373],
         [0.5412, 0.5255, 0.4980,  ..., 0.7451, 0.7451, 0.7451]],

        [[0.5137, 0.5137, 0.5059,  ..., 0.5020, 0.5098, 0.5098],
         [0.5098, 0.5098, 0.5020,  ..., 0.5020, 0.5059, 0.5059],
         [0.5059, 0.5020, 0.4980,  ..., 0.4980, 0.5059, 0.5020],
         ...,
         [0.4431, 0.4235, 0.3961,  ..., 0.7373, 0.7373, 0.7373],
         [0.4549, 0.4353, 0.4078,  ..., 0.7373, 0.7373, 0.7373],
         [0.4941, 0.4784, 0.4510,  ..., 0.7490, 0.7490, 0.7490]]])
```

**transforms.Normalize(） 归一化处理**

```text
img = Image.open("./demo.jpg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img)
print(img)
```

输出：

```text
tensor([[[-0.0902, -0.0902, -0.1059,  ...,  0.0431,  0.0588,  0.0588],
         [-0.0980, -0.0980, -0.1137,  ...,  0.0431,  0.0510,  0.0510],
         [-0.1059, -0.1137, -0.1216,  ...,  0.0353,  0.0510,  0.0431],
         ...,
         [ 0.1059,  0.0667,  0.0118,  ...,  0.5843,  0.5843,  0.5843],
         [ 0.1294,  0.0902,  0.0353,  ...,  0.5843,  0.5843,  0.5843],
         [ 0.1765,  0.1451,  0.0902,  ...,  0.5686,  0.5686,  0.5686]],

        [[-0.0039, -0.0039, -0.0196,  ...,  0.0118,  0.0275,  0.0275],
         [-0.0118, -0.0118, -0.0275,  ...,  0.0118,  0.0196,  0.0196],
         [-0.0196, -0.0275, -0.0353,  ...,  0.0039,  0.0196,  0.0118],
         ...,
         [ 0.0118, -0.0275, -0.0824,  ...,  0.4745,  0.4745,  0.4745],
         [ 0.0353, -0.0039, -0.0588,  ...,  0.4745,  0.4745,  0.4745],
         [ 0.0824,  0.0510, -0.0039,  ...,  0.4902,  0.4902,  0.4902]],

        [[ 0.0275,  0.0275,  0.0118,  ...,  0.0039,  0.0196,  0.0196],
         [ 0.0196,  0.0196,  0.0039,  ...,  0.0039,  0.0118,  0.0118],
         [ 0.0118,  0.0039, -0.0039,  ..., -0.0039,  0.0118,  0.0039],
         ...,
         [-0.1137, -0.1529, -0.2078,  ...,  0.4745,  0.4745,  0.4745],
         [-0.0902, -0.1294, -0.1843,  ...,  0.4745,  0.4745,  0.4745],
         [-0.0118, -0.0431, -0.0980,  ...,  0.4980,  0.4980,  0.4980]]])
```

##### 均值与标准差：

- 0.1307；
- 0.3081；

<img src="assets/image-20240525145605611.png" alt="image-20240525145605611" style="zoom:67%;" />

#### Model

![image-20240525150055239](assets/image-20240525150055239.png)

> `view(-1,784)`,`-1`自动计算`batch_size`；
>
> 图像 四维`[batch_size,1,28,28]` 转为 张量 二维`[batch_size,28*28=784]`；



#### 损失 优化器：

![image-20240525150828527](assets/image-20240525150828527.png)

> `momentum`冲量优化 优化器；



#### Train:

![image-20240525151018635](assets/image-20240525151018635.png)

> 优化器清零；



#### Text

![image-20240525151225736](assets/image-20240525151225736.png)

- `no_grad()`不计算梯度；

- `max` `dim=1`沿着第一个维度的方向找并返回 **最大值** 和 **最大值的下标**；
- `label.size(0)` 返回样本总个数；

> **数字1**其实可以写为`dim=1`，这里简写为1，python也可以自动识别，`dim=1`表示输出所在行的最大值，若改写成`dim=0`则输出所在列的最大值



```
correct += (predicted == labels).sum().item()
```

这里面`(predicted == labels)`是布尔型，为什么可以接sum()呢？

我做了个测试，如果这里的predicted和labels是列表形式就会报错，如果是numpy的数组格式，会返回一个值，如果是tensor形式，就会返回一个张量。

举个例子：

```python
import torch

a = torch.tensor([1,2,3])
b = torch.tensor([1,3,2])

print((a == b).sum())
```

上述代码的输出结果：

```python
tensor(1)
```

如果将a和b改成numpy下的数组格式：

```python
import numpy as np

a = np.array([1,2,3])
b = np.array([1,3,2])

print((a == b).sum())
```

上述代码的输出结果：

```python
1
```

如果将a和b改成列表：

```python
a = [1,2,3]
b = [1,3,2]

print((a == b).sum())
```

上述代码的输出结果：

```python
Traceback (most recent call last):
  File "路径", line 4, in <module>
    print((a == b).sum())
AttributeError: 'bool' object has no attribute 'sum'

Process finished with exit code 1
```

Added：

.item()用于取出tensor中的值。

![image-20240525152525000](assets/image-20240525152525000.png)

![image-20240525152952921](assets/image-20240525152952921.png)





## 卷积神经网络：

### 全连接神经网络：

![image-20240525153207508](assets/image-20240525153207508.png)

### 流程：

![image-20240525154041206](assets/image-20240525154041206.png)

![image-20240526222718907](assets/image-20240526222718907.png)

### 栅格图像结构

![image-20240525155712229](assets/image-20240525155712229.png)

> 扫描采样；

### 卷积与下采样：

#### 单通道卷积：

![image-20240525160019021](assets/image-20240525160019021.png)

![image-20240525160139845](assets/image-20240525160139845.png)

#### 多通道卷积：

![image-20240525160251768](assets/image-20240525160251768.png)

> 块 与 卷积核  相乘求和



![image-20240525160623861](assets/image-20240525160623861.png)



![image-20240525161200295](assets/image-20240525161200295.png)

> 卷积核通道 和 输入通道 一样多



![image-20240525161401451](assets/image-20240525161401451.png)



### Code:

![image-20240525161916846](assets/image-20240525161916846.png)

> `input.shape`:`batch_size`,`in_channel`,`W`,`H`;
>
> `output.shape`:`batch_size`,`in_channel`,`W`,`H`;
>
> - 输出图像大小：`input_size`-`kernel_size`+1
>
> `conv_layer.shape`:`output_channels`,`input_channel`,`k_W`,`K_H`;

#### `Conv2d`（卷积层）必须参数:

- `in_channals`输入通道数量；
- `out_channals`输出通道数量；

- `kerenal_size`卷积层大小；

##### 卷积张量：

```python
torch.Tensor(tensor1).view(Batch_size,channels,Width,Height)#卷积张量转化
```

##### padding:填充

> 使 输入 和 输出大小 一致

![image-20240525162218314](assets/image-20240525162218314.png)

![image-20240525162419642](assets/image-20240525162419642.png)

> `torch.Tensor().view(B,C,W,H)`转置矩阵为对应`Conv2d`输入张量`(B,C,W,H)`
>
> `torch.Tensor().view(B,C,W,H)`转置矩阵为对应卷积核张量`(B,C,W,H)`
>
> `kernel` 做 `w`  ,
>
> 不使用偏置量`bias=False`;



##### stride:跳的步长

> 缩小宽高；



![image-20240525162805377](assets/image-20240525162805377.png)

![image-20240525162824530](assets/image-20240525162824530.png)

> `stride=2`每次采用间隔为2；

#### Max Pooling（池化层）:下采样

> **通道数目不变**；
>
> **图像大小改变；**



![image-20240525163012691](assets/image-20240525163012691.png)

![image-20240525163032839](assets/image-20240525163032839.png)

> 给出卷积核大小；

### 卷积神经网络：

![image-20240525163940505](assets/image-20240525163940505.png)



#### 改GPU:

- 检测是否有显卡；
- 模型迁移

- 数据迁移；

> 模型

![image-20240525164224977](assets/image-20240525164224977.png)

> 训练

![image-20240525164309447](assets/image-20240525164309447.png)

> 测试

![image-20240525164437829](assets/image-20240525164437829.png)

## 卷积神经网络（高级）：

### GooleNet：

![image-20240525165205646](assets/image-20240525165205646.png)

#### Inception Module:

![image-20240525165352998](assets/image-20240525165352998.png)



##### `1x1 Conv`:

> 使修改卷积通道；

- 取决于输入张量的；
- 一个通道配置一个 kenal



![image-20240525165909084](assets/image-20240525165909084.png)



![image-20240525170015143](assets/image-20240525170015143.png)



##### 对比：

![image-20240525171231487](assets/image-20240525171231487.png)



#### Code:

![image-20240525172157613](assets/image-20240525172157613.png)



![image-20240525172218530](assets/image-20240525172218530.png)



>  # kernel_size -2*padding+1为图像边长变化；调整二者可以变化后边长不变；

> `cat`拼接：沿着`dim=1`  即`channel`拼接；

![image-20240525172325661](assets/image-20240525172325661.png)





![image-20240525172946304](assets/image-20240525172946304.png)

> `Linear`输入层数  可由卷积层后;





### Residual net:



![image-20240525173727991](assets/image-20240525173727991.png)

> 使用跳链接 防止梯度消失；

![image-20240525174327676](assets/image-20240525174327676.png)



![image-20240527212548778](assets/image-20240527212548778.png)



## RNN循环神经网络

> 处理序列数据；
>
> 权值共享；



### RNN Cell:



![image-20240525221448886](assets/image-20240525221448886.png)

> 将引层`h`结果加入下一次`RNN Cell`;`hidden`**既做输出有做输入**；
>
> `h0`可以取 `h1`一样维度的**[0,0,0,0]**全0向量；



#### 计算过程：

![image-20240525222306367](assets/image-20240525222306367.png)

<img src="assets/image-20240525222528562.png" alt="image-20240525222528562" style="zoom:50%;" />



#### code:

> 确定输入`input_size`输出`hidden_size`两个参数维度；





![image-20240525222916953](assets/image-20240525222916953.png)

<img src="assets/image-20240525221258618.png" alt="image-20240525221258618" style="zoom:50%;" />

![image-20240525223312462](assets/image-20240525223312462.png)

> `hidden_size`代替了原`output_`
>
> `dataset.shape`:序列长度seqlen+BatchSize+inputsize； **`seqlen`**序列维度放在最前面；



![image-20240525223354430](assets/image-20240525223354430.png)





### RNN_Code:

![image-20240525223629347](assets/image-20240525223629347.png)



> `out` 接受每次的`h`;
>
> `hidden`输出指向**当前层**最后的`hn`;

![image-20240527225727973](assets/image-20240527225727973.png)

> `input`:序列长度+batch+输入样本长度；
>
> 输入的`hidden`:RNN层数+batch+hidden_size;



> `output`接收当前层所有`hidden`输出:序列长度+batch+hidden_size;
>
> 输出当前层的`hidden_N`:RNN层数+batch+hidden_size;

<img src="assets/image-20240527231439046.png" alt="image-20240527231439046" style="zoom: 50%;" />

### numLayers:



![image-20240525224101739](assets/image-20240525224101739.png)



### Code：

![image-20240525224206616](assets/image-20240525224206616.png)



### Example:

#### 字符数据转向量：

![image-20240525224704066](assets/image-20240525224704066.png)

> `inputsize` = 4;



![image-20240525225046969](assets/image-20240525225046969.png)

> 接交叉熵损失； 
>
> 交叉熵标准`target`必须是`LongTensor`
>
> 在交叉熵损失函数中，输入必须是二维的张量 `[N, C]`，其中 `N` 是批量大小，`C` 是类别数，而目标是一个一维的张量 `[N]`，其中 `N` 是批量大小。



### Embedding：

#### 嵌入层结构

![image-20240528173104357](assets/image-20240528173104357.png)

> `inputsize`=4 扩展`embedding`长度转为矩阵；

> 转置后取出特殊列；

![image-20240528173658497](assets/image-20240528173658497.png)

> 输入为长整型`LonTensor`;
>
> 输出为[Seq,W]型向量；

### `nn.Embedding()`:

![image-20240528173949675](assets/image-20240528173949675.png)

> `num_embedding`独热向量维度`input_size`；
>
> `embedding_dim`嵌入层宽度；

> 输入必须为`LongTenser`；

> `embedding`层会代替`input`层，相当于添加一个基础维度`embedding`；



是的，使用 `batch_first=True` 相当于避免了手动进行矩阵转置

### `nn.Linear`

![image-20240528212409071](assets/image-20240528212409071.png)

> `Linear`支持输入`[N,in_features]`二维以上张量；
>
> - `N`：批量大小，指一批数据中的样本数量。
> - `*`：表示可以有任意数量的附加维度，这意味着输入数据可以是高维度的，例如二维、三维等。
> - `in_features`：每个输入样本的特征数，即输入向量的维度。
>
> 二维以上张量形状为`[N(外侧维度),*(中间的其他维度)...,in_features(内层维度)]`;`[5,1,32,32]`

### `CrossEntropyLoss`

![image-20240528214047183](assets/image-20240528214047183.png)

> 同理`CrossEntropyLoss`也支持多维输入；
>
> - 必须保证`Target`比`Input`小一个维度；保证映射关系；



[LSTM模型](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#lstm)

```python
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
```

> **需要hidden元组的输入；**

1. **Hidden State (h_t)**：
   - 隐藏状态（hidden state）是与传统 RNN 中的隐藏状态类似的部分。它代表了 LSTM 在当前时间步的输出。
   - Hidden state 主要用于决定当前时间步的输出。
2. **Cell State (c_t)**：
   - 细胞状态（cell state）是 LSTM 独有的一个部分。它用于携带长期的信息，通过时间步传播。
   - Cell state 可以看作是一个贯穿整个序列的传送带，信息可以选择性地添加或删除，从而保持长期的记忆。



## RNN_H

### 流程：

![image-20240528225408078](assets/image-20240528225408078.png)

![image-20240528225454150](assets/image-20240528225454150.png)

### Code_Train:

![image-20240528225948147](assets/image-20240528225948147.png)
