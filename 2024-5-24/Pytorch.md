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



### Model:

![image-20240524224931508](assets/image-20240524224931508.png)

> `xy.shape[0]` 数据行数 即`len`

> `__getitem__`返回元组；





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
