import matplotlib

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display# 显示图片

# Data
transform = transforms.ToTensor()

mnist_train = torchvision.datasets.FashionMNIST(root="../data",train=True,transform=transform,download=False)

mnist_test = torchvision.datasets.FashionMNIST(root="../data",train=False,transform=transform,download=False)

print(len(mnist_train),len(mnist_test),mnist_train[0][0].shape)

# ------------将数字标签转换为对应的文本标签----------------
def get_fashion_label(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']

    return [text_labels[int(i)] for i in labels]
# --------------------------------------------------------
#imgs：这是一个包含图像的列表或张量。每个图像可以是 PyTorch 张量或Numpy组。
#num_rows 和 num_cols：指定图像显示的行数和列数
#scale：控制整个图像网格的缩放比例。

def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize = (num_cols*scale,num_rows*scale)
    _,axes = d2l.plt.subplots(num_rows,num_cols,figsize=figsize)#使用 plt.subplots 创建一个 num_rows 行、num_cols 列的子图网格
    axes = axes.flatten()# 展平为一维数组
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)

        #隐藏每个子图的 x 和 y 轴。
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        # 设置表标题
        if titles:
            ax.set_title(titles[i])
    d2l.plt.show()  # 确保显示图像
    return axes

# 展示图片
# x,y = next(iter(data.DataLoader(mnist_train,batch_size=18)))
# show_images(x.reshape(18,28,28),3,6,titles=get_fashion_label(y))

# ---------------------------------------------------------

batch_size = 256
train_loader = data.DataLoader(mnist_train,
                               batch_size,
                               num_workers=4,
                               shuffle=True)

timer = d2l.Timer()

if __name__=='__main__':
    for x,y in train_loader:
        continue
    print(f'{timer.stop():.2f}sec')