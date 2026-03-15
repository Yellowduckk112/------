import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils import data
from d2l import torch as d2l

d2l.use_svg_display()

def get_fashion_mnist_labels(labels): #@save
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in labels]

def show_images(images, num_rows, num_cols, titles=None, scale = 1.5): #@save
    figsize = (num_cols * scale, num_rows * scale)
    print(figsize)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])    

def get_dataloader_workers(): #@save
    """使用4个进程来读取数据。"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None): #@save
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

b = 18
train_iter, test_iter = load_data_fashion_mnist(b)
X, y = next(iter(train_iter))
print(X.shape)
show_images(X.reshape(b, 28, 28), 2, 9, get_fashion_mnist_labels(y))

batch_size = 256
data_iter, ignore = d2l.load_data_fashion_mnist(batch_size) 
timer = d2l.Timer()
for X, y in data_iter:
    continue
print(f'{timer.stop():.2f} sec')

plt.show()