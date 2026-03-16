import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils import data
from d2l import torch as d2l
import psutil
import os
import time

d2l.use_svg_display()
# 添加进程监测函数
def monitor_processes(stage_name=""):
    """监测当前进程和子进程的数量"""
    current_pid = os.getpid()
    current_process = psutil.Process(current_pid)
    
    print(f"\n{'='*50}")
    print(f"【{stage_name}】进程监测")
    print(f"{'='*50}")
    print(f"主进程 PID: {current_pid}")
    print(f"主进程名称: {current_process.name()}")
    
    # 获取子进程
    children = current_process.children()
    print(f"子进程数量: {len(children)}")
    
    for i, child in enumerate(children):
        try:
            print(f"  子进程 {i+1}: PID={child.pid}, 名称={child.name()}, 状态={child.status()}")
        except:
            print(f"  子进程 {i+1}: PID={child.pid} (状态无法获取)")
    
    # 获取所有Python相关进程（可选）
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except:
            pass
    
    print(f"\n系统中Python进程总数: {len(python_processes)}")
    return len(children)

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
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

# 开始监测
monitor_processes("程序启动时")

b = 18
print("\n" + "="*50)
print("第一次加载 FashionMNIST (自定义函数，workers=4)")
print("="*50)
train_iter, test_iter = load_data_fashion_mnist(b)
monitor_processes("第一次加载后")

# 给进程一点时间启动
time.sleep(1)
monitor_processes("等待1秒后")

X, y = next(iter(train_iter))
print(f"\nX.shape: {X.shape}")
monitor_processes("取第一个batch后")

show_images(X.reshape(b, 28, 28), 2, 9, get_fashion_mnist_labels(y))

batch_size = 256
print("\n" + "="*50)
print("第二次加载 FashionMNIST (d2l库函数)")
print("="*50)
data_iter, ignore = d2l.load_data_fashion_mnist(batch_size) 
monitor_processes("第二次加载后")

time.sleep(1)
monitor_processes("等待1秒后")

timer = d2l.Timer()
batch_count = 0
for X, y in data_iter:
    batch_count += 1
    if batch_count == 1:
        monitor_processes("开始遍历第一个batch时")
    if batch_count % 50 == 0:  # 每50个batch监测一次
        print(f"\n已处理 {batch_count} 个batch")
        monitor_processes(f"处理 {batch_count} 个batch后")
    continue

print(f'\n遍历完成，共 {batch_count} 个batch')
print(f'耗时: {timer.stop():.2f} sec')
monitor_processes("遍历完成后")

plt.show()

# 最终监测
monitor_processes("程序结束前")