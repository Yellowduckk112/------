import torchvision
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn
from d2l import torch as d2l

import multiprocessing; multiprocessing. set_start_method('spawn', force=True)

def get_fashion_labels(labels): #@save
    return [str(i) for i in labels]

def get_dataloader_workers(): #@save
    return 4

def load_data_mnist(batch_size, resize=None): #@save
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def softmax(X):
    X = torch.exp(X)
    sum_exp = X.sum(1, keepdim=True)
    return X / sum_exp

def accurancy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: 
        y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())

def evaluate_accuracy(net, test_iter):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            metric.add(accurancy(net(X), y), y.shape[0])
    return metric[0] / metric[1]

def train_an_epoch(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(l.sum(), accurancy(y_hat, y), y.shape[0])
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_an_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, key):
        return self.data[key]

batch_size = 256
train_iter, test_iter = load_data_mnist(batch_size)
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 20

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_labels(y)
    preds = get_fashion_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)    
plt.show()