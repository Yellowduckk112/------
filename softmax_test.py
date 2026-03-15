from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt
from IPython import display
import torchvision
from torchvision import transforms
from torch.utils import data

def softmax(X):
    X_exp = torch.exp(X)
    sum_exp = X_exp.sum(1, keepdim=True)
    return X_exp / sum_exp

def load_data_fashion_mnist(batch_size, resize=None): #@save
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=d2l.get_dataloader_workers()))

def net(X):
    return softmax(torch.matmul(X.reshape(-1, w.shape[0]), w) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(y.shape[0]), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmd = y_hat.type(y.dtype) == y
    return float(cmd.sum())


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_an_epoch(net, train_iter, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        los = cross_entropy(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            los.mean().backward()
            updater.step()
        else:
            los.sum().backward()
            updater(len(X))
        metric.add(float(los.sum()), accuracy(y_hat, y), X.shape[0])
    return metric[0] / metric[2], metric[1] / metric[2]

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(data)
    
    def __getitem__(self, key):
        return self.data[key]

def train_ch3(epoch_num, net, updater, loss, train_iter, test_iter):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, epoch_num], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(epoch_num):
        train_metrics = train_an_epoch(net, train_iter, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics

def updater(batch_size):
    return d2l.sgd([w, b], lr, batch_size)

def predict_ch3(net, test_iter, n=6):  #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

print(type(evaluate_accuracy))
input_num = 784
output_num = 10
w = torch.normal(0, 0.01, size=(input_num, output_num), requires_grad=True)
b = torch.zeros(output_num, requires_grad=True)

lr = 0.1
num_epochs = 10
train_ch3(num_epochs, net, updater, cross_entropy, train_iter, test_iter)

_, test_iter_2 = load_data_fashion_mnist(10)
for i, X, y in zip(range(10), *test_iter_2):
    pass
true_labels = d2l.get_fashion_mnist_labels(y)
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
print(true_labels)
print(pred_labels)
plt.show()