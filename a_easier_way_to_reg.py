import random
import torch
from torch import nn
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    for X, y in data_iter:
        L = loss(net(X), y)
        trainer.zero_grad()
        L.backward()
        trainer.step()
    with torch.no_grad():
        train_l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {train_l:f}')

print(net[0].weight.data - true_w, net[0].bias.data - true_b)