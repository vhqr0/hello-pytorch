import torch
from torch import nn, optim

lr = 0.03
num_epochs = 3
num_examples = 100
batch_size = 10


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(num_examples, 1)


true_w, true_b = torch.tensor([2, -3.4]), torch.tensor(4.2)
features, labels = synthetic_data(true_w, true_b, num_examples)

print("[true_w true_b] shape:", true_w.shape, true_b.shape)  # (2,), ()
print("[features labels] shape:", features.shape, labels.shape)  # (100, 2), (100, 1)


# import matplotlib.pyplot as plt
# scatter = lambda x, y: (plt.cla(), plt.scatter(x, y), plt.show())
# scatter(features[:, 0], labels[:, 0])
# scatter(features[:, 1], labels[:, 0])


def data_iter(batch_size, features, labels):
    ## hard:
    # from random import shuffle
    # num_examples = len(features)
    # indices = list(range(num_examples))
    # shuffle(indices)
    # for i in range(0, num_examples, batch_size):
    #     j = min(i + batch_size, num_examples)
    #     batch_indices = torch.tensor(indices[i:j])
    #     yield features[batch_indices], labels[batch_indices]
    from torch.utils import data

    dataset = data.TensorDataset(features, labels)
    return iter(data.DataLoader(dataset, batch_size, shuffle=True))


X, y = next(data_iter(batch_size, features, labels))
print("batched data shape:", X.shape, y.shape)


def linreg(X, w, b):
    return X @ w + b


def squared_loss(y_hat, y):
    return (y_hat.reshape(y.shape) - y) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


## hard:
# w = torch.normal(0, 0.01, size=(2,), requires_grad=True)
# b = torch.tensor(0.0, requires_grad=True)
# net = lambda X: linreg(X, w, b)
# loss = squared_loss
# optimize = lambda: sgd([w, b], lr, batch_size)

net = nn.Sequential(nn.Linear(2, 1))
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
optimize = lambda: (optimizer.step(), optimizer.zero_grad())


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        loss(net(X), y).sum().backward()
        optimize()
    with torch.no_grad():
        train_loss = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {train_loss.mean()}")
