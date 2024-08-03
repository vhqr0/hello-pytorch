import torch
from torch import nn, optim

lr = 0.03
num_epochs = 20
num_inputs, num_outputs = 256, 10
num_examples = 100
batch_size = 20


def synthetic_data(W, b, num_examples):
    num_inputs, num_outputs = W.shape
    X = torch.normal(0, 1, (num_examples, num_inputs))
    y = X @ W + b
    y += torch.normal(0, 0.5, y.shape)
    return X, y.reshape(num_examples, num_outputs)


true_W = torch.normal(0, 1, (num_inputs, num_outputs), requires_grad=True) * 0.01
true_b = torch.normal(0, 1, (num_outputs,), requires_grad=True)

train_features, train_labels = synthetic_data(true_W, true_b, int(0.8 * num_examples))
test_features, test_labels = synthetic_data(true_W, true_b, int(0.2 * num_examples))


def data_iter(batch_size, features, labels):
    from torch.utils import data

    dataset = data.TensorDataset(features, labels)
    return iter(data.DataLoader(dataset, batch_size, shuffle=True))


def linreg(X, W, b):
    return X @ W + b


def squared_loss(y_hat, y):
    return (y_hat.reshape(y.shape) - y) ** 2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def l2_penalty(W):
    return (W**2).sum() / 2


def train(l):
    import matplotlib.pyplot as plt

    ## hard:
    # W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    # b = torch.zeros(num_outputs, requires_grad=True)
    # net = lambda X: linreg(X, W, b)
    # loss = lambda y_hat, y: squared_loss(y_hat, y) + l * l2_penalty(W)
    # optimize = lambda: sgd([W, b], lr, batch_size)

    net = nn.Sequential(nn.Linear(num_inputs, num_outputs))
    loss = nn.MSELoss()
    optimizer = optim.SGD(
        [{"params": net[0].weight, "weight_decay": l}, {"params": net[0].bias}],
        lr=lr,
    )
    optimize = lambda: (optimizer.step(), optimizer.zero_grad())

    train_losses, test_losses = [], []

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, train_features, train_labels):
            loss(net(X), y).sum().backward(retain_graph=True)
            optimize()
        with torch.no_grad():
            train_loss = loss(net(train_features), train_labels).mean().item()
            test_loss = loss(net(test_features), test_labels).mean().item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"epoch {epoch + 1}, train loss {train_loss}, test_loss {test_loss}")

    xs = list(range(len(train_losses)))
    plt.cla()
    plt.plot(xs, train_losses, "o")
    plt.plot(xs, test_losses, "x")
    plt.show()


train(0)
train(3)
