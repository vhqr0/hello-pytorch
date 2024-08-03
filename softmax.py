import torch
from torch import nn, optim
import torchvision

lr = 0.1
num_epochs = 10
num_inputs, num_hiddens, num_outputs = 28 * 28, 256, 10
batch_size = 256

mnist_train = torchvision.datasets.FashionMNIST(
    root="./data",
    download=True,
    train=True,
    transform=torchvision.transforms.ToTensor(),
)

mnist_test = torchvision.datasets.FashionMNIST(
    root="./data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor(),
)

print("[train test] length:", len(mnist_train), len(mnist_test))  # 60000, 10000

X, y = mnist_test[0]
print("X shape:", X.shape)  # (1, 28, 28)
print("y type:", type(y))  # int


def data_iter(dataset, batch_size):
    from torch.utils import data

    return iter(data.DataLoader(dataset, batch_size=batch_size, shuffle=True))


mnist_titles = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]


def get_mnist_titles(labels):
    return [mnist_titles[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    import matplotlib.pyplot as plt

    plt.cla()
    figsize = (num_cols * scale, num_cols * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    # What's your problem, pyright?
    axes = axes.flatten()  # type: ignore
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            img = img.numpy()
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes, plt.show()


# X, y = next(data_iter(mnist_test, 18))
# titles = get_mnist_titles(y)
# show_images(X.reshape(18, 28, 28), 2, 9, titles=titles)


def relu(X):
    return torch.max(X, torch.zeros_like(X))


def softmax(X):
    X_exp = torch.exp(X)
    return X_exp / X_exp.sum(1, keepdim=True)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y])


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


## hard 1 layer:
# W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs, requires_grad=True)
# net = lambda X: softmax(X.reshape(-1, num_inputs) @ W + b)
# loss = cross_entropy
# optimize = lambda: sgd([W, b], lr, batch_size)

## easy 1 layer:
# net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))
# loss = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=lr)
# optimize = lambda: (optimizer.step(), optimizer.zero_grad())

## hard 2 layer:
# W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
# b1 = torch.zeros((), requires_grad=True)
# W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
# b2 = torch.zeros((), requires_grad=True)
# net = lambda X: softmax(relu(X.reshape(-1, num_inputs) @ W1 + b1) @ W2 + b2)
# loss = cross_entropy
# optimize = lambda: sgd([W1, b1, W2, b2], lr, batch_size)


## easy 2 layer:
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
optimize = lambda: (optimizer.step(), optimizer.zero_grad())


for epoch in range(num_epochs):
    for X, y in data_iter(mnist_train, batch_size):
        loss(net(X), y).sum().backward()
        optimize()
    with torch.no_grad():
        X, y = next(data_iter(mnist_test, batch_size))
        train_loss = loss(net(X), y)
        print(f"epoch {epoch + 1}, loss {train_loss.mean()}")


X, y = next(data_iter(mnist_test, 18))
true_titles = get_mnist_titles(y)
pred_titles = get_mnist_titles(net(X).argmax(1))
titles = ["\n".join(titles) for titles in zip(true_titles, pred_titles)]
show_images(X.reshape(18, 28, 28), 2, 9, titles=titles)
