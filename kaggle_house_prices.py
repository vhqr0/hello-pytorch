import torch
from torch import nn, optim
import pandas as pd

train_path = "./data/house-prices-advanced-regression-techniques/train.csv"
test_path = "./data/house-prices-advanced-regression-techniques/test.csv"
submission_path = "./data/house-prices-advanced-regression-techniques/submission.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

num_examples = train_data.shape[0]

train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)

features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_indexes = features.dtypes[features.dtypes != "object"].index
# features[numeric_indexes] = features[numeric_indexes].apply(lambda x: (x - x.mean()) / x.std())
features[numeric_indexes] = features[numeric_indexes].fillna(0)
features = pd.get_dummies(features, dummy_na=True)
features = features.to_numpy(dtype=float)

train_features = torch.tensor(features[:num_examples], dtype=torch.float32)
test_features = torch.tensor(features[num_examples:], dtype=torch.float32)


def data_iter(batch_size, features, labels):
    from torch.utils import data

    dataset = data.TensorDataset(features, labels)
    return iter(data.DataLoader(dataset, batch_size, shuffle=True))


lr, wd = 0.03, 1
num_epochs = 500
num_inputs = train_features.shape[1]
num_hiddens1, num_hiddens2 = 77, 18
batch_size = 100

net = nn.Sequential(
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(num_hiddens2, 1),
)
loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

net.train()

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, train_features, train_labels):
        loss(net(X), y).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        l = loss(net(train_features), train_labels).mean()
        print(f"epoch {epoch + 1}, loss {l}")


net.eval()

test_labels = net(test_features).detach().numpy()

submission = pd.concat(
    (test_data["Id"], pd.Series(test_labels.reshape(-1), name="SalePrice")), axis=1
)

submission.to_csv(submission_path, index=False)
