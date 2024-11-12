""" This script contains function to training model
.. moduleauthor:: Marius THORRE, Dang-Dinh NGUYEN
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import relu


class Model(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.gru = nn.GRU(hidden_size=d, input_size=d, batch_first=True, bias=False)
        self.fc1 = nn.Linear(d, 32)
        self.fc2 = nn.Linear(32, vocab_size)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def fit(X_train, y_train, X_test, y_test, vocab_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    model = Model(vocab_size=vocab_size, d=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = Adam(lr=1e-4, params=model.parameters())
    batch_size = 64
    nb_epochs = 10

    history = {
        "train_loss": [],
        "test_loss": []
    }

    num_samples = X_train.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).long()
            batch_Y = batch_Y.to(device).long()

            output = model(batch_X)
            loss = criterion(output, batch_Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(device).long()
                batch_Y = batch_Y.to(device).long()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)

                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)



