""" This script contains function to training model
.. moduleauthor:: Marius THORRE, Dang-Dinh NGUYEN
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import relu
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_output, d):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size_input, d, padding_idx=0)
        self.gru = nn.GRU(input_size=d, hidden_size=d, batch_first=True, bias=False)
        self.fc1 = nn.Linear(d, 32)
        self.fc2 = nn.Linear(32, vocab_size_output)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def fit(X_train, y_train, X_test, y_test, vocab_size_input, vocab_size_output):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    model = Model(vocab_size_input=vocab_size_input, vocab_size_output=vocab_size_output, d=32).to(device)
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

    PAD_ID = 0

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).long()
            batch_Y = batch_Y.to(device).long()

            output = model(batch_X)

            output = output.view(-1, vocab_size_output)
            batch_Y = batch_Y.view(-1)

            mask = batch_Y != PAD_ID
            output = output[mask]
            batch_Y = batch_Y[mask]

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


                outputs = outputs.view(-1, vocab_size_output)
                batch_Y = batch_Y.view(-1)
                mask = batch_Y != PAD_ID
                outputs = outputs[mask]
                batch_Y = batch_Y[mask]

                loss = criterion(outputs, batch_Y)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)

        print(f"Epoch {epoch+1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    # Vous pouvez sauvegarder le modèle et les vocabulaires ici si nécessaire
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'form_vocab': form_vocab_train,
    #     'upos_vocab': upos_vocab_train,
    # }, 'model_checkpoint.pt')

    return model, history




