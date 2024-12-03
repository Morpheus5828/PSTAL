"""This script contains code to train tp3 model
..moduleauthor:: Marius THORRE
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import relu


class MorphModel(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_output, d, embedding_size, hidden_layer=32):
        super(MorphModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size_input, embedding_size, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=d, batch_first=True, bias=False, num_layers=2)
        self.fc1 = nn.Linear(d, hidden_layer)
        self.dropout = nn.Dropout(0.4)
        self.fc_number = nn.Linear(hidden_layer, vocab_size_output)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = relu(self.fc1(x))
        x = self.dropout(x)
        number = self.fc_number(x)
        return number


def fit(
        train_data,
        dev_data,
        vocab_size_input,
        vocab_size_output,
        nb_epochs,
        batch_size,
        device,
        embedding_size,
        hidden_layer,
        hidden_dim
):
    model = MorphModel(
        vocab_size_input=vocab_size_input,
        vocab_size_output=vocab_size_output,
        d=hidden_dim,
        embedding_size=embedding_size,
        hidden_layer=hidden_layer
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=2)
    optim = Adam(lr=1e-4, params=model.parameters())

    history = {
        "train_loss": [],
        "test_loss": []
    }

    train_dataset = TensorDataset(in_enc_train, ends_train, feats_number_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(in_enc_dev, ends_dev, feats_number_dev)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_ends, batch_Y in train_dataloader:
            batch_X = batch_X.to(device).long()
            batch_ends = batch_ends.to(device).long()
            batch_Y = batch_Y.to(device).long()

            outputs = model(batch_X)

            batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1,
                                                                 outputs.size(2))
            gathered = outputs.gather(1, batch_ends_expanded)

            gathered = gathered.view(-1, vocab_size_output)
            batch_Y = batch_Y.view(-1)

            loss = criterion(gathered, batch_Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_X, batch_ends, batch_Y in test_loader:
                batch_X = batch_X.to(device).long()
                batch_ends = batch_ends.to(device).long()
                batch_Y = batch_Y.to(device).long()

                outputs = model(batch_X)
                batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, outputs.size(2))
                gathered = outputs.gather(1, batch_ends_expanded)
                gathered = gathered.view(-1, vocab_size_output)
                batch_Y = batch_Y.view(-1)

                loss = criterion(gathered, batch_Y)
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        history["test_loss"].append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
    }, 'model_checkpoint.pt')


    return model, history
