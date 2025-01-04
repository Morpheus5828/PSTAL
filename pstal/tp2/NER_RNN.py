import os
import pickle
import sys

import torch
from matplotlib import pyplot as plt
from torch import nn, relu
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader


class NER_RNN(nn.Module):
    def __init__(self, d_in, d_out, d_emb, d_hidden=128):
        super().__init__()
        self.d_in = d_in
        self.d_embed = d_emb
        self.d_hidden = d_hidden
        self.d_out = d_out

        # Embedding layer
        self.embedding = nn.Embedding(d_in, d_emb, padding_idx=0)

        # GRU Layer (Increase hidden size and add more layers if needed)
        self.gru = nn.GRU(input_size=d_emb, hidden_size=d_hidden, bidirectional=True,  batch_first=True, bias=False, num_layers=2)

        # Dense layers
        self.dense1 = nn.Linear(d_hidden * 2, d_hidden)  # *2 for bidirectional GRU
        self.dropout = nn.Dropout(0.3)  # Lower dropout
        self.dense2 = nn.Linear(d_hidden, d_out)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


def pad_tensor(X: list, max_len: int, padding_value: int = 0, device=None) -> torch.Tensor:
    result = torch.full((len(X), max_len), padding_value, device=device, dtype=torch.long)
    for i, row in enumerate(X):
        x_len = min(max_len, len(row))
        result[i, :x_len] = torch.tensor(row[:x_len], device=device, dtype=torch.long)
    return result


def extract_ner(
        file_path: str,
        all_form_vocab: dict,
        all_bio_vocab: dict,
        update_vocab: bool = True,
        max_len: int = 30,
        name_tag: str = 'parseme:ne',
        bio_stype: str = 'bio'
):
    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = CoNLLUReader(file).readConllu()

        all_form, all_bio = [], []

        for tokenList in tokenLists:
            current_X = []
            current_Y = []
            forms = [token['form'] for token in tokenList]
            BIOs = CoNLLUReader.to_bio(tokenList, bio_style=bio_stype, name_tag=name_tag)

            for form, bio in zip(forms,BIOs):
                if form not in all_form_vocab:
                    if update_vocab:
                        all_form_vocab[form] = len(all_form_vocab)
                    else:
                        all_form_vocab[form] = all_form_vocab.get("UNK_ID", 1)
                current_X.append(all_form_vocab.get(form, all_form_vocab["UNK_ID"]))

                if bio not in all_bio_vocab:
                    if update_vocab:
                        all_bio_vocab[bio] = len(all_bio_vocab)
                    else:
                        all_bio_vocab[bio] = all_bio_vocab.get("UNK_ID", 1)
                current_Y.append(all_bio_vocab.get(bio, all_bio_vocab["UNK_ID"]))

            all_form.append(current_X)
            all_bio.append(current_Y)

        all_form = pad_tensor(all_form, max_len)
        all_bio = pad_tensor(all_bio, max_len)

        return all_form, all_bio, all_form_vocab, all_bio_vocab


def save(model, vocabulary: dict, tags: dict, history: dict, output_path: str):
    parameters = {'model': model, 'vocabulary': vocabulary, 'tags': tags, 'history': history}

    with open(output_path, "wb") as f:
        pickle.dump(parameters, f)
    print(f"Model saved to {output_path}")


def fit(model, X_train, y_train, X_dev, y_dev, nb_epochs, batch_size):
    optimizer = Adam(lr=0.001, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    histories = {
        "train_loss": [],
        "dev_loss": []
    }

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    dev_dataset = TensorDataset(X_dev, y_dev)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(nb_epochs):
        model.train()
        total_train_loss = 0

        for batch_X, batch_Y in train_dataloader:
            # Forward pass
            output = model(batch_X)
            output = output.view(-1, model.d_out)
            batch_Y = batch_Y.view(-1)
            mask = (batch_Y != 0)

            output = output[mask]
            batch_Y = batch_Y[mask]
            loss = criterion(output, batch_Y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        histories["train_loss"].append(avg_train_loss)

        # Evaluate on the test set
        model.eval()
        total_dev_loss = 0

        with torch.no_grad():
            for batch_X, batch_Y in dev_loader:
                output = model(batch_X)

                output = output.view(-1, model.d_out)
                batch_Y = batch_Y.view(-1)

                mask = batch_Y != 0
                output = output[mask]
                batch_Y = batch_Y[mask]

                loss = criterion(output, batch_Y)
                total_dev_loss += loss.item()

        avg_dev_loss = total_dev_loss / len(dev_loader)
        histories["dev_loss"].append(avg_dev_loss)

        print(f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")

    return model, histories


if __name__ == "__main__":
    sequoia_train_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.train")
    sequoia_dev_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.dev")

    all_form_vocab, all_bio_vocab = {"PAD_ID": 0, "UNK_ID": 1}, {"PAD_ID": 0, "UNK_ID": 1}

    d_emb = 128
    d_hidden = 128
    batch_size = 64
    nb_epochs = 50
    max_len = 30
    name_tag = "parseme:ne"

    X_train, y_train, form_vocab_train, ner_vocab_train = extract_ner(
        file_path=sequoia_train_path,
        all_form_vocab=all_form_vocab,
        all_bio_vocab=all_bio_vocab,
        update_vocab=True,
        max_len=max_len,
    )

    X_dev, y_dev, form_vocab_dev, ner_vocab_dev = extract_ner(
        file_path=sequoia_dev_path,
        all_form_vocab=all_form_vocab,
        all_bio_vocab=all_bio_vocab,
        update_vocab=False,
        max_len=max_len,
    )

    ner_rnn_model = NER_RNN(d_in=len(form_vocab_train), d_out=len(ner_vocab_train), d_emb=d_emb, d_hidden=32)

    model, histories = fit(ner_rnn_model, X_train, y_train, X_dev, y_dev, nb_epochs, batch_size)

    # Save the model
    save(model, form_vocab_train, ner_vocab_train, histories, "./simple.train.emission.parameters.pkl")

    fig = plt.figure()
    plt.plot(histories["dev_loss"], label="Dev loss", c="orange")
    plt.plot(histories["train_loss"], label="Train loss", c="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("RNN for NER training loss")
    plt.show()
    fig.savefig("./RNN for NER training loss", dpi=fig.dpi)



