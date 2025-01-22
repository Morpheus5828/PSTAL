import os
import pickle
import sys
from abc import ABC, abstractmethod

import torch
from matplotlib import pyplot as plt
from pytorchcrf import CRF
from torch import nn, relu, tensor
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from pstal.tp2.process_data import extract_ner, first_letter_uppercase, all_letters_uppercase, contains_digits, \
    contains_hyphen, is_noun, is_propn

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)


def save(model, optimizer, criterion, vocabulary: dict, tags: dict, history: dict, batch_size: int, output_path: str):
    parameters = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'vocabulary': vocabulary,
                  'tags': tags, 'history': history, 'batch_size': batch_size}

    with open(output_path, "wb") as f:
        pickle.dump(parameters, f)
    print(f"Model saved to {output_path}")


class BaseNERModel(nn.Module, ABC):
    """Abstract Base Class for Named Entity Recognition models."""

    def __init__(self, d_in, d_emb, d_out, d_hidden):
        super().__init__()

        self.d_in = d_in
        self.d_emb = d_emb
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.embedding = nn.Embedding(d_in, d_emb, padding_idx=0)
        self.gru = nn.GRU(input_size=d_emb, hidden_size=d_hidden, bidirectional=True, batch_first=True, num_layers=2,
                          dropout=0.1)  # Bidirectional GRU
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(d_hidden * 2, d_hidden)
        self.decision = nn.Linear(d_hidden, d_out)

    @abstractmethod
    def forward(self, x, features=None):
        pass


class NER_RNN(BaseNERModel):
    def forward(self, x, features=None):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = relu(self.dense(x))
        x = self.dropout(x)

        return self.decision(x)


class NER_RNN_with_features(BaseNERModel):
    def __init__(self, d_in, d_emb, d_features, d_out, d_hidden):
        super().__init__(d_in, d_emb, d_out, d_hidden)
        self.d_features = d_features
        self.gru = nn.GRU(input_size=d_emb + d_features, hidden_size=d_hidden,
                          bidirectional=True, batch_first=True, num_layers=2, dropout=0.1)

    def forward(self, x, features):
        x = self.embedding(x)
        x = torch.cat((x, features), dim=-1)
        x, _ = self.gru(x)
        x = relu(self.dense(x))
        x = self.dropout(x)

        return self.decision(x)


class ModelTrainer:
    """Encapsulates training and evaluation logic."""

    def __init__(self, model, optimizer, criterion, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.history = {"train_loss": [], "dev_loss": []}

    def train(self, X_train, features_train, y_train, X_dev, features_dev, y_dev, nb_epochs):
        train_dataset = TensorDataset(X_train, features_train,
                                      y_train) if features_train is not None else TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        dev_dataset = TensorDataset(X_dev, features_dev, y_dev) if features_dev is not None else TensorDataset(X_dev,
                                                                                                               y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(nb_epochs):
            self.model.train()
            avg_train_loss = self._run_epoch(train_loader, train=True)

            self.model.eval()
            avg_dev_loss = self._run_epoch(dev_loader, train=False)

            print(
                f"Epoch {epoch + 1}/{nb_epochs}, Training Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}")

        return self.history

    def _run_epoch(self, dataloader, train):
        total_loss = 0
        for batch in dataloader:
            X, *features, Y = batch
            features = features[0] if features else None

            self.optimizer.zero_grad()
            output = self.model(X, features)
            loss = self._compute_loss(output, Y)
            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_key = "train_loss" if train else "dev_loss"
        self.history[loss_key].append(avg_loss)

        return avg_loss

    def _compute_loss(self, output, target):
        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        mask = target != 0
        return self.criterion(output[mask], target[mask])


class Config:
    """Configuration settings for model training."""
    D_EMB = 128
    D_HIDDEN = 32
    BATCH_SIZE = 128
    NB_EPOCHS = 50
    MAX_LEN = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.0005
    FEATURE_FUNCTIONS = [
        first_letter_uppercase,
        all_letters_uppercase,
        contains_digits,
        contains_hyphen,
        is_noun,
        is_propn
    ]


if __name__ == "__main__":
    sequoia_train_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.train")
    sequoia_dev_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.dev")

    vocab = {"PAD_ID": 0, "UNK_ID": 1}
    tag_vocab = {"PAD_ID": 0}

    X_train, features_train, y_train, form_vocab_train, ner_vocab_train = extract_ner(
        sequoia_train_path, vocab, tag_vocab, update_vocab=True, max_len=Config.MAX_LEN
    )
    X_dev, features_dev, y_dev, _, _ = extract_ner(
        sequoia_dev_path, form_vocab_train, ner_vocab_train, update_vocab=False, max_len=Config.MAX_LEN
    )

    """
    X_train, features_train, y_train, form_vocab_train, ner_vocab_train = extract_ner(
        sequoia_train_path, vocab, tag_vocab, update_vocab=True, feature_functions=Config.FEATURE_FUNCTIONS, max_len=Config.MAX_LEN
    )
    X_dev, features_dev, y_dev, _, _ = extract_ner(
        sequoia_dev_path, form_vocab_train, ner_vocab_train, update_vocab=False, feature_functions=Config.FEATURE_FUNCTIONS, max_len=Config.MAX_LEN
    )
    """

    # Choose model with or without features
    # model = NER_RNN_with_features(len(form_vocab_train), Config.D_EMB, len(Config.FEATURE_FUNCTIONS), len(ner_vocab_train), Config.D_HIDDEN)
    model = NER_RNN(len(form_vocab_train), Config.D_EMB, len(ner_vocab_train), Config.D_HIDDEN)

    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    trainer = ModelTrainer(model, optimizer, criterion, Config.BATCH_SIZE)
    history = trainer.train(X_train, features_train, y_train, X_dev, features_dev, y_dev, Config.NB_EPOCHS)

    save(model, optimizer, criterion, form_vocab_train, ner_vocab_train, history, Config.BATCH_SIZE,
         "./rnn_model.parameters.pkl")

    plt.plot(history["dev_loss"], label="Dev loss", color="orange")
    plt.plot(history["train_loss"], label="Train loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("NER Model Training Loss")
    plt.show()
