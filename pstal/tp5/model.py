"""This script contains model from tp5
..moduleauthor::Marius THORRE
"""

from torch import nn
from torch.nn.functional import relu


class Model(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_output, d, embedding_size, hidden_layer=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_input, embedding_size, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=d)
        self.fc1 = nn.Linear(d, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 32)
        self.fc3 = nn.Linear(32, vocab_size_output)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x