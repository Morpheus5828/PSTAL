"""This script contains model from tp4
..moduleauthor::Marius THORRE
"""

from torch import nn
from torch.nn.functional import relu


class MLP(nn.Module):
    def __init__(self, d, vocab_size_output, hidden_layer=32):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, vocab_size_output)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x