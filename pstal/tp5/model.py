"""This script contains model from tp5
..moduleauthor::Marius THORRE
"""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size_input, vocab_size_output, d, embedding_size, hidden_layer=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size_input, embedding_size, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=d, batch_first=True, bias=False, num_layers=2)
        self.fc1 = nn.Linear(d * 4, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 32)
        self.fc3 = nn.Linear(32, vocab_size_output)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x, feats):
        emb = self.embedding(x)
        out_rnn, _ = self.gru(emb)

        B, L, d = out_rnn.shape
        feats_expanded = feats.unsqueeze(2).expand(-1, -1, d)
        hidden_feats = torch.gather(out_rnn, 1, feats_expanded)
        hidden_feats = hidden_feats.reshape(B, 4 * d)

        x = torch.relu(self.fc1(hidden_feats))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits