"""This script contains different model for tp3
..moduleauthor:: Marius THORRE
"""


from torch import nn
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


class MultiTaskMorphModel(nn.Module):
    def __init__(self, vocab_size_input, all_feats_dict, d, embedding_size, hidden_layer=64):
        super(MultiTaskMorphModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size_input, embedding_size, padding_idx=0)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=d, batch_first=True, bias=False, num_layers=2)
        self.fc1 = nn.Linear(d, hidden_layer)
        self.fc_layers = nn.ModuleDict({
            tm: nn.Linear(hidden_layer, len(vocab)) for tm, vocab in all_feats_dict.items()
        })
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        outputs = {tm: fc_layer(x) for tm, fc_layer in self.fc_layers.items()}
        return outputs

