"""This script contains model from tp4
..moduleauthor::Marius THORRE
"""

from torch import nn
from torch.nn.functional import relu


class MLP(nn.Module):
    def __init__(self, d, vocab_size_output, hidden_layer=32):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 32)
        self.fc3 = nn.Linear(32, vocab_size_output)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_FT(nn.Module):
    def __init__(self, pre_trained_model, num_labels, freeze_layers=2, hidden_layer=64):
        super().__init__()
        self.pre_trained_model = pre_trained_model

        for param in self.pre_trained_model.parameters():
            param.requires_grad = False
        for layer in self.pre_trained_model.encoder.layer[-freeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.fc1 = nn.Linear(pre_trained_model.config.hidden_size, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_labels)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input_ids, attention_mask):
        outputs = self.pre_trained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        x = relu(self.fc1(last_hidden_state))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

