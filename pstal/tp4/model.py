"""This script contains model from tp4
..moduleauthor::Marius THORRE
"""
import torch
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
    def __init__(self, pre_trained_model, num_labels, freeze_layers, hidden_layer=32):
        super().__init__()
        self.pre_trained_model = pre_trained_model

        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        for layer in self.pre_trained_model.encoder.layer[-freeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.fc1 = nn.Linear(768, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_labels)
        self.dropout = nn.Dropout()

    def forward(self, x, word_ids, attention_mask, device):
        batch_size, seq_length = x.size()
        all_avg_embeddings = []

        outputs = self.pre_trained_model(input_ids=x.to(device), attention_mask=attention_mask)

        for batch_idx in range(batch_size):
            word_ids_seq = word_ids[batch_idx]
            word_ids_filtered = [w_id.item() if w_id.item() != -1 else None for w_id in word_ids_seq]

            word_ids_filtered = word_ids_filtered[1:-1]
            last_hidden = outputs.last_hidden_state[batch_idx, 1:-1, :]

            avg_embeddings = []
            for i, w_id in enumerate(word_ids_filtered):
                subtoken_indices = [j for j, word_id in enumerate(word_ids_filtered) if word_id == i]
                subtoken_indices = [j for j in subtoken_indices if j < last_hidden.size(0)]
                if subtoken_indices:
                    subtokens_emb = last_hidden[subtoken_indices, :]
                    avg_emb = subtokens_emb.mean(dim=0)
                    avg_embeddings.append(avg_emb)

            sentence_embedding = torch.stack(avg_embeddings).mean(dim=0)
            all_avg_embeddings.append(sentence_embedding)

        input_embeddings = torch.stack(all_avg_embeddings)

        x = relu(self.fc1(input_embeddings))
        x = self.dropout(x)
        x = self.fc2(x)
        return x