from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import torch


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    word_ids = [item['word_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    word_ids = pad_sequence(word_ids, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'word_ids': word_ids,
        'labels': labels
    }


class SupersenseDataset(Dataset):
    def __init__(self, input_ids, attention_masks, word_ids, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.word_ids = word_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'word_ids': torch.tensor(self.word_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }