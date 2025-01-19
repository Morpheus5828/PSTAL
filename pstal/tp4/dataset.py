from torch.utils.data import Dataset

import torch

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