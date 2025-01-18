"""
Main script for training and evaluating the super-sense tagger
..moduleauthor:: Marius THORRE
"""

import os
import sys
import torch
import warnings
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from dataset import SupersenseDataset
from pstal.tp4.model import MLP_FT
from pstal.tp4.train_ssense import fit_ft
from pstal.tp4.predict_ssense import download_pred_file_ft
from pstal.tp4.process_data import prepare_data_for_finetuning, target_encoding

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_train_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_dev_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.dev")
sequoia_test_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

nb_epochs = 15
batch_size = 128
hidden_layer = 64
freeze_layers = 2
learning_rate = 1e-4
max_length = 50

tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
pre_trained_model = AutoModel.from_pretrained("almanach/camembert-base")
encoding_dict = {"*": 0}

print("Preparing data...")
inputs_train, targets_train, word_ids_train, attention_masks_train, encoding_dict = prepare_data_for_finetuning(
    sequoia_train_path, tokenizer, encoding_dict=encoding_dict, max_length=max_length
)
print(inputs_train.shape)

inputs_dev, targets_dev, word_ids_dev, attention_masks_dev, encoding_dict = prepare_data_for_finetuning(
    sequoia_dev_path, tokenizer, encoding_dict=encoding_dict, max_length=max_length
)
train_dataset = SupersenseDataset(inputs_train, attention_masks_train, word_ids_train, targets_train)
dev_dataset = SupersenseDataset(inputs_dev, attention_masks_dev, word_ids_dev, targets_dev)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

num_labels = len(encoding_dict)
model_ft = MLP_FT(
    pre_trained_model=pre_trained_model,
    num_labels=num_labels,
    freeze_layers=freeze_layers,
    hidden_layer=hidden_layer
).to(device)

print("Starting training...")
trained_model, history = fit_ft(
    model=model_ft,
    train_loader=train_loader,
    dev_loader=dev_loader,
    nb_epochs=nb_epochs,
    device=device,
    learning_rate=learning_rate
)

inputs_test, targets_test, word_ids_test, attention_masks_test, _ = prepare_data_for_finetuning(
    sequoia_test_path, tokenizer, encoding_dict=encoding_dict, max_length=max_length
)

test_dataset = SupersenseDataset(inputs_test, attention_masks_test, word_ids_test, targets_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

output_file_path = os.path.join(project_path, "pstal/tp4/prediction_ft.conllu")
download_pred_file_ft(
    model=trained_model,
    test_loader=test_loader,
    encoding_dict=encoding_dict,
    sequoia_test_path=sequoia_test_path,
    output_file_path=output_file_path,
    device=device
)

plt.figure()
plt.plot(range(len(history["train_loss"])), history["train_loss"], label="Training Loss")
plt.plot(range(len(history["validation_loss"])), history["validation_loss"], label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.savefig(os.path.join(project_path, "pstal/tp4/loss_ft.png"))
plt.show()

