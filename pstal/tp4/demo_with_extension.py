"""
This script is the main function to run model with extension from tp4
..moduleauthor::Marius THORRE
"""

import argparse
import os
import sys
import time
import torch
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel
from pstal.tp4.tool.dataset import SupersenseDataset

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

from pstal.tp4.tool.model import MLP_FT
from pstal.tp4.tool.train_ssense import fit_ft
from pstal.tp4.tool.predict_ssense import download_pred_file_ft
from pstal.tp4.tool.process_data import prepare_data_for_finetuning
from pstal.tp4.tool.dataset import collate_fn

sequoia_train_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_dev_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.dev")
sequoia_test_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nb_epochs = 50
batch_size = 128
hidden_layer = 128
freeze_layers = 4
learning_rate = 2e-5
max_length = 45
encoding_dict = {"*": 0}

camembert_tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
camembert_pre_trained_model = AutoModel.from_pretrained("almanach/camembert-base")

bert_multi_tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
bert_multi_pre_trained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

distilbert_tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-multilingual-cased")
distilbert_pre_trained_model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SuperSense running model")
    parser.add_argument("--pre_trained_model", required=True,
                        choices=["CamemBERT", "Multi-BERT", "Distilbert"],
                        help="Choose between CamemBERT or Multi-BERT")

    args = parser.parse_args()

    tokenizer, pre_trained_model, title = None, None, None

    if args.pre_trained_model == "CamemBERT":
        tokenizer = camembert_tokenizer
        pre_trained_model = camembert_pre_trained_model
        title = "Camembert"
    elif args.pre_trained_model == "Multi-BERT":
        tokenizer = camembert_tokenizer
        pre_trained_model = camembert_pre_trained_model
        title = "Multi-BERT"
    elif args.pre_trained_model == "Distilbert":
        tokenizer = distilbert_tokenizer
        pre_trained_model = distilbert_pre_trained_model
        title = "Distilbert"

        
    print("Preparing data, please hold on :-) \n")
    start = time.time()
    inputs_train, targets_train, word_ids_train, attention_masks_train, encoding_dict = prepare_data_for_finetuning(
        path=sequoia_train_path,
        tokenizer=tokenizer,
        encoding_dict=encoding_dict,
        max_length=max_length
    )
    print(f"Input shape: {inputs_train.shape} \n")

    inputs_dev, targets_dev, word_ids_dev, attention_masks_dev, encoding_dict = prepare_data_for_finetuning(
        path=sequoia_dev_path,
        tokenizer=tokenizer,
        encoding_dict=encoding_dict,
        max_length=max_length
    )

    train_dataset = SupersenseDataset(
        input_ids=inputs_train,
        attention_masks=attention_masks_train,
        word_ids=word_ids_train,
        labels=targets_train
    )
    dev_dataset = SupersenseDataset(
        input_ids=inputs_dev,
        attention_masks=attention_masks_dev,
        word_ids=word_ids_dev,
        labels=targets_dev
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    print(f"Labels: {encoding_dict} \n")

    step = time.time()
    print(f"Extract process time: {step - start}")

    num_labels = len(encoding_dict)
    model_ft = MLP_FT(
        pre_trained_model=pre_trained_model,
        pre_trained_model_name=title,
        num_labels=num_labels,
        freeze_layers=freeze_layers,
        hidden_layer=hidden_layer,
    ).to(device)

    print("Starting training...\n")

    trained_model, history = fit_ft(
        model=model_ft,
        train_loader=train_loader,
        dev_loader=dev_loader,
        nb_epochs=nb_epochs,
        device=device,
        learning_rate=learning_rate
    )

    end = time.time()
    print(f"Training process time: {end - step}")
    print(f"All process time: {end - start}")

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

    output_file_path = os.path.join(project_path, f"pstal/tp4/prediction/prediction_ft_{title}.conllu")

    download_pred_file_ft(
        model=trained_model,
        test_loader=test_loader,
        encoding_dict=encoding_dict,
        sequoia_test_path=sequoia_test_path,
        output_file_path=output_file_path,
        device=device
    )

    print(f"Prediction file saved here: {output_file_path}")

    plt.plot(range(len(history["train_loss"])), history["train_loss"], label="Training Loss")
    plt.plot(range(len(history["validation_loss"])), history["validation_loss"], label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss using {title} pre-trained model")
    plt.savefig(os.path.join(project_path, f"pstal/tp4/loss_ft_{title}.png"))

