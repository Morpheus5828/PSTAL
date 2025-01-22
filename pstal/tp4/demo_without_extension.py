"""
This script is the main function to run model without extension from tp4
thor::Marius THORRE
"""

import os, sys
import argparse
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel
import torch
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)


from pstal.tp4.tool.process_data import contextual_embeddings, target_encoding
from pstal.tp4.tool.predict_ssense import download_pred_file
from pstal.tp4.tool.train_ssense import fit
from pstal.tp4.tool.model import MLP

sequoia_train_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_test_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.test")
sequoia_dev_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.dev")
sequoia_simple_small_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

nb_epochs = 20
batch_size = 128
hidden_layer = 128
learning_rate = 1e-4
encoding_dict = {}

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

    inputs_train, targets_train = contextual_embeddings(sequoia_train_path, tokenizer, pre_trained_model)
    targets_train, encoding_dict = target_encoding(targets_train, encoding_dict, update_dict=True)

    print(f"Input shape: {inputs_train.shape} \n")

    inputs_dev, targets_dev = contextual_embeddings(sequoia_dev_path, tokenizer, pre_trained_model)
    targets_dev, encoding_dict = target_encoding(targets_dev, encoding_dict, update_dict=False)

    print(f"Labels: {encoding_dict} \n")

    step = time.time()
    print(f"Extract process time: {step-start}")

    model = MLP(
        d=768,
        vocab_size_output=len(encoding_dict),
        hidden_layer=hidden_layer).to(device)

    print("Starting training...\n")

    model, history = fit(
        model=model,
        input_train=inputs_train,
        target_train=targets_train,
        input_dev=inputs_dev,
        target_dev=targets_dev,
        vocab_size=len(encoding_dict),
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )

    end = time.time()
    print(f"Training process time: {end - step}")
    print(f"All process time: {end - start}")

    inputs_test, targets_test = contextual_embeddings(sequoia_test_path, tokenizer, pre_trained_model)
    targets_test, _ = target_encoding(targets_test, encoding_dict, update_dict=False)


    output_file_path = os.path.join(project_path, f"pstal/tp4/prediction/prediction_{title}.conllu")


    download_pred_file(
        model=model,
        inputs_test=inputs_test,
        device=device,
        output_file_path=output_file_path,
        encoding_dict=encoding_dict,
        sequoia_test_path=sequoia_test_path
    )

    print(f"Prediction file saved here: {output_file_path}")

    plt.plot(range(len(history["train_loss"])), history["train_loss"], label="train_loss")
    plt.plot(range(len(history["validation_loss"])), history["validation_loss"], label="validation_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")

    plt.savefig(os.path.join(project_path, f"pstal/tp4/loss_{title}.png"))

