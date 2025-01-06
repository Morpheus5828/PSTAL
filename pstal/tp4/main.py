"""This script is the main function to run model from tp4
..moduleauthor::Marius THORRE
"""

import os, sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_train_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_test_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.test")
sequoia_dev_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.dev")
sequoia_simple_small_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.small")

from process_data import contextual_embeddings, target_encoding
from predict_ssense import download_pred_file
from train_postag import fit
from model import MLP
from transformers import AutoTokenizer, AutoModel
import torch
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Device used: {device}")

if __name__ == "__main__":
    tokenize = AutoTokenizer.from_pretrained("almanach/camembert-base")
    pre_trained_model = AutoModel.from_pretrained("almanach/camembert-base")
    nb_epochs = 20
    batch_size = 128
    hidden_layer = 128
    encoding_dict = {"*": 0}

    inputs_train, targets_train = contextual_embeddings(sequoia_train_path, tokenize, pre_trained_model)
    targets_train, encoding_dict = target_encoding(targets_train, encoding_dict, update_dict=True)

    inputs_dev, targets_dev = contextual_embeddings(sequoia_dev_path, tokenize, pre_trained_model)
    targets_dev, encoding_dict = target_encoding(targets_dev, encoding_dict, update_dict=False)

    print(encoding_dict)

    model = MLP(d=768, vocab_size_output=len(encoding_dict), hidden_layer=hidden_layer).to(device)

    print("Start model: \n")


    print(inputs_train.shape, targets_train.shape)

    model, history = fit(
        model=model,
        input_train=inputs_train,
        target_train=targets_train,
        input_dev=inputs_dev,
        target_dev=targets_dev,
        vocab_size=len(encoding_dict),
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        device=device
    )

    inputs_test, targets_test = contextual_embeddings(sequoia_test_path, tokenize, pre_trained_model)
    targets_test, _ = target_encoding(targets_test, encoding_dict, update_dict=False)

    download_pred_file(
        model=model,
        inputs_test=inputs_test,
        device=device,
        output_file_path=os.path.join(project_path, "pstal/tp4/prediction.conllu"),
        encoding_dict=encoding_dict,
        sequoia_test_path=sequoia_test_path
    )

    plt.plot(range(len(history["train_loss"])), history["train_loss"], label="train_loss")
    plt.plot(range(len(history["validation_loss"])), history["validation_loss"], label="validation_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(os.path.join(project_path, "pstal/tp4/loss.png"))
