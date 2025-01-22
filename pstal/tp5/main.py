"""This script is the main function to run model from tp5
..moduleauthor::Marius THORRE
"""

import os, sys

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_train_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_test_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.test")
sequoia_dev_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.dev")
sequoia_simple_small_path = os.path.join(project_path, "lib/corpus/sequoia-ud.parseme.frsemcor.simple.small")

from model import Model
from predict_tbparser import download_pred_file
from train_tbparser import fit
import torch
from preprocessing import prepare_data
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == "__main__":
    embedding_size = 32
    hidden_layer = 32
    max_len = 10
    nb_epochs = 1
    batch_size = 128
    d = 32

    input_vocab = {"PAD_IDX": 0, "UNK_IDX": 1}
    target_vocab = {}

    input_train, feats_train, target_train, target_vocab, input_vocab = prepare_data(
        path=sequoia_train_path,
        input_vocab=input_vocab,
        target_vocab=target_vocab,
        max_len=max_len
    )

    input_dev, feats_dev, target_dev, target_vocab, input_vocab = prepare_data(
        path=sequoia_dev_path,
        input_vocab=input_vocab,
        target_vocab=target_vocab,
        max_len=max_len
    )

    print(input_train.shape, feats_train.shape, target_train.shape)

    model = Model(
        vocab_size_input=len(input_vocab),
        vocab_size_output=len(target_vocab),
        d=d,
        embedding_size=embedding_size
    ).to(device)

    model, history = fit(
        model=model,
        input_train=input_train,
        feats_train=feats_train,
        target_train=target_train,
        input_dev=input_dev,
        feats_dev=feats_dev,
        target_dev=target_dev,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        device=device
    )

    download_pred_file(
        model=model,
        device=device,
        corpus_path=sequoia_test_path,
        vocab=input_vocab,
        output_file_path=os.path.join(project_path, "pstal/tp5/prediction.conllu"),
        max_len=max_len
    )

    plt.plot(range(len(history["train_loss"])), history["train_loss"], label="train_loss")
    plt.plot(range(len(history["validation_loss"])), history["validation_loss"], label="validation_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.savefig(os.path.join(project_path, "pstal/tp5/loss.png"))
