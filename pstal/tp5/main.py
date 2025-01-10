"""This script is the main function to run model from tp5
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

from model import Model
from train_tbparser import fit
import torch
from preprocessing import prepare_data
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_size_input = 0
    vocab_size_output = 0
    embedding_size = 0
    hidden_layer = 32
    max_len = 10

    input_vocab = {"PAD_IDX": 0, "UNK_IDX": 1}
    target_vocab = {}

    input_train, feats_train, target_train = prepare_data(
        path=sequoia_simple_small_path,
        input_vocab=input_vocab,
        target_vocab=target_vocab,
        max_len=max_len
    )

    print(input_train.shape, feats_train.shape, len(target_train))










