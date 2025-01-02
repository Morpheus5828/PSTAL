"""This script is the main function to run model from tp5
..moduleauthor::Marius THORRE
"""

import os, sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")
sequoia_simple_small_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.small")

from model import Model
from train_tbparser import fit
import torch
import matplotlib.pyplot as plt
from lib.conllulib import CoNLLUReader, TransBasedSent, TransBasedConfig
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    vocab_size_input = 0
    vocab_size_output = 0
    embedding_size = 0
    hidden_layer = 32

    reader = CoNLLUReader(open(sequoia_simple_small_path))

    for sent in reader.readConllu():
        print(sent)
        print(TransBasedSent(sent, actions_only=True))
        print(TransBasedConfig(sent).is_final())
        break

'''    model = Model(
        vocab_size_input=vocab_size_input,
        vocab_size_output=vocab_size_output,
        d=hidden_layer,
        embedding_size=embedding_size
    ).to(device)
'''







