"""This script is the main function to run model from tp4
..moduleauthor::Marius THORRE
"""

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")
sequoia_simple_small_path = os.path.join(project_path, "pstal/sequoia/sequoia-ud.parseme.frsemcor.simple.small")


from model import MLP
from process_data import contextual_embeddings
from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":
    tokenize = AutoTokenizer.from_pretrained("almanach/camembert-base")
    pre_trained_model = AutoModel.from_pretrained("almanach/camembert-base")

    inputs, targets = contextual_embeddings(sequoia_simple_small_path, tokenize, pre_trained_model)
    print(len(inputs), len(targets))
    """
    model = AutoModel.from_pretrained("almanach/camembert-base")
    predict = model(**input[:3])["last_hidden_state"][0]
    print(predict.shape)
    """