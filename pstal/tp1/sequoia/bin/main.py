""" This script is a tools script which contains code run model fitting code
..moduleauthor:: Marius Thorre
"""

import os, sys
import torch
import matplotlib.pyplot as plt
from tool import extract_data, download_pred_file

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import pstal.tp1.sequoia.bin.train_postag as train_postag

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/sequoia-ud.parseme.frsemcor.simple.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/sequoia-ud.parseme.frsemcor.simple.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/sequoia-ud.parseme.frsemcor.simple.dev")


if __name__ == "__main__":
    all_form_vocab, all_upos_vocab = {"PAD_ID": 0, "UNK_ID": 1}, {"PAD_ID": 0, "UNK_ID": 1}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 64
    nb_epochs = 120
    max_len = 30
    embedding_size = 100
    hidden_layer = 32

    X_train, y_train, form_vocab_train, upos_vocab_train = extract_data(
        path=sequoia_train_path,
        all_form_vocab=all_form_vocab,
        all_upos_vocab=all_upos_vocab,
        update_vocab=True,
        max_len=max_len
    )

    X_dev, y_dev, form_vocab_train, upos_vocab_train = extract_data(
        path=sequoia_dev_path,
        all_form_vocab=all_form_vocab,
        all_upos_vocab=all_upos_vocab,
        update_vocab=True,
        max_len=max_len
    )

    model, history = train_postag.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_dev,
        y_test=y_dev,
        vocab_size_input=len(form_vocab_train),
        vocab_size_output=len(upos_vocab_train),
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        hidden_layer=hidden_layer,
        embedding_size=embedding_size,
        device=device
    )

    X_test, y_test, _, _ = extract_data(
        path=sequoia_test_path,
        all_form_vocab=all_form_vocab,
        all_upos_vocab=all_upos_vocab,
        update_vocab=False
    )

    plt.plot(history["test_loss"], label="Test loss", c="orange")
    plt.plot(history["train_loss"], label="Train loss", c="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss")

    download_pred_file(
        model=model,
        test_file_path=sequoia_test_path,
        form_vocab=form_vocab_train,
        upos_vocab=upos_vocab_train,
        device=device,
        batch_size=batch_size,
        output_file_path=os.path.join(project_path, 'pstal/tp1/sequoia/bin/predictions.conllu')
    )

    plt.show()




