import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
from conllu import parse_incr
import pstal.tp3.train_morph as train_morph
from pstal.tp3.tool import extract_data, download_pred_file
import matplotlib.pyplot as plt

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")

all_char_vocab = {"<pad>": 0, "<unk>": 1, "<esp>": 2}
all_feats = {"Sing": 0, "Plur": 1, "<N/A>": 2}

if __name__ == "__main__":
    nb_epochs = 10
    batch_size = 64
    hidden_layer = 32
    embedding_size = 50
    hidden_dim = 64
    max_len = 200
    max_w = 20

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.abspath(os.path.join(current_dir, '../..'))
    if project_path not in sys.path:
        sys.path.append(project_path)

    sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
    sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")

    all_char_vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<esp>": 2
    }
    all_feats = {
        "Sing": 0,
        "Plur": 1,
        "<N/A>": 2
    }

    in_enc_train, ends_train, feats_number_train, all_char_vocab = extract_data(
        path=sequoia_train_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats,
        max_len=max_len,
        max_w=max_w
    )

    in_enc_dev, ends_dev, feats_number_dev, all_char_vocab = extract_data(
        path=sequoia_dev_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats,
        max_len=max_len,
        max_w=max_w
    )

    vocab_size_input = len(all_char_vocab)
    vocab_size_output = len(all_feats)

    model, history = train_morph.fit(
        in_enc_train=in_enc_train,
        ends_train=ends_train,
        feats_number_train=feats_number_train,
        in_enc_dev=in_enc_dev,
        ends_dev=ends_dev,
        feats_number_dev=feats_number_dev,
        vocab_size_input=vocab_size_input,
        vocab_size_output=vocab_size_output,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        device=device,
        embedding_size=embedding_size,
        hidden_layer=hidden_layer,
        hidden_dim=hidden_dim
    )

    plt.plot(history["test_loss"], label="Test loss", c="orange")
    plt.plot(history["train_loss"], label="Train loss", c="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training loss for Number")
    #plt.show()

    in_enc_test, ends_test, feats_number_test, all_char_vocab = extract_data(
        path=sequoia_test_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats
    )

    with open(sequoia_test_path, 'r', encoding='utf-8') as f:
        sentences = list(parse_incr(f))

    download_pred_file(
        model=model,
        sentences=sentences,
        in_enc_test=in_enc_test,
        ends_test=ends_test,
        all_feats=all_feats,
        device=device,
        output_file_path=os.path.join(project_path, 'pstal/tp3/predictions.conllu')
    )


