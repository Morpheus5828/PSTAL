import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
import pstal.tp3.train_morph as train_morph
from pstal.tp3.tool import preprocess_data, download_pred_file
import matplotlib.pyplot as plt

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")

all_char_vocab = {"<pad>": 0, "<unk>": 1, "<esp>": 2}
all_feats = {"Sing": 0, "Plur": 1, "<N/A>": 2}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    nb_epochs = 10
    batch_size = 64
    hidden_layer = 32
    embedding_size = 50
    hidden_dim = 64
    max_len = 200
    max_w = 20

    train_data, dev_data, test_data = preprocess_data(
        sequoia_train_path=sequoia_train_path,
        sequoia_dev_path=sequoia_dev_path,
        max_len=max_len,
        max_w=max_w,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats
    )

    model, history = train_morph.fit(
        train_data=train_data,
        dev_data=dev_data,
        vocab_size_input=len(all_char_vocab),
        vocab_size_output=len(all_feats),
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

    download_pred_file(
        model=model,
        test_data=test_data,
        sequoia_test_path=sequoia_test_path,
        device=device,
        output_file_path=os.path.join(project_path, 'pstal/tp3/predictions.conllu')
    )


