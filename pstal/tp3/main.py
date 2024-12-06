import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

import torch
from pstal.tp3.train_morph import fit_multitask, fit
from pstal.tp3.tool import preprocess_data_multitask, download_pred_file_multitask, preprocess_data, download_pred_file
import matplotlib.pyplot as plt

sequoia_train_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.train")
sequoia_test_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.test")
sequoia_dev_path = os.path.join(project_path, "pstal/tp1/sequoia/src/sequoia-ud.parseme.frsemcor.dev")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    nb_epochs = 10
    batch_size = 64
    hidden_layer = 32
    embedding_size = 50
    hidden_dim = 64
    max_len = 200
    max_w = 20

    all_char_vocab = {"<pad>": 0, "<unk>": 1, "<esp>": 2}
    all_feats = {"Sing": 0, "Plur": 1, "<N/A>": 2}

    train_data, dev_data, test_data = preprocess_data(
        sequoia_train_path=sequoia_train_path,
        sequoia_dev_path=sequoia_dev_path,
        sequoia_test_path=sequoia_test_path,
        max_len=max_len,
        max_w=max_w,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats
    )

    print("Start training first model: \n")

    model, history = fit(
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

    download_pred_file(
        model=model,
        test_data=test_data,
        sequoia_test_path=sequoia_test_path,
        device=device,
        output_file_path=os.path.join(project_path, 'pstal/tp3/predictions.conllu')
    )

    all_feats_dict = {
        "Definite": {"Def": 0, "Ind": 1, "<N/A>": 2},
        "Foreign": {"Yes": 0, "<N/A>": 1},
        "Gender": {"Masc": 0, "Fem": 1, "Neut": 2, "<N/A>": 3},
        "Mood": {"Ind": 0, "Sub": 1, "Imp": 2, "<N/A>": 3},
        "NumType": {"Card": 0, "Ord": 1, "<N/A>": 2},
        "Number": {"Sing": 0, "Plur": 1, "<N/A>": 2},
        "Person": {"1": 0, "2": 1, "3": 2, "<N/A>": 3},
        "Polarity": {"Pos": 0, "Neg": 1, "<N/A>": 2},
        "Poss": {"Yes": 0, "<N/A>": 1},
        "PronType": {"Prs": 0, "Rel": 1, "Int": 2, "<N/A>": 3},
        "Reflex": {"Yes": 0, "<N/A>": 1},
        "Tense": {"Past": 0, "Pres": 1, "Fut": 2, "<N/A>": 3},
        "VerbForm": {"Inf": 0, "Part": 1, "Fin": 2, "<N/A>": 3},
        "Voice": {"Act": 0, "Pass": 1, "<N/A>": 2}
    }

    train_data, dev_data, test_data = preprocess_data_multitask(
        sequoia_train_path=sequoia_train_path,
        sequoia_dev_path=sequoia_dev_path,
        sequoia_test_path=sequoia_test_path,
        max_len=max_len,
        max_w=max_w,
        all_char_vocab=all_char_vocab,
        all_feats_dict=all_feats_dict,
    )

    print("Start training second model: \n")

    model2, history2 = fit_multitask(
        train_data=train_data,
        dev_data=dev_data,
        vocab_size_input=len(all_char_vocab),
        all_feats_dict=all_feats_dict,
        nb_epochs=nb_epochs,
        batch_size=batch_size,
        device=device,
        embedding_size=embedding_size,
        hidden_layer=hidden_layer,
        hidden_dim=hidden_dim
    )

    output_file_path = os.path.join(project_path, 'pstal/tp3/predictions_multitask.conllu')

    download_pred_file_multitask(
        model=model2,
        sequoia_test_path=sequoia_test_path,
        test_data=test_data,
        all_feats_dict=all_feats_dict,
        device=device,
        output_file_path=output_file_path
    )


    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    axes[0].plot(history["validation_loss"], label="Validation loss", c="orange")
    axes[0].plot(history["train_loss"], label="Train loss", c="blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss for Model 1 (Single Task)")
    axes[0].legend()

    axes[1].plot(history2["validation_loss"], label="Validation loss", c="orange")
    axes[1].plot(history2["train_loss"], label="Train loss", c="blue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training and Validation Loss for Model 2 (Multi Task)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("pstal/tp3/Figure_3.png")




