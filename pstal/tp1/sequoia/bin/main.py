
import os, sys
import torch
from conllu import parse_incr


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../'))
if project_path not in sys.path:
    sys.path.append(project_path)

import pstal.tp1.sequoia.bin.train_postag as train_postag

sequoia_train_path = os.path.join(project_path, "sequoia-ud.parseme.frsemcor.simple.train")
sequoia_test_path = os.path.join(project_path, "sequoia-ud.parseme.frsemcor.simple.test")


def pad_tensor(X, max_len):
    res = torch.full((len(X), max_len), 0)
    for (i, row) in enumerate(X) :
        x_len = min(max_len, len(X[i]))
        res[i,:x_len] = torch.LongTensor(X[i][:x_len])
    return res


def extract_data(path: str):
    all_form_vocab, all_upos_vocab = {"PAD_ID": 0, "UNK_ID": 1}, {"PAD_ID": 0, "UNK_ID": 1}
    max_length = 0
    all_form, all_upos = [], []
    for sent in parse_incr(open(path, encoding="UTF-8")):
        current_X = []
        current_Y = []
        if len(sent) > max_length:
            max_length = len(sent)
        for tok in sent:
            if tok["form"] not in all_form_vocab.keys():
                all_form_vocab[tok["form"]] = len(all_form_vocab)
            current_X.append(all_form_vocab[tok["form"]])

            if tok["upos"] not in all_upos_vocab.keys():
                all_upos_vocab[tok["upos"]] = len(all_upos_vocab)
            current_Y.append(all_upos_vocab[tok["upos"]])
        all_form.append(current_X)
        all_upos.append(current_Y)

    all_form = pad_tensor(all_form, 80)
    all_upos = pad_tensor(all_upos, 80)

    return all_form, all_upos, all_form_vocab, all_upos_vocab


if __name__ == "__main__":
    X_train, y_train, form_vocab_train, upos_vocab_train = extract_data(sequoia_train_path)
    X_test, y_test, form_vocab_test, upos_vocab_test = extract_data(sequoia_test_path)

    train_postag.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        vocab_size=len(form_vocab_train)
    )



