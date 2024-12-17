"""This script is for process data before training model
..moduleauthor::Marius THORRE
"""

import torch
from conllu import parse_incr


def tokenize_data(
        sequoia_train_path: str,
        tokenize,
        upos_list: list = ["NOUN", "PROPN", "NUM"],
):
    input_data, target_data, all_ids = [], [], []
    for sent in parse_incr(open(sequoia_train_path, encoding="UTF-8")):
        for tok in sent:
            if tok["upos"] in upos_list:
                target_data.append(tok["frsemcor:noun"])
            else:
                target_data.append("*")
        sent = [tok["form"] for tok in sent]
        tok_sent = tokenize(sent, is_split_into_words=True, return_tensors='pt')
        all_ids.append(tok_sent.word_ids())
        input_data.append(tok_sent["input_ids"])

    return input_data, target_data, all_ids


def clean_data(
        all_ids,
        input_data
    ):
    result = []
    for idx, ids in enumerate(all_ids):
        dico = {}
        current_data = input_data[idx].squeeze()
        for current_ids in ids:
            if current_ids is not None:
                if current_ids not in dico.keys():
                    dico[current_ids] = current_data[current_ids]
                else:
                    dico[current_ids] = (dico[current_ids] + current_data[current_ids]) / 2

        print(torch.Tensor(list(dico.values())))
        result.append(list(dico.values()))
        return

    return result


    '''for sent in parse_incr(open(sequoia_train_path, encoding="UTF-8")):
        current_target = []
        sent = [tok["form"] for tok in sent]
        dico = {}
        tok_sent = tokenize(sent, is_split_into_words=True, return_tensors='pt')
        for element in tok_sent.word_ids():
            if element is not None:
                if element not in dico.keys():
                    dico[element] = tok_sent["input_ids"].squeeze()[element]
                dico[element] = (dico[element] + tok_sent["input_ids"].squeeze()[element]) / 2
        input.append(list(dico.values()))
        target.append(current_target)

    return input, target'''
