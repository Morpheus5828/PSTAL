"""
This script is for process data before training model
Authors : Marius THORRE & Dang Dinh NGUYEN
"""
import os
import sys

import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../../lib'))
if project_path not in sys.path:
    sys.path.append(project_path)

from conllu import parse_incr


def contextual_embeddings(
        path: str,
        tokenize,
        model,
        upos_list: list = ["NOUN", "PROPN", "NUM"],
):
    input_embeddings, target_labels = [], []

    for sentence in parse_incr(open(path, encoding="UTF-8")):
        words, labels, upos_tags = [], [], []

        for token in sentence:
            words.append(token["form"])
            upos_tags.append(token["upos"])
            if token["upos"] in upos_list:
                labels.append(token["frsemcor:noun"])

        tok_sent = tokenize(words, is_split_into_words=True, return_tensors='pt')
        word_ids = tok_sent.word_ids()
        with torch.no_grad():
            output = model(**tok_sent)['last_hidden_state'][0]

        output = output[1:-1]
        word_ids = word_ids[1:-1]

        current_result = []
        current_idx = None
        label_idx = 0

        for idx, value in enumerate(word_ids):
            if current_idx is None:
                current_idx = value
                current_result.append(output[idx])
            elif value == current_idx:
                current_result.append(output[idx])
            else:
                if upos_tags[current_idx] in upos_list:
                    average = torch.mean(torch.stack(current_result), dim=0)
                    input_embeddings.append(average)
                    target_labels.append(labels[label_idx])
                    label_idx += 1

                current_result = [output[idx]]
                current_idx = value

        if len(current_result) > 0 and upos_tags[current_idx] in upos_list:
            average = torch.mean(torch.stack(current_result), dim=0)
            input_embeddings.append(average)
            target_labels.append(labels[label_idx])

    return torch.stack(input_embeddings), target_labels


def target_encoding(target: list, encoding_dict: dict, update_dict: bool = True):
    result = []
    for word in target:
        if word not in encoding_dict:
            if update_dict:
                encoding_dict[word] = len(encoding_dict)
            else:
                encoding_dict[word] = encoding_dict["*"]
        result.append(encoding_dict[word])
    return torch.tensor(result), encoding_dict
