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
        upos_list: list = ["NOUN","PROPN","NUM"]
):
    input_embeddings = []
    target_labels = []

    with open(path, encoding="UTF-8") as f:
        for sentence in parse_incr(f):
            words = [token["form"] for token in sentence]

            tok_sent = tokenize(words, is_split_into_words=True, return_tensors='pt')
            word_ids = tok_sent.word_ids()

            with torch.no_grad():
                output = model(**tok_sent)['last_hidden_state'][0]

            output = output[1:-1]
            word_ids = word_ids[1:-1]

            for i, token in enumerate(sentence):
                if token["upos"] in upos_list and token["frsemcor:noun"] != "*":
                    subtoken_indices = [idx for idx, w_id in enumerate(word_ids) if w_id == i]
                    if not subtoken_indices:
                        continue

                    subtokens_emb = output[subtoken_indices]
                    avg_emb = subtokens_emb.mean(dim=0)

                    input_embeddings.append(avg_emb)
                    target_labels.append(token["frsemcor:noun"])

    input_embeddings = torch.stack(input_embeddings)
    return input_embeddings, target_labels


def target_encoding(target: list, encoding_dict: dict, update_dict: bool = True):
    result = []
    for word in target:
        if word not in encoding_dict:
            if update_dict:
                encoding_dict[word] = len(encoding_dict)
        result.append(encoding_dict[word])
    return torch.tensor(result), encoding_dict