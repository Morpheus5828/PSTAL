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

                    subtokens_emb = output[subtoken_indices]
                    avg_emb = subtokens_emb.mean(dim=0)

                    input_embeddings.append(avg_emb)
                    target_labels.append(token["frsemcor:noun"])

    input_embeddings = torch.stack(input_embeddings)
    return input_embeddings, target_labels


def prepare_data_for_finetuning(path, tokenize, upos_list=["NOUN", "PROPN", "NUM"], max_length=30, pad_label_value=-100):
    data, targets, all_words_ids = [], [], []
    with open(path, encoding="UTF-8") as f:
        for sentence in parse_incr(f):
            words = [token["form"] for token in sentence]
            tok_sent = tokenize(
                words,
                is_split_into_words=True,
                return_tensors='pt',
                padding='max_length',
                max_length=max_length,
                truncation=True
            )
            words_ids = tok_sent.word_ids()
            words_ids = [a if a is not None else -1 for a in words_ids]
            all_words_ids.append(torch.tensor(words_ids))
            data.append(tok_sent["input_ids"].squeeze())
            sentence_labels = [pad_label_value] * max_length
            for idx, token in enumerate(sentence):
                if idx >= max_length:
                    break
                if token["upos"] in upos_list and token["frsemcor:noun"] != "*":
                    sentence_labels[idx] = token["frsemcor:noun"]
            targets.append(sentence_labels)
    data = torch.stack(data)
    all_words_ids = torch.stack(all_words_ids)
    return data, targets, all_words_ids

'''def prepare_data_for_finetuning(path, tokenize, upos_list: list = ["NOUN","PROPN","NUM"]):
    data, targets, all_words_ids = [], [], []

    with open(path, encoding="UTF-8") as f:
        for sentence in parse_incr(f):
            words = [token["form"] for token in sentence]
            tok_sent = tokenize(words, is_split_into_words=True, return_tensors='pt', padding='max_length', max_length=42, truncation=True)
            words_ids = tok_sent.word_ids()
            words_ids = [a if a is not None else -1 for a in words_ids]

            all_words_ids.append(torch.Tensor(words_ids))
            data.append(tok_sent["input_ids"].squeeze())

            for i, token in enumerate(sentence):
                if token["upos"] in upos_list and token["frsemcor:noun"] != "*":
                    targets.append(token["frsemcor:noun"])

    data = torch.stack(data)
    all_words_ids = torch.stack(all_words_ids)
    return data, targets, all_words_ids'''


def target_encoding(target: list, encoding_dict: dict, update_dict: bool = True):
    result = []
    for sequence in target:
        seq_indices = []
        for word in sequence:
            if word not in encoding_dict:
                if update_dict:
                    encoding_dict[word] = len(encoding_dict)
            seq_indices.append(encoding_dict.get(word, -100))
        result.append(seq_indices)
    return torch.tensor(result), encoding_dict