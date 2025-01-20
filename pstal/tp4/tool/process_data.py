"""
This script is for process data before training model
Authors : Marius THORRE & Dang Dinh NGUYEN
"""
import os
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

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


def prepare_data_for_finetuning(path, tokenizer, encoding_dict, upos_list=["NOUN", "PROPN", "NUM"], max_length=30, pad_label_value=-100):
    input_ids = []
    attention_masks = []
    word_ids_list = []
    labels = []

    with open(path, encoding="utf-8") as f:
        for sentence in parse_incr(f):
            words = [token["form"] for token in sentence]
            tok_sent = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True
            )

            word_ids = tok_sent.word_ids()
            word_ids = [w if w is not None else -1 for w in word_ids]
            word_ids_list.append(torch.tensor(word_ids))
            input_ids.append(tok_sent["input_ids"].squeeze(0))
            attention_masks.append(tok_sent["attention_mask"].squeeze(0))

            sentence_labels = [pad_label_value] * max_length
            for idx, token in enumerate(sentence):
                if idx >= max_length:
                    break
                if token["upos"] in upos_list:
                    label = token.get("frsemcor:noun")

                    if label not in encoding_dict:
                        encoding_dict[label] = len(encoding_dict)
                    sentence_labels[idx] = encoding_dict[label]
            labels.append(sentence_labels)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    word_ids_list = torch.stack(word_ids_list)

    return input_ids, labels, word_ids_list, attention_masks, encoding_dict



def target_encoding(target: list, encoding_dict: dict, update_dict: bool = True):
    result = []
    for word in target:
        if word not in encoding_dict:
            if update_dict:
                encoding_dict[word] = len(encoding_dict)
        result.append(encoding_dict[word])
    return torch.tensor(result), encoding_dict