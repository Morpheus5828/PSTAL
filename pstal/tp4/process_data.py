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


def prepare_data_for_finetuning(
        path,
        tokenizer,
        upos_list=["NOUN", "PROPN", "NUM"],
        max_len=30
):

    all_input_ids = []
    all_word_ids = []
    all_labels = []
    all_attention_mask = []

    with open(path, encoding="utf-8") as f:
        for sentence in parse_incr(f):
            words = [token["form"] for token in sentence]

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                return_tensors='pt',
                padding='max_length',
                max_length=max_len,
                truncation=True
            )
            all_attention_mask.append(encoding["attention_mask"])
            word_ids = encoding.word_ids()
            word_ids = [w if w is not None else -1 for w in word_ids]

            all_input_ids.append(encoding["input_ids"].squeeze(0))
            all_word_ids.append(torch.tensor(word_ids))

            sentence_labels = []
            for idx, token in enumerate(sentence):
                if idx >= max_len:
                    break
                if token["upos"] in upos_list and token["frsemcor:noun"] != "*":
                    sentence_labels.append(token["frsemcor:noun"])


            if len(sentence_labels) < max_len:
                pad_len = max_len - len(sentence_labels)
                sentence_labels.extend(["*"] * pad_len)

            all_labels.append(sentence_labels)

    all_input_ids = torch.stack(all_input_ids, dim=0)
    all_word_ids = torch.stack(all_word_ids, dim=0)
    all_attention_mask = torch.stack(all_attention_mask)

    return all_input_ids, all_labels, all_word_ids, all_attention_mask


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