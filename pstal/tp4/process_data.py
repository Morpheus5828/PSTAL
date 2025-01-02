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

from lib.conllulib import CoNLLUReader


def contextual_embeddings(
        path: str,
        tokenize,
        model,
        upos_list: list = ["NOUN", "PROPN", "NUM"],
):

    input_embeddings, target_labels = [], []

    with open(path, "r", encoding="UTF-8") as file:
        sentences = CoNLLUReader(file).readConllu()

        for sentence in sentences:
            words, labels, word_indices = [], [], []

            for token in sentence:
                words.append(token["form"])
                if token["upos"] in upos_list:
                    labels.append(token["frsemcor:noun"])
                    word_indices.append(len(words) - 1)


            tok_sent = tokenize(words, is_split_into_words=True, return_tensors='pt')
            word_ids = tok_sent.word_ids()  # Alignement words <-> sub-tokens

            with torch.no_grad():  # Freeze embeddings
                outputs = model(**tok_sent)
                embeddings = outputs["last_hidden_state"][0]  # Shape: (num_tokens, 768)

            for idx in word_indices:
                subtoken_indices = [i for i, word_id in enumerate(word_ids) if word_id == idx]
                if subtoken_indices:  # If the word contains sub-tokens
                    subtoken_embs = embeddings[subtoken_indices]
                    mean_emb = subtoken_embs.mean(dim=0)  # Mean of embeddings
                    input_embeddings.append(mean_emb.numpy())
                    target_labels.append(labels.pop(0))

        return torch.tensor(input_embeddings), target_labels


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
