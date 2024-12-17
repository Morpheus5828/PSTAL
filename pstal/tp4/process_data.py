"""
This script is for process data before training model
Authors : Marius THORRE & Dang Dinh NGUYEN
"""
import os
import sys

import torch

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader


def contextual_embeddings(
        sequoia_train_path: str,
        tokenize,
        model,
        upos_list: list = ["NOUN", "PROPN", "NUM"],
):
    # TO DO : convert targets_labels into list of index
    input_embeddings, target_labels = [], []

    with open(sequoia_train_path, "r", encoding="UTF-8") as file:
        sentences = CoNLLUReader(file).readConllu()

        for sentence in sentences:
            words, labels, word_indices = [], [], []
            # print(sentence)

            for token in sentence:
                words.append(token["form"])
                if token["upos"] in upos_list:
                    """
                    print(token)
                    print(token["upos"])
                    print(token["frsemcor:noun"])
                    """
                    labels.append(token["frsemcor:noun"])
                    word_indices.append(len(words) - 1)

            """
            print(words)
            print(labels)
            print(word_indices)
            """

            # Tokenize sentence
            tok_sent = tokenize(words, is_split_into_words=True, return_tensors='pt')
            word_ids = tok_sent.word_ids()  # Alignement words <-> sub-tokens
            """
            print(tok_sent["input_ids"])
            print(tok_sent.word_ids())
            """

            # Extract embeddings
            with torch.no_grad():  # Freeze embeddings
                outputs = model(**tok_sent)
                embeddings = outputs.last_hidden_state[0]  # Shape: (num_tokens, 768)
                # print(embeddings.shape)

            # Mean of contextual embeddings for each desired word
            for idx in word_indices:
                subtoken_indices = [i for i, word_id in enumerate(word_ids) if word_id == idx]
                # print(subtoken_indices)
                if subtoken_indices:  # If the word contains sub-tokens
                    subtoken_embs = embeddings[subtoken_indices]
                    mean_emb = subtoken_embs.mean(dim=0)  # Mean of embeddings
                    # print(mean_emb)
                    # print(mean_emb.shape)
                    input_embeddings.append(mean_emb.numpy())
                    target_labels.append(labels.pop(0))
            #print()

        return input_embeddings, target_labels