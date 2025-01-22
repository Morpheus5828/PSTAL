import inspect
import os
import sys

import torch
from conllu import TokenList

# Add the correct directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../'))
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.conllulib import CoNLLUReader


def pad_tensor(X: list, max_len: int, padding_value: int = 0, device=None) -> torch.Tensor:
    result = torch.full((len(X), max_len), padding_value, device=device, dtype=torch.long)
    for i, row in enumerate(X):
        x_len = min(max_len, len(row))
        result[i, :x_len] = torch.tensor(row[:x_len], device=device, dtype=torch.long)
    return result


def extract_features(tokenList: TokenList, feature_functions, max_len=30):
    """
    Extract features from tokens using provided feature extraction functions.

    Parameters:
    - tokenList (TokenList): tokens to be extracted
    - feature_functions (list): List of feature extraction functions.
    - max_len (int): Maximum length for the feature vector list.

    Returns:
    - Features tensor for each token.
    """
    features = []
    forms = [token['form'] for token in tokenList]
    upos = [token['upos'] for token in tokenList]

    for token, tag in zip(forms, upos):
        feature_vector = []
        for func in feature_functions:
            # Use inspect to get argument names
            func_args = inspect.signature(func).parameters
            if 'token' in func_args and 'tag' in func_args:
                feature_vector.append(func(token, tag))
            elif 'token' in func_args:
                feature_vector.append(func(token))
            elif 'tag' in func_args:
                feature_vector.append(func(tag))
            else:
                feature_vector.append(func())  # For functions that don't need any arguments
        features.append(feature_vector)

    # Ensure the feature list is padded or truncated to max_len
    if len(features) < max_len:
        features = features + [[0] * len(feature_functions) for _ in range(max_len - len(features))]
    else:
        features = features[:max_len]

    # Convert the feature list into a tensor of shape (max_len, num_features)
    features_tensor = torch.tensor(features, dtype=torch.float)

    return features_tensor


def first_letter_uppercase(token, *args):
    return int(token[0].isupper())


def all_letters_uppercase(token, *args):
    return int(token.isupper())


def contains_digits(token, *args):
    return int(any(char.isdigit() for char in token))


def contains_hyphen(token, *args):
    return int("-" in token)


def is_noun(*args, tag):
    return int(tag == 'NOUN')


def is_propn(*args, tag):
    return int(tag == 'PROPN')


def extract_ner(
        file_path: str,
        all_form_vocab: dict,
        all_bio_vocab: dict,
        update_vocab: bool = True,
        max_len: int = 30,
        feature_functions=None,
        name_tag: str = 'parseme:ne',
        bio_stype: str = 'bio'
):
    with open(file_path, "r", encoding="UTF-8") as file:
        tokenLists = CoNLLUReader(file).readConllu()

        all_form, all_bio, all_features = [], [], []

        for tokenList in tokenLists:
            current_X = []
            current_Y = []

            forms = [token['form'] for token in tokenList]
            BIOs = CoNLLUReader.to_bio(tokenList, bio_style=bio_stype, name_tag=name_tag)

            for form, bio in zip(forms, BIOs):
                if form not in all_form_vocab:
                    if update_vocab:
                        all_form_vocab[form] = len(all_form_vocab)
                current_X.append(all_form_vocab.get(form, all_form_vocab["UNK_ID"]))

                if bio not in all_bio_vocab:
                    if update_vocab:
                        all_bio_vocab[bio] = len(all_bio_vocab)
                current_Y.append(all_bio_vocab.get(bio))

            all_form.append(current_X)
            all_bio.append(current_Y)

            if feature_functions:
                current_features = extract_features(tokenList, feature_functions, max_len)
                all_features.append(current_features)

        all_form = pad_tensor(all_form, max_len)
        all_bio = pad_tensor(all_bio, max_len)

        if feature_functions:
            all_features = torch.stack(all_features)
            return all_form, all_features, all_bio, all_form_vocab, all_bio_vocab

        return all_form, all_bio, all_form_vocab, all_bio_vocab
