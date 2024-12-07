"""
This script is a tools script which contains code for fit function
..moduleauthor:: Marius Thorre
"""

import torch
#from pstal.tp1.sequoia.bin.tool import pad_tensor
from torch.utils.data import TensorDataset, DataLoader
from conllu import parse_incr


def pad_tensor(X: list, max_len: int, padding_value: int = 0, device=None) -> torch.Tensor:
    result = torch.full((len(X), max_len), padding_value, device=device, dtype=torch.long)
    for i, row in enumerate(X):
        x_len = min(max_len, len(row))
        result[i, :x_len] = torch.tensor(row[:x_len], device=device, dtype=torch.long)
    return result


def preprocess_data_multitask(
        sequoia_train_path,
        sequoia_dev_path,
        sequoia_test_path,
        max_len,
        max_w,
        all_char_vocab,
        all_feats_dict
):
    def extract_data(path, all_char_vocab, all_feats_dict):
        in_enc, ends, feats_dict = [], [], {tm: [] for tm in all_feats_dict.keys()}

        for sent in parse_incr(open(path, encoding="UTF-8")):
            current_chars, current_ends = [all_char_vocab["<pad>"]], []
            tm_features = {tm: [] for tm in all_feats_dict.keys()}

            for tok in sent:
                for letter in tok["lemma"]:
                    if letter not in all_char_vocab:
                        all_char_vocab[letter] = len(all_char_vocab)
                    current_chars.append(all_char_vocab[letter])
                current_chars.append(all_char_vocab["<esp>"])

                for tm, vocab in all_feats_dict.items():
                    feat_value = tok["feats"].get(tm, "<N/A>") if tok["feats"] else "<N/A>"
                    tm_features[tm].append(vocab.get(feat_value, vocab["<N/A>"]))

            current_ends = [idx for idx in range(len(current_chars) - 1) if current_chars[idx + 1] == all_char_vocab["<esp>"]]
            in_enc.append(current_chars[:-1])
            ends.append(current_ends)
            for tm in feats_dict.keys():
                feats_dict[tm].append(tm_features[tm])

        # Padding des données et transformation en tensors PyTorch
        in_enc = pad_tensor(in_enc, max_len)
        ends = pad_tensor(ends, max_w)
        for tm in feats_dict.keys():
            feats_dict[tm] = pad_tensor(feats_dict[tm], max_w)

        return {
            "in_enc": in_enc,
            "ends": ends,
            "feats_dict": feats_dict
        }

    # Utilisation de extract_data pour chaque jeu de données
    train_data = extract_data(sequoia_train_path, all_char_vocab, all_feats_dict)
    dev_data = extract_data(sequoia_dev_path, all_char_vocab, all_feats_dict)
    test_data = extract_data(sequoia_test_path, all_char_vocab, all_feats_dict)

    return train_data, dev_data, test_data


def _extract_data(
        path: str,
        all_char_vocab: dict,
        all_feats: dict,
        max_len: int = 200,
        max_w: int = 20,
        poly: bool = False
):
    in_enc = []
    ends = []
    feats_number = []
    max_length = 0
    for sent in parse_incr(open(path, encoding="UTF-8")):
        current_chars = [all_char_vocab["<pad>"]]
        current_feats_number = []
        if len(sent) > max_length:
            max_length = len(sent)
        for tok in sent:
            for letter in tok["lemma"]:
                if letter not in all_char_vocab:
                    all_char_vocab[letter] = len(all_char_vocab)
                current_chars.append(all_char_vocab[letter])
            current_chars.append(all_char_vocab["<esp>"])
            if poly:
                if tok["feats"] is None or "Number" not in tok["feats"]:
                    current_feats_number.append(all_feats["<N/A>"])
                else:
                    number = tok["feats"]["Number"]
                    if number not in all_feats.keys():
                        all_feats[number] = len(all_feats)
                    current_feats_number.append(all_feats[number])
            else:
                if tok["feats"] is None or "Number" not in tok["feats"]:
                    current_feats_number.append(all_feats["<N/A>"])
                else:
                    number = tok["feats"]["Number"]
                    current_feats_number.append(all_feats.get(number, all_feats["<N/A>"]))
        current_ends = [idx for idx in range(len(current_chars)-1) if current_chars[idx+1] == all_char_vocab["<esp>"]]

        in_enc.append(current_chars[:-1])
        ends.append(current_ends)
        feats_number.append(current_feats_number)

    in_enc = pad_tensor(in_enc, max_len)
    ends = pad_tensor(ends, max_w)
    feats_number = pad_tensor(feats_number, max_w)

    return in_enc, ends, feats_number, all_char_vocab, all_feats


def preprocess_data(
        sequoia_train_path: str,
        sequoia_dev_path: str,
        sequoia_test_path: str,
        max_len: int,
        max_w: int,
        all_char_vocab: dict,
        all_feats: dict,
        poly: bool = False
):

    in_enc_train, ends_train, feats_number_train, all_char_vocab, _ = _extract_data(
        path=sequoia_train_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats,
        max_len=max_len,
        max_w=max_w
    )

    in_enc_dev, ends_dev, feats_number_dev, all_char_vocab, _ = _extract_data(
        path=sequoia_dev_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats,
        max_len=max_len,
        max_w=max_w
    )

    in_enc_test, ends_test, feats_number_test, all_char_vocab, _ = _extract_data(
        path=sequoia_test_path,
        all_char_vocab=all_char_vocab,
        all_feats=all_feats
    )

    train_data = {
        "in_enc_train": in_enc_train,
        "ends_train": ends_train,
        "feats_number_train": feats_number_train,
        "all_char_vocab": all_char_vocab,
        "all_feats": all_feats
    }

    dev_data = {
        "in_enc_dev": in_enc_dev,
        "ends_dev": ends_dev,
        "feats_number_dev": feats_number_dev,
        "all_char_vocab": all_char_vocab
    }

    test_data = {
        "in_enc_test": in_enc_test,
        "ends_test": ends_test,
        "feats_number_test": feats_number_test,
        "all_char_vocab": all_char_vocab,
        "all_feats": all_feats
    }

    return train_data, dev_data, test_data


def download_pred_file(
    model,
    sequoia_test_path,
    test_data,
    device,
    output_file_path
):

    with open(sequoia_test_path, 'r', encoding='utf-8') as f:
        sentences = list(parse_incr(f))
    idx_to_feat = {idx: feat for feat, idx in test_data["all_feats"].items()}

    model.eval()

    test_dataset = torch.utils.data.TensorDataset(test_data["in_enc_test"], test_data["ends_test"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        for i, (batch_X, batch_ends) in enumerate(test_loader):
            batch_X = batch_X.to(device).long()
            batch_ends = batch_ends.to(device).long()

            sent = sentences[i]

            with torch.no_grad():
                outputs = model(batch_X)
                batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, outputs.size(2))
                gathered = outputs.gather(1, batch_ends_expanded)
                gathered = gathered.view(-1, outputs.size(2))
                _, predicted_indices = torch.max(gathered, dim=1)

            predicted_feats = [idx_to_feat.get(idx.item(), '<N/A>') for idx in predicted_indices]

            for tok, pred_feat in zip(sent, predicted_feats):
                if pred_feat == "<N/A>":
                    if tok["feats"] and "Number" in tok["feats"]:
                        del tok["feats"]["Number"]
                else:
                    if tok["feats"] is None:
                        tok["feats"] = {}
                    tok["feats"]["Number"] = pred_feat

            out_f.write(sent.serialize())
            out_f.write('\n')


def download_pred_file_multitask(
    model,
    sequoia_test_path: str,
    test_data,
    all_feats_dict,
    device,
    output_file_path: str
):

    with open(sequoia_test_path, 'r', encoding='utf-8') as f:
        sentences = list(parse_incr(f))

    idx_to_feat_dict = {tm: {idx: feat for feat, idx in vocab.items()} for tm, vocab in all_feats_dict.items()}

    model.eval()

    test_dataset = torch.utils.data.TensorDataset(test_data["in_enc"], test_data["ends"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open(output_file_path, 'w', encoding='utf-8') as out_f:
        for i, (batch_X, batch_ends) in enumerate(test_loader):
            batch_X = batch_X.to(device).long()
            batch_ends = batch_ends.to(device).long()

            sent = sentences[i]

            with torch.no_grad():
                outputs = model(batch_X)

                predicted_feats_dict = {tm: [] for tm in outputs.keys()}

                for tm, output in outputs.items():
                    batch_ends_expanded = batch_ends.unsqueeze(2).expand(-1, -1, output.size(2))
                    gathered = output.gather(1, batch_ends_expanded)
                    gathered = gathered.view(-1, output.size(2))
                    _, predicted_indices = torch.max(gathered, dim=1)

                    predicted_feats_dict[tm] = [idx_to_feat_dict[tm].get(idx.item(), '<N/A>') for idx in predicted_indices]

            for tok, feats in zip(sent, zip(*[predicted_feats_dict[tm] for tm in outputs.keys()])):
                for tm, feat in zip(outputs.keys(), feats):
                    if feat == "<N/A>":
                        if tok["feats"] and tm in tok["feats"]:
                            del tok["feats"][tm]
                    else:
                        if tok["feats"] is None:
                            tok["feats"] = {}
                        tok["feats"][tm] = feat

            out_f.write(sent.serialize())
            out_f.write('\n')





