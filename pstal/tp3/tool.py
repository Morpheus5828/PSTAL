"""
This script is a tools script which contains code for fit function
..moduleauthor:: Marius Thorre
"""

import torch
from pstal.tp1.sequoia.bin.tool import pad_tensor
from torch.utils.data import TensorDataset, DataLoader
from conllu import parse_incr


def extract_data(
        path: str,
        all_char_vocab: dict,
        all_feats: dict,
        max_len: int = 200,
        max_w: int = 20
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
            # Traitement du Number
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

    return in_enc, ends, feats_number, all_char_vocab


def download_pred_file(
    model,
    sentences,
    in_enc_test,
    ends_test,
    all_feats,
    device,
    output_file_path
):
    idx_to_feat = {idx: feat for feat, idx in all_feats.items()}

    model.eval()

    test_dataset = torch.utils.data.TensorDataset(in_enc_test, ends_test)
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




