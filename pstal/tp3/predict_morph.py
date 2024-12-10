"""This script contains code to predict tp3 model
..moduleauthor:: Marius THORRE
"""

import torch
from conllu import parse_incr


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