"""
This script download a prediction file
Authors : Marius THORRE
"""

import torch
from conllu import parse_incr


def download_pred_file(
    model,
    inputs_test,
    device,
    output_file_path,
    encoding_dict,
    sequoia_test_path,
    upos_list=["NOUN", "PROPN", "NUM"]
) -> None:
    model.eval()
    rev_encoding_dict = {v: k for k, v in encoding_dict.items()}
    idx_word = 0
    with open(sequoia_test_path, "r", encoding="utf-8") as f_in, \
         open(output_file_path, "w", encoding="utf-8") as f_out:
        sentences = list(parse_incr(f_in))
        for sentence in sentences:
            for token in sentence:
                if token["upos"] in upos_list and token["frsemcor:noun"] != "*":
                    emb = inputs_test[idx_word].unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(emb)
                        _, pred_idx = torch.max(logits, dim=1)
                        pred_label = rev_encoding_dict[pred_idx.item()]
                    idx_word += 1
                    token["frsemcor:noun"] = pred_label
            f_out.write(sentence.serialize())
            f_out.write("\n")


def download_pred_file_ft(
        model,
        test_loader,
        encoding_dict,
        sequoia_test_path,
        output_file_path,
        device,
        upos_list=["NOUN", "PROPN", "NUM"]):
    model.eval()
    rev_encoding_dict = {v: k for k, v in encoding_dict.items()}

    with open(sequoia_test_path, "r", encoding="utf-8") as f_in:
        sentences = list(parse_incr(f_in))

    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            _, pred_idxs = torch.max(outputs, dim=2)
            all_predictions.extend(pred_idxs.cpu().numpy())

    with open(output_file_path, "w", encoding="utf-8") as f_out:
        pred_idx_iter = iter(all_predictions)
        for sentence in sentences:
            predictions = next(pred_idx_iter)
            for idx, token in enumerate(sentence):
                if token["upos"] in upos_list and token.get("frsemcor:noun", "*") != "*":
                    if idx < len(predictions):
                        pred_label_idx = predictions[idx]
                        pred_label = rev_encoding_dict.get(pred_label_idx, "*")
                        token["frsemcor:noun"] = pred_label
                    else:
                        token["frsemcor:noun"] = "*"
            f_out.write(sentence.serialize())
            f_out.write("\n")





