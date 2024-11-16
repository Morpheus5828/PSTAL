"""
This script is a tools script which contains code for fit function
..moduleauthor:: Marius Thorre
"""

import torch
from conllu import parse_incr


def pad_tensor(X: torch.Tensor, max_len: int) -> torch.Tensor:
    result = torch.full((len(X), max_len), 0)
    for i, row in enumerate(X):
        x_len = min(max_len, len(X[i]))
        result[i, :x_len] = torch.LongTensor(X[i][:x_len])
    return result


def extract_data(
        path: str,
        all_form_vocab: dict,
        all_upos_vocab: dict,
        update_vocab: bool = True,
        max_len: int = 20
):
    max_length = 0
    all_form, all_upos = [], []
    for sent in parse_incr(open(path, encoding="UTF-8")):
        current_X = []
        current_Y = []
        if len(sent) > max_length:
            max_length = len(sent)
        for tok in sent:
            if tok["form"] not in all_form_vocab:
                if update_vocab:
                    all_form_vocab[tok["form"]] = len(all_form_vocab)
                else:
                    all_form_vocab[tok["form"]] = all_form_vocab.get("UNK_ID", 1)
            current_X.append(all_form_vocab.get(tok["form"], all_form_vocab["UNK_ID"]))

            if tok["upos"] not in all_upos_vocab:
                if update_vocab:
                    all_upos_vocab[tok["upos"]] = len(all_upos_vocab)
                else:
                    all_upos_vocab[tok["upos"]] = all_upos_vocab.get("UNK_ID", 1)
            current_Y.append(all_upos_vocab.get(tok["upos"], all_upos_vocab["UNK_ID"]))
        all_form.append(current_X)
        all_upos.append(current_Y)

    all_form = pad_tensor(all_form, max_len)
    all_upos = pad_tensor(all_upos, max_len)

    return all_form, all_upos, all_form_vocab, all_upos_vocab


def download_pred_file(
        model,
        test_file_path: str,
        form_vocab: list,
        upos_vocab: list,
        device,
        batch_size: int,
        output_file_path: str = 'predictions.conllu'
) -> None:

    model.eval()

    idx_to_upos = {idx: tag for tag, idx in upos_vocab.items()}

    with open(test_file_path, 'r', encoding='UTF-8') as f_in, open(output_file_path, 'w', encoding='UTF-8') as f_out:
        sentences = list(parse_incr(f_in))
        all_forms = [tok['form'] for sent in sentences for tok in sent]

        X = [form_vocab.get(form, form_vocab.get("UNK_ID", 1)) for form in all_forms]

        X_tensor = torch.tensor(X).to(device).long()

        num_tokens = X_tensor.size(0)
        num_batches = (num_tokens + batch_size - 1) // batch_size

        predictions = []
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_tokens)
                batch_X = X_tensor[start_idx:end_idx].unsqueeze(0)

                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=-1).squeeze(0).tolist()
                predictions.extend(preds)

        pred_idx = 0
        for sent in sentences:
            for tok in sent:
                pred_tag_idx = predictions[pred_idx]
                pred_tag = idx_to_upos.get(pred_tag_idx, "X")
                tok['upos'] = pred_tag
                pred_idx += 1
            f_out.write(sent.serialize())

    print(f"File saved at: {output_file_path}")