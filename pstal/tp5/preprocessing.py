import torch
from lib.conllulib import CoNLLUReader, TransBasedSent


def pad_tensor(X: list, max_len: int, padding_value: int = 0, device=None) -> torch.Tensor:
    result = torch.full((len(X), max_len), padding_value, device=device, dtype=torch.long)
    for i, row in enumerate(X):
        x_len = min(max_len, len(row))
        result[i, :x_len] = torch.tensor(row[:x_len], device=device, dtype=torch.long)
    return result


def prepare_data(
    path: str,
    input_vocab: dict,
    target_vocab: dict,
    max_len: int,
):
    all_inputs = []
    all_feats = []
    all_actions = []

    reader = CoNLLUReader(open(path, 'r', encoding='utf-8'))

    for sent in reader.readConllu():
        encoded_sent = []
        for token in sent:
            w = token["form"]
            if w not in input_vocab:
                input_vocab[w] = len(input_vocab)
            encoded_sent.append(input_vocab[w])

        tbs = TransBasedSent(sent, actions_only=False)
        for config, action in tbs.get_configs_oracle():
            b = config.buff[0] if len(config.buff) > 0 else 0
            s0 = config.stack[-1] if len(config.stack) > 0 else 0
            s1 = config.stack[-2] if len(config.stack) > 1 else 0
            s2 = config.stack[-3] if len(config.stack) > 2 else 0

            if b >= max_len or s0 >= max_len or s1 >= max_len or s2 >= max_len:
                continue

            all_feats.append([b, s0, s1, s2])
            act_str = str(action)
            if act_str not in target_vocab:
                target_vocab[act_str] = len(target_vocab)
            all_actions.append(target_vocab[act_str])

            all_inputs.append(encoded_sent)

    inputs_tensor = pad_tensor(all_inputs, max_len)
    feats_tensor = torch.tensor(all_feats, dtype=torch.long)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    return inputs_tensor, feats_tensor, actions_tensor, target_vocab, input_vocab

