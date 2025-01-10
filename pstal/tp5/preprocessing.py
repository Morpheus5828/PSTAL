
import torch
from lib.conllulib import CoNLLUReader, TransBasedConfig, TransBasedSent


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
    max_len: int
):
    all_target = []
    all_input = []
    all_feats = []

    reader = CoNLLUReader(open(path))

    for sent in reader.readConllu():
        current_input = []
        for word in sent:
            word = word["form"]
            if word not in input_vocab.keys():
                input_vocab[word] = len(input_vocab)
            current_input.append(input_vocab[word])

        all_input.append(current_input)

        ########################

        target = TransBasedSent(sent, actions_only=True)

        b = None

        all_stack = []

        for idx, couple in enumerate(target.get_configs_oracle()):
            if b is None:
                b = couple[0].buff[0]
            all_stack.append(couple[0].stack.copy())
            #print(couple[0].buff, couple[0].stack)

        if len(all_stack[-1]) == 0:
            all_feats.append(torch.concat((torch.Tensor([b]), torch.zeros(3))))
        elif len(all_stack[-1]) == 1:
            all_stack[-1].append(0)
            all_stack[-1].append(0)
            all_feats.append(torch.concat((torch.Tensor([b]), torch.Tensor(all_stack[-1]))))
        elif len(all_stack[-1]) == 2:
            all_stack[-1].append(0)
            all_feats.append(torch.concat((torch.Tensor([b]), torch.Tensor(all_stack[-1]))))
        elif len(all_stack[-1]) == 3:
            all_feats.append(torch.concat((torch.Tensor([b]), torch.Tensor(all_stack[-1]))))

        ########################

        target = TransBasedSent(sent, actions_only=True)
        current_target = []
        for t in str(target).split(" "):
            if t not in target_vocab.keys():
                target_vocab[t] = len(target_vocab)
            current_target.append(target_vocab[t])

        all_target.append(current_target)


    return pad_tensor(all_input, max_len), torch.stack(all_feats), all_target



