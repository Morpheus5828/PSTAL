import torch
from lib.conllulib import CoNLLUReader, TransBasedSent
from preprocessing import pad_tensor


def download_pred_file(model, device, corpus_path, vocab, max_len):
    model.eval()

    with torch.no_grad():
        reader = CoNLLUReader(open(corpus_path, 'r', encoding='utf-8'))

        for sent in reader.readConllu():
            all_inputs = []
            all_feats = []

            encoded_sent = []
            for token in sent:
                w = token["form"]
                if w not in vocab:
                    vocab[w] = len(vocab)
                encoded_sent.append(vocab[w])

            tbs = TransBasedSent(sent, actions_only=False)
            for config, action in tbs.get_configs_oracle():
                b = config.buff[0] if len(config.buff) > 0 else 0
                s0 = config.stack[-1] if len(config.stack) > 0 else 0
                s1 = config.stack[-2] if len(config.stack) > 1 else 0
                s2 = config.stack[-3] if len(config.stack) > 2 else 0

                all_feats.append([b, s0, s1, s2])
                all_inputs.append(encoded_sent)

            input_tensor = pad_tensor(all_inputs, max_len, device=device)
            feats_tensor = torch.tensor(all_feats, dtype=torch.long, device=device)

            output = model(input_tensor, feats_tensor).to(device)


