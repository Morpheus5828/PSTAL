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
):
    model.eval()
    rev_encoding_dict = {v: k for k, v in encoding_dict.items()}

    idx_word = 0

    with open(sequoia_test_path, "r", encoding="utf-8") as f_in, \
         open(output_file_path, "w", encoding="utf-8") as f_out:

        sentences = list(parse_incr(f_in))

        for sentence in sentences:
            for token in sentence:
                if token["upos"] in upos_list:
                    emb = inputs_test[idx_word].unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(emb)
                        _, pred_idx = torch.max(logits, dim=1)
                        pred_label = rev_encoding_dict[pred_idx.item()]
                    idx_word += 1

                    token["frsemcor:noun"] = pred_label
                else:
                    token["frsemcor:noun"] = "*"

            f_out.write(sentence.serialize())
            f_out.write("\n")
