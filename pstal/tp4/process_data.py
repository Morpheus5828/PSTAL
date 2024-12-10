"""This script is for process data before training model
..moduleauthor::Marius THORRE
"""

from conllu import parse_incr


def extract_data(
        sequoia_train_path: str,
        tokenize,
        upos_list: list = ["NOUN", "PROPN", "NUM"],
):
    input, target = [], []
    for sent in parse_incr(open(sequoia_train_path, encoding="UTF-8")):
        current_target = []
        for tok in sent:
            if tok["upos"] in upos_list:
                current_target.append(tok["frsemcor:noun"])
            else:
                current_target.append("*")
        sent = [tok["form"] for tok in sent]
        dico = {}
        tok_sent = tokenize(sent, is_split_into_words=True, return_tensors='pt')
        for element in tok_sent.word_ids():
            if element is not None:
                if element not in dico.keys():
                    dico[element] = tok_sent["input_ids"].squeeze()[element]
                dico[element] = (dico[element] + tok_sent["input_ids"].squeeze()[element]) / 2
        input.append(list(dico.values()))
        target.append(current_target)

    return input, target
