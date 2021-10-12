import torch


ALPHABET = '^йцукенгшщзхъфывапролджэячсмитьбю '
assert len(ALPHABET) == 34


def get_maps():
    id2sym = {x[0]: x[1] for x in enumerate(ALPHABET)}
    sym2id = {x[1]: x[0] for x in enumerate(ALPHABET)}
    return id2sym, sym2id


def preprocess_text(text, sym2id):
    text = list(filter(ALPHABET.__contains__, text.lower()))
    encoded = torch.tensor(list(map(lambda x: sym2id[x], text)))
    return encoded
