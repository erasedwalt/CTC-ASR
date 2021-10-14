import torch
import string


# ALPHABET = '^йцукенгшщзхъфывапролджэячсмитьбю '
# assert len(ALPHABET) == 34

ALPHABET = '^' + ''.join(set(string.ascii_letters.lower())) + ' ' # '^qwertyuiopasdfghjklzxcvbnm '
assert len(ALPHABET) == 28


def get_maps():
    id2sym = {x[0]: x[1] for x in enumerate(ALPHABET)}
    sym2id = {x[1]: x[0] for x in enumerate(ALPHABET)}
    return id2sym, sym2id


def preprocess_text(text, sym2id):
    def process_sym(x):
        if x in ALPHABET:
            return sym2id[x]
        return sym2id[' ']
    encoded = torch.tensor([process_sym(x) for x in text.lower()])
    return encoded
