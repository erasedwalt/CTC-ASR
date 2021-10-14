import torch

from typing import Dict
from collections import defaultdict
from tqdm import tqdm


class GreedyDecoder:
    def __init__(self, id2sym: Dict[int, str]):
        self.id2sym = id2sym

    def __call__(self, logits, input_lens):
        return greedy_decoder(logits, input_lens, self.id2sym)


class BeamSearchDecoder:
    def __init__(self, id2sym: Dict[int, str], beam_size: int = 100):
        self.beam_size = beam_size
        self.id2sym = id2sym

    def __call__(self, logits, input_lens):
        logits = logits.transpose(1, 2)
        texts = []
        for i, line in enumerate(tqdm(logits)):
            line = line[:input_lens[i]]
            text = beam_search(line, self.beam_size, self.id2sym)[0][0]
            texts.append(text)
        return texts


def greedy_decoder(logits: torch.Tensor, input_lens: torch.Tensor, id2sym: Dict[int, str]) -> str:
    argmaxes = logits.argmax(dim=1).numpy()
    texts = []
    for j, sentence in enumerate(argmaxes):
        sentence = sentence[:input_lens[j]]
        text = []
        for i, idx in enumerate(sentence):
            if idx == 0:
                continue
            if i > 0 and sentence[i] == sentence[i-1]:
                continue
            text.append(sentence[i])
        text = ''.join(map(lambda x: id2sym[x], text))
        texts.append(text)
    return texts


def beam_search(probs, beam_size, id2sym):
  paths = {('', '^'): 1.0}
  for next_char_probs in probs:
    paths = extend_and_merge(next_char_probs, paths, id2sym)
    paths = truncate_beam(paths, beam_size)
  return [(prefix, score) for (prefix, _), score in sorted(paths.items(), key=lambda x: x[1], reverse=True)]


def extend_and_merge(next_char_probs, src_paths, id2sym):
  new_paths = defaultdict(float)
  for next_char_ind, next_char_prob in enumerate(next_char_probs):
    next_char = id2sym[next_char_ind]
    for (text, last_char), path_prob in src_paths.items():
      new_prefix = text if next_char == last_char else text + next_char
      new_prefix = new_prefix.replace('^', '')
      new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
  return new_paths


def truncate_beam(paths, beam_size):
  return dict(sorted(paths.items(), key=lambda x: x[1], reverse=True)[:beam_size])
