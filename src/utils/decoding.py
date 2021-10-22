import torch
from fast_ctc_decode import beam_search
from pyctcdecode import build_ctcdecoder

from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm

from .text import ALPHABET, get_maps


class GreedyDecoder:
    """
    Simple argmax decoder
    """
    def __init__(self) -> None:
        self.id2sym, _ = get_maps()

    def __call__(self, logits: torch.Tensor, input_lens: torch.Tensor) -> List[str]:
        return greedy_decoder(logits, input_lens, self.id2sym)


class BeamSearchDecoder:
    """
    Vanilla beam search decoder

    Args:
        beam_size (int): Beam size
    """
    def __init__(self, beam_size: int = 100) -> None:
        self.beam_size = beam_size

    def __call__(self, logits: torch.Tensor, input_lens: torch.Tensor) -> List[str]:
        logits = logits.transpose(1, 2)
        texts = []
        for i, line in enumerate(logits):
            line = line[:input_lens[i]]
            text = beam_search(line.exp().numpy(), ALPHABET, beam_size=self.beam_size)[0]
            texts.append(text)
        return texts


class LMBeamSearchDecoder:
    """
    Beam search with LM shallow fusion decoder

    Args:
        lm_path (str): Path to KenLM language model
        beam_size (int): Beam size
        alpha (float, optional): Alpha in shallow fusion
        beta (float, optional): Beat in shallow fusion
    """
    def __init__(self, lm_path: str, beam_size: int, alpha: float = 0.5, beta: float = 1.0) -> None:
        print('Loading LM...')
        self.decoder = build_ctcdecoder(
            list(ALPHABET)[1:],
            lm_path,
            alpha=alpha,
            beta=beta
        )
        print('Done!')

    def __call__(self, logits: torch.Tensor, input_lens: torch.Tensor) -> List[str]:
        logits = logits.transpose(1, 2)
        # some strange transforms to get right input for decoder
        logits = torch.cat([logits[:, :, 1:], logits[:, :, 0].unsqueeze(2)], dim=2).numpy()
        texts = []
        for i, line in enumerate(logits):
            line = line[:input_lens[i]]
            text = self.decoder.decode(line)
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
