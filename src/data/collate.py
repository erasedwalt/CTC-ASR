import torch

from math import ceil
from typing import Tuple

class Collater:
    """
    Collater that takes into account model time downsampling

    Args:
        model_name (str): Model name
    """
    def __init__(self, model_name):
        assert model_name in ['Jasper', 'QuartzNet', 'Citrinet']

        if model_name in ['Jasper', 'QuartzNet']:
            self.downsample = 2
        elif model_name == 'Citrinet':
            self.downsample = 4

    def __call__(self, batch: list) -> Tuple[torch.Tensor]:
        max_input_len = max(batch, key=lambda x: x[2])[2]
        max_target_len = max(batch, key=lambda x: x[3])[3]
        specs = []
        texts = []
        input_lens = []
        target_lens = []
        for i in range(len(batch)):
            if batch[i][2] == 0:
                continue
            spec = batch[i][0]
            spec = torch.cat([spec, torch.zeros(1, 64, max_input_len - spec.shape[2])], dim=2)
            specs.append(spec)
            text = batch[i][1]
            text = torch.cat([text, torch.zeros(max_target_len - text.shape[0])])
            texts.append(text.unsqueeze(0))
            input_lens.append(ceil(batch[i][2] / self.downsample))
            target_lens.append(batch[i][3])
        specs = torch.cat(specs)
        texts = torch.cat(texts)
        input_lens = torch.tensor(input_lens)
        target_lens = torch.tensor(target_lens)
        return specs, texts, input_lens, target_lens
