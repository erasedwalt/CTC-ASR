import torch
from torch.utils.data import Dataset
import torchaudio

import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

from utils import preprocess_text, get_maps
from augmentations import AudioAugmentator, SpecAugmentator


class BaseDataset(Dataset):
    """
    Base dataset class
    
    Args:
        sample_rate (int): Sample rate
        n_mels (int): Number of mels in MelSpectrogram
        augmantation (bool): Whether to use augmentation
        Other parameters see in augmentations.
    """
    def __init__(
            self,
            sample_rate: int = 16000,
            n_mels: int = 64,
            augmentation: bool = False,
            tempo: float = 0.2,
            pitch: int = 300,
            noise_factor: int = 10,
            time: int = 30,
            freq: int = 10
    ) -> None:

        self.sample_rate = sample_rate
        self.computer = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels
        )
        if augmentation:
            self.augment_audio = AudioAugmentator(tempo, pitch, noise_factor)
            self.augment_spec = SpecAugmentator(time, freq)
        else:
            self.augment_audio = None
            self.augment_spec = None

        _, self.sym2id = get_maps()

    def load_one(self, audio_path: str, text: str) -> Tuple[torch.Tensor, str, int]:
        text_preprocess = preprocess_text(text, self.sym2id)
        try:
            a, r = torchaudio.sox_effects.apply_effects_file(
                    audio_path,
                    [['channels', '1'],
                    ['rate', str(self.sample_rate)]]
            )
            if self.augment_audio:
                a, r = self.augment_audio(a, r)
            spec = self.computer(a)
            spec = self.computer(a).clamp(min=1e-5).log()
            # spec = spec / (torch.max(torch.abs(spec) + 1e-7))
            if self.augment_spec:
                spec = self.augment_spec(spec)
            return spec, text_preprocess, spec.shape[-1], len(text_preprocess), a, r, text
        except:
            return torch.tensor(0), torch.tensor(0), 0, 0, 0, 0, 0
