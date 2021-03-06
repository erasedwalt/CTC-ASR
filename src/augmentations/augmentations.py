import torch
from torch.distributions import Normal
import torchaudio
import numpy as np

from typing import Tuple


class AudioAugmentator:
    """
    Audio tensor augmentation class.

    Args:
        tempo (float): Range in which to time stretch audio
        pitch (int): Range in which to pitch audio
        noise_factor (int): Factor in which noise have to be quiter than audio
    """
    def __init__(self, tempo: float, pitch: int, noise_factor: int) -> None:
        self.tempo = tempo
        self.pitch = pitch
        self.noise_factor = noise_factor
        self.noiser = Normal(0, 0.05)

    def __call__(self, audio: torch.Tensor, r: int) -> Tuple[torch.Tensor, int]:
        effects = [
                ['tempo', str(1 + np.random.uniform(-1, 1) * self.tempo)],
                ['pitch', str(int(np.random.uniform(-1, 1) * self.pitch))],
                ['rate', str(r)]
        ]
        audio, r = torchaudio.sox_effects.apply_effects_tensor(audio, r, effects)

        # add noise
        audio_norm = torch.norm(audio)
        noise_norm = (torch.rand(1) * audio_norm) / self.noise_factor
        noise = self.noiser.sample(audio.shape)
        noise = (noise / torch.norm(noise)) * noise_norm
        audio += noise
        return audio, r


class SpecAugmentator:
    """
    Spectrogram tensor augmentation class.

    Args:
        time (int): Range of number of channels in time axis to be zeroed
        freq (int): Range of number of channels in frequency axis to be zeroed
    """
    def __init__(self, time: int, freq: int) -> None:
        self.time = time
        self.freq = freq

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        spec = torchaudio.transforms.TimeMasking(self.time)(spec)
        spec = torchaudio.transforms.FrequencyMasking(self.freq)(spec)
        return spec
