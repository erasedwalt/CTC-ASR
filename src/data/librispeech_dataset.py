import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import LIBRISPEECH

import os
import subprocess

from utils import preprocess_text, get_maps
from augmentations import AudioAugmentator, SpecAugmentator


def download_librispeech(part):
    if 'datasets' not in os.listdir('../'):
        subprocess.run(['mkdir', '../datasets'])
    print('Download dataset...')
    subprocess.run(['wget', '-O', f'../datasets/{part}.tar.gz', f'https://www.openslr.org/resources/12/{part}.tar.gz'])
    print('Done!')
    print('Unarchive dataset...')
    subprocess.run(['tar', '-xf', f'../datasets/{part}.tar.gz', '-C', '../datasets/'])
    print('Done!')
    return '../datasets/'


class LibriSpeechDataset(Dataset):
    def __init__(
        self,
        task: str,
        path: str,
        part: str,
        sample_rate: int = 16000,
        n_mels: int = 64,
        augmentation: bool = False,
        tempo: float = 0.2,
        pitch: int = 300,
        noise_factor: int = 10,
        time: int = 30,
        freq: int = 10
    ) -> None:

        if len(path) == 0:
            path = download_librispeech(part)

        self.dataset = LIBRISPEECH(path, part)
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
        self.effects = [['channels', '1'], ['rate', str(sample_rate)]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            a, r, text, _, _, _ = self.dataset[idx]
            text_preprocess = preprocess_text(text, self.sym2id)

            a, r = torchaudio.sox_effects.apply_effects_tensor(a, r, self.effects)
            if self.augment_audio:
                a, r = self.augment_audio(a, r)
            spec = self.computer(a).clamp(min=1e-5).log()
            # spec = spec / (torch.max(torch.abs(spec) + 1e-7))

            if self.augment_spec:
                spec = self.augment_spec(spec)
            return spec, text_preprocess, spec.shape[-1], len(text_preprocess), a, r, text
        except:
            return torch.tensor(0), torch.tensor(0), 0, 0, 0, 0, 0
