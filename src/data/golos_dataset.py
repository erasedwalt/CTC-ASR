import torch

import os
import json
from tqdm import tqdm
from typing import Tuple

from .base_dataset import BaseDataset


def read_manifest(task: str):
    if task == 'train':
        path = '../datasets/train_opus/manifest.jsonl'
    elif task == 'test':
        path = '../datasets/test_opus/crowd/manifest.jsonl'
    with open(path, 'r') as fp:
        jsons = list(fp)
    dataset_info = []
    for json_line in tqdm(jsons):
        dataset_info.append(json.loads(json_line))
    dataset_info = list(filter(lambda x: x['duration'] <= 10., dataset_info))
    return dataset_info


class GolosDataset(BaseDataset):
    def __init__(
            self,
            task: str,
            sample_rate: int = 16000,
            n_mels: int = 64,
            augmentation: bool = False,
            tempo: float = 0.2,
            pitch: int = 300,
            noise_factor: int = 10,
            time: int = 30,
            freq: int = 10
    ) -> None:

        self.manifest = read_manifest(task)
        self.task = task
        if task == 'train':
            self.audios_path = '../datasets/train_opus/'
        elif task == 'test':
            self.audios_path = '../datasets/test_opus/crowd'
        if task == 'test':
            augmentation = False
        super().__init__(
            sample_rate=sample_rate,
            n_mels=n_mels,
            augmentation=augmentation,
            tempo=tempo,
            pitch=pitch,
            time=time,
            freq=freq
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        row = self.manifest[idx]
        audio_path = os.path.join(self.audios_path, row['audio_filepath'])
        return super().load_one(audio_path, row['text'])
