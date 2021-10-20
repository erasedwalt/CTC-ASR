import os
import pandas as pd

from .base_dataset import BaseDataset


class CommonVoiceDataset(BaseDataset):
    def __init__(
        self,
        task: str,
        path: str,
        sample_rate: int = 16000,
        n_mels: int = 64,
        augmentation: bool = False,
        tempo: float = 0.2,
        pitch: int = 300,
        noise_factor: int = 10,
        time: int = 30,
        freq: int = 10 
    ) -> None:

        if task == 'train':
            self.info = pd.read_csv(os.path.join(path, 'train.tsv'), sep='\t')
        elif task == 'test':
            self.info = pd.read_csv(os.path.join(path, 'test.tsv'), sep='\t')
        self.info.index = range(self.info.shape[0])
        self.path = path

        super().__init__(
            sample_rate=sample_rate,
            n_mels=n_mels,
            augmentation=augmentation,
            tempo=tempo,
            pitch=pitch,
            time=time,
            freq=freq
        )

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, idx):
        row = self.info.loc[idx]
        audio_path = os.path.join(self.path, 'clips', row['path'])
        text = row['sentence']
        return super().load_one(audio_path, text)
