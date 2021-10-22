import os
import subprocess
import pandas as pd

from .base_dataset import BaseDataset


def download_lj_speech():
    if 'datasets' not in os.listdir('../'):
        subprocess.run(['mkdir', '../datasets'])
    if 'LJSpeech-1.1' in os.listdir('../datasets'):
        return '../datasets/LJSpeech-1.1'
    print('Download dataset...')
    subprocess.run(['wget', '-O', f'../datasets/LJSpeech.tar.bz2', 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'])
    print('Done!')
    print('Unarchive dataset...')
    subprocess.run(['tar', '-xf', '../datasets/LJSpeech.tar.bz2', '-C', '../datasets/'])
    print('Done!')
    return '../datasets/LJSpeech-1.1'


class LJSpeechDataset(BaseDataset):
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

        if len(path) == 0:
            path = download_lj_speech()

        self.task = task
        self.path = path
        meta_path = os.path.join(path, 'metadata.csv')
        if task == 'train':
            self.info = pd.read_csv(meta_path, sep='|', header=None).loc[:10000]
        elif task == 'test':
            self.info = pd.read_csv(meta_path, sep='|', header=None).loc[10001:]
        self.info.index = range(self.info.shape[0])
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

    def __len__(self):
        return self.info.shape[0]

    def __getitem__(self, idx):
        row = self.info.loc[idx]
        audio_path = os.path.join(self.path, 'wavs', row[0] + '.wav')
        text = str(row[2])
        return super().load_one(audio_path, text)
