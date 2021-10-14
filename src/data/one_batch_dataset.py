import pandas as pd
from tqdm import tqdm
import os

from .base_dataset import BaseDataset


class OneBatchDataset(BaseDataset):
    def __init__(self, task, length):
        self.main_ds = super().__init__()
        self.length = length
        self.info = pd.read_csv('../one_batch/metadata.csv', index_col=0)
        self.info.index = range(self.info.shape[0])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            row = self.info.loc[idx % 20]
            audio_path = os.path.join('../one_batch/wavs', row[0] + '.wav')
            text = str(row[2])
        except:
            print(row)
            exit(1)
        return super().load_one(audio_path, text)
