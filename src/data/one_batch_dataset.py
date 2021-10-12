import torch
import torchaudio
from torch.utils.data import Dataset

from tqdm import tqdm

from .golos_dataset import read_manifest
from utils import preprocess_text, get_maps


class OneBatchDataset(Dataset):
    def __init__(self, task, length):
        self.length = length
        self.manifest = read_manifest('test')[:20]
        self.specs = []
        self.computer = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
        for idx in tqdm(range(len(self.manifest))):
            a, r = torchaudio.sox_effects.apply_effects_file(
                    '../datasets/test_opus/crowd/' + self.manifest[idx]['audio_filepath'],
                    effects = [['channels', '1'], ['rate', '16000']]
            )
            spec = self.computer(a)
            spec = spec / (torch.max(torch.abs(spec) + 1e-7))
            self.specs.append(spec)
        _, self.sym2id = get_maps()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % 20
        text = preprocess_text(self.manifest[idx]['text'], self.sym2id)
        return self.specs[idx], text, self.specs[idx].shape[-1], len(text)
