import wandb
import torch
import random

from typing import Dict


class WandbLogger:
    def __init__(self, config):
        self.config = config
        self.init_wandb()

    def init_wandb(self):
        wandb.login(key=self.config['wandb_key'])
        wandb.init(project=self.config['wandb_project'])

    def log_metrics(self, info: Dict[str, float]):
        wandb.log(info)

    def log_spec_and_audio(self):
        train_rand = random.randint(0, len(self.train_dataset) - 1)
        test_rand = random.randint(0, len(self.test_dataset) - 1)
        train_spec, _, _, _, train_a, train_r, train_text = self.train_dataset[train_rand]
        test_spec, _, _, _, test_a, test_r, test_text = self.test_dataset[test_rand]
        if train_a == 0 or test_a == 0:
            return
        wandb.log(
            {
                'train_audio': wandb.Audio(train_a.squeeze().numpy(), sample_rate=train_r, caption=train_text),
                'test_audio': wandb.Audio(test_a.squeeze().numpy(), sample_rate=test_r, caption=test_text),
                'train_spec': wandb.Image(train_spec.squeeze(), caption=train_text),
                'test_spec': wandb.Image(test_spec.squeeze(), caption=test_text)
            }
        )

    def log_text(self, name: str, text: str):
        wandb.log({name: wandb.Html(text.replace('\n', '<br>'))})

    def init_datasets(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
