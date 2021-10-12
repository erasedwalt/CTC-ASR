import wandb
import torch

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

    def log_audio(self, name: str, audio: torch.Tensor, rate: int):
        wandb.log({name: wandb.Audio(audio.squeeze().numpy(), sample_rate=rate)})

    def log_spec(self, name: str, spec: torch.Tensor):
        wandb.log({name: wandb.Image(spec.squeeze().numpy())})

    def log_text(self, name: str, text: str):
        wandb.log({name: wandb.Html(text.replace('\n', '<br>'))})
