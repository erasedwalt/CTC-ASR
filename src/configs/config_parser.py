import json
from typing import Tuple, Dict
from collections.abc import Generator

import wandb
import torch
import torch_optimizer as optim
from torch.utils.data import DataLoader, ConcatDataset

import models
import data
import utils
import loggers
import schedulers
from warmup_wrapper import WarmupWrapper
from data import Collater


class Config:
    """
    Config parser.

    Args:
        config_path (str): Path to config
    """
    def __init__(self, config_path: str) -> None:
        with open(config_path, 'r') as fp:
            self.config = json.load(fp)
        
        self.name = self.config['name']
        if self.config['device'] == 'cpu':
            self.device = 'cpu'
            print('Work on CPU')
        elif self.config['device'] == 'cuda':
            self.device_ids = self.config['device_ids']
            self.device = 'cuda:' + str(self.device_ids[0])
            print(f'Work on GPU: {self.device_ids}')
        self.clip_grad = self.config['clip_grad']
        self.eval_interval = self.config['eval_interval']
        self.best_wer = float(self.config['best_wer'])
        self.epochs = self.config['epochs']
        self.logger = None
        print(self.config)


    def get_logger(self):
        """
        Create a logger.
        """
        try:
            logger_class = getattr(loggers, self.config['logging'])
            logger = logger_class(self.config)
            self.logger = logger
            return logger
        except:
            self.logger = None
            return None

    def get_model(self) -> torch.nn.Module:
        """
        Create a model and maybe load weights from checkpoint.
        """
        model_class = getattr(models, self.config['arch']['name'])
        model = model_class(**self.config['arch']['args'])
        if len(self.config['chkpt_path']) > 0:
            model.load_state_dict(torch.load(self.config['chkpt_path'], map_location='cpu'))
        if self.config['device'] == 'cuda':
            model = torch.nn.DataParallel(model.to(self.device), device_ids=self.device_ids)
        elif self.config['device'] == 'cpu':
            model = model.to(self.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f'Trainable parameters: {trainable_params}')
        return model

    def get_dataloaders(self, task: str) -> DataLoader:
        """
        Create train and test dataloaders.
        """
        if task == 'train':
            drop_last = True
            shuffle = self.config['data']['train_shuffle']
        elif task == 'test':
            drop_last = False
            shuffle = False
        collater = Collater(self.config['arch']['name'])
        datasets = []
        for dataset_info in self.config['data'][task]:
            dataset_class = getattr(data, dataset_info['name'])
            dataset = dataset_class(task, **dataset_info['args'], **self.config['data'][f'{task}_args'])
            datasets.append(dataset)
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)

        dataloader = DataLoader(
            dataset,
            batch_size=self.config['data']['bsz'],
            shuffle=shuffle,
            collate_fn=collater,
            num_workers=self.config['data']['num_workers'],
            drop_last=drop_last
        )
        if self.logger:
            self.logger.init_datasets(task, dataset)
        return dataloader

    def get_optimizer(self, model_parameters: Generator) -> torch.optim.Optimizer:
        """
        Create a optimizer.
        """
        try:
            optimizer_class = getattr(optim, self.config['optimizer']['name'])
        except:
            optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        optimizer = optimizer_class(model_parameters, **self.config['optimizer']['args'])
        if 'warmup' in self.config['optimizer']:
            optimizer = WarmupWrapper(self.config['optimizer']['warmup'], optimizer, self.config['optimizer']['args']['lr'])
            scheduler = None
        elif 'scheduler' in self.config['optimizer']:
            scheduler_class = getattr(schedulers, self.config['optimizer']['scheduler']['name'])
            scheduler = scheduler_class(optimizer, **self.config['optimizer']['scheduler']['args'])
        print(optimizer)
        print(scheduler)
        return optimizer, scheduler

    def get_criterion(self) -> torch.nn.Module:
        """
        Create a criterion to train.
        """
        criterion_class = getattr(torch.nn, self.config['criterion']['name'])
        criterion = criterion_class(**self.config['criterion']['args'])
        print(f'Using criterion: {criterion}')
        return criterion

    def get_decoder(self):
        """
        Create a decoder.
        """
        decoder_class = getattr(utils, self.config['decoder']['name'])
        decoder = decoder_class(**self.config['decoder']['args'])
        return decoder
