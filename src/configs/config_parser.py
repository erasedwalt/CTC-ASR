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

    def get_dataloaders(self) -> Tuple[DataLoader]:
        """
        Create train and test dataloaders.
        """
        collater = Collater(self.config['arch']['name'])
        train_datasets = []
        for train_dataset_info in self.config['data']['train']:
            train_dataset_class = getattr(data, train_dataset_info['name'])
            train_dataset = train_dataset_class('train', **train_dataset_info['args'], **self.config['data']['train_args'])
            train_datasets.append(train_dataset)
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = ConcatDataset(train_datasets)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['bsz'],
            shuffle=self.config['data']['train_shuffle'],
            collate_fn=collater,
            num_workers=self.config['data']['num_workers'],
            drop_last=True
        )

        test_datasets = []
        for test_dataset_info in self.config['data']['test']:
            test_dataset_class = getattr(data, test_dataset_info['name'])
            test_dataset = test_dataset_class('test', **test_dataset_info['args'], **self.config['data']['test_args'])
            test_datasets.append(test_dataset)
        if  len(test_datasets) == 1:
            test_dataset = test_datasets[0]
        else:
            test_dataset = ConcatDataset(test_datasets)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['bsz'],
            shuffle=False,
            collate_fn=collater,
            num_workers=self.config['data']['num_workers'],
            drop_last=False
        )
        if self.logger:
            self.logger.init_datasets(train_dataset, test_dataset)
        print('Dataloaders are ready')
        return train_dataloader, test_dataloader

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
