import json

import wandb
import torch
from torch.utils.data import DataLoader

import models
import data
import utils
import loggers


class Config:
    def __init__(self, config_path):
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
        print(self.config)


    def get_logger(self):
        try:
            logger_class = getattr(loggers, self.config['logging'])
            logger = logger_class(self.config)
            return logger
        except:
            return None

    def get_model(self):
        model_class = getattr(models, self.config['arch']['name'])
        model = model_class(**self.config['arch']['args'])
        if len(self.config['chkpt_path']) > 0:
            model.load_state_dict(torch.load(self.config['chkpt_path']))
        if self.config['device'] == 'cuda':
            model = torch.nn.DataParallel(model.to(self.device), device_ids=self.device_ids)
        elif self.config['device'] == 'cpu':
            model = model.to(self.device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f'Trainable parameters: {trainable_params}')
        return model

    def get_dataloaders(self):
        train_dataset_class = getattr(data, self.config['data']['train'])
        train_dataset = train_dataset_class('train', **self.config['data']['train_args'])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['bsz'],
            shuffle=self.config['data']['train_shuffle'],
            collate_fn=data.collate,
            num_workers=self.config['data']['num_workers'],
            drop_last=True
        )
        test_dataset_class = getattr(data, self.config['data']['test'])
        test_dataset = test_dataset_class('test', **self.config['data']['test_args'])
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config['data']['bsz'],
            shuffle=False,
            collate_fn=data.collate,
            num_workers=self.config['data']['num_workers'],
            drop_last=False
        )
        print('Dataloaders are ready')
        return train_dataloader, test_dataloader

    def get_optimizer(self, model_parameters):
        optimizer_class = getattr(torch.optim, self.config['optimizer']['name'])
        optimizer = optimizer_class(model_parameters, **self.config['optimizer']['args'])
        print(optimizer)
        return optimizer

    def get_criterion(self):
        criterion_class = getattr(torch.nn, self.config['criterion']['name'])
        criterion = criterion_class(**self.config['criterion']['args'])
        print(f'Using criterion: {criterion}')
        return criterion

    def get_decoder(self, id2sym):
        decoder_class = getattr(utils, self.config['decoder'])
        if self.config['decoder'] == 'BeamSearchDecoder':
            decoder = decoder_class(id2sym, self.config['beam_size'])
        else:
            decoder = decoder_class(id2sym)
        return decoder
