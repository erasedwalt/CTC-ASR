import wandb
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

import argparse
from tqdm import tqdm
from typing import Dict

from utils import get_maps
from metrics import wer, cer
from configs import Config


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train(
        epochs: int,
        model: nn.Module, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        train_dl: DataLoader, 
        test_dl: DataLoader,
        logger,
        device: str,
        clip: float,
        id2sym: Dict[int, str],
        eval_interval: int,
        best_wer: float,
        exp_name: str,
        decoder
) -> None:

    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        for i, (specs, texts, input_lens, target_lens) in enumerate(tqdm(train_dl)):
            try:
                optimizer.zero_grad()

                specs = specs.to(device)
                preds = model(specs)

                loss = criterion(preds.permute(2, 0, 1), texts, input_lens, target_lens)
                loss.backward()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

                if bool(torch.isnan(norm)):
                    continue
                optimizer.step()

                if scheduler:
                    scheduler.step()
                    lr = scheduler.get_lr()[0]
                else:
                    lr = optimizer.rate()

                to_log = {'train_loss': float(loss.item()), 'grad_norm': float(norm), 'lr': float(lr)}
                if logger:
                    logger.log_metrics(to_log)
                    if i == 0:
                        logger.log_spec_and_audio()
                else:
                    print(to_log)

                if (i + 1) % eval_interval == 0:
                    best_wer = evaluate(model, criterion, test_dl, logger, best_wer, device, id2sym,
                                        exp_name, decoder)
                    model.train()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("OOM on batch. Skipping batch.")
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        logger,
        best_wer: float,
        device: str,
        id2sym: Dict[int, str],
        exp_name: str,
        decoder
) -> float:

    model.eval()
    total_wer = []
    total_cer = []
    aver_test_loss = 0.
    with torch.no_grad():
        for j, (specs, texts, input_lens, target_lens) in enumerate(tqdm(dataloader)):
            specs = specs.to(device)
            preds = model(specs)

            loss = criterion(preds.permute(2, 0, 1), texts, input_lens, target_lens)
            aver_test_loss += loss.item()

            rand = int(np.random.random_integers(0, texts.shape[0], 1))
            pred_texts = decoder(preds.detach().cpu(), input_lens)
 
            for i in range(texts.shape[0]):
                text = ''.join(map(lambda x: id2sym[x], texts[i].numpy()))
                text = text[:int(target_lens[i])]
                pred_text = pred_texts[i]
                total_wer.append(wer(text, pred_text))
                total_cer.append(cer(text, pred_text))
                if i == rand and j % 100 == 0:
                    logging_text = []
                    logging_text.append('------------------------------------')
                    logging_text.append('Truth:\n\t' + text)
                    logging_text.append('Pred:\n\t' + pred_text)
                    logging_text = '\n'.join(logging_text)
                    if logger:
                        logger.log_text('text', logging_text)
                    else:
                        print(logging_text)

        aver_wer = np.mean(total_wer)
        aver_cer = np.mean(total_cer)
        aver_test_loss /= len(dataloader)
        if aver_wer < best_wer:
            best_wer = aver_wer
            torch.save(model.module.state_dict(), f'../chkpt/{exp_name}_best.pt')
        torch.save(model.module.state_dict(), f'../chkpt/{exp_name}_last.pt')
        to_log = {'wer': aver_wer, 'cer': aver_cer, 'best_wer': best_wer, 'test_loss': aver_test_loss}
        if logger:
            logger.log_metrics(to_log)
        else:
            print(to_log)
    return float(best_wer)


def _parse_args():
    parser = argparse.ArgumentParser(description='Train argparser')
    parser.add_argument(
        '-c', '--config',
        help='Path to config',
        required=True
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    config = Config(args.config)

    id2sym, sym2id = get_maps()

    model = config.get_model()
    optimizer, scheduler = config.get_optimizer(model.parameters())
    criterion = config.get_criterion()
    logger = config.get_logger()
    train_dataloader, test_dataloader = config.get_dataloaders()
    decoder = config.get_decoder(id2sym)

    train(config.epochs, model, optimizer, scheduler, criterion, train_dataloader,
          test_dataloader, logger, config.device, config.clip_grad, id2sym,
          config.eval_interval, config.best_wer, config.name, decoder)
