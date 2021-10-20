import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

import argparse
from tqdm import tqdm
from typing import Dict

from configs import Config
from metrics import wer, cer
from utils import get_maps


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        dataloader: DataLoader,
        device: str,
        id2sym: Dict[int, str],
        decoder
) -> float:

    model.eval()
    total_wer = []
    total_cer = []
    aver_test_loss = 0.
    output = []
    with torch.no_grad():
        for j, (specs, texts, input_lens, target_lens) in enumerate(tqdm(dataloader)):
            specs = specs.to(device)
            preds = model(specs)

            loss = criterion(preds.permute(2, 0, 1), texts, input_lens, target_lens)
            aver_test_loss += loss.item()

            pred_texts = decoder(preds.detach().cpu(), input_lens)
            for i in range(texts.shape[0]):
                text = ''.join(map(lambda x: id2sym[x], texts[i].numpy()))
                text = text[:int(target_lens[i])]
                pred_text = pred_texts[i]
                total_wer.append(wer(text, pred_text))
                total_cer.append(cer(text, pred_text))
                output.append('---------------------------------------------------\n')
                output.append('Truth:\n\t' + text + '\n' + 'Pred:\n\t' + pred_text + '\n')

        aver_wer = np.mean(total_wer)
        aver_cer = np.mean(total_cer)
        aver_test_loss /= len(dataloader)
        with open('output.txt', 'w') as fp:
            fp.write(''.join(output))
    print(f'WER: {aver_wer: .3f}\nCER: {aver_cer: .3f}\nLoss {aver_test_loss: .3f}')


def _parse_args():
    parser = argparse.ArgumentParser(description='Test argparser')
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
    _, _ = config.get_optimizer(model.parameters())
    criterion = config.get_criterion()
    _ = config.get_logger()
    _, test_dataloader = config.get_dataloaders()
    decoder = config.get_decoder()

    evaluate(model, criterion, test_dataloader, config.device, id2sym, decoder)
