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
from models import QuartzNet
from data import LibriSpeechDataset, Collater
from utils import GreedyDecoder, BeamSearchDecoder, LMBeamSearchDecoder


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
        decoder,
        output_name: str
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
        with open(output_name, 'w') as fp:
            fp.write(''.join(output))
    print(f'WER: {aver_wer: .5f}\nCER: {aver_cer: .5f}\nLoss {aver_test_loss: .5f}')


def _parse_args():
    parser = argparse.ArgumentParser(description='Test argparser')
    parser.add_argument(
        '--chkpt',
        help='Path to model checkpoint',
        required=True
    )
    parser.add_argument(
        '--ds',
        help='Path to LibriSpeech dataset',
        required=False
    )
    parser.add_argument(
        '--decoder',
        help='Path to language model',
        required=False
    )
    parser.add_argument(
        '--device',
        help='GPU number',
        required=False
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.device:
        device = torch.device('cuda:' + args.device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2sym, sym2id = get_maps()

    model = QuartzNet(1, 5, 64, 28)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))
    model.to(device)
    criterion = nn.CTCLoss()

    if args.ds:
        path = args.ds
    else:
        path = ''
    dataset_clean = LibriSpeechDataset('test', path, 'test-clean')
    dataset_other = LibriSpeechDataset('test', path, 'test-other')
    dataloader_clean = DataLoader(dataset_clean, batch_size=32, drop_last=False, shuffle=False, collate_fn=Collater('QuartzNet'), num_workers=20)
    dataloader_other = DataLoader(dataset_other, batch_size=32, drop_last=False, shuffle=False, collate_fn=Collater('QuartzNet'), num_workers=20)

    if args.decoder == 'greedy':
        decoder = GreedyDecoder()
    elif args.decoder == 'vanilla':
        decoder = BeamSearchDecoder(beam_size=100)
    else:
        decoder = LMBeamSearchDecoder(args.decoder, beam_size=100)

    print('test-clean results:')
    evaluate(model, criterion, dataloader_clean, device, id2sym, decoder, 'output_clean.txt')
    print('\ntest-other results:')
    evaluate(model, criterion, dataloader_other, device, id2sym, decoder, 'output_other.txt')
