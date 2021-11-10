# CTC based ASR

This repository contains implementations of [Jasper](https://arxiv.org/pdf/1904.03288.pdf), [QuartzNet](https://arxiv.org/pdf/1910.10261.pdf), 
[Citrinet](https://arxiv.org/pdf/2104.01721.pdf) and pipeline for training and inference CTC-based ASR models. 

Three types of decoding are available: greedy argmax decoding, vanilla beam search and beam search with language model shallow fusion. 
[KenLM](https://kheafield.com/code/kenlm/) models used to shallow fusion. One can add your own implementation of CTC based ASR model and train it.

## Usage

### Setup

You should install dependent libraries with

```
pip3 install -r requirements.txt
```

and create folders `lm` and `chkpt`.

### Training

You should write config like [this](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_1.json). After this you
can start training with

```
python3 train.py -c /path/to/config.json
```
where decoding name is one of `greedy` or `vanilla`.
### Test

You can test your model on LibriSpeech dataset with 

```
python3 test.py --chkpt /path/to/chkpt --decoder <decoding-name or path to lm>
```

## Pretrained models

I've trained QuartzNet on LibriSpeech, LJ and CommonVoice to 8 WER with 4-gram KenLM on LibriSpeech test-clean.
One can download QuartzNet [here](https://www.dropbox.com/s/ga8zxnb7p6gtorm/qnet_5x5_22_wer_other_with_lm.pt?dl=0),
and can download LM's too: [2-gram](https://drive.google.com/uc?id=1LqEFoHQ1vq9ni_Fqtp5LB6w7CIhoekKY), 
[3-gram](https://drive.google.com/uc?id=1-1tIFykkoX6xhxNPn47ZjvLa_nptbJUs), [4-gram](https://drive.google.com/uc?id=1EzRB8qugZSO-RhOAJCQ16RIUHW-JfKw2).

One can reproduce this model by running `qnet_stage_i.json` configs step by step.
