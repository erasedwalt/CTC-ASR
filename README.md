# ASR Homework

Подготовка среды описана в разделе "Подготовка среды", там и чекпоинты, и языковая модель тоже лежат.

## Результаты

В табличке приведены WER лучшей модели на тестовых сетах LibriSpeech. Языковые модели использовал с shallow fusion.

| Decoding  | test-clean | test-other |
| ------------- | ------------- | ------------- |
| Greedy | 15.5 | 34.9 |
| Vanilla Beam Search | 15.0 | 34.3 |
| 2-gram LM | 8.8 | 23.4 |
| 3-gram LM | 8.2 | 22.2 |
| 4-gram LM | 8.1 | 22.1 |

## Отчет

Лучший результат получился с QuartzNet 5x5. Из общего для всех стадий - лучшие (по характеристике качество/время обучения) результаты получились без регуляризации и аугментаций. Количество степов, которое обучал каждую стадию, можно посмотреть в логах. Конкретные стадии, конфиги и логи:

0) Попробовал потренировать на русский язык на датасете Golos от Сбера. Что-то получилось, но решил все-таки потренировать на английский.
1) Как рекомендовалось, начал с тренировки на LJSpeech, сначала взял большой QuartzNet 15x5, но он вообще не учился. Попробовал маленький 5x5, он уже очень бодро что-то выучил. Оптимизатор взял Adam с `lr=3e-4` и небольшим `warmup=100`. [Конфиг](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_1.json), [логи](https://wandb.ai/erasedwalt/QuartzNet-LJ/runs/342ixycg?workspace=user-erasedwalt), [логи](https://wandb.ai/erasedwalt/QuartzNet-LJ/runs/20nx2bvt?workspace=user-erasedwalt). Тут вторые логи - это продолжение обучения первых. А почему там WER так, будто учил с нуля, опишу чуть ниже (пока в логах нет аудио и спектрограмм, я их на следующих стадиях завезу).
2) Ну, вроде достаточно потренировал на LJ, решил перейти на LibriSpeech. Тренировал только на train-clean частях, валидировался на test-clean. Все параметры оптимизатора взял из статьи про архитектуру, кроме регуляризации и шедулера: NovoGrad с `lr=0.05`, `betas=[0.95, 0.5]` и `warmup=1000`. Но тут случился подвох: у меня для каждой тренировки раньше создавался алфавит в каком-то рандомном порядке ([вот эта злая строка](https://github.com/erasedwalt/asr-hw/blob/deeae75b784ce6b996df1a42f90c1d6cd29f7954/src/utils/text.py#L11)). Именно поэтому модель в тренировке не сразу начала выдавать более-менее результаты после LJ. [Конфиг](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_2.json), [логи](https://wandb.ai/erasedwalt/QuartzNet-LibriSpeech/runs/3gc0yavo?workspace=user-erasedwalt). К счастью, после этой стадии модель уже выдавала читаемый результат, так что я вручную "раскодировал" порядок алфавита.
3) Добавил к тренировочной части train-other-500, валидировался всё ещё на test-clean. Оптимизатор тот же, его настройки тоже, но при этом добавил `CosineAnnealingWithWarmup` шедулер, как в статье, с периодом в 300к степов. [Конфиг](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_3.json), [логи](https://wandb.ai/erasedwalt/QuartzNet-LibriSpeech/runs/2ajzgsc9?workspace=user-erasedwalt).
4) Просто поменял валидационный сет на test-other. [Конфиг](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_4.json), [логи](https://wandb.ai/erasedwalt/QuartzNet-LibriSpeech/runs/3805fx3o?workspace=user-erasedwalt), [логи](https://wandb.ai/erasedwalt/QuartzNet-LibriSpeech/runs/356tfqnk?workspace=user-erasedwalt) (тренировка падала). В принципе, тут уже получил конечный результат, который в табличке.
5) Бежит прямо сейчас, пока пишу. Добавил CommonVoice и аугментации, валидация на test-other. Увеличил период шедулера до 1М. Учится долго, но многообещающе по логам. [Конфиг](https://github.com/erasedwalt/asr-hw/blob/main/configs/qnet_stage_5.json), [логи](https://wandb.ai/erasedwalt/QuartzNet-LibriSpeech/runs/2v6mtycz?workspace=user-erasedwalt).

Кроме тренировки акустической модели, обучал и языковую модель [KenLM](https://kheafield.com/code/kenlm/). На сайте всё понятно расписано про тренировку. Обучал на нормализованном тексте [отсюда](https://www.openslr.org/11), причем предварительно ещё почистил от знаков препинаний и т.д.

Вроде бы, все implementation penalties выполнил. Из бонусных баллов сделал только языковую модель.

## Подготовка среды

Для удобства проверки сделал [ноутбук](https://colab.research.google.com/drive/1a21E7wNWBGRjOzb7paU_meLeJ3z10NQR?usp=sharing).

Для установки зависимостей достаточно сделать:

```
pip3 install -r requirements.txt
```

[Ссылка](https://www.dropbox.com/s/ga8zxnb7p6gtorm/qnet_5x5_22_wer_other_with_lm.pt?dl=0) на чекпоинт, ссылки на языковые модели: [2-gram](https://drive.google.com/uc?id=1LqEFoHQ1vq9ni_Fqtp5LB6w7CIhoekKY), [3-gram](https://drive.google.com/uc?id=1-1tIFykkoX6xhxNPn47ZjvLa_nptbJUs), [4-gram](https://drive.google.com/uc?id=1EzRB8qugZSO-RhOAJCQ16RIUHW-JfKw2). Языковые модели заархивированы, для использования их нужно разархивировать. Можно вот так:

```
import gzip
import shutil
with gzip.open('n-gram.arpa.gz', 'rb') as f_in:
    with open('n-gram.arpa', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
```
### Тестирование

Простестировать модель на test-clean и test-other можно с помощью `test.py`. Скрипту нужно передать путь к чекпоинту `--chkpt`, вид декодинга `--decoder` и номер девайса, на котором вычислять логиты `--device`. Также, если LibriSpeech тест сеты уже скачаны, то можно передать аргумент `--ds` - путь к датасету. Под капотом датасета я использую торчовский класс `LIBRISPEECH`, поэтому, если путь к датасету такой: `/path/to/LibriSpeech`, то нужно написать только `--ds /path/to`. Если же датасеты не скачаны, то код сам скачает их, аргумент `--ds` передавать не нужно. Про виды декодинга: если хочется Greedy, нужно написать `--decoder greedy`, если хочется ванильный Beam Search, нужно написать `--decoder vanilla`, если хочется LM Beam Search, то нужно написать `--decoder /path/to/lm`. Запускать скрипт нужно из `src`.

Скрипт запринтует результаты WER, CER и Loss для каждого сета. После каждого запуска скрипта в файлы `output_clean.txt` и `output_other.txt` запишутся результаты распознавания в виде `Truth: ...\n Pred: ...`
### Обучение

Обучение запускается скриптом `train.py`, ему в параметр `-c` нужно передать путь к конфигу. В конфигах не указаны пути к датасетам, то есть они по умолчанию будут скачиваться в папку `datasets` в корне (кроме CommonVoice, там, кажется, ссылка протухает). Если же датасеты уже скачаны, то нужно указать `path` для датасета в конфиге. Как это делать для LibriSpeech было выше, для LJ нужно указать так: `/path/to/LJSpeech-1.1`, а для CommonVoice так: `/path/to/cv-corpus-7.0-2021-07-21/en`. Для тестирования того, что скрипт работает, можно заиспользовать конфиг `qnet_stage_1.json`. За 3 эпохи выдает примерно 50 CER и 93 WER. Это на LJSpeech. С остальными конфигами не заработает, потому что у них есть зависимость в виде чекпоинта предыдущей стадии. То есть по идее мое обучение можно воспроизвести, по очереди запустив все конфиги (при этом каждый конфиг должен пробежать столько, сколько степов было сделано у меня). Но у меня был ещё косяк с алфавитом первые две стадии, то есть в идеале вообще воспроизвести и этот косяк тоже)). Если хочется использовать wandb, то в конфиге в `logger` нужно написать `WandbLogger` и добавить `wandb_key`.
