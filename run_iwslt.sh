#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --vocab=iwslt_vocab.json --word_freq=islt_word_freq.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en outputs/test_outputs.txt --cuda
elif [ "$1" = "dev" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en outputs/test_outputs_dev.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json
elif [ "$1" = "train_local_rarity" ]; then
	python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json --order_name rarity
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "download" ]; then
	python -c $"from torchnlp.datasets import iwslt_dataset
train, dev, test = iwslt_dataset(language_extensions=['fr', 'en'], train=True, dev=True, test=True)"
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en iwslt_vocab.json iwslt_word_freq.json --freq-cutoff 5
else
	echo "Invalid Option Selected"
fi
