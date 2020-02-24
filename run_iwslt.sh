#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --vocab=iwslt_vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en outputs/test_outputs.txt --cuda
elif [ "$1" = "dev" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en outputs/test_outputs_dev.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./data/wmt16_en_de/train.tok.clean.bpe.32000.en --train-tgt=./data/wmt16_en_de/train.tok.clean.bpe.32000.de --dev-src=./data/wmt16_en_de/newstest2013.tok.bpe.32000.en --dev-tgt=./data/wmt16_en_de/newstest2013.tok.bpe.32000.de --vocab=wmt_vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
