# CUDA_VISIBLE_DEVICES=0 python run.py train \
#     --train-src=./data/iwslt/fr-en/train.fr-en.fr \
#     --train-tgt=./data/iwslt/fr-en/train.fr-en.en \
#     --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr \
#     --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en \
#     --test-src=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr \
#     --test-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en \
#     --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json \
#     --cuda
CUDA_VISIBLE_DEVICES=0 python run.py train \
    --train-src=./data/iwslt/fr-en/train.fr-en.fr \
    --train-tgt=./data/iwslt/fr-en/train.fr-en.en \
    --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr \
    --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en \
    --test-src=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr \
    --test-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en \
    --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json \
    --cuda --order-name=rarity --pacing-name=linear --save-to=linear.bin --ignore-test-bleu 1
# CUDA_VISIBLE_DEVICES=0 python run.py train \
#     --train-src=./data/iwslt/fr-en/train.fr-en.fr \
#     --train-tgt=./data/iwslt/fr-en/train.fr-en.en \
#     --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr \
#     --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en \
#     --test-src=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr \
#     --test-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en \
#     --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json \
#     --cuda --order-name=rarity --pacing-name=root --save-to=root.bin