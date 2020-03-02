# curriculum-nmt

# Setup

0. Create conda environment: `conda create --name curriculum_nmt python=3.7`
1. Install requirements in `requirements.txt`
1. Run `bash run_iwslt.sh download` to download the IWSLT dataset
1. Run `bash run_iwslt.sh vocab` to generate vocab files. This generates a
    `iwslt_vocab.json` and `iwslt_word_freq.json`

# Usage
1. Train the model locally on IWSLT with `bash run_iwslt.sh train_local` (with "none" ordering)

2. Train the model with desired scoring and pacing functions locally on IWSLT e.g. `bash run_iwslt.sh train_local rarity linear` (with "rarity" ordering and "linear" pacing. see `scoring.py` and `pacing.py` for more options)

# References
1. Fine-Tuning by Curriculum Learning for Non-Autoregressive
Neural Machine Translation [arXiv](https://arxiv.org/abs/1911.08717)
2. On The Power of Curriculum Learning in Training Deep Networks [arXiv](https://arxiv.org/abs/1904.03626) [code](https://github.com/GuyHacohen/curriculum_learning)
3. Competence-based Curriculum Learning for Neural Machine Translation [arXiv](https://arxiv.org/abs/1903.09848)
4. Improving Neural Machine Translation Models with Monolingual Data [arXiv](https://arxiv.org/abs/1511.06709)
