from torchnlp.datasets import wmt_dataset

train = wmt_dataset(train=True)

# https://pytorch.org/docs/master/nn.html?highlight=nn%20transformer#torch.nn.Transformer
# https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html
# https://github.com/PetrochukM/PyTorch-NLP/blob/master/examples/snli/train.py