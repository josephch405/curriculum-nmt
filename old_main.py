from functools import partial

import glob
import itertools
import os
import time

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

import torch
import torch.optim as optim
import torch.nn as nn

from torchnlp.samplers import BucketBatchSampler
from torchnlp.datasets import iwslt_dataset
from torchnlp.encoders.text import WhitespaceEncoder
from torchnlp.encoders import LabelEncoder
from torchnlp import word_to_vector

# from model import RNNEncoder, RNNDecoder
# from util import get_args, makedirs, collate_fn

# args = get_args()

#if args.gpu >= 0:
#  torch.cuda.set_device(args.gpu)

# load dataset
train, dev, test = iwslt_dataset(language_extensions=['fr', 'en'], train=True, dev=True, test=True)
'''
src_key = 'en'
tar_key = 'de'

# Preprocess
for row in itertools.chain(train, dev, test):
  row[src_key] = row[src_key].lower()
  row[tar_key] = row[tar_key].lower()

# Make Encoders
src_corpus = [row[src_key] for row in itertools.chain(train, dev, test)]
src_encoder = WhitespaceEncoder(src_corpus)

tar_corpus = [row[tar_key] for row in itertools.chain(train, dev, test)]
tar_encoder = WhitespaceEncoder(tar_corpus)

# Encode
for row in itertools.chain(train, dev, test):
  row[src_key] = src_encoder.encode(row[src_key])
  row[tar_key] = tar_encoder.encode(row[tar_key])

# DONE UP TO HERE

config = args
config.n_embed = sentence_encoder.vocab_size
config.d_out = label_encoder.vocab_size
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
  config.n_cells *= 2

if args.resume_snapshot:
  model = torch.load(
    args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
  model = SNLIClassifier(config)
  if args.word_vectors:
    # Load word vectors
    word_vectors = word_to_vector.aliases[args.word_vectors]()
    for i, token in enumerate(sentence_encoder.vocab):
      model.embed.weight.data[i] = word_vectors[token]

  if args.gpu >= 0:
    model.cuda()

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'
    .split(','))
log_template = ' '.join(
    '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

for epoch in range(args.epochs):
  n_correct, n_total = 0, 0

  train_sampler = SequentialSampler(train)
  train_batch_sampler = BucketBatchSampler(
      train_sampler, args.batch_size, True, sort_key=lambda r: len(row['premise']))
  train_iterator = DataLoader(
      train,
      batch_sampler=train_batch_sampler,
      collate_fn=collate_fn,
      pin_memory=torch.cuda.is_available(),
      num_workers=0)
  for batch_idx, (premise_batch, hypothesis_batch, label_batch) in enumerate(train_iterator):

    # switch model to training mode, clear gradient accumulators
    model.train()
    torch.set_grad_enabled(True)
    opt.zero_grad()

    iterations += 1

    # forward pass
    answer = model(premise_batch, hypothesis_batch)

    # calculate accuracy of predictions in the current batch
    n_correct += (torch.max(answer, 1)
                  [1].view(label_batch.size()) == label_batch).sum()
    n_total += premise_batch.size()[1]
    train_acc = 100. * n_correct / n_total

    # calculate loss of the network output with respect to training labels
    loss = criterion(answer, label_batch)

    # backpropagate and update optimizer learning rate
    loss.backward()
    opt.step()

    # checkpoint model periodically
    if iterations % args.save_every == 0:
      snapshot_prefix = os.path.join(args.save_path, 'snapshot')
      snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(
          train_acc, loss.item(), iterations)
      torch.save(model, snapshot_path)
      for f in glob.glob(snapshot_prefix + '*'):
        if f != snapshot_path:
          os.remove(f)

    # evaluate performance on validation set periodically
    if iterations % args.dev_every == 0:

      # switch model to evaluation mode
      model.eval()
      torch.set_grad_enabled(False)

      # calculate accuracy on validation set
      n_dev_correct, dev_loss = 0, 0

      # dev_sampler = SequentialSampler(train)
      dev_batch_sampler = BucketBatchSampler(
          dev, args.batch_size, True, sort_key=lambda r: len(row['premise']))
      dev_iterator = DataLoader(
          dev,
          batch_sampler=dev_batch_sampler,
          collate_fn=partial(collate_fn, train=False),
          pin_memory=torch.cuda.is_available(),
          num_workers=0)
      for dev_batch_idx, (premise_batch, hypothesis_batch,
                          label_batch) in enumerate(dev_iterator):
        answer = model(premise_batch, hypothesis_batch)
        n_dev_correct += (torch.max(answer,
                                    1)[1].view(label_batch.size()) == label_batch).sum()
        dev_loss = criterion(answer, label_batch)
      dev_acc = 100. * n_dev_correct / len(dev)

      print(
        dev_log_template.format(time.time() - start, epoch, iterations, 1 + batch_idx,
                                  len(train_sampler),
                                  100. * (1 + batch_idx) /
                                  len(train_sampler), loss.item(),
                                  dev_loss.item(), train_acc, dev_acc))

      # update best validation set accuracy
      if dev_acc > best_dev_acc:

        # found a model with better validation set accuracy

        best_dev_acc = dev_acc
        snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
        snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(
            dev_acc, dev_loss.item(), iterations)

        # save model, delete previous 'best_snapshot' files
        torch.save(model, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)

    elif iterations % args.log_every == 0:

        # print progress message
      print(
          log_template.format(time.time() - start, epoch, iterations, 1 + batch_idx,
                              len(train_sampler), 100. *
                              (1 + batch_idx) / len(train_sampler),
                              loss.item(), ' ' * 8, n_correct / n_total * 100, ' ' * 12))
                              '''
