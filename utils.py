import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')
from tqdm import tqdm

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)

    lens = map(lambda words: len(words), sents)
    max_len = max(lens)
    for sent in sents:
        _sent = sent[:]
        _sent += [pad_token] * (max_len - len(_sent))
        sents_padded += [_sent]

    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, space_tokenize=False, dev_mode=False):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param space_tokenize (bool): Whether to tokenize with just spaces. Useful
        for BPE input
    @param dev_mode (bool): Only reads first 100 lines; for fast iteration
    """
    data = []
    i = 0
    for line in tqdm(open(file_path)):
        sent = nltk.word_tokenize(line) if not space_tokenize else line.strip().split()
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)
        # TODO: nicer iteration dev flag
        i += 1
        if i > 100 and dev_mode:
            break
    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def get_pacing_batch(data, batch_size, shuffle=False):
    """ Returns (not yields) a single batch of source and target sentences
    @param data (list of (src_sent, tgt_sent)): list of tuples containing src and tgt sents
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    index_array = list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    indices = index_array[:batch_size]
    examples = [data[idx] for idx in indices]
    examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
    src_sents = [e[0] for e in examples]
    tgt_sents = [e[1] for e in examples]
    
    return src_sents, tgt_sents
    

